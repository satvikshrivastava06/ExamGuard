"""MySQL database integration for exam monitoring data.

Provides a thread-safe, pooled connection layer for storing session metadata,
monitoring events, and raw face detections.

Schema Design:
-------------
  1. `sessions`: Core session metadata.
     (session_id [PK], start_time, end_time, final_state)
  2. `events`: Debounced state changes and point events.
     (id [PK], session_id [FK], type, ts_utc, ts_mono, frame_id, details [JSON])
  3. `detections`: Raw per-frame detection results.
     (id [PK], session_id [FK], frame_id, box_x, box_y, box_w, box_h, confidence)

Integration:
------------
  - `DatabaseSink` implements `EventSink` protocol; can be used in `EventBus`.
  - `DatabaseManager` handles the connection pool and schema initialization.
  - `DetectionStore` provides optimized bulk-insert for raw detection data.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol, Union

try:
    import mysql.connector
    from mysql.connector import pooling
    _MYSQL_ERROR: Any = mysql.connector.Error
    HAS_MYSQL = True
except ImportError:
    mysql = None  # type: ignore
    pooling = None  # type: ignore
    _MYSQL_ERROR = Exception
    HAS_MYSQL = False

from .events import Event, EventSink

log = logging.getLogger(__name__)

@dataclass(frozen=True)
class DBConfig:
    """MySQL connection parameters."""
    host: str = "localhost"
    port: int = 3306
    user: str = "root"
    password: str = ""
    database: str = "exam_monitor"
    pool_name: str = "exam_monitor_pool"
    pool_size: int = 5

def _to_mysql_datetime(ts_utc: Optional[str]) -> Optional[str]:
    """Convert an ISO-8601 UTC timestamp string to MySQL DATETIME(6) format."""
    if not ts_utc:
        return None
    if isinstance(ts_utc, datetime):
        dt = ts_utc
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    try:
        s = str(ts_utc).strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    except Exception:
        return str(ts_utc)


def ensure_database_exists(cfg: DBConfig) -> None:
    """Create the target database if it does not exist yet."""
    if not HAS_MYSQL:
        raise ImportError(
            "mysql-connector-python is required to use the MySQL database backend. "
            "Install it with: pip install mysql-connector-python"
        )
    conn = mysql.connector.connect(
        host=cfg.host,
        port=cfg.port,
        user=cfg.user,
        password=cfg.password,
    )
    try:
        cursor = conn.cursor()
        safe_db = cfg.database.replace("`", "``")
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS `{safe_db}` "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
        )
        conn.commit()
    finally:
        cursor.close()
        conn.close()


class DatabaseManager:
    """
    Thread-safe manager for MySQL connection pooling and schema setup.

    Singleton-style management of the connection pool to ensure that
    multiple monitoring workers share a limited set of connections.
    """
    def __init__(self, cfg: DBConfig):
        self.cfg = cfg
        self._pool: Optional[pooling.MySQLConnectionPool] = None
        self._lock = threading.Lock()

    def initialize_pool(self) -> None:
        """Creates the connection pool if it doesn't exist."""
        if not HAS_MYSQL:
            raise ImportError(
                "mysql-connector-python is required to use the MySQL database backend. "
                "Install it with: pip install mysql-connector-python"
            )
        with self._lock:
            if self._pool is None:
                try:
                    self._pool = pooling.MySQLConnectionPool(
                        pool_name=self.cfg.pool_name,
                        pool_size=self.cfg.pool_size,
                        host=self.cfg.host,
                        port=self.cfg.port,
                        user=self.cfg.user,
                        password=self.cfg.password,
                        database=self.cfg.database
                    )
                    log.info("MySQL connection pool initialized (%s)", self.cfg.pool_name)
                except _MYSQL_ERROR as exc:
                    if getattr(exc, "errno", None) == 1049:
                        ensure_database_exists(self.cfg)
                        self._pool = pooling.MySQLConnectionPool(
                            pool_name=self.cfg.pool_name,
                            pool_size=self.cfg.pool_size,
                            host=self.cfg.host,
                            port=self.cfg.port,
                            user=self.cfg.user,
                            password=self.cfg.password,
                            database=self.cfg.database
                        )
                        log.info("MySQL database created and pool initialized (%s)", self.cfg.pool_name)
                    else:
                        log.error("Failed to initialize MySQL pool: %s", exc)
                        raise

    def get_connection(self):
        """Retrieves a connection from the pool."""
        if self._pool is None:
            self.initialize_pool()
        return self._pool.get_connection()

    def setup_schema(self) -> None:
        """Initializes the database tables if they do not exist."""
        queries = [
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id VARCHAR(64) PRIMARY KEY,
                start_time DATETIME(6),
                end_time DATETIME(6),
                final_state VARCHAR(32)
            ) ENGINE=InnoDB;
            """,
            """
            CREATE TABLE IF NOT EXISTS events (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(64),
                type VARCHAR(32),
                ts_utc DATETIME(6),
                ts_mono DOUBLE,
                frame_id INT,
                details JSON,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            ) ENGINE=InnoDB;
            """,
            """
            CREATE TABLE IF NOT EXISTS detections (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(64),
                frame_id INT,
                box_x INT,
                box_y INT,
                box_w INT,
                box_h INT,
                confidence DOUBLE,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            ) ENGINE=InnoDB;
            """,
            """
            CREATE TABLE IF NOT EXISTS session_summaries (
                session_id VARCHAR(64) PRIMARY KEY,
                total_monitored_s DOUBLE,
                face_present_s DOUBLE,
                face_absent_s DOUBLE,
                multiple_faces_s DOUBLE,
                detection_unavailable_s DOUBLE,
                presence_ratio DOUBLE,
                camera_availability_ratio DOUBLE,
                detection_coverage_ratio DOUBLE,
                effective_monitoring_ratio DOUBLE,
                avg_confidence DOUBLE,
                p50_confidence DOUBLE,
                p95_confidence DOUBLE,
                max_confidence DOUBLE,
                confidence_samples INT,
                longest_absence_s DOUBLE,
                n_prolonged_absences INT,
                n_camera_outages INT,
                n_detection_failures INT,
                n_interruptions INT,
                is_sufficient BOOLEAN,
                sufficiency_reasons VARCHAR(255),
                summary_json JSON,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            ) ENGINE=InnoDB;
            """,
            "CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);"
        ]

        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            for q in queries:
                # Note: MySQL doesn't support 'CREATE INDEX IF NOT EXISTS' in all versions,
                # using a try-except block for index creation.
                try:
                    cursor.execute(q)
                except _MYSQL_ERROR as e:
                    if getattr(e, 'errno', None) == 1061: # Duplicate key name
                        pass
                    else:
                        raise
            conn.commit()
            log.info("Database schema verified/initialized.")
        finally:
            cursor.close()
            conn.close()

class DatabaseSink:
    """
    MySQL implementation of the EventSink protocol.
    Can be used as a primary sink or a subscriber in an EventBus.
    """
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def write(self, event: Event) -> None:
        """Stores an event in the database. Ensures the session exists first."""
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()

            # 1. Ensure session is registered (upsert session start)
            # We only set start_time if it's a SESSION_STARTED event,
            # otherwise we just ensure the row exists.
            ts_mysql = _to_mysql_datetime(event.ts_utc)
            start_time = ts_mysql if event.type.value == "SESSION_STARTED" else None

            cursor.execute(
                "INSERT INTO sessions (session_id, start_time) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE start_time=COALESCE(start_time, VALUES(start_time))",
                (event.session_id, start_time)
            )

            # 2. Write the event
            cursor.execute(
                "INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (event.session_id, event.type.value, ts_mysql, event.ts_mono,
                 event.frame_id, json.dumps(event.details))
            )

            # 3. Update final state if session is ending
            if event.type.value == "SESSION_ENDED":
                final_state = event.details.get("final_state")
                cursor.execute(
                    "UPDATE sessions SET end_time = %s, final_state = %s WHERE session_id = %s",
                    (ts_mysql, final_state, event.session_id)
                )

            conn.commit()
        except _MYSQL_ERROR as exc:
            log.error("DatabaseSink failed to write event %s: %s", event.type.value, exc)
        finally:
            cursor.close()
            conn.close()

    def close(self) -> None:
        """No-op for pooled connections."""
        pass

class DetectionStore:
    """
    Optimized store for raw face detections.
    Bypasses the EventSink protocol for bulk insertion of per-frame data.
    """
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def store_detections(self, session_id: str, frame_id: int, detections: list) -> None:
        """Bulk insert detections for a single frame."""
        if not detections:
            return

        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()

            # Ensure session exists
            cursor.execute(
                "INSERT INTO sessions (session_id) VALUES (%s) "
                "ON DUPLICATE KEY UPDATE session_id=session_id",
                (session_id,)
            )

            # Prepare bulk data: (session_id, frame_id, x, y, w, h, conf)
            data = [
                (
                    session_id,
                    int(frame_id),
                    int(d.box[0]),
                    int(d.box[1]),
                    int(d.box[2]),
                    int(d.box[3]),
                    float(d.confidence),
                )
                for d in detections
            ]

            cursor.executemany(
                "INSERT INTO detections (session_id, frame_id, box_x, box_y, box_w, box_h, confidence) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                data
            )
            conn.commit()
        except _MYSQL_ERROR as exc:
            log.error("DetectionStore failed to store detections for frame %d: %s", frame_id, exc)
        finally:
            cursor.close()
            conn.close()

    def store_batch(self, items: list[tuple[str, int, Any]]) -> None:
        """Bulk insert detections from multiple frames in a single DB transaction."""
        if not items:
            return

        rows: list[tuple[str, int, int, int, int, int, float]] = []
        session_ids: set[str] = set()
        for session_id, frame_id, detections in items:
            if not detections:
                continue
            session_ids.add(session_id)
            for d in detections:
                rows.append((
                    session_id,
                    int(frame_id),
                    int(d.box[0]),
                    int(d.box[1]),
                    int(d.box[2]),
                    int(d.box[3]),
                    float(d.confidence),
                ))

        if not rows:
            return

        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            for sid in session_ids:
                cursor.execute(
                    "INSERT INTO sessions (session_id) VALUES (%s) "
                    "ON DUPLICATE KEY UPDATE session_id=session_id",
                    (sid,)
                )
            cursor.executemany(
                "INSERT INTO detections (session_id, frame_id, box_x, box_y, box_w, box_h, confidence) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                rows
            )
            conn.commit()
        except _MYSQL_ERROR as exc:
            log.error("DetectionStore failed to store batch (%d rows): %s", len(rows), exc)
        finally:
            cursor.close()
            conn.close()


class BufferedDetectionStore:
    """Non-blocking background writer for per-frame detections and confidence scores."""

    def __init__(self, db_manager: DatabaseManager, max_queue_size: int = 2000, batch_size: int = 32):
        self._store = DetectionStore(db_manager)
        self._batch_size = batch_size
        self._q: queue.Queue[Optional[tuple[str, int, tuple]]] = queue.Queue(maxsize=max_queue_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._drain, daemon=True, name="mysql-detection-writer")
        self._thread.start()

    def enqueue(self, session_id: str, frame_id: int, detections: tuple) -> None:
        """Queue detections from one frame without blocking the CV worker."""
        if not detections or self._stop.is_set():
            return
        try:
            self._q.put_nowait((session_id, frame_id, detections))
        except queue.Full:
            try:
                self._q.get_nowait()
                self._q.put_nowait((session_id, frame_id, detections))
            except queue.Empty:
                pass

    def _drain(self) -> None:
        while True:
            batch: list[tuple[str, int, Any]] = []
            try:
                item = self._q.get(timeout=0.1)
            except queue.Empty:
                if self._stop.is_set():
                    break
                continue
            if item is None:
                while not self._q.empty():
                    try:
                        extra = self._q.get_nowait()
                        if extra is not None:
                            batch.append(extra)
                    except queue.Empty:
                        break
                if batch:
                    self._store.store_batch(batch)
                break
            batch.append(item)
            while len(batch) < self._batch_size:
                try:
                    extra = self._q.get_nowait()
                except queue.Empty:
                    break
                if extra is None:
                    if batch:
                        self._store.store_batch(batch)
                    return
                batch.append(extra)
            self._store.store_batch(batch)

    def close(self, timeout: float = 5.0) -> None:
        """Flush all pending detections and stop the background writer."""
        self._stop.set()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout)
        remaining: list[tuple[str, int, Any]] = []
        while not self._q.empty():
            try:
                item = self._q.get_nowait()
                if item is not None:
                    remaining.append(item)
            except queue.Empty:
                break
        if remaining:
            self._store.store_batch(remaining)


class SummaryStore:
    """Persists full AccuracyReport + DataSufficiency metrics into `session_summaries`."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def save_summary(self, report, criteria=None) -> None:
        """Upsert an AccuracyReport summary for a session."""
        suff = report.sufficiency(criteria)
        reasons = ";".join(r.value for r in suff.reasons)
        payload = report.to_json(indent=None, criteria=criteria)
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO sessions (session_id, start_time, end_time, final_state) "
                "VALUES (%s, %s, %s, %s) "
                "ON DUPLICATE KEY UPDATE "
                "start_time=COALESCE(start_time, VALUES(start_time)), "
                "end_time=COALESCE(VALUES(end_time), end_time), "
                "final_state=COALESCE(VALUES(final_state), final_state)",
                (
                    report.session_id,
                    _to_mysql_datetime(report.start_time_utc),
                    _to_mysql_datetime(report.end_time_utc),
                    report.final_state,
                ),
            )
            cursor.execute(
                """
                INSERT INTO session_summaries (
                    session_id, total_monitored_s, face_present_s, face_absent_s,
                    multiple_faces_s, detection_unavailable_s, presence_ratio,
                    camera_availability_ratio, detection_coverage_ratio, effective_monitoring_ratio,
                    avg_confidence, p50_confidence, p95_confidence, max_confidence,
                    confidence_samples, longest_absence_s, n_prolonged_absences,
                    n_camera_outages, n_detection_failures, n_interruptions,
                    is_sufficient, sufficiency_reasons, summary_json
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s
                )
                ON DUPLICATE KEY UPDATE
                    total_monitored_s=VALUES(total_monitored_s),
                    face_present_s=VALUES(face_present_s),
                    face_absent_s=VALUES(face_absent_s),
                    multiple_faces_s=VALUES(multiple_faces_s),
                    detection_unavailable_s=VALUES(detection_unavailable_s),
                    presence_ratio=VALUES(presence_ratio),
                    camera_availability_ratio=VALUES(camera_availability_ratio),
                    detection_coverage_ratio=VALUES(detection_coverage_ratio),
                    effective_monitoring_ratio=VALUES(effective_monitoring_ratio),
                    avg_confidence=VALUES(avg_confidence),
                    p50_confidence=VALUES(p50_confidence),
                    p95_confidence=VALUES(p95_confidence),
                    max_confidence=VALUES(max_confidence),
                    confidence_samples=VALUES(confidence_samples),
                    longest_absence_s=VALUES(longest_absence_s),
                    n_prolonged_absences=VALUES(n_prolonged_absences),
                    n_camera_outages=VALUES(n_camera_outages),
                    n_detection_failures=VALUES(n_detection_failures),
                    n_interruptions=VALUES(n_interruptions),
                    is_sufficient=VALUES(is_sufficient),
                    sufficiency_reasons=VALUES(sufficiency_reasons),
                    summary_json=VALUES(summary_json)
                """,
                (
                    report.session_id,
                    round(report.total_monitored_s, 3),
                    round(report.face_present_s, 3),
                    round(report.face_absent_s, 3),
                    round(report.multiple_faces_s, 3),
                    round(report.detection_unavailable_s, 3),
                    round(report.presence_ratio, 4),
                    round(report.camera_availability_ratio, 4),
                    round(report.detection_coverage_ratio, 4),
                    round(report.effective_monitoring_ratio, 4),
                    round(report.avg_confidence, 4) if report.avg_confidence is not None else None,
                    round(report.p50_confidence, 4) if report.p50_confidence is not None else None,
                    round(report.p95_confidence, 4) if report.p95_confidence is not None else None,
                    round(report.max_confidence, 4) if report.max_confidence is not None else None,
                    len(report.confidence_samples),
                    round(report.longest_absence_s, 3),
                    report.n_prolonged_absences,
                    report.n_camera_unavailable + report.n_camera_permission_denied,
                    report.n_detection_failure,
                    report.n_interruptions,
                    bool(suff.is_sufficient),
                    reasons,
                    payload,
                ),
            )
            conn.commit()
        except _MYSQL_ERROR as exc:
            log.error("SummaryStore failed to save summary for %s: %s", report.session_id, exc)
        finally:
            cursor.close()
            conn.close()


def fetch_detection_confidences(session_id: str, db_manager: DatabaseManager) -> list[float]:
    """Fetch per-frame max confidence values recorded in `detections` for a session."""
    conn = db_manager.get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT MAX(confidence) FROM detections WHERE session_id = %s GROUP BY frame_id ORDER BY frame_id ASC",
            (session_id,),
        )
        return [float(row[0]) for row in cursor.fetchall() if row[0] is not None and row[0] > 0]
    finally:
        cursor.close()
        conn.close()


def fetch_session_events(session_id: str, db_manager: DatabaseManager) -> list[dict]:
    """Retrieve all logged events for a session ordered by ts_mono."""
    conn = db_manager.get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT type, ts_mono, ts_utc, frame_id, details "
            "FROM events WHERE session_id = %s ORDER BY ts_mono ASC",
            (session_id,)
        )
        rows = cursor.fetchall()
        events = []
        for r in rows:
            details = r[4]
            if isinstance(details, str):
                try:
                    details = json.loads(details)
                except Exception:
                    details = {}
            events.append({
                "session_id": session_id,
                "type": r[0],
                "ts_mono": float(r[1]) if r[1] is not None else 0.0,
                "ts_utc": str(r[2]) if r[2] is not None else "",
                "frame_id": r[3],
                "details": details or {},
            })
        return events
    finally:
        cursor.close()
        conn.close()


def compute_accuracy_from_db(session_id: str, db_manager: DatabaseManager):
    """Compute an AccuracyReport from events stored in MySQL for a session."""
    from .accuracy_report import compute_accuracy_from_events
    events = fetch_session_events(session_id, db_manager)
    if not events:
        raise ValueError(f"No events found in database for session {session_id}")
    return compute_accuracy_from_events(session_id, events)


def export_db_summaries(
    db_manager: DatabaseManager,
    output_path: Union[Path, str],
    fmt: str = "json",
    criteria=None,
) -> Path:
    """Export summaries for all sessions in MySQL to JSON or CSV."""
    from .accuracy_report import export_summaries_csv, export_summaries_json
    conn = db_manager.get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT session_id FROM sessions")
        session_ids = [row[0] for row in cursor.fetchall()]
    finally:
        cursor.close()
        conn.close()

    reports = []
    for sid in session_ids:
        try:
            reports.append(compute_accuracy_from_db(sid, db_manager))
        except Exception as exc:
            log.warning("Could not compute accuracy for session %s from DB: %s", sid, exc)

    if fmt.lower() == "csv":
        return export_summaries_csv(reports, output_path, criteria=criteria)
    return export_summaries_json(reports, output_path, criteria=criteria)
