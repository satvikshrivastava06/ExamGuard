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
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Protocol, Union

import mysql.connector
from mysql.connector import pooling

from .events import Event, EventSink

log = logging.getLogger(__name__)

@dataclass(frozen=True)
class DBConfig:
    """MySQL connection parameters."""
    host: str = "localhost"
    user: str = "root"
    password: str = ""
    database: str = "exam_monitor"
    pool_name: str = "exam_monitor_pool"
    pool_size: int = 5

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
        with self._lock:
            if self._pool is None:
                try:
                    self._pool = pooling.MySQLConnectionPool(
                        pool_name=self.cfg.pool_name,
                        pool_size=self.cfg.pool_size,
                        host=self.cfg.host,
                        user=self.cfg.user,
                        password=self.cfg.password,
                        database=self.cfg.database
                    )
                    log.info("MySQL connection pool initialized (%s)", self.cfg.pool_name)
                except mysql.connector.Error as exc:
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
                except mysql.connector.Error as e:
                    if e.errno == 1061: # Duplicate key name
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
            start_time = None
            if event.type.value == "SESSION_STARTED":
                start_time = event.ts_utc

            cursor.execute(
                "INSERT INTO sessions (session_id, start_time) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE session_id=session_id",
                (event.session_id, start_time)
            )

            # 2. Write the event
            cursor.execute(
                "INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (event.session_id, event.type.value, event.ts_utc, event.ts_mono,
                 event.frame_id, json.dumps(event.details))
            )

            # 3. Update final state if session is ending
            if event.type.value == "SESSION_ENDED":
                final_state = event.details.get("final_state")
                cursor.execute(
                    "UPDATE sessions SET end_time = %s, final_state = %s WHERE session_id = %s",
                    (event.ts_utc, final_state, event.session_id)
                )

            conn.commit()
        except mysql.connector.Error as exc:
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
                (session_id, frame_id, d.box[0], d.box[1], d.box[2], d.box[3], d.confidence)
                for d in detections
            ]

            cursor.executemany(
                "INSERT INTO detections (session_id, frame_id, box_x, box_y, box_w, box_h, confidence) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                data
            )
            conn.commit()
        except mysql.connector.Error as exc:
            log.error("DetectionStore failed to store detections for frame %d: %s", frame_id, exc)
        finally:
            cursor.close()
            conn.close()
