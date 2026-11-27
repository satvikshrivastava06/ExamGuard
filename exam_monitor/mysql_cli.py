"""Shared CLI helpers for live MySQL sync (--mysql) and session reporting."""
from __future__ import annotations

import argparse
import getpass
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .accuracy_report import AccuracyReport, compute_accuracy
from .database import (
    BufferedDetectionStore,
    DBConfig,
    DatabaseManager,
    DatabaseSink,
    SummaryStore,
    compute_accuracy_from_db,
    fetch_detection_confidences,
)
from .event_publisher import BufferedPublisher, EventBus, SinkPublisher
from .events import EventSink, JsonlSink


def add_mysql_args(parser: argparse.ArgumentParser) -> None:
    """Attach --mysql and connection flags to any CLI parser."""
    grp = parser.add_argument_group("MySQL live sync")
    grp.add_argument("--mysql", action="store_true",
                     help="Sync events, per-frame detections/confidence, and session summaries to MySQL in real time")
    grp.add_argument("--mysql-host", default=os.getenv("MYSQL_HOST", "localhost"),
                     help="MySQL host (default: localhost)")
    grp.add_argument("--mysql-port", type=int, default=int(os.getenv("MYSQL_PORT", "3307")),
                     help="MySQL port (default: 3307 or MYSQL_PORT env var)")
    grp.add_argument("--mysql-user", default=os.getenv("MYSQL_USER", "root"),
                     help="MySQL user (default: root)")
    grp.add_argument("--mysql-password", default=None,
                     help="MySQL password (defaults to MYSQL_PASSWORD env var; prompts if needed)")
    grp.add_argument("--mysql-database", default=os.getenv("MYSQL_DATABASE", "exam_monitor"),
                     help="MySQL database name (default: exam_monitor)")


def connect_mysql_from_args(args: argparse.Namespace) -> DatabaseManager:
    """Connect to MySQL using CLI args, auto-creating database & tables if needed."""
    password = args.mysql_password if args.mysql_password is not None else os.getenv("MYSQL_PASSWORD", None)
    tried_prompt = False

    if password is None:
        password = ""
    else:
        tried_prompt = True

    while True:
        cfg = DBConfig(
            host=args.mysql_host,
            port=args.mysql_port,
            user=args.mysql_user,
            password=password,
            database=args.mysql_database,
        )
        mgr = DatabaseManager(cfg)
        try:
            mgr.setup_schema()
            return mgr
        except Exception as exc:
            errno = getattr(exc, "errno", None)
            if errno == 1045 and not tried_prompt:
                tried_prompt = True
                try:
                    password = getpass.getpass(
                        f"Enter MySQL password for '{args.mysql_user}'@{args.mysql_host}:{args.mysql_port}: "
                    )
                    continue
                except (EOFError, KeyboardInterrupt):
                    raise SystemExit("MySQL connection aborted.")
            raise SystemExit(
                f"Cannot connect to MySQL at {args.mysql_host}:{args.mysql_port} "
                f"(user={args.mysql_user}, db={args.mysql_database}): {exc}"
            )


@dataclass
class MysqlSyncContext:
    """Holds live MySQL writers for a monitoring session."""
    db_manager: DatabaseManager
    detection_writer: BufferedDetectionStore
    summary_store: SummaryStore

    def make_sink(self, log_path: Path) -> EventSink:
        """Create an EventBus writing to both local JSONL and buffered MySQL DatabaseSink."""
        jsonl = JsonlSink(log_path)
        db_sink = DatabaseSink(self.db_manager)
        bus = EventBus(primary=jsonl)
        bus.subscribe(BufferedPublisher(SinkPublisher(db_sink, name="mysql"), name="mysql"), name="mysql")
        return bus

    def finalize_session(self, session_id: str, log_path: Optional[Path] = None) -> Optional[AccuracyReport]:
        """Flush buffered detections, compute full report (with DB confidence), and save summary."""
        self.detection_writer.close(timeout=5.0)
        report: Optional[AccuracyReport] = None
        if log_path and log_path.exists():
            try:
                report = compute_accuracy(log_path)
            except Exception:
                report = None
        if report is None:
            try:
                report = compute_accuracy_from_db(session_id, self.db_manager)
            except Exception:
                return None

        # Enrich confidence samples with per-frame detections stored in MySQL
        try:
            frame_confs = fetch_detection_confidences(session_id, self.db_manager)
            if len(frame_confs) > len(report.confidence_samples):
                report.confidence_samples = frame_confs
        except Exception:
            pass

        self.summary_store.save_summary(report)
        return report


def start_mysql_sync(args: argparse.Namespace) -> Optional[MysqlSyncContext]:
    """Create a MysqlSyncContext if --mysql was passed, else return None."""
    if not getattr(args, "mysql", False):
        return None
    mgr = connect_mysql_from_args(args)
    return MysqlSyncContext(
        db_manager=mgr,
        detection_writer=BufferedDetectionStore(mgr),
        summary_store=SummaryStore(mgr),
    )
