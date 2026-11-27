r"""Import JSONL event logs into MySQL database.

Usage:
  .venv\Scripts\python.exe import_logs_to_mysql.py --password YOUR_PASSWORD
  .venv\Scripts\python.exe import_logs_to_mysql.py --port 3307 --password YOUR_PASSWORD
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path

try:
    import mysql.connector
except ImportError:
    print("[FAIL] mysql-connector-python is required. Run using .venv:")
    print(r"  .venv\Scripts\python.exe import_logs_to_mysql.py")
    sys.exit(1)

from exam_monitor.accuracy_report import compute_accuracy
from exam_monitor.database import DBConfig, DatabaseManager, DatabaseSink, SummaryStore
from exam_monitor.events import Event, EventType


def ensure_database_exists(host: str, port: int, user: str, password: str, database: str = "exam_monitor") -> None:
    """Connects to MySQL server and creates the database if it doesn't exist."""
    conn = mysql.connector.connect(host=host, port=port, user=user, password=password)
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS {database} "
            f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
        )
        conn.commit()
    finally:
        cursor.close()
        conn.close()


def get_db_manager(user: str = "root", password: str = "", port: int = 3307) -> DatabaseManager:
    ports = [port, 3306] if port != 3306 else [3306, 3307]
    last_err = None

    for p in ports:
        try:
            # 1. Ensure database exists
            ensure_database_exists(host="localhost", port=p, user=user, password=password)
            # 2. Connect and setup tables
            cfg = DBConfig(host="localhost", port=p, user=user, password=password, database="exam_monitor")
            mgr = DatabaseManager(cfg)
            conn = mgr.get_connection()
            conn.close()
            mgr.setup_schema()
            return mgr
        except mysql.connector.Error as exc:
            last_err = exc
            if exc.errno == 1045:
                # Wrong password or access denied - trying another port with the same wrong password won't help
                raise exc
            continue
        except Exception as exc:
            last_err = exc
            continue

    if last_err:
        raise last_err
    raise ConnectionError(f"Could not connect to MySQL on ports {ports}.")


def import_log_file(file_path: Path, sink: DatabaseSink) -> int:
    count = 0
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            event = Event(
                type=EventType(raw["type"]),
                session_id=raw["session_id"],
                ts_utc=raw.get("ts_utc") or raw.get("ts_local", ""),
                ts_mono=float(raw.get("ts_mono", 0.0)),
                frame_id=raw.get("frame_id"),
                details=raw.get("details", {})
            )
            sink.write(event)
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Import JSONL logs into MySQL database")
    parser.add_argument("--log", help="Specific JSONL file to import. Defaults to all in logs/")
    parser.add_argument("--user", default=os.getenv("MYSQL_USER", "root"), help="MySQL user (default: root)")
    parser.add_argument("--port", type=int, default=int(os.getenv("MYSQL_PORT", "3307")), help="MySQL port (default: 3307)")
    parser.add_argument("--password", default=None, help="MySQL password (or set MYSQL_PASSWORD env var)")
    args = parser.parse_args()

    password = args.password
    if password is None:
        password = os.getenv("MYSQL_PASSWORD", None)

    # If password is still None, try empty password first; if that fails with 1045, prompt interactively
    if password is None:
        try:
            mgr = get_db_manager(user=args.user, password="", port=args.port)
            print("[OK] Connected to MySQL with empty password.")
        except mysql.connector.Error as exc:
            if exc.errno == 1045:
                # Prompt user for password
                try:
                    password = getpass.getpass(f"Enter MySQL password for user '{args.user}': ")
                except (EOFError, KeyboardInterrupt):
                    print("\nAborted.")
                    return 1
                try:
                    mgr = get_db_manager(user=args.user, password=password, port=args.port)
                except Exception as inner_e:
                    print(f"\n[FAIL] {inner_e}")
                    print("\nTip: Run with: .venv\\Scripts\\python.exe import_logs_to_mysql.py --password \"your_mysql_password\"")
                    return 1
            else:
                print(f"[FAIL] {exc}")
                return 1
        except Exception as e:
            print(f"[FAIL] {e}")
            return 1
    else:
        try:
            mgr = get_db_manager(user=args.user, password=password, port=args.port)
            print("[OK] Connected to MySQL and verified schema.")
        except Exception as e:
            print(f"[FAIL] {e}")
            print("\nTip: Make sure your password is correct, or run:")
            print(r'  .venv\Scripts\python.exe import_logs_to_mysql.py --password "your_password"')
            return 1

    sink = DatabaseSink(mgr)
    summary_store = SummaryStore(mgr)

    if args.log:
        files = [Path(args.log)]
    else:
        files = sorted(Path("logs").glob("*.jsonl"))

    if not files:
        print("No log files found to import.")
        return 0

    total_events = 0
    for f in files:
        if not f.exists():
            continue
        try:
            c = import_log_file(f, sink)
            print(f"Imported {c} events from {f.name}")
            total_events += c
            try:
                report = compute_accuracy(f)
                summary_store.save_summary(report)
                print(f"  Saved session summary for {report.session_id} "
                      f"(avg_conf={report.avg_confidence or 0:.3f}, sufficient={report.is_sufficient()})")
            except Exception as sum_err:
                print(f"  Skipped summary for {f.name}: {sum_err}")
        except Exception as e:
            print(f"Error importing {f.name}: {e}")

    print(f"\n[DONE] Imported total of {total_events} events into exam_monitor database.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
