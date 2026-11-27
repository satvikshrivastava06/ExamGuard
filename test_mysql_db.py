"""Test script for MySQL database integration.

This script demonstrates:
1. Setting up the MySQL database connection
2. Creating the schema (tables)
3. Writing test events and detections
4. Querying the stored data to verify it works

Prerequisites:
- MySQL server running (localhost:3306 by default)
- Database 'exam_monitor' created
- User 'root' with appropriate permissions (or update DBConfig)
"""

import time
from datetime import datetime, timezone

from exam_monitor.config import Config
from exam_monitor.database import DBConfig, DatabaseManager, DatabaseSink, DetectionStore
from exam_monitor.events import Event, EventType, JsonlSink
from exam_monitor.session_binding import MonitoringRegistry
from exam_monitor.detector import Detection


def test_database_connection():
    """Test basic database connection and schema creation.
    Automatically tries port 3306 first, then 3307 if that fails."""
    print("=" * 60)
    print("TEST 1: Database Connection & Schema Creation")
    print("=" * 60)

    # Try ports in order: 3306 (default), then 3307 (common alternative)
    ports_to_try = [3306, 3307]
    db_manager = None

    # Allow password via environment variable or default to empty
    import os
    password = os.getenv("MYSQL_PASSWORD", "")

    for port in ports_to_try:
        db_config = DBConfig(
            host="localhost",
            port=port,
            user="root",
            password=password,
            database="exam_monitor",
            pool_name="test_pool",
            pool_size=3
        )

        try:
            db_manager = DatabaseManager(db_config)
            db_manager.setup_schema()
            print(f"[OK] Database connection successful on port {port}")
            print(f"[OK] Schema created/verified (tables: sessions, events, detections)")
            return db_manager
        except Exception as e:
            print(f"  Port {port}: {e}")

    # If all ports failed
    print(f"\n[FAIL] Database connection failed on all ports: {ports_to_try}")
    print("\nTroubleshooting:")
    print("1. Ensure MySQL server is running")
    print("2. Create database: CREATE DATABASE exam_monitor;")
    print("3. Set MYSQL_PASSWORD env var if root has a password:")
    print("   $env:MYSQL_PASSWORD='your_password'  (PowerShell)")
    print("   export MYSQL_PASSWORD='your_password'  (bash)")
    print("4. Check user permissions: GRANT ALL PRIVILEGES ON exam_monitor.* TO 'root'@'localhost';")
    return None


def test_event_storage(db_manager):
    """Test storing events via DatabaseSink."""
    print("\n" + "=" * 60)
    print("TEST 2: Event Storage (DatabaseSink)")
    print("=" * 60)

    sink = DatabaseSink(db_manager)
    session_id = f"TEST-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    # Create test events
    now = time.monotonic()
    utc_now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    events = [
        Event(
            type=EventType.SESSION_STARTED,
            session_id=session_id,
            ts_utc=utc_now,
            ts_mono=now,
            frame_id=None,
            details={}
        ),
        Event(
            type=EventType.FACE_PRESENT,
            session_id=session_id,
            ts_utc=utc_now,
            ts_mono=now + 1.0,
            frame_id=10,
            details={"face_count": 1}
        ),
        Event(
            type=EventType.MULTIPLE_FACES,
            session_id=session_id,
            ts_utc=utc_now,
            ts_mono=now + 2.0,
            frame_id=20,
            details={"face_count": 2, "previous_state": "FACE_PRESENT"}
        ),
        Event(
            type=EventType.FACE_ABSENT,
            session_id=session_id,
            ts_utc=utc_now,
            ts_mono=now + 5.0,
            frame_id=50,
            details={"previous_state": "MULTIPLE_FACES", "previous_duration_s": 3.0}
        ),
        Event(
            type=EventType.PROLONGED_ABSENCE,
            session_id=session_id,
            ts_utc=utc_now,
            ts_mono=now + 15.0,
            frame_id=150,
            details={"duration_s": 10.0, "started_at_mono": now + 5.0}
        ),
        Event(
            type=EventType.SESSION_ENDED,
            session_id=session_id,
            ts_utc=utc_now,
            ts_mono=now + 20.0,
            frame_id=None,
            details={"final_state": "FACE_ABSENT"}
        ),
    ]

    print(f"Writing {len(events)} events for session: {session_id}")
    for ev in events:
        sink.write(ev)
        print(f"  [OK] {ev.type.value} (frame={ev.frame_id})")

    return session_id


def test_detection_storage(db_manager, session_id):
    """Test storing raw detections via DetectionStore."""
    print("\n" + "=" * 60)
    print("TEST 3: Raw Detection Storage (DetectionStore)")
    print("=" * 60)

    store = DetectionStore(db_manager)

    # Create mock detections (box = x, y, w, h)
    detections_frame_10 = [
        Detection(box=(100, 100, 80, 80), confidence=0.95, detector_type="haar"),
        Detection(box=(300, 120, 70, 70), confidence=0.88, detector_type="haar"),
    ]

    detections_frame_20 = [
        Detection(box=(150, 110, 90, 90), confidence=0.92, detector_type="haar"),
        Detection(box=(280, 130, 75, 75), confidence=0.85, detector_type="haar"),
        Detection(box=(400, 100, 60, 60), confidence=0.78, detector_type="haar"),
    ]

    print(f"Storing detections for session: {session_id}")
    store.store_detections(session_id, 10, detections_frame_10)
    print(f"  [OK] Frame 10: {len(detections_frame_10)} detections")

    store.store_detections(session_id, 20, detections_frame_20)
    print(f"  [OK] Frame 20: {len(detections_frame_20)} detections")

    # Test empty detections (should be skipped)
    store.store_detections(session_id, 30, [])
    print(f"  [OK] Frame 30: 0 detections (skipped)")


def query_stored_data(db_manager, session_id):
    """Query and display the stored data."""
    print("\n" + "=" * 60)
    print("TEST 4: Query Stored Data")
    print("=" * 60)

    conn = db_manager.get_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        # Query sessions
        print(f"\n--- SESSIONS (session_id={session_id}) ---")
        cursor.execute("SELECT * FROM sessions WHERE session_id = %s", (session_id,))
        sessions = cursor.fetchall()
        for s in sessions:
            print(f"  session_id: {s['session_id']}")
            print(f"  start_time: {s['start_time']}")
            print(f"  end_time: {s['end_time']}")
            print(f"  final_state: {s['final_state']}")

        # Query events
        print(f"\n--- EVENTS (session_id={session_id}) ---")
        cursor.execute("""
            SELECT id, type, ts_utc, ts_mono, frame_id, details
            FROM events
            WHERE session_id = %s
            ORDER BY id
        """, (session_id,))
        events = cursor.fetchall()
        for e in events:
            print(f"  id={e['id']} | {e['type']:20s} | frame={e['frame_id']} | ts_mono={e['ts_mono']:.2f} | details={e['details']}")

        # Query detections
        print(f"\n--- DETECTIONS (session_id={session_id}) ---")
        cursor.execute("""
            SELECT id, frame_id, box_x, box_y, box_w, box_h, confidence
            FROM detections
            WHERE session_id = %s
            ORDER BY frame_id, id
        """, (session_id,))
        detections = cursor.fetchall()
        for d in detections:
            print(f"  id={d['id']} | frame={d['frame_id']} | box=({d['box_x']},{d['box_y']},{d['box_w']},{d['box_h']}) | conf={d['confidence']:.2f}")

        # Summary counts
        print(f"\n--- SUMMARY ---")
        cursor.execute("SELECT COUNT(*) as cnt FROM sessions WHERE session_id = %s", (session_id,))
        print(f"  Sessions: {cursor.fetchone()['cnt']}")
        cursor.execute("SELECT COUNT(*) as cnt FROM events WHERE session_id = %s", (session_id,))
        print(f"  Events: {cursor.fetchone()['cnt']}")
        cursor.execute("SELECT COUNT(*) as cnt FROM detections WHERE session_id = %s", (session_id,))
        print(f"  Detections: {cursor.fetchone()['cnt']}")

    finally:
        cursor.close()
        conn.close()


def test_full_integration():
    """Test full integration with MonitoringRegistry and DatabaseSink."""
    print("\n" + "=" * 60)
    print("TEST 5: Full Integration (MonitoringRegistry + DatabaseSink)")
    print("=" * 60)

    # Try ports in order: 3306 (default), then 3307 (common alternative)
    ports_to_try = [3306, 3307]
    db_manager = None

    import os
    password = os.getenv("MYSQL_PASSWORD", "")

    for port in ports_to_try:
        db_config = DBConfig(
            host="localhost",
            port=port,
            user="root",
            password=password,
            database="exam_monitor",
            pool_name="integration_pool",
            pool_size=3
        )

        try:
            db_manager = DatabaseManager(db_config)
            db_manager.setup_schema()
            print(f"[OK] Database connection successful on port {port}")
            break
        except Exception as e:
            print(f"  Port {port}: {e}")

    if db_manager is None:
        print(f"\n[FAIL] Database connection failed on all ports: {ports_to_try}")
        return

    # Create a sink factory that uses DatabaseSink
    def sink_factory(session_id: str):
        return DatabaseSink(db_manager)

    cfg = Config()
    registry = MonitoringRegistry(cfg, sink_factory)

    session_id = f"INTEGRATION-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    # Bind a monitor (this writes SESSION_STARTED)
    monitor = registry.bind(session_id)
    print(f"[OK] Session bound: {session_id}")

    # Simulate some observations
    now = time.monotonic()
    monitor.observe(EventType.FACE_PRESENT, now, frame_id=100, details={"face_count": 1})
    print(f"  [OK] Observed FACE_PRESENT")

    monitor.observe(EventType.FACE_ABSENT, now + 5.0, frame_id=150, details={})
    print(f"  [OK] Observed FACE_ABSENT")

    # Release (writes SESSION_ENDED)
    final_status = registry.release(session_id)
    print(f"[OK] Session released, final state: {final_status.state}")

    # Query the data
    query_stored_data(db_manager, session_id)


def show_mysql_commands():
    """Show useful MySQL commands for manual inspection."""
    print("\n" + "=" * 60)
    print("USEFUL MYSQL COMMANDS FOR MANUAL INSPECTION")
    print("=" * 60)

    commands = [
        ("Connect to MySQL (port 3306)", "mysql -u root -p -P 3306"),
        ("Connect to MySQL (port 3307)", "mysql -u root -p -P 3307"),
        ("Create database", "CREATE DATABASE exam_monitor CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"),
        ("Show tables", "USE exam_monitor; SHOW TABLES;"),
        ("Describe sessions table", "DESCRIBE sessions;"),
        ("Describe events table", "DESCRIBE events;"),
        ("Describe detections table", "DESCRIBE detections;"),
        ("View all sessions", "SELECT * FROM sessions;"),
        ("View events for a session", "SELECT * FROM events WHERE session_id = 'YOUR_SESSION_ID' ORDER BY id;"),
        ("View detections for a session", "SELECT * FROM detections WHERE session_id = 'YOUR_SESSION_ID' ORDER BY frame_id, id;"),
        ("Count events per session", "SELECT session_id, COUNT(*) as event_count FROM events GROUP BY session_id;"),
        ("Count detections per session", "SELECT session_id, COUNT(*) as detection_count FROM detections GROUP BY session_id;"),
        ("View latest session events with JSON details", "SELECT id, type, ts_utc, frame_id, JSON_PRETTY(details) FROM events ORDER BY id DESC LIMIT 20;"),
    ]

    for desc, cmd in commands:
        print(f"\n-- {desc}")
        print(f"   {cmd}")


def main():
    print("MySQL Database Integration Test for Exam Monitor")
    print("=" * 60)

    # Run tests
    db_manager = test_database_connection()

    if db_manager is None:
        print("\n[ERROR] Cannot proceed without database connection.")
        show_mysql_commands()
        return

    # Test event storage
    session_id = test_event_storage(db_manager)

    # Test detection storage
    test_detection_storage(db_manager, session_id)

    # Query stored data
    query_stored_data(db_manager, session_id)

    # Test full integration
    test_full_integration()

    # Show useful commands
    show_mysql_commands()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()