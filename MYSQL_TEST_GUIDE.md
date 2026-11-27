# MySQL Database Testing Guide for Exam Monitor

## Prerequisites

### 1. Install MySQL Server
- **Windows**: Download from [MySQL Installer](https://dev.mysql.com/downloads/installer/)
- **macOS**: `brew install mysql` then `brew services start mysql`
- **Linux**: `sudo apt-get install mysql-server` (Ubuntu/Debian)

### 2. Create Database and User
```sql
-- Connect as root (use -P for port if not 3306)
mysql -u root -p -P 3306    # Default port
# OR
mysql -u root -p -P 3307    # Alternative port

-- Create database
CREATE DATABASE exam_monitor CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- Create a dedicated user (optional but recommended)
CREATE USER 'exam_monitor'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON exam_monitor.* TO 'exam_monitor'@'localhost';
FLUSH PRIVILEGES;
```

### 3. Install Python Dependencies
```bash
cd d:\exam_monitor_cv
.venv\Scripts\activate
pip install -r requirements.txt
```
This installs `mysql-connector-python>=8.0`

---

## Port Configuration

The code supports **both port 3306 (default) and 3307** automatically:

| File | Default Port | Override |
|------|-------------|----------|
| `exam_monitor/database.py` | 3306 | Pass `port=3307` to `DBConfig()` |
| `test_mysql_db.py` | **Auto-detect** (tries 3306, then 3307) | Edit `ports_to_try` list |

---

## Quick Test: Run the Test Script

```bash
cd d:\exam_monitor_cv
.venv\Scripts\python.exe test_mysql_db.py
```

The test script **automatically tries port 3306 first, then 3307** - no config needed!

This will:
1. Connect to MySQL (auto-detects port)
2. Create tables (`sessions`, `events`, `detections`)
3. Insert test events and detections
4. Query and display the stored data

---

## Database Schema Overview

### Tables Created

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `sessions` | Session metadata | `session_id` (PK), `start_time`, `end_time`, `final_state` |
| `events` | Debounced state changes & point events | `id` (PK), `session_id` (FK), `type`, `ts_utc`, `ts_mono`, `frame_id`, `details` (JSON) |
| `detections` | Raw per-frame face detections | `id` (PK), `session_id` (FK), `frame_id`, `box_x`, `box_y`, `box_w`, `box_h`, `confidence` |

### Event Types Stored
- `SESSION_STARTED` - Monitoring began
- `SESSION_ENDED` - Monitoring stopped
- `SESSION_INTERRUPTED` - Gap in observations (system sleep, etc.)
- `FACE_PRESENT` - Single face detected (debounced)
- `FACE_ABSENT` - No face detected (debounced)
- `MULTIPLE_FACES` - More than one face (debounced)
- `DETECTION_UNAVAILABLE` - Cannot judge (dark/blown-out frame)
- `CAMERA_UNAVAILABLE` - Camera not delivering frames
- `CAMERA_PERMISSION_DENIED` - OS blocked camera access
- `PROLONGED_ABSENCE` - Absence exceeded threshold (point event)

---

## How Detection Data Is Stored

### Via DatabaseSink (Events)
Events are written through the `DatabaseSink` which implements `EventSink` protocol:

```python
from exam_monitor.database import DBConfig, DatabaseManager, DatabaseSink
from exam_monitor.events import Event, EventType

db_config = DBConfig(host="localhost", user="root", password="", database="exam_monitor")
db_manager = DatabaseManager(db_config)
db_manager.setup_schema()

sink = DatabaseSink(db_manager)

# Write an event
event = Event(
    type=EventType.FACE_PRESENT,
    session_id="EXAM-2026-S0001",
    ts_utc="2026-11-23T10:30:00.000Z",
    ts_mono=12345.67,
    frame_id=150,
    details={"face_count": 1}
)
sink.write(event)
```

### Via DetectionStore (Raw Detections)
Raw per-frame detections are bulk-inserted for performance:

```python
from exam_monitor.database import DetectionStore
from exam_monitor.detector import Detection

store = DetectionStore(db_manager)

# Store detections for a frame
detections = [
    Detection(box=(100, 100, 80, 80), confidence=0.95, detector_type="haar"),
    Detection(box=(300, 120, 70, 70), confidence=0.88, detector_type="haar"),
]
store.store_detections("EXAM-2026-S0001", frame_id=150, detections=detections)
```

### Automatic Integration with MonitoringService
When using `MonitoringService` with a `DatabaseSink`:

```python
from exam_monitor.config import Config
from exam_monitor.monitoring_service import MonitoringService
from exam_monitor.session_binding import MonitoringRegistry
from exam_monitor.database import DBConfig, DatabaseManager, DatabaseSink

db_config = DBConfig(host="localhost", user="root", password="", database="exam_monitor")
db_manager = DatabaseManager(db_config)
db_manager.setup_schema()

def sink_factory(session_id):
    return DatabaseSink(db_manager)

cfg = Config()
registry = MonitoringRegistry(cfg, sink_factory)
service = MonitoringService(cfg, registry)

# Start monitoring - automatically writes SESSION_STARTED
service.start_monitoring("EXAM-2026-S0001", source=0, detector="haar")

# ... monitoring runs, events written automatically ...

# Stop monitoring - automatically writes SESSION_ENDED
service.stop_monitoring("EXAM-2026-S0001")
```

---

## Querying Stored Data

### Using MySQL CLI
```sql
-- Connect
mysql -u root -p exam_monitor

-- View all sessions
SELECT * FROM sessions;

-- View events for a session
SELECT id, type, ts_utc, frame_id, JSON_PRETTY(details) 
FROM events 
WHERE session_id = 'EXAM-2026-S0001' 
ORDER BY id;

-- View detections for a session
SELECT frame_id, box_x, box_y, box_w, box_h, confidence 
FROM detections 
WHERE session_id = 'EXAM-2026-S0001' 
ORDER BY frame_id, id;

-- Count events per type per session
SELECT session_id, type, COUNT(*) as count
FROM events
GROUP BY session_id, type;

-- Average confidence per session
SELECT session_id, AVG(confidence) as avg_confidence, COUNT(*) as detection_count
FROM detections
GROUP BY session_id;
```

### Using Python
```python
conn = db_manager.get_connection()
cursor = conn.cursor(dictionary=True)

# Get session summary
cursor.execute("""
    SELECT s.session_id, s.start_time, s.end_time, s.final_state,
           COUNT(DISTINCT e.id) as event_count,
           COUNT(DISTINCT d.id) as detection_count
    FROM sessions s
    LEFT JOIN events e ON s.session_id = e.session_id
    LEFT JOIN detections d ON s.session_id = d.session_id
    WHERE s.session_id = %s
    GROUP BY s.session_id
""", (session_id,))

result = cursor.fetchone()
print(result)

cursor.close()
conn.close()
```

---

## Troubleshooting

### Connection Refused
```
mysql.connector.errors.InterfaceError: 2003: Can't connect to MySQL server
```
- Ensure MySQL service is running
- Check host/port in DBConfig (default: localhost:3306)

### Access Denied
```
mysql.connector.errors.ProgrammingError: 1045: Access denied for user 'root'@'localhost'
```
- Check password in DBConfig
- Ensure user has privileges: `GRANT ALL ON exam_monitor.* TO 'root'@'localhost';`

### Database Doesn't Exist
```
mysql.connector.errors.ProgrammingError: 1049: Unknown database 'exam_monitor'
```
- Create database first: `CREATE DATABASE exam_monitor;`

### Table Creation Errors
- MySQL 8.0+ required for `JSON` column type and `DATETIME(6)`
- Use InnoDB engine (default in modern MySQL)

---

## Configuration

Update `DBConfig` in your code or create a config file:

```python
from exam_monitor.database import DBConfig

db_config = DBConfig(
    host="localhost",           # MySQL host
    port=3306,                  # MySQL port (default 3306)
    user="exam_monitor",        # Database user
    password="your_password",   # Database password
    database="exam_monitor",    # Database name
    pool_name="exam_monitor_pool",
    pool_size=5                 # Connection pool size
)
```

For production, use environment variables:
```python
import os
db_config = DBConfig(
    host=os.getenv("MYSQL_HOST", "localhost"),
    user=os.getenv("MYSQL_USER", "root"),
    password=os.getenv("MYSQL_PASSWORD", ""),
    database=os.getenv("MYSQL_DATABASE", "exam_monitor"),
)
```

---

## Example: Full Monitoring Session with MySQL

```python
"""Complete example: Run a monitoring session with MySQL persistence."""
import time
from exam_monitor.config import Config
from exam_monitor.monitoring_service import MonitoringService
from exam_monitor.session_binding import MonitoringRegistry
from exam_monitor.database import DBConfig, DatabaseManager, DatabaseSink

# 1. Setup database
db_config = DBConfig(host="localhost", user="root", password="", database="exam_monitor")
db_manager = DatabaseManager(db_config)
db_manager.setup_schema()

# 2. Create sink factory using DatabaseSink
def sink_factory(session_id):
    return DatabaseSink(db_manager)

# 3. Setup monitoring
cfg = Config()
registry = MonitoringRegistry(cfg, sink_factory)
service = MonitoringService(cfg, registry)

# 4. Start monitoring (writes SESSION_STARTED to MySQL)
session_id = "EXAM-2026-S0001"
service.start_monitoring(session_id, source=0, detector="haar", publish_frames=True)

print(f"Monitoring started for {session_id}")
print("Press Ctrl+C to stop...")

try:
    # 5. Monitor for 30 seconds
    time.sleep(30)
except KeyboardInterrupt:
    pass

# 6. Stop monitoring (writes SESSION_ENDED to MySQL)
result = service.stop_monitoring(session_id)
print(f"Monitoring stopped. Clean: {result.worker_stopped_cleanly}")

# 7. Query results
conn = db_manager.get_connection()
cursor = conn.cursor(dictionary=True)

cursor.execute("SELECT * FROM sessions WHERE session_id = %s", (session_id,))
print("Session:", cursor.fetchone())

cursor.execute("SELECT type, COUNT(*) as count FROM events WHERE session_id = %s GROUP BY type", (session_id,))
print("Events:", cursor.fetchall())

cursor.execute("SELECT COUNT(*) as count FROM detections WHERE session_id = %s", (session_id,))
print("Detections:", cursor.fetchone())

cursor.close()
conn.close()
```

---

## Files in This Project

| File | Purpose |
|------|---------|
| `exam_monitor/database.py` | MySQL integration (DatabaseManager, DatabaseSink, DetectionStore) |
| `test_mysql_db.py` | Test script to verify database works |
| `MYSQL_TEST_GUIDE.md` | This guide |

---

## Next Steps

1. Run `python test_mysql_db.py` to verify setup
2. Integrate `DatabaseSink` into your monitoring pipeline
3. Use `DetectionStore` for high-volume raw detection storage
4. Build dashboards querying the `events` and `detections` tables
5. Set up automated cleanup for old sessions if needed