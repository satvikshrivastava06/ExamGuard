-- Exam Monitor Database Setup Script
-- Open and run this file in MySQL Workbench (or copy-paste into a Query tab and click the Lightning icon)

-- 1. Create the database schema
CREATE DATABASE IF NOT EXISTS exam_monitor 
  CHARACTER SET utf8mb4 
  COLLATE utf8mb4_unicode_ci;

USE exam_monitor;

-- 2. Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    session_id VARCHAR(64) PRIMARY KEY,
    start_time DATETIME(6),
    end_time DATETIME(6),
    final_state VARCHAR(32)
) ENGINE=InnoDB;

-- 3. Events table
CREATE TABLE IF NOT EXISTS events (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(64),
    type VARCHAR(32),
    ts_utc DATETIME(6),
    ts_mono DOUBLE,
    frame_id INT,
    details JSON,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
    INDEX idx_events_session (session_id)
) ENGINE=InnoDB;

-- 4. Detections table (raw bounding boxes)
CREATE TABLE IF NOT EXISTS detections (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(64),
    frame_id INT,
    box_x INT,
    box_y INT,
    box_w INT,
    box_h INT,
    confidence DOUBLE,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
    INDEX idx_detections_session (session_id)
) ENGINE=InnoDB;

-- 5. Session Summaries table (accuracy, confidence, camera availability, data sufficiency)
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

-- Verify creation
SHOW TABLES;
