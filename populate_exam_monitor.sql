-- ============================================================
-- Exam Monitor Full Schema & Data Script for MySQL Workbench
-- Run this entire script in MySQL Workbench (click the ⚡ icon)
-- ============================================================
CREATE DATABASE IF NOT EXISTS exam_monitor CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE exam_monitor;

-- 1. Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    session_id VARCHAR(64) PRIMARY KEY,
    start_time DATETIME(6),
    end_time DATETIME(6),
    final_state VARCHAR(32)
) ENGINE=InnoDB;

-- 2. Events table
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

-- 3. Detections table
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

-- 4. Session Summaries table (accuracy, confidence, camera availability, data sufficiency)
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

-- 5. Insert Sessions & Events Data
INSERT INTO sessions (session_id, start_time) VALUES ('exam_session_001', '2026-10-18 18:51:38.451') ON DUPLICATE KEY UPDATE session_id=session_id;
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('exam_session_001', 'SESSION_STARTED', '2026-10-18 18:51:38.451', 23016.666366, NULL, '{}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('exam_session_001', 'FACE_PRESENT', '2026-10-18 18:51:42.280', 23020.4913144, 2, '{"n_faces": 1, "max_confidence": 3.858, "detector": "haar", "previous_state": null, "previous_duration_s": null}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('exam_session_001', 'FACE_ABSENT', '2026-10-18 18:52:36.939', 23075.1481195, 1092, '{"n_faces": 0, "max_confidence": 0.0, "detector": "haar", "previous_state": "FACE_PRESENT", "previous_duration_s": 54.657}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('exam_session_001', 'FACE_PRESENT', '2026-10-18 18:52:43.799', 23082.0099609, 1230, '{"n_faces": 1, "max_confidence": 4.025, "detector": "haar", "previous_state": "FACE_ABSENT", "previous_duration_s": 6.862}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('exam_session_001', 'SESSION_ENDED', '2026-10-18 18:53:57.169', 23155.3850708, NULL, '{"final_state": "FACE_PRESENT"}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('exam_session_001', 'SESSION_STARTED', '2026-11-23 15:45:52.744', 30588.2133089, NULL, '{}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('exam_session_001', 'FACE_ABSENT', '2026-11-23 15:45:57.726', 30593.1920717, 68, '{"n_faces": 0, "max_confidence": 0.0, "detector": "haar", "previous_state": null, "previous_duration_s": null}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('exam_session_001', 'SESSION_ENDED', '2026-11-23 15:46:04.311', 30599.7806902, NULL, '{"final_state": "FACE_ABSENT"}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('exam_session_001', 'SESSION_STARTED', '2026-11-23 15:47:45.573', 30701.0423691, NULL, '{}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('exam_session_001', 'SESSION_ENDED', '2026-11-23 15:47:49.472', 30704.9421183, NULL, '{"final_state": null}');
INSERT INTO sessions (session_id, start_time) VALUES ('EXAM-2026-S0001', '2026-10-18 19:31:27.096') ON DUPLICATE KEY UPDATE session_id=session_id;
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'SESSION_STARTED', '2026-10-18 19:31:27.096', 25405.3119215, NULL, '{}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_PRESENT', '2026-10-18 19:31:32.753', 25410.9574353, 48, '{"n_faces": 1, "max_confidence": 3.912, "detector": "haar", "previous_state": null, "previous_duration_s": null}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'SESSION_ENDED', '2026-10-18 19:32:22.397', 25460.6130477, NULL, '{"final_state": "FACE_PRESENT"}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'SESSION_STARTED', '2026-10-21 20:31:53.851', 15121.2157024, NULL, '{}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_PRESENT', '2026-10-21 20:31:59.526', 15126.8800002, 48, '{"n_faces": 1, "max_confidence": 3.262, "detector": "haar", "previous_state": null, "previous_duration_s": null}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_ABSENT', '2026-10-21 20:32:05.658', 15133.0138406, 172, '{"n_faces": 0, "max_confidence": 0.0, "detector": "haar", "previous_state": "FACE_PRESENT", "previous_duration_s": 6.134}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_PRESENT', '2026-10-21 20:32:09.451', 15136.8023434, 248, '{"n_faces": 1, "max_confidence": 5.181, "detector": "haar", "previous_state": "FACE_ABSENT", "previous_duration_s": 3.789}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_ABSENT', '2026-10-21 20:32:19.193', 15146.5464278, 444, '{"n_faces": 0, "max_confidence": 0.0, "detector": "haar", "previous_state": "FACE_PRESENT", "previous_duration_s": 9.744}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_PRESENT', '2026-10-21 20:32:29.941', 15157.2971706, 660, '{"n_faces": 1, "max_confidence": 4.226, "detector": "haar", "previous_state": "FACE_ABSENT", "previous_duration_s": 10.751}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_ABSENT', '2026-10-21 20:33:02.476', 15189.8283387, 1314, '{"n_faces": 0, "max_confidence": 0.0, "detector": "haar", "previous_state": "FACE_PRESENT", "previous_duration_s": 32.531}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_PRESENT', '2026-10-21 20:33:25.867', 15213.2209104, 1784, '{"n_faces": 1, "max_confidence": 3.478, "detector": "haar", "previous_state": "FACE_ABSENT", "previous_duration_s": 23.393}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_ABSENT', '2026-10-21 20:33:34.318', 15221.6740557, 1954, '{"n_faces": 0, "max_confidence": 0.0, "detector": "haar", "previous_state": "FACE_PRESENT", "previous_duration_s": 8.453}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_PRESENT', '2026-10-21 20:34:25.262', 15272.6181349, 2978, '{"n_faces": 1, "max_confidence": 6.663, "detector": "haar", "previous_state": "FACE_ABSENT", "previous_duration_s": 50.944}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_ABSENT', '2026-10-21 20:35:05.068', 15312.4239755, 3778, '{"n_faces": 0, "max_confidence": 0.0, "detector": "haar", "previous_state": "FACE_PRESENT", "previous_duration_s": 39.806}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_PRESENT', '2026-10-21 20:35:08.144', 15315.4975836, 3840, '{"n_faces": 1, "max_confidence": 3.581, "detector": "haar", "previous_state": "FACE_ABSENT", "previous_duration_s": 3.074}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_ABSENT', '2026-10-21 20:35:32.740', 15340.0897326, 4334, '{"n_faces": 0, "max_confidence": 0.0, "detector": "haar", "previous_state": "FACE_PRESENT", "previous_duration_s": 24.592}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_PRESENT', '2026-10-21 20:35:50.057', 15357.405072, 4682, '{"n_faces": 1, "max_confidence": 5.236, "detector": "haar", "previous_state": "FACE_ABSENT", "previous_duration_s": 17.315}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_ABSENT', '2026-10-21 20:36:01.679', 15369.0357145, 4916, '{"n_faces": 0, "max_confidence": 0.0, "detector": "haar", "previous_state": "FACE_PRESENT", "previous_duration_s": 11.631}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_PRESENT', '2026-10-21 20:36:05.858', 15373.2100057, 5000, '{"n_faces": 1, "max_confidence": 2.913, "detector": "haar", "previous_state": "FACE_ABSENT", "previous_duration_s": 4.174}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_ABSENT', '2026-10-21 20:36:13.925', 15381.2754911, 5162, '{"n_faces": 0, "max_confidence": 0.0, "detector": "haar", "previous_state": "FACE_PRESENT", "previous_duration_s": 8.065}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_PRESENT', '2026-10-21 20:36:24.469', 15391.8200653, 5374, '{"n_faces": 1, "max_confidence": 8.349, "detector": "haar", "previous_state": "FACE_ABSENT", "previous_duration_s": 10.545}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_ABSENT', '2026-10-21 20:36:33.923', 15401.2757373, 5564, '{"n_faces": 0, "max_confidence": 0.0, "detector": "haar", "previous_state": "FACE_PRESENT", "previous_duration_s": 9.456}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_PRESENT', '2026-10-21 20:36:36.212', 15403.5643549, 5610, '{"n_faces": 1, "max_confidence": 6.641, "detector": "haar", "previous_state": "FACE_ABSENT", "previous_duration_s": 2.289}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_ABSENT', '2026-10-21 20:36:56.715', 15424.068126, 6022, '{"n_faces": 0, "max_confidence": 0.0, "detector": "haar", "previous_state": "FACE_PRESENT", "previous_duration_s": 20.504}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_PRESENT', '2026-10-21 20:36:58.595', 15425.9482211, 6060, '{"n_faces": 1, "max_confidence": 11.741, "detector": "haar", "previous_state": "FACE_ABSENT", "previous_duration_s": 1.88}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_ABSENT', '2026-10-21 20:37:12.337', 15439.6930585, 6384, '{"n_faces": 0, "max_confidence": 0.0, "detector": "haar", "previous_state": "FACE_PRESENT", "previous_duration_s": 13.745}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_PRESENT', '2026-10-21 20:37:28.517', 15455.8702301, 6868, '{"n_faces": 1, "max_confidence": 7.544, "detector": "haar", "previous_state": "FACE_ABSENT", "previous_duration_s": 16.177}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_ABSENT', '2026-10-21 20:37:31.666', 15459.0215656, 6962, '{"n_faces": 0, "max_confidence": 0.0, "detector": "haar", "previous_state": "FACE_PRESENT", "previous_duration_s": 3.151}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_PRESENT', '2026-10-21 20:37:42.770', 15470.1184392, 7232, '{"n_faces": 1, "max_confidence": 7.169, "detector": "haar", "previous_state": "FACE_ABSENT", "previous_duration_s": 11.097}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_ABSENT', '2026-10-21 20:37:57.428', 15484.7827, 7568, '{"n_faces": 0, "max_confidence": 0.0, "detector": "haar", "previous_state": "FACE_PRESENT", "previous_duration_s": 14.664}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_PRESENT', '2026-10-21 20:38:24.760', 15512.1119971, 8206, '{"n_faces": 1, "max_confidence": 6.262, "detector": "haar", "previous_state": "FACE_ABSENT", "previous_duration_s": 27.329}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_ABSENT', '2026-10-21 20:38:39.669', 15527.0241311, 8520, '{"n_faces": 0, "max_confidence": 0.0, "detector": "haar", "previous_state": "FACE_PRESENT", "previous_duration_s": 14.912}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_PRESENT', '2026-10-21 20:38:50.266', 15537.6173801, 8788, '{"n_faces": 1, "max_confidence": 9.65, "detector": "haar", "previous_state": "FACE_ABSENT", "previous_duration_s": 10.593}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_ABSENT', '2026-10-21 20:39:13.234', 15560.5912882, 9362, '{"n_faces": 0, "max_confidence": 0.0, "detector": "haar", "previous_state": "FACE_PRESENT", "previous_duration_s": 22.974}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_PRESENT', '2026-10-21 20:41:02.125', 15669.4780935, 11588, '{"n_faces": 1, "max_confidence": 2.565, "detector": "haar", "previous_state": "FACE_ABSENT", "previous_duration_s": 108.887}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_ABSENT', '2026-10-21 20:41:20.331', 15687.6865279, 11954, '{"n_faces": 0, "max_confidence": 0.0, "detector": "haar", "previous_state": "FACE_PRESENT", "previous_duration_s": 18.208}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_PRESENT', '2026-10-21 20:41:29.180', 15696.5345509, 12132, '{"n_faces": 1, "max_confidence": 9.323, "detector": "haar", "previous_state": "FACE_ABSENT", "previous_duration_s": 8.848}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_ABSENT', '2026-10-21 20:41:48.399', 15715.7516135, 12516, '{"n_faces": 0, "max_confidence": 0.0, "detector": "haar", "previous_state": "FACE_PRESENT", "previous_duration_s": 19.217}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_PRESENT', '2026-10-21 20:41:51.765', 15719.1228313, 12584, '{"n_faces": 1, "max_confidence": 2.757, "detector": "haar", "previous_state": "FACE_ABSENT", "previous_duration_s": 3.371}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'SESSION_STARTED', '2026-10-30 16:20:43.499', 3210.5776157, NULL, '{}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'SESSION_INTERRUPTED', '2026-10-30 16:20:48.803', 3215.8769983, 2, '{"gap_s": 2.161}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_ABSENT', '2026-10-30 16:20:52.000', 3219.0647275, 34, '{"n_faces": 0, "max_confidence": 0.0, "detector": "haar", "previous_state": null, "previous_duration_s": null}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'SESSION_ENDED', '2026-10-30 16:20:58.243', 3225.3224947, NULL, '{"final_state": "FACE_ABSENT"}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'SESSION_STARTED', '2026-10-30 16:23:17.404', 3364.4833685, NULL, '{}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_ABSENT', '2026-10-30 16:23:23.096', 3370.1574388, 14, '{"n_faces": 0, "max_confidence": 0.0, "detector": "haar", "previous_state": null, "previous_duration_s": null}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'SESSION_ENDED', '2026-10-30 16:23:25.880', 3372.9593271, NULL, '{"final_state": "FACE_ABSENT"}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'SESSION_STARTED', '2026-10-30 16:23:48.078', 3395.1574354, NULL, '{}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'DETECTION_FAILURE', '2026-10-30 16:23:55.907', 3402.9854264, 48, '{"reason": "too_dark_or_lens_covered", "previous_state": null, "previous_duration_s": null}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_ABSENT', '2026-10-30 16:24:08.396', 3415.4646891, 174, '{"n_faces": 0, "max_confidence": 0.0, "detector": "haar", "previous_state": "DETECTION_FAILURE", "previous_duration_s": 12.479}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'DETECTION_FAILURE', '2026-10-30 16:24:21.857', 3428.9355538, 310, '{"reason": "too_dark_or_lens_covered", "previous_state": "FACE_ABSENT", "previous_duration_s": 13.471}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'SESSION_ENDED', '2026-10-30 16:24:32.278', 3439.3572651, NULL, '{"final_state": "DETECTION_FAILURE"}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'SESSION_STARTED', '2026-11-03 12:18:36.980', 555.0333541, NULL, '{}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'SESSION_INTERRUPTED', '2026-11-03 12:18:41.048', 559.0958842, 2, '{"gap_s": 2.025}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_PRESENT', '2026-11-03 12:18:41.580', 559.6261063, 18, '{"n_faces": 1, "max_confidence": 8.551, "detector": "haar", "previous_state": null, "previous_duration_s": null}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_ABSENT', '2026-11-03 12:20:39.210', 677.2040267, 2350, '{"n_faces": 0, "max_confidence": 0.0, "detector": "haar", "previous_state": "FACE_PRESENT", "previous_duration_s": 117.578}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_PRESENT', '2026-11-03 12:20:48.930', 686.9545521, 2546, '{"n_faces": 1, "max_confidence": 3.941, "detector": "haar", "previous_state": "FACE_ABSENT", "previous_duration_s": 9.751}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'SESSION_ENDED', '2026-11-03 12:21:16.713', 714.766325, NULL, '{"final_state": "FACE_PRESENT"}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'SESSION_STARTED', '2026-11-03 12:34:28.083', 1506.1364382, NULL, '{}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_PRESENT', '2026-11-03 12:34:31.884', 1509.9369185, 2, '{"n_faces": 1, "max_confidence": 10.581, "detector": "haar", "previous_state": null, "previous_duration_s": null}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'SESSION_ENDED', '2026-11-03 12:35:03.505', 1541.5582835, NULL, '{"final_state": "FACE_PRESENT"}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'SESSION_STARTED', '2026-11-03 18:13:11.658+05:30', 2029.7112678, NULL, '{}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'FACE_PRESENT', '2026-11-03 18:13:15.522+05:30', 2033.5746561, 2, '{"n_faces": 1, "max_confidence": 7.863, "detector": "haar", "previous_state": null, "previous_duration_s": null}');
INSERT INTO events (session_id, type, ts_utc, ts_mono, frame_id, details) VALUES ('EXAM-2026-S0001', 'SESSION_ENDED', '2026-11-03 18:13:24.149+05:30', 2042.2024572, NULL, '{"final_state": "FACE_PRESENT"}');

-- 6. Queries to verify
SELECT * FROM sessions;
SELECT session_id, type, ts_utc, frame_id, details FROM events ORDER BY id ASC LIMIT 50;