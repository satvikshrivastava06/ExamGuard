# Sprint 2 (Team 2): webcam unavailable and permission-denied handling

Code: `exam_monitor/camera_probe.py`, wiring in `exam_monitor/monitoring_service.py`, command `check_camera.py`.

## Behaviour
| Moment | What happens |
|---|---|
| Before a session (UI) | `service.preflight_camera(source)` or `python check_camera.py` opens the camera briefly and returns `CameraCheck(ok, problem, permission, detail, hint)` |
| `start_monitoring(..., require_camera=True)` | Preflight runs first. If unusable: `CameraCheckFailed` (carries the `CameraCheck`); no session bound, no log, no thread |
| `start_monitoring(...)` default | Session starts regardless; the problem is recorded as events, so a camera that fails mid-exam and one that never worked are handled the same way |
| Camera down while monitoring | Worker keeps running, re-reads OS state every `camera_diag_interval_s` (2 s), emits `CAMERA_PERMISSION_DENIED` or `CAMERA_UNAVAILABLE` with `problem / permission / detail / hint` |
| Fixed mid-session | Worker's existing reopen logic reconnects; the next confirmed state (`FACE_PRESENT` etc.) is the recovery event |

Problems: `no_device`, `permission_denied`, `unavailable`, `black_frames`, `open_timeout`.
Permission states: `granted`, `denied`, `not_determined`, `unknown`.

## What each OS lets us detect
| OS | Signal | Result |
|---|---|---|
| Linux | `/dev/video<N>` missing | `no_device` |
| Linux | node present, no read/write access (user not in `video` group) | `permission_denied` |
| Windows | `ConsentStore\webcam` `Value` = `Deny` under HKCU / HKLM, also under `NonPackaged` (desktop apps); policy `LetAppsAccessCamera` = 2; policy `AllowCamera` = 0 | `permission_denied` |
| macOS | AVFoundation authorization status 0 (not determined) / 1 (restricted) / 2 (denied), via optional `pyobjc-framework-AVFoundation` | `permission_denied` |
| macOS without pyobjc | cannot read status; a denied app receives only black frames | `black_frames` in preflight; at runtime the worker reports permission-denied only if the OS check says blocked, otherwise `DETECTION_FAILURE` |
| Any OS | camera in use by another app, driver fault, unplugged (Windows/macOS) | `unavailable`. OpenCV gives no reason for a failed open, so none is invented |
| Any OS | open/read hangs | `open_timeout` (preflight, helper thread, 5 s default) |

Design rules: permission is only reported when the OS positively says so; a dark frame is only re-labelled
permission-denied when the OS also says blocked (otherwise a covered lens would be mislabelled); good frames always win over a stale "deny" reading.

## Not verified here
- Windows registry reads and macOS pyobjc calls have only been tested through injected fakes (the sandbox is Linux, no camera, runs as root, so a real `EACCES` could not be produced).
- Windows 11 is adding per-desktop-app camera switches in Insider builds; this probe reads the global/desktop-app switches, so a per-app deny there would show as `unavailable` or `black_frames`, not `permission_denied`.
- Linux index-to-node mapping assumes OpenCV index N is `/dev/videoN`.
- Linux sandboxed apps (Snap/Flatpak) use portal permissions that are not visible through file access.

## Sources (retrieved and read while building this)
- Windows camera privacy keys (`HKCU/HKLM ...\CapabilityAccessManager\ConsentStore\webcam`, `NonPackaged`, values Allow/Deny): https://www.elevenforum.com/t/enable-or-disable-apps-access-to-camera-in-windows-11.17140/ and https://geekrewind.com/?p=53150 (both community guides, not Microsoft docs)
- Windows policy keys (`LetAppsAccessCamera`, `AllowCamera`): https://r-pufky.github.io/docs/operating-systems/windows/10/20H2/settings/privacy/camera.html
- Windows 11 per-desktop-app camera control (Insider): https://www.elevenforum.com/t/enable-or-disable-apps-access-to-camera-in-windows-11.17140/post-329581
- Apple: denied or undecided permission yields only black video frames: https://developer.apple.com/tutorials/data/documentation/avfoundation/avcapturedevice/authorizationstatus(for:).md
- AVAuthorizationStatus values 0 not determined, 1 restricted, 2 denied, 3 authorized: https://docs.rs/crate/crabcamera/0.9.2/source/src/permissions.rs
- macOS permission belongs to the launching app (a bare CLI inherits its host app's grant): https://docs.rs/crate/moq-video/latest/source/src/capture/avfoundation.rs
- Linux V4L2: user needs read access to `/dev/video*`, usually via the `video` group: https://docs.pjsip.org/en/latest/specific-guides/video-troubleshooting/problems/invalid_video_device.html
- Linux "Permission denied" on `/dev/video0`, root:video, add user to group and re-login: https://bbs.archlinux.org/viewtopic.php?id=193009
- OpenCV VideoCapture on Linux without root, video group: https://forum.opencv.org/t/how-to-run-opencv-videocapture-without-root-priviledge-on-linux/18209

The implementation is original to this project; nothing was copied from these pages.
