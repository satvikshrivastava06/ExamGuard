"""Tell "no camera" from "camera blocked", as far as each OS lets us.

What is actually detectable (see docs/camera_unavailable_permission.md for sources):
  Linux    /dev/video<N> missing -> NO_DEVICE; present but not readable/writable -> PERMISSION_DENIED
  Windows  camera privacy switches in the registry (user, desktop-app, machine, policy) -> PERMISSION_DENIED
  macOS    AVFoundation authorization status (needs optional pyobjc-framework-AVFoundation).
           Without it, a denied camera shows up as black frames -> BLACK_FRAMES
  Anything else (busy device, driver fault, unplugged on Windows/macOS) is reported honestly as
  UNAVAILABLE. OpenCV does not give a reason for a failed open, so we never guess one.
"""
from __future__ import annotations

import os
import sys
import threading
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Callable, Optional, Union

from .camera import Camera
from .config import Config
from .preprocess import resize_to_width, to_gray

Source = Union[int, str]


class CameraProblem(str, Enum):
    NONE = "none"
    NO_DEVICE = "no_device"                  # positively known: node/file does not exist
    PERMISSION_DENIED = "permission_denied"  # OS says the app may not use the camera
    UNAVAILABLE = "unavailable"              # could not open or read; cause unknowable (busy, driver, unplugged)
    BLACK_FRAMES = "black_frames"            # opens and streams, but only black
    OPEN_TIMEOUT = "open_timeout"            # open/read did not finish in time (hung driver)


class PermissionState(str, Enum):
    GRANTED = "granted"                      # no blocking setting found (not a guarantee the camera works)
    DENIED = "denied"
    NOT_DETERMINED = "not_determined"        # macOS: user has not answered the prompt yet
    UNKNOWN = "unknown"                      # this OS/environment does not let us tell


@dataclass(frozen=True)
class OsCheck:
    permission: PermissionState
    device_present: Optional[bool]           # None = cannot tell
    detail: str

    @property
    def blocked(self) -> bool:
        return self.permission in (PermissionState.DENIED, PermissionState.NOT_DETERMINED)


@dataclass(frozen=True)
class CameraCheck:
    ok: bool
    problem: CameraProblem
    permission: PermissionState
    detail: str
    hint: str
    platform: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["problem"], d["permission"] = self.problem.value, self.permission.value
        return d


# ----------------------------------------------------------------------------- OS inspection
_WIN_BASE = r"SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam"


def _winreg_reader(hive: str, path: str, name: str):
    import winreg  # Windows only; imported lazily so other platforms never touch it
    root = {"HKCU": winreg.HKEY_CURRENT_USER, "HKLM": winreg.HKEY_LOCAL_MACHINE}[hive]
    try:
        with winreg.OpenKey(root, path) as key:
            return winreg.QueryValueEx(key, name)[0]
    except OSError:
        return None


def _macos_status() -> Optional[int]:
    """AVAuthorizationStatus: 0 not determined, 1 restricted, 2 denied, 3 authorized. None if pyobjc is absent."""
    try:
        import AVFoundation  # type: ignore  (pip install pyobjc-framework-AVFoundation)
        return int(AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(AVFoundation.AVMediaTypeVideo))
    except Exception:
        return None


def _inspect_linux(index: int, exists: Callable[[str], bool], access: Callable[[str, int], bool]) -> OsCheck:
    path = f"/dev/video{index}"
    if not exists(path):
        return OsCheck(PermissionState.UNKNOWN, False, f"{path} does not exist")
    if not access(path, os.R_OK | os.W_OK):
        return OsCheck(PermissionState.DENIED, True, f"no read/write access to {path}")
    return OsCheck(PermissionState.GRANTED, True, f"{path} is accessible")


def _inspect_windows(reader: Callable[[str, str, str], object]) -> OsCheck:
    layers = [
        ("HKCU", _WIN_BASE, "camera access is switched off for this user"),
        ("HKCU", _WIN_BASE + r"\NonPackaged", "desktop-app camera access is switched off for this user"),
        ("HKLM", _WIN_BASE, "camera access is switched off for all users"),
        ("HKLM", _WIN_BASE + r"\NonPackaged", "desktop-app camera access is switched off for all users"),
    ]
    for hive, path, msg in layers:
        value = reader(hive, path, "Value")
        if isinstance(value, str) and value.strip().lower() == "deny":
            return OsCheck(PermissionState.DENIED, None, f"Windows privacy setting: {msg}")
    if reader("HKLM", r"SOFTWARE\Policies\Microsoft\Windows\AppPrivacy", "LetAppsAccessCamera") == 2:
        return OsCheck(PermissionState.DENIED, None, "camera access forced off by policy")
    if reader("HKLM", r"SOFTWARE\Policies\Microsoft\Camera", "AllowCamera") == 0:
        return OsCheck(PermissionState.DENIED, None, "camera disabled by policy")
    return OsCheck(PermissionState.GRANTED, None, "no Windows setting blocks camera access")


def _inspect_macos(status: Callable[[], Optional[int]]) -> OsCheck:
    value = status()
    if value is None:
        return OsCheck(PermissionState.UNKNOWN, None,
                       "macOS permission not readable (install pyobjc-framework-AVFoundation)")
    mapping = {
        0: (PermissionState.NOT_DETERMINED, "camera permission has not been granted yet"),
        1: (PermissionState.DENIED, "camera access is restricted (e.g. parental controls / device management)"),
        2: (PermissionState.DENIED, "camera access was denied"),
        3: (PermissionState.GRANTED, "camera access is authorised"),
    }
    state, detail = mapping.get(value, (PermissionState.UNKNOWN, f"unrecognised authorization status {value}"))
    return OsCheck(state, None, detail)


def inspect_os(source: Source, platform: Optional[str] = None, *,
               exists: Callable[[str], bool] = os.path.exists,
               access: Callable[[str, int], bool] = os.access,
               reg_reader: Callable[[str, str, str], object] = _winreg_reader,
               mac_status: Callable[[], Optional[int]] = _macos_status) -> OsCheck:
    """Cheap, non-opening look at OS-level camera state. Safe to call repeatedly."""
    plat = platform or sys.platform
    if not isinstance(source, int):
        found = exists(source)
        return OsCheck(PermissionState.UNKNOWN, found, "source exists" if found else f"{source!r} not found")
    if plat.startswith("linux"):
        return _inspect_linux(source, exists, access)
    if plat.startswith("win"):
        return _inspect_windows(reg_reader)
    if plat == "darwin":
        return _inspect_macos(mac_status)
    return OsCheck(PermissionState.UNKNOWN, None, f"no permission probe for platform {plat!r}")


# ----------------------------------------------------------------------------- user-facing hints
def hint_for(problem: CameraProblem, platform: str, os_check: Optional[OsCheck] = None) -> str:
    p = platform
    if problem == CameraProblem.PERMISSION_DENIED:
        if p.startswith("win"):
            return ("Open Settings > Privacy & security > Camera. Turn on 'Camera access', "
                    "'Let apps access your camera' and 'Let desktop apps access your camera'. "
                    "If your device is managed by an institution, ask its IT admin.")
        if p == "darwin":
            return ("Open System Settings > Privacy & Security > Camera and enable the app you started "
                    "Python from (Terminal, VS Code, ...), then restart that app.")
        if p.startswith("linux"):
            return "Add your user to the 'video' group (sudo usermod -aG video $USER), then log out and back in."
        return "Allow camera access for this application in your operating system's privacy settings."
    if problem == CameraProblem.NO_DEVICE:
        return "No camera was found. Plug in a webcam, or try another camera index (0, 1, 2)."
    if problem == CameraProblem.BLACK_FRAMES:
        base = "The camera is sending only black. Remove the lens cover or privacy shutter and check the room light."
        return base + (" On macOS this also happens when camera permission is not granted." if p == "darwin" else "")
    if problem == CameraProblem.OPEN_TIMEOUT:
        return "The camera did not respond. Unplug and replug it, close other apps using it, then retry."
    if problem == CameraProblem.UNAVAILABLE:
        return ("The camera could not be opened. Close other apps that may be using it (video calls, "
                "browser tabs), check it is plugged in, or try another camera index.")
    return ""


# ----------------------------------------------------------------------------- full preflight
def _mean_luma(frame) -> float:
    small, _ = resize_to_width(frame, 64)
    return float(to_gray(small).mean())


def check_camera(source: Source, cfg: Config, *, open_timeout_s: float = 5.0, frames: int = 8,
                 platform: Optional[str] = None,
                 inspect: Callable[..., OsCheck] = inspect_os,
                 camera_factory: Callable[..., Camera] = Camera) -> CameraCheck:
    """Preflight: can we get real frames from `source`? Opens the camera briefly, so call it BEFORE
    start_monitoring (or let start_monitoring(require_camera=True) do it), never while a worker runs."""
    plat = platform or sys.platform
    is_cam = isinstance(source, int)
    osc = inspect(source, plat)

    def result(problem: CameraProblem, detail: str) -> CameraCheck:
        return CameraCheck(problem == CameraProblem.NONE, problem, osc.permission, detail,
                           hint_for(problem, plat, osc), plat)

    if is_cam and osc.blocked:
        return result(CameraProblem.PERMISSION_DENIED, osc.detail)
    if osc.device_present is False:
        return result(CameraProblem.NO_DEVICE, osc.detail)

    outcome: dict = {}

    def work() -> None:                      # runs in a helper thread: open() can hang on a bad driver
        cam = None
        try:
            cam = camera_factory(source, cfg)
            outcome["opened"] = cam.is_open
            lumas = []
            for _ in range(frames if cam.is_open else 0):
                ok, frame = cam.read()
                if ok:
                    lumas.append(_mean_luma(frame))
            outcome["lumas"] = lumas
        except Exception as exc:
            outcome["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if cam is not None:
                cam.release()

    t = threading.Thread(target=work, name="camera-preflight", daemon=True)
    t.start()
    t.join(open_timeout_s)
    if t.is_alive():                          # join() returns None either way; is_alive() says if it timed out
        return result(CameraProblem.OPEN_TIMEOUT, f"no result within {open_timeout_s:.1f}s")
    if "error" in outcome:
        return result(CameraProblem.UNAVAILABLE, outcome["error"])
    if not outcome.get("opened"):
        return result(CameraProblem.UNAVAILABLE if is_cam else CameraProblem.NO_DEVICE,
                      "could not open the camera" if is_cam else f"cannot open source {source!r}")
    lumas = outcome.get("lumas", [])
    if not lumas:
        return result(CameraProblem.UNAVAILABLE, "opened, but no frames could be read")
    if max(lumas) < cfg.black_frame_luma:     # max over several frames: webcams start dark while exposure settles
        return result(CameraProblem.BLACK_FRAMES, f"{len(lumas)} frames, brightest mean grey level {max(lumas):.1f}")
    return result(CameraProblem.NONE, f"{len(lumas)} frames read")
