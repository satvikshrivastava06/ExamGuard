import threading
import time

import numpy as np
import pytest

from exam_monitor.camera_probe import (
    CameraProblem as P, OsCheck, PermissionState as S, check_camera, hint_for, inspect_os,
)
from exam_monitor.config import Config

CFG = Config()


# ------------------------------------------------------------------ Linux
def test_linux_node_missing_is_no_device():
    r = inspect_os(0, "linux", exists=lambda p: False, access=lambda p, m: True)
    assert r.device_present is False and not r.blocked


def test_linux_node_not_accessible_is_permission_denied():
    r = inspect_os(0, "linux", exists=lambda p: True, access=lambda p, m: False)
    assert r.permission == S.DENIED and r.blocked and "/dev/video0" in r.detail


def test_linux_accessible_is_granted():
    r = inspect_os(2, "linux", exists=lambda p: True, access=lambda p, m: True)
    assert r.permission == S.GRANTED and "/dev/video2" in r.detail


# ------------------------------------------------------------------ Windows (registry faked)
BASE = r"SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam"


def reg(values):
    return lambda hive, path, name: values.get((hive, path, name))


@pytest.mark.parametrize("key", [
    ("HKCU", BASE, "Value"), ("HKCU", BASE + r"\NonPackaged", "Value"),
    ("HKLM", BASE, "Value"), ("HKLM", BASE + r"\NonPackaged", "Value"),
])
def test_windows_any_deny_layer_blocks(key):
    r = inspect_os(0, "win32", reg_reader=reg({key: "Deny"}))
    assert r.permission == S.DENIED


def test_windows_policies_block():
    r = inspect_os(0, "win32", reg_reader=reg({("HKLM", r"SOFTWARE\Policies\Microsoft\Windows\AppPrivacy",
                                                "LetAppsAccessCamera"): 2}))
    assert r.permission == S.DENIED and "policy" in r.detail
    r = inspect_os(0, "win32", reg_reader=reg({("HKLM", r"SOFTWARE\Policies\Microsoft\Camera", "AllowCamera"): 0}))
    assert r.permission == S.DENIED


def test_windows_allow_or_missing_is_granted():
    assert inspect_os(0, "win32", reg_reader=reg({})).permission == S.GRANTED
    allow = {("HKCU", BASE, "Value"): "Allow", ("HKCU", BASE + r"\NonPackaged", "Value"): "Allow"}
    assert inspect_os(0, "win32", reg_reader=reg(allow)).permission == S.GRANTED


def test_windows_value_comparison_is_case_and_space_insensitive():
    r = inspect_os(0, "win32", reg_reader=reg({("HKCU", BASE, "Value"): " deny "}))
    assert r.permission == S.DENIED


# ------------------------------------------------------------------ macOS (status faked)
@pytest.mark.parametrize("code,state,blocked", [(0, S.NOT_DETERMINED, True), (1, S.DENIED, True),
                                                (2, S.DENIED, True), (3, S.GRANTED, False)])
def test_macos_status_mapping(code, state, blocked):
    r = inspect_os(0, "darwin", mac_status=lambda: code)
    assert r.permission == state and r.blocked is blocked


def test_macos_without_pyobjc_is_unknown_not_guessed():
    r = inspect_os(0, "darwin", mac_status=lambda: None)
    assert r.permission == S.UNKNOWN and not r.blocked


def test_unknown_platform_and_non_camera_source():
    assert inspect_os(0, "freebsd").permission == S.UNKNOWN
    assert inspect_os("nope.mp4", "linux", exists=lambda p: False).device_present is False


# ------------------------------------------------------------------ check_camera with fakes
class FakeCam:
    def __init__(self, frames=None, opened=True, hang=None):
        self._frames, self.is_open, self._hang, self.released = list(frames or []), opened, hang, False

    def read(self):
        if self._hang:
            self._hang.wait(3)
        return (True, self._frames.pop(0)) if self._frames else (False, None)

    def release(self):
        self.released = True


GREY = np.full((120, 160, 3), 120, np.uint8)
BLACK = np.zeros((120, 160, 3), np.uint8)
OS_OK = lambda src, plat=None: OsCheck(S.GRANTED, True, "ok")


def run(cam, inspect=OS_OK, **kw):
    return check_camera(0, CFG, inspect=inspect, camera_factory=lambda s, c: cam, platform="linux", **kw)


def test_good_camera_ok_and_released():
    cam = FakeCam([GREY] * 8)
    res = run(cam)
    assert res.ok and res.problem == P.NONE and cam.released


def test_permission_denied_short_circuits_without_opening():
    def boom(*a):
        raise AssertionError("camera must not be opened when the OS already denies it")
    res = check_camera(0, CFG, inspect=lambda s, p=None: OsCheck(S.DENIED, True, "denied"),
                       camera_factory=boom, platform="win32")
    assert res.problem == P.PERMISSION_DENIED and not res.ok
    assert "Privacy & security" in res.hint            # Windows-specific guidance


def test_macos_not_determined_reported_as_permission_problem():
    res = check_camera(0, CFG, inspect=lambda s, p=None: OsCheck(S.NOT_DETERMINED, None, "not granted yet"),
                       camera_factory=lambda s, c: FakeCam(), platform="darwin")
    assert res.problem == P.PERMISSION_DENIED and res.permission == S.NOT_DETERMINED
    assert "System Settings" in res.hint


def test_missing_device_node():
    res = run(FakeCam(), inspect=lambda s, p=None: OsCheck(S.UNKNOWN, False, "/dev/video0 does not exist"))
    assert res.problem == P.NO_DEVICE


def test_cannot_open_is_unavailable_not_guessed():
    assert run(FakeCam(opened=False)).problem == P.UNAVAILABLE


def test_opened_but_no_frames_is_unavailable():
    assert run(FakeCam([])).problem == P.UNAVAILABLE


def test_all_black_frames():
    res = run(FakeCam([BLACK] * 8))
    assert res.problem == P.BLACK_FRAMES and not res.ok


def test_dark_warmup_frames_then_light_is_ok():
    assert run(FakeCam([BLACK] * 3 + [GREY] * 5)).ok     # exposure settling must not fail the check


def test_hung_open_times_out_promptly():
    gate = threading.Event()
    t0 = time.monotonic()
    res = run(FakeCam([GREY] * 8, hang=gate), open_timeout_s=0.3)
    gate.set()
    assert res.problem == P.OPEN_TIMEOUT and time.monotonic() - t0 < 1.5


def test_factory_exception_is_reported_not_raised():
    def boom(s, c):
        raise RuntimeError("driver exploded")
    res = check_camera(0, CFG, inspect=OS_OK, camera_factory=boom, platform="linux")
    assert res.problem == P.UNAVAILABLE and "driver exploded" in res.detail


def test_file_source_missing_and_present(tmp_path):
    assert check_camera(str(tmp_path / "x.mp4"), CFG, platform="linux").problem == P.NO_DEVICE
    import cv2
    img = tmp_path / "ok.png"
    cv2.imwrite(str(img), GREY)
    assert check_camera(str(img), CFG, platform="linux").ok


def test_hints_exist_for_every_problem_and_are_platform_specific():
    for p in (P.PERMISSION_DENIED, P.NO_DEVICE, P.BLACK_FRAMES, P.OPEN_TIMEOUT, P.UNAVAILABLE):
        assert hint_for(p, "linux")
    assert "video" in hint_for(P.PERMISSION_DENIED, "linux")
    assert "System Settings" in hint_for(P.PERMISSION_DENIED, "darwin")
    assert hint_for(P.NONE, "linux") == ""


def test_to_dict_is_json_ready():
    import json
    json.dumps(run(FakeCam([GREY] * 8)).to_dict())
