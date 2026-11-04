"""Browser-based frame source for video-meeting exam monitoring.

Launches a real Chromium browser window, navigates to a meeting URL
(Google Meet, Zoom web client, Teams, etc.), and exposes the student's
**own local camera preview** as a frame stream with the same read()
interface as :class:`exam_monitor.camera.Camera`.

The capture mechanism injects a small JavaScript snippet that:
  1. Finds the <video> element fed by the local MediaStream.
  2. Paints each video frame to an off-screen <canvas>.
  3. Serialises the canvas to a base-64 JPEG and hands it to Python via
     Playwright's ``page.expose_function`` bridge.

Requirements
------------
  playwright >= 1.40    (``pip install playwright; playwright install chromium``)
  numpy, opencv-python  (already used by the rest of the pipeline)
"""

from __future__ import annotations

import base64
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Platform adapters – tell the capture script how to find the local preview
# ---------------------------------------------------------------------------

_PLATFORM_HINTS: dict[str, str] = {
    # Google Meet  – local tile is the <video> inside [data-self-name] or jsname="FViaS"
    "meet.google.com": "document.querySelector('video[data-is-muted]') || document.querySelector('video')",
    # Zoom web client – local preview video
    "zoom.us": "document.querySelector('video.self-view-video') || document.querySelector('video')",
    # Microsoft Teams
    "teams.microsoft.com": "document.querySelector('video[data-tid=\"localVideoEl\"]') || document.querySelector('video')",
    # Jitsi / 8x8
    "meet.jit.si": "document.querySelector('#localVideo_container video') || document.querySelector('video')",
    # Fallback for anything else
    "_default": "document.querySelector('video')",
}

# JS injected into the page after the meeting is loaded.
# It polls for the video element and, once found, starts pushing JPEG frames.
_CAPTURE_JS = """
(function startCapture(frameCallback, targetFps) {
    const interval = 1000 / targetFps;
    let _video = null;
    const canvas = document.createElement('canvas');
    const ctx    = canvas.getContext('2d');

    function tryFind() {
        _video = {selector};
        if (!_video || _video.readyState < 2) {
            setTimeout(tryFind, 500);
            return;
        }
        canvas.width  = _video.videoWidth  || 640;
        canvas.height = _video.videoHeight || 480;
        tick();
    }

    function tick() {
        if (!_video || _video.paused || _video.ended) { setTimeout(tick, interval); return; }
        ctx.drawImage(_video, 0, 0, canvas.width, canvas.height);
        const jpeg = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
        frameCallback(jpeg);
        setTimeout(tick, interval);
    }

    tryFind();
})
"""


def _selector_for_url(url: str) -> str:
    for host, sel in _PLATFORM_HINTS.items():
        if host in url and host != "_default":
            return sel
    return _PLATFORM_HINTS["_default"]


# ---------------------------------------------------------------------------
# MeetCapture
# ---------------------------------------------------------------------------

class MeetCapture:
    """Drop-in replacement for :class:`exam_monitor.camera.Camera` that pulls
    frames from a browser-joined meeting instead of a physical webcam.

    Parameters
    ----------
    meet_url:
        The full meeting URL (e.g. ``https://meet.google.com/abc-defg-hij``).
    target_fps:
        How many frames per second to request from the JS injector.
    join_wait_s:
        Seconds to wait after navigation before injecting the capture script,
        giving the meeting UI time to load and request camera permission.
    """

    def __init__(self, meet_url: str, target_fps: int = 15, join_wait_s: float = 8.0):
        self.meet_url = meet_url
        self.target_fps = target_fps
        self.join_wait_s = join_wait_s

        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_count = 0
        self._error: Optional[str] = None
        self._closed = False

        # Camera-compatible attributes expected by pipeline / overlay
        self.consecutive_failures = 0
        self.is_webcam = False   # not a real webcam index – disables throttle logic in preview.py

        # Start the browser in a background thread (Playwright is sync-API here)
        self._thread = threading.Thread(target=self._browser_loop, daemon=True, name="meet-capture")
        self._thread.start()

    # ------------------------------------------------------------------
    # Camera-compatible interface
    # ------------------------------------------------------------------

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return the most recent frame captured from the meeting.

        Mirrors :meth:`exam_monitor.camera.Camera.read`.
        """
        with self._lock:
            if self._error:
                self.consecutive_failures += 1
                return False, None
            if self._latest_frame is None:
                # Still waiting for the first frame – not an error yet
                self.consecutive_failures += 1
                return False, None
            self.consecutive_failures = 0
            return True, self._latest_frame.copy()

    def maybe_reopen(self, _now: float) -> bool:  # noqa: D401
        """No-op – browser reconnect is not implemented; just satisfy the interface."""
        return False

    def release(self) -> None:
        """Signal the browser thread to stop and wait for it."""
        with self._lock:
            self._closed = True
        self._thread.join(timeout=10)

    @property
    def is_open(self) -> bool:
        return self._thread.is_alive() and not self._closed

    # ------------------------------------------------------------------
    # Internal – runs in a daemon thread
    # ------------------------------------------------------------------

    def _on_frame(self, jpeg_b64: str) -> None:
        """Called from JavaScript via Playwright's expose_function bridge."""
        try:
            data = base64.b64decode(jpeg_b64)
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None and frame.size > 0:
                with self._lock:
                    self._latest_frame = frame
                    self._frame_count += 1
        except Exception:
            pass  # never let a bad frame crash the thread

    def _browser_loop(self) -> None:
        try:
            from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
        except ImportError:
            with self._lock:
                self._error = ("playwright not installed – "
                               "run: pip install playwright && playwright install chromium")
            return

        selector = _selector_for_url(self.meet_url)
        # Build the IIFE source with the platform-specific video selector injected
        js_source = _CAPTURE_JS.replace("{selector}", selector)
        js_call   = f"({js_source})(frameCallback, {self.target_fps})"

        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=False,
                args=[
                    # Grant camera/mic automatically so the meeting can start
                    "--use-fake-ui-for-media-stream",
                    # Suppress the "Chrome is being controlled" bar
                    "--disable-infobars",
                    # Disable popup blockers that might intercept the join dialog
                    "--disable-popup-blocking",
                ],
            )
            context = browser.new_context(
                permissions=["camera", "microphone"],
                # Claim to be a normal Chrome so meeting platforms don't block headless detection
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
            )
            page = context.new_page()

            # Expose a Python callback that JS can call with each JPEG frame
            page.expose_function("frameCallback", self._on_frame)

            print(f"[MeetCapture] Navigating to {self.meet_url} …")
            page.goto(self.meet_url, wait_until="domcontentloaded", timeout=30_000)

            print(f"[MeetCapture] Waiting {self.join_wait_s}s for meeting UI to load …")
            time.sleep(self.join_wait_s)

            # Auto-click common "Join now" / "Ask to join" buttons
            _try_join_buttons(page)

            print("[MeetCapture] Injecting frame-capture script …")
            page.evaluate(js_call)
            print("[MeetCapture] Capture running. Close the browser or press q to stop.")

            # Keep the browser alive until release() is called
            while not self._closed:
                time.sleep(0.2)

            context.close()
            browser.close()


def _try_join_buttons(page) -> None:
    """Best-effort: click common 'Join' buttons across platforms."""
    selectors = [
        # Google Meet
        "button[data-idom-class*='join']",
        "button[jsname='Qx7uuf']",          # "Join now"
        "button[jsname='Cg5QOb']",          # "Ask to join"
        # Zoom
        ".zm-btn--primary",
        "#joinBtn",
        # Generic
        "button:has-text('Join now')",
        "button:has-text('Join')",
        "button:has-text('Ask to join')",
    ]
    for sel in selectors:
        try:
            btn = page.query_selector(sel)
            if btn and btn.is_visible():
                btn.click()
                time.sleep(0.5)
                break
        except Exception:
            continue
