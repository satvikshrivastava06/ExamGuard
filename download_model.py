"""Download the YuNet face detector (ONNX, ~230 KB) used by --detector yunet."""
import sys
import urllib.request
from pathlib import Path

URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
       "face_detection_yunet_2023mar.onnx")
DEST = Path("models/face_detection_yunet_2023mar.onnx")

DEST.parent.mkdir(exist_ok=True)
try:
    urllib.request.urlretrieve(URL, DEST)
except Exception as exc:
    sys.exit(f"Download failed: {exc}\nFetch it manually from {URL} into {DEST}")
print(f"saved {DEST} ({DEST.stat().st_size} bytes)")
