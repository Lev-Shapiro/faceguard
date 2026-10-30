import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import sys
import glob
import shutil
from datetime import datetime
import cv2
import numpy as np

BASE_DIR = os.path.expanduser("~/.faceguard")
THRESHOLD = 0.363

# Read the image path passed by LidWatcher.app (or fallback to latest_capture.jpg)
img_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, "latest_capture.jpg")
frame = cv2.imread(img_path)

if frame is None:
    sys.exit(1)

detector = cv2.FaceDetectorYN.create(os.path.join(BASE_DIR, "yunet.onnx"), "", (320, 320))
recognizer = cv2.FaceRecognizerSF.create(os.path.join(BASE_DIR, "sface.onnx"), "")
my_embedding = np.load(os.path.join(BASE_DIR, "my_face.npy"))

h, w, _ = frame.shape
detector.setInputSize((w, h))
_, faces = detector.detect(frame)

is_owner = False
max_score = 0.0

if faces is not None:
    for face in faces:
        aligned = recognizer.alignCrop(frame, face)
        feat = recognizer.feature(aligned).flatten()
        feat = feat / np.linalg.norm(feat)
        score = float(np.dot(my_embedding, feat))
        if score > max_score:
            max_score = score
        if score >= THRESHOLD:
            is_owner = True
            break

if not is_owner:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"intruder_{timestamp}_score_{max_score:.2f}.jpg"
    local_path = os.path.join(BASE_DIR, "intruders", filename)
    cv2.imwrite(local_path, frame)

    icloud_dir = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/SecurityAlerts")
    os.makedirs(icloud_dir, exist_ok=True)
    shutil.copy2(local_path, os.path.join(icloud_dir, filename))

    gdrive_mounts = glob.glob(os.path.expanduser("~/Library/CloudStorage/GoogleDrive-*/My Drive"))
    if gdrive_mounts:
        gdrive_dir = os.path.join(gdrive_mounts[0], "SecurityAlerts")
        os.makedirs(gdrive_dir, exist_ok=True)
        shutil.copy2(local_path, os.path.join(gdrive_dir, filename))
