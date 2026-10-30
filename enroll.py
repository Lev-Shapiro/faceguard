import os
import glob
import cv2
import numpy as np

BASE_DIR = os.path.expanduser("~/.faceguard")
detector = cv2.FaceDetectorYN.create(os.path.join(BASE_DIR, "yunet.onnx"), "", (320, 320))
recognizer = cv2.FaceRecognizerSF.create(os.path.join(BASE_DIR, "sface.onnx"), "")

embeddings = []
image_paths = glob.glob(os.path.join(BASE_DIR, "me", "*.*"))

for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        continue
    
    h, w, _ = img.shape
    detector.setInputSize((w, h))
    _, faces = detector.detect(img)
    
    if faces is not None and len(faces) > 0:
        # Align face using detected eye/nose landmarks and extract 128-D vector
        aligned_face = recognizer.alignCrop(img, faces[0])
        feat = recognizer.feature(aligned_face)
        # L2 normalize the feature vector
        feat = feat / np.linalg.norm(feat)
        embeddings.append(feat.flatten())
        print(f"[OK] Enrolled: {os.path.basename(path)}")
    else:
        print(f"[SKIP] No face found in: {os.path.basename(path)}")

if not embeddings:
    raise RuntimeError("No faces detected in ~/.faceguard/me/!")

# Compute the centroid (mean) embedding and re-normalize it
centroid = np.mean(embeddings, axis=0)
centroid = centroid / np.linalg.norm(centroid)

np.save(os.path.join(BASE_DIR, "my_face.npy"), centroid)
print(f"\nSuccess! Master embedding saved from {len(embeddings)} photos.")
