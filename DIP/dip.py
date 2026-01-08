import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(img):
    """Detects the first face in the image and returns the grayscale cropped face."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]
    return face
def extract_hog_features(face):
    """Extract HOG (Histogram of Oriented Gradients) features from a face image."""
    face_resized = cv2.resize(face, (64, 64))
    features = hog(face_resized, orientations=9, pixels_per_cell=(8,8),
                   cells_per_block=(2,2), block_norm='L2-Hys', visualize=False)
    return features

known_folder = "known_faces"
X, y = [], []

print("\n🔍 Extracting features from known faces...")

for person_name in os.listdir(known_folder):
    person_folder = os.path.join(known_folder, person_name)
    if not os.path.isdir(person_folder):
        continue  
    for filename in os.listdir(person_folder):
        path = os.path.join(person_folder, filename)
        img = cv2.imread(path)
        if img is None:
            print(f"[ERROR] Cannot load {filename}")
            continue
        
        face = detect_face(img)
        if face is None:
            print(f"[WARNING] No face found in {filename}")
            continue
        
        features = extract_hog_features(face)
        X.append(features)
        y.append(person_name) 

if len(X) == 0:
    raise ValueError("No training faces found. Check 'known_faces/' folder.")

X = np.array(X)
y = np.array(y)
print(f"Loaded {len(X)} training samples for {len(set(y))} people.\n")

clf = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
clf.fit(X, y)
print("SVM training complete!\n")

test_path = "test_photo.jpg"
image = cv2.imread(test_path)
if image is None:
    print(f"[ERROR] Cannot load test image: {test_path}")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

recognized_names = set()
unknown_count = 0
marked_count = 0

os.makedirs("marked_faces", exist_ok=True)
os.makedirs("unrecognized_faces", exist_ok=True)

print("Detecting and recognizing faces...\n")

for (x, y1, w, h) in faces:
    face = gray[y1:y1+h, x:x+w]
    features = extract_hog_features(face).reshape(1, -1)

    pred_name = clf.predict(features)[0]
    pred_prob = clf.predict_proba(features).max()

    if pred_prob > 0.6:
        name = pred_name
        if name not in recognized_names:
            recognized_names.add(name)
            marked_count += 1

            cropped_face = image[y1:y1+h, x:x+w]
            cv2.imwrite(f"marked_faces/{name}_{marked_count}.jpg", cropped_face)
            print(f"Marked Attendance for: {name}")

        color = (0, 255, 0)
        label = f"{name} ({pred_prob:.2f})"
    else:
        unknown_count += 1
        name = "Unknown"
        unknown_face = image[y1:y1+h, x:x+w]
        cv2.imwrite(f"unrecognized_faces/Unknown_{unknown_count}.jpg", unknown_face)
        print(f"Unknown face saved: Unknown_{unknown_count}.jpg")
        color = (0, 0, 255)
        label = "Unknown"

    cv2.rectangle(image, (x, y1), (x+w, y1+h), color, 2)
    cv2.putText(image, label, (x, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

cv2.imwrite("output_with_boxes.jpg", image)
print("\nAnnotated image saved as: output_with_boxes.jpg")
attendance_file = "attendance.csv"
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

if recognized_names:
    df = pd.DataFrame({"Name": list(recognized_names), "Timestamp": [now] * len(recognized_names)})
    if os.path.exists(attendance_file):
        df.to_csv(attendance_file, mode='a', index=False, header=False)
    else:
        df.to_csv(attendance_file, index=False)
    print(f"Attendance saved to {attendance_file}")
else:
    print("No known faces recognized.")
