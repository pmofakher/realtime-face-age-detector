import os
import time
import urllib.request
import cv2
import mediapipe as mp

# 1. Auto-download age detection model files on first run
MODEL_PROTO = "age_deploy.prototxt"
MODEL_WEIGHTS = "age_net.caffemodel"

if not os.path.exists(MODEL_PROTO):
    print("Downloading model structure file (Prototxt)...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt",
        MODEL_PROTO
    )

if not os.path.exists(MODEL_WEIGHTS):
    print("Downloading age neural network weights (~44 MB - please wait)...")
    urllib.request.urlretrieve(
        "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel",
        MODEL_WEIGHTS
    )
    print("Download completed successfully!")

# 2. Load the age detection network into OpenCV's DNN engine
age_net = cv2.dnn.readNet(MODEL_WEIGHTS, MODEL_PROTO)

# Age brackets, midpoints for weighted average, and mean values for normalization
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
AGE_MIDPOINTS = [1, 5, 10, 18, 28, 40, 50, 75]
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# 3. Initialize MediaPipe Face Detection and Webcam
mp_face_detection = mp.solutions.face_detection
cap = cv2.VideoCapture(0)

blur_mode = False  # State flag for privacy face blurring
prev_time = 0      # Timestamp for real-time FPS calculation

with mp_face_detection.FaceDetection(min_detection_confidence=0.6) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            break

        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box

                # Compute tight face box coordinates and clamp to frame boundaries
                x1 = max(0, int(bboxC.xmin * w))
                y1 = max(0, int(bboxC.ymin * h))
                x2 = min(w, int((bboxC.xmin + bboxC.width) * w))
                y2 = min(h, int((bboxC.ymin + bboxC.height) * h))

                # Add 25% padding around the face box to improve age model accuracy
                pad_w = int((x2 - x1) * 0.25)
                pad_h = int((y2 - y1) * 0.25)
                crop_x1, crop_y1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
                crop_x2, crop_y2 = min(w, x2 + pad_w), min(h, y2 + pad_h)

                # Crop the padded region for age prediction
                face_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                if face_img.shape[0] > 20 and face_img.shape[1] > 20:
                    # Predict age on the clear, unblurred image first
                    blob = cv2.dnn.blobFromImage(
                        face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False
                    )
                    age_net.setInput(blob)
                    age_preds = age_net.forward()[0]

                    # Calculate weighted average for a smooth, exact age estimate
                    exact_age = int(sum(age_preds * AGE_MIDPOINTS))

                    # Apply Gaussian blur to the tight face box if privacy mode is active
                    if blur_mode and (y2 - y1) > 0 and (x2 - x1) > 0:
                        tight_face = frame[y1:y2, x1:x2]
                        frame[y1:y2, x1:x2] = cv2.GaussianBlur(tight_face, (99, 99), 30)

                    # Draw bounding box and prediction label
                    score = int(detection.score[0] * 100)
                    label = f"Age: ~{exact_age} ({score}%)"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Calculate real-time FPS
        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time)) if prev_time > 0 else 0
        prev_time = curr_time

        # Count active faces and render top-left telemetry dashboard
        face_count = len(results.detections) if results.detections else 0
        status_text = f"Faces: {face_count} | FPS: {fps} | Blur: {'ON' if blur_mode else 'OFF'}"
        cv2.putText(frame, status_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow('Real-Time Face & Age Detection', frame)

        # Handle keyboard shortcuts
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            blur_mode = not blur_mode  # Toggle privacy blur mode
        elif key == ord('s'):
            filename = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved snapshot: {filename}")

cap.release()
cv2.destroyAllWindows()