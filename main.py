import os
import urllib.request
import cv2
import mediapipe as mp
import time

# 1. Auto-download the age detection model files (only runs on first execution)
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

# 2. Load the age detection model into OpenCV's DNN engine
age_net = cv2.dnn.readNet(MODEL_WEIGHTS, MODEL_PROTO)

# Standard age range categories of this model and mean color values for image normalization
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# Using the midpoint of each age range to calculate an exact age instead of a range
AGE_MIDPOINTS = [1, 5, 10, 18, 28, 40, 50, 75]
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

mp_face_detection = mp.solutions.face_detection
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(min_detection_confidence=0.6) as face_detection:
    blur_mode = False
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box

                # Compute main face box coordinates (Clamping to frame bounds)
                x1 = max(0, int(bboxC.xmin * w))
                y1 = max(0, int(bboxC.ymin * h))
                x2 = min(w, int((bboxC.xmin + bboxC.width) * w))
                y2 = min(h, int((bboxC.ymin + bboxC.height) * h))

                # Add 25% padding around the face box to improve age model accuracy
                pad_w = int((x2 - x1) * 0.25)
                pad_h = int((y2 - y1) * 0.25)
                crop_x1, crop_y1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
                crop_x2, crop_y2 = min(w, x2 + pad_w), min(h, y2 + pad_h)

                # Crop the padded face region for the age network
                face_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                if face_img.shape[0] > 20 and face_img.shape[1] > 20:
                    # 1. Predict age FIRST on the clear, unblurred image
                    blob = cv2.dnn.blobFromImage(
                        face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False
                    )
                    age_net.setInput(blob)
                    age_preds = age_net.forward()[0]
                    exact_age = int(sum(age_preds * AGE_MIDPOINTS))

                    # 2. Apply Gaussian blur ONLY to the tight face box (matching dimensions)
                    if blur_mode and (y2 - y1) > 0 and (x2 - x1) > 0:
                        tight_face = frame[y1:y2, x1:x2]
                        frame[y1:y2, x1:x2] = cv2.GaussianBlur(tight_face, (99, 99), 30)

                    # 3. Draw bounding box and age label
                    score = int(detection.score[0] * 100)
                    label = f"Age: ~{exact_age} ({score}%)"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
        # Calculating FPS (Processing Speed)
        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time)) if prev_time > 0 else 0
        prev_time = curr_time

        # Real-time count of detected faces
        face_count = len(results.detections) if results.detections else 0

        # Display information in the top-left corner of the image
        cv2.putText(frame, f"Faces: {face_count} | FPS: {fps}", (15, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)            
        cv2.imshow('Production Face & Age Detection', frame)


        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            blur_mode = not blur_mode  #Change the blur mode by pressing the b key.

cap.release()
cv2.destroyAllWindows()