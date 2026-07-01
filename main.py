import cv2
import mediapipe as mp
import pyautogui
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

pyautogui.FAILSAFE = False


# Configuration
SENSITIVITY = 3
BASELINE = 0.4
SMOOTHING = 5
BLINK_THRESHOLD = 0.015

base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()
x_coords, y_coords = [], []

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = detector.detect(mp_image)

    if results.face_landmarks:
        landmarks = results.face_landmarks[0]
        
        # 1. Mouse Movement (Right eye pupil: 468)
        iris_right = landmarks[468]
        x_coords.append(iris_right.x); y_coords.append(iris_right.y)
        if len(x_coords) > SMOOTHING:
            x_coords.pop(0); y_coords.pop(0)
        
        pyautogui.moveTo(int((sum(x_coords)/len(x_coords) - BASELINE) * SENSITIVITY * screen_w), 
                         int((sum(y_coords)/len(y_coords) - BASELINE) * SENSITIVITY * screen_h))

        # 2. Draw Green Dots on BOTH eyes (468 for right, 473 for left)
        cv2.circle(frame, (int(landmarks[468].x * frame_w), int(landmarks[468].y * frame_h)), 5, (0, 255, 0), -1)
        cv2.circle(frame, (int(landmarks[473].x * frame_w), int(landmarks[473].y * frame_h)), 5, (0, 255, 0), -1)

        # 3. Blink Detection
        if abs(landmarks[159].y - landmarks[145].y) < BLINK_THRESHOLD:
            pyautogui.click()
            cv2.waitKey(300) 

    cv2.imshow('Eye Controlled Mouse', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#python3 main.py use this in terminal to run the code