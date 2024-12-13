import cv2 as cv
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Start Webcam
cap = cv.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert BGR image to RGB
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
    # Perform pose estimation
    results = pose.process(rgb_frame)
    
    # Draw landmarks and connections
    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Display the result
    cv.imshow("Pose Estimation", frame)
    
    # Exit on 'q' key
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
