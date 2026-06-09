import cv2
from ultralytics import YOLO

# Load your trained YOLO model
model = YOLO("best.pt")

# Try webcam index 0 first
cap = cv2.VideoCapture(1)

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to read webcam frame")
        break
    
    results = model(frame)
    annotated = results[0].plot()

    cv2.imshow("YOLO Webcam", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
