from ultralytics import YOLO
import cv2
from qa_engine import ApplianceQASystem


def lock_device_from_camera(model_path="best.pt", camera_index=1):
    """
    Use YOLO + webcam to detect a device and lock its class name.
    Press 'l' to lock the current device, 'q' to quit.
    Returns: locked device name as string (e.g. 'Mouse') or None.
    """
    model = YOLO(model_path)
    qa = ApplianceQASystem("kb.csv")

    # Use AVFOUNDATION backend on Mac with camera index 1
    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    locked_device = None
    current_label = None

    print("Opening camera...")
    print("Press 'l' to LOCK the current detected device.")
    print("Press 'q' to quit without locking.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read webcam frame")
            break

        results = model(frame)
        annotated = results[0].plot()

        # Get best detection (highest confidence)
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            confs = boxes.conf.cpu().numpy()
            best_idx = confs.argmax()
            cls_id = int(boxes.cls[best_idx].item())
            current_label = model.names[cls_id]
        else:
            current_label = None

        # Show current label on image
        if current_label is not None:
            text = f"Detected: {current_label} (press 'l' to lock)"
        else:
            text = "No device detected..."
        cv2.putText(
            annotated,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Smart Assistant - Lock Device", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            # quit without locking
            break
        elif key == ord("l"):
            if current_label is not None:
                locked_device = current_label
                print(f"\nLocked device: {locked_device}")
                break
            else:
                print("No device detected to lock yet.")

    cap.release()
    cv2.destroyAllWindows()

    if locked_device is None:
        print("No device locked.")
        return

    # After locking, start Q&A loop in terminal
    print("\nYou can now ask questions about:", locked_device)
    print("Type 'exit' to stop.\n")

    while True:
        question = input(f"Your question about {locked_device}: ")
        if question.strip().lower() in ["exit", "quit", "q"]:
            print("Exiting Q&A.")
            break

        answer = qa.answer(locked_device, question)
        print("Assistant:", answer, "\n")


if __name__ == "__main__":
    # Your working camera index = 1
    lock_device_from_camera(model_path="best.pt", camera_index=1)
