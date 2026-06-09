import cv2
import pyttsx3
from ultralytics import YOLO
from qa_engine import ApplianceQASystem
import speech_recognition as sr
import threading


# ----------------------------
# Global Text-to-Speech engine
# ----------------------------
engine = pyttsx3.init()
engine.setProperty("rate", 175)
engine.setProperty("volume", 1.0)


def speak(text: str):
    if not text:
        return
    print(f"Assistant: {text}")
    engine.say(text)
    engine.runAndWait()


# ----------------------------
# Generic speech-to-text helper
# ----------------------------
def stt_listen_once(timeout=5, phrase_time_limit=6):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.5)
        audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
    try:
        text = r.recognize_google(audio)
        print(f"🗣 Heard: {text}")
        return text.lower()
    except sr.UnknownValueError:
        print("❌ Could not understand audio")
        return ""
    except sr.RequestError as e:
        print(f"❌ STT service error: {e}")
        return ""
    except sr.WaitTimeoutError:
        return ""


# ---------------------------------------
# Voice listener for "hey uniq lock this"
# ---------------------------------------
def voice_lock_listener(state):
    """
    Runs in a background thread.
    Listens for phrases like 'hey uniq lock this' and sets state["request"] = True.
    """
    r = sr.Recognizer()
    mic = sr.Microphone()
    print("🎧 Voice lock listener started (say: 'hey uniq, lock this').")

    while not state["stop"]:
        try:
            with mic as source:
                # We don't spam prints every loop, keep it quiet
                r.adjust_for_ambient_noise(source, duration=0.5)
                audio = r.listen(source, timeout=5, phrase_time_limit=5)
        except sr.WaitTimeoutError:
            # No speech, just loop again
            continue

        try:
            text = r.recognize_google(audio).lower()
            print(f"🎙 Lock listener heard: {text}")
        except sr.UnknownValueError:
            continue
        except sr.RequestError as e:
            print(f"❌ Voice lock STT error: {e}")
            break

        if "hey uniq" in text and "lock" in text:
            # user said something like "hey uniq lock this"
            state["request"] = True
        elif "stop listening" in text:
            state["stop"] = True

    print("🎧 Voice lock listener stopped.")


# ---------------------------------------
# Part 1: YOLO webcam detection + locking
# ---------------------------------------
def detect_and_lock_device(model_path="best.pt", camera_index=1):
    """
    Open webcam, run YOLO, and let the user LOCK a detected device.

    Controls:
      - Press 'l' to lock via keyboard.
      - Say 'hey uniq, lock this' to lock via voice.
      - Press 'q' to quit without locking.

    Returns:
      locked_device (str) or None
    """
    print("Loading YOLO model...")
    model = YOLO(model_path)

    print("Opening webcam...")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return None

    locked_device = None
    current_label = None

    print("\n📷 Camera is ON.")
    print("➡ Show one device clearly to the camera (Mouse, KeyboardLayer, Food_Processor, Water_Heater).")
    print("➡ When the label looks correct, either:")
    print("   - Press 'l' on the keyboard to lock, OR")
    print("   - Say: 'hey uniq, lock this' to lock by voice.")
    print("➡ Press 'q' to quit without locking.\n")

    # Start voice lock listener in background
    state = {"request": False, "stop": False}
    listener_thread = threading.Thread(target=voice_lock_listener, args=(state,), daemon=True)
    listener_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read webcam frame.")
            break

        # Run YOLO detection
        results = model(frame)
        annotated = results[0].plot()
        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            confs = boxes.conf.cpu().numpy()
            best_idx = confs.argmax()
            cls_id = int(boxes.cls[best_idx].item())
            current_label = model.names[cls_id]
        else:
            current_label = None

        # Video status text
        if current_label is not None:
            text = f"Detected: {current_label}  ('l' or say 'hey uniq lock this' to lock, 'q' to quit)"
        else:
            text = "No device detected... (show one device to camera)"
        cv2.putText(
            annotated,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        cv2.imshow("UniQ - Device Detection", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("❌ Quit without locking.")
            break
        elif key == ord("l"):
            if current_label is not None and current_label != "undefined":
                locked_device = current_label
                print(f"\n🔒 Locked device (keyboard): {locked_device}")
                break
            else:
                print("⚠ No valid device to lock yet.")

        # Check for voice lock request
        if state["request"]:
            if current_label is not None and current_label != "undefined":
                locked_device = current_label
                print(f"\n🔒 Locked device (voice): {locked_device}")
                break
            else:
                print("⚠ Voice lock requested but no valid device detected yet.")
                # reset request so we don't spam
                state["request"] = False

    # Stop listener and clean up
    state["stop"] = True
    cap.release()
    cv2.destroyAllWindows()
    # give the listener a moment to exit
    listener_thread.join(timeout=1.0)

    return locked_device


# -----------------------------------
# Part 2: Q&A loop for locked device
# -----------------------------------
def question_answer_loop(locked_device: str, kb_path="kb.csv"):
    """
    Q&A loop after a device is locked.
    - Type questions directly
    - Type 'voice' to ask by speaking
    - Type 'change' to pick a new device
    - Type 'exit' to stop
    """
    qa = ApplianceQASystem(kb_path)

    print(f"\n🧠 Q&A mode for device: {locked_device}")
    print("➡ Type your questions about this device.")
    print("➡ Type 'voice' to ask by speaking.")
    print("➡ Type 'change' to pick a new device.")
    print("➡ Type 'exit' to stop.\n")

    while True:
        raw = input(f"Your question about {locked_device} (or 'voice', 'change', 'exit'): ").strip()

        # exit assistant
        if raw.lower() in ["exit", "quit", "q"]:
            print("👋 Exiting assistant.")
            return "exit"

        # change device
        if raw.lower() in ["change", "change device", "new device"]:
            print("\n🔄 Changing device...")
            return "change"

        # voice question
        if raw.lower() in ["voice", "v"]:
            print("🎤 Speak your question (you can start with 'hey uniq ...')")
            text = stt_listen_once()
            if not text:
                continue
            # optional wake word stripping
            if text.startswith("hey uniq"):
                text = text.replace("hey uniq", "", 1).strip()
            question = text
        else:
            question = raw
            if question.lower().startswith("hey uniq"):
                question = question[8:].strip()

        # Get answer and speak
        answer = qa.answer(locked_device, question)
        speak(answer)


# ---------------
# Main application
# ---------------
def main():
    while True:
        locked_device = detect_and_lock_device(
            model_path="best.pt",
            camera_index=1,  # change if needed
        )

        if locked_device is None:
            break

        result = question_answer_loop(locked_device, kb_path="kb.csv")

        if result == "exit":
            break
        # if "change", loop again → reopen camera and lock new device


if __name__ == "__main__":
    main()
