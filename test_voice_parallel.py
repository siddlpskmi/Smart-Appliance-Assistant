# test_voice_parallel.py
import cv2
import speech_recognition as sr
import threading

state = {"request": False, "stop": False}

def listener():
    r = sr.Recognizer()
    mic = sr.Microphone()
    print("🎧 Listening (say: hey uniq lock this)")
    while not state["stop"]:
        try:
            with mic as source:
                r.adjust_for_ambient_noise(source, duration=0.3)
                audio = r.listen(source, timeout=5, phrase_time_limit=5)
            text = r.recognize_google(audio).lower()
            print("Heard:", text)
            if "hey uniq" in text and "lock" in text:
                state["request"] = True
        except:
            pass
    print("Listener stopped.")
    

# start thread
threading.Thread(target=listener, daemon=True).start()

# Show webcam only
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
while True:
    ret, frame = cap.read()
    cv2.imshow("Test Window", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if state["request"]:
        print("🎉 Voice trigger detected!")
        break

state["stop"] = True
cap.release()
cv2.destroyAllWindows()
