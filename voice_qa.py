import speech_recognition as sr
import pyttsx3
from qa_engine import ApplianceQASystem

# Initialize QA engine (MiniLM + TF-IDF)
qa = ApplianceQASystem("kb.csv")

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Initialize TTS
engine = pyttsx3.init()
engine.setProperty("rate", 170)
engine.setProperty("volume", 1.0)

def speak(text):
    print(f"Assistant: {text}")
    engine.say(text)
    engine.runAndWait()

def listen():
    with sr.Microphone() as source:
        print("\n🎤 Say something (start with 'Hey UniQ')...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print(f"🗣 You said: {text}")
        return text.lower()

    except sr.UnknownValueError:
        print("❌ Could not understand audio")
        return ""
    except sr.RequestError:
        print("❌ Speech recognition service error")
        return ""

def process_voice_input():
    while True:
        user_text = listen()

        if not user_text:
            continue

        # EXIT COMMAND
        if "stop" in user_text or "exit" in user_text or "quit" in user_text:
            speak("Okay, stopping voice mode.")
            break

        # WAKE WORD HANDLING
        if user_text.startswith("hey uniq"):
            cleaned = user_text.replace("hey uniq", "").strip()

            if cleaned == "":
                speak("Yes? How can I help you?")
                continue

            # Ask for locked device
            device = input("Enter locked device (Mouse, Food_Processor, etc.): ")

            answer = qa.answer(device, cleaned)
            speak(answer)

        else:
            print("‼️ Say 'Hey UniQ' before your question.")

if __name__ == "__main__":
    print("🎧 Voice Q&A Mode — say 'Hey UniQ' before your question.")
    process_voice_input()
