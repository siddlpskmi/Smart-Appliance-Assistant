import pyttsx3

engine = pyttsx3.init()

print("Voices:")
for v in engine.getProperty("voices"):
    print("-", v.id)

engine.setProperty("rate", 175)
engine.setProperty("volume", 1.0)

print("Speaking...")
engine.say("Hello, this is a test of your smart assistant.")
engine.runAndWait()
print("Done.")
