from qa_engine import ApplianceQASystem

qa = ApplianceQASystem("kb.csv")

tests = [
    # WATER HEATER
    ("Water_Heater", "How do I start using it?"),         # we already tested this
    ("Water_Heater", "How do I turn it on?"),
    ("Water_Heater", "how to switch this on"),
    ("Water_Heater", "how can I make it stop?"),
    ("Water_Heater", "how do I turn this off?"),

    # FOOD PROCESSOR
    ("Food_Processor", "How do I wash this thing?"),      # already tested
    ("Food_Processor", "How to clean this?"),
    ("Food_Processor", "how do I clean this machine"),
    ("Food_Processor", "how should I maintain it?"),

    # MOUSE
    ("Mouse", "how do I start using it?"),
    ("Mouse", "how to turn it on"),
    ("Mouse", "how to stop using it"),
    ("Mouse", "is using this safe for my wrist"),

    # KEYBOARD
    ("KeyboardLayer", "how to start using this keyboard"),
    ("KeyboardLayer", "how do I turn this off"),
    ("KeyboardLayer", "how should I type properly"),
    ("KeyboardLayer", "is this keyboard safe to use for long hours"),
]

for device, q in tests:
    print("=" * 70)
    print("Device :", device)
    print("Question:", q)
    print("Answer  :", qa.answer(device, q))
