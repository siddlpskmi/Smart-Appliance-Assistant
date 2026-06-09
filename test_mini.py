from qa_engine import ApplianceQASystem

qa = ApplianceQASystem("kb.csv")

tests = [
    ("Water_Heater", "How do I start using it?"),
    ("Water_Heater", "How can I switch it off?"),
    ("Food_Processor", "How do I wash this thing?"),
    ("Mouse", "Is it safe for my wrist?"),
]

for device, q in tests:
    print("=" * 50)
    print("Device:", device)
    print("Question:", q)
    print("Answer:", qa.answer(device, q))
