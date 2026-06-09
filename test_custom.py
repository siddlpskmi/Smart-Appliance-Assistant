from qa_engine import ApplianceQASystem

qa = ApplianceQASystem("kb.csv")

question = "How do I start using it?"
device = "Water_Heater"

print("Device:", device)
print("Question:", question)
print("Answer:", qa.answer(device, question))
