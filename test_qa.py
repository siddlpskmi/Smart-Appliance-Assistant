from qa_engine import ApplianceQASystem

qa = ApplianceQASystem("kb.csv")

device = "Mouse"
question = "How do I turn this on?"

answer = qa.answer(device, question)
print("Device:", device)
print("Question:", question)
print("Answer:", answer)
