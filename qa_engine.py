import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SUPPORTED_DEVICES = ["Mouse", "KeyboardLayer", "Food_Processor", "Water_Heater"]

DEVICE_CONTEXT = {
    "Mouse": "a computer mouse used for navigating a computer",
    "KeyboardLayer": "a computer keyboard used for typing",
    "Food_Processor": "a kitchen food processor appliance used for chopping, blending and processing food",
    "Water_Heater": "a water heater appliance used for heating water at home"
}


class ApplianceQASystem:
    def __init__(self, kb_path: str = "kb.csv"):
        # kb_path kept for compatibility but not used anymore
        pass

    def answer(self, device: str, question: str) -> str:
        """
        Use Groq LLM to answer any question about the detected device.
        """
        device_description = DEVICE_CONTEXT.get(device, f"a {device} appliance")

        prompt = f"""You are UniQ, a smart home appliance assistant. 
You are an expert on {device_description}.

Answer the following question about this device clearly and helpfully in 2-3 sentences.
If the question is not related to this device, politely say you can only answer questions about {device}.

Device: {device}
Question: {question}

Answer:"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )

        return response.choices[0].message.content.strip()