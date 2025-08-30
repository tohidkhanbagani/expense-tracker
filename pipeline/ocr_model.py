from langchain_google_genai import ChatGoogleGenerativeAI
import os
import base64
import cv2
from langchain.schema import HumanMessage
from langchain_core.output_parsers import JsonOutputParser 
from dotenv import load_dotenv



load_dotenv()

class ExpenseExtractor:
    def __init__(self, model_name: str = "gemini-1.5-flash", system_prompt_file: str = "system_prompts/ocr_system_prompt.txt"):
        self.model_name = model_name
        self.system_prompt_file = system_prompt_file

        # ✅ Pass API key explicitly
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        with open(self.system_prompt_file, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

    def image_to_base64(self, image_input):
        if isinstance(image_input, str):  # file path
            with open(image_input, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        elif isinstance(image_input, (bytes, bytearray)):
            return base64.b64encode(image_input).decode("utf-8")
        else:  # assume numpy array
            _, buffer = cv2.imencode(".jpg", image_input)
            return base64.b64encode(buffer).decode("utf-8")

    def extract_expense(self, image_input):
        img_b64 = self.image_to_base64(image_input)

        msg = HumanMessage(content=[
            {"type": "text", "text": self.system_prompt},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_b64}"}
        ])

        response = self.llm.invoke([msg])
        parser = JsonOutputParser()
        return parser.parse(response.content)
