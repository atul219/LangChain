from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model = "gpt-4o-mini", temperature= 0.6, max_completion_tokens= 10)

result = model.invoke("What is the capital of India")
print(result.content)
