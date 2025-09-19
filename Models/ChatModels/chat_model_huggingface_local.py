from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
import os

load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")


llm = HuggingFacePipeline.from_model_id(
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
    task = 'text-generation',
    pipeline_kwargs= dict(
        temperature = 0.5,
        max_new_tokens = 100
    ))

model = ChatHuggingFace(llm)
result = model.invoke("Who was the first prime minister of India")

print(result)