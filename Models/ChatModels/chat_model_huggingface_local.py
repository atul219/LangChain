from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
import os

load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")


llm = HuggingFacePipeline.from_model_id(model_id = "meta-llama/Meta-Llama-3-8B-Instruct", 
                          task = 'text-generation')

model = ChatHuggingFace(llm)
result = model.invoke("Who was the first prime minister of India")
print(result.content)