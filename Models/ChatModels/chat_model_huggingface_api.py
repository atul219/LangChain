from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# repo_id = "deepseek-ai/DeepSeek-R1-0528"
repo_id="meta-llama/Meta-Llama-3-8B-Instruct"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="text-generation",
    max_new_tokens=100,
    do_sample=False,
)

model = ChatHuggingFace(llm = llm)
result = model.invoke("Who was the first prime minister of India")

print(result.content)