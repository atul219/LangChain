from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv


load_dotenv()
model = ChatOpenAI()

messages = [
    SystemMessage(content = "You are a helpful assistant"),
    HumanMessage(content = "tell me about Langchain")
]

result = model.invoke(messages)

messages.append(AIMessage(content = result.content))

print(messages)
