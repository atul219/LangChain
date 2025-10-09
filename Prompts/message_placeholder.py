from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

# chat template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(
        variable_name= 'chat_history'
    ),
    ('human', '{query}')

])
# load chat history
chat_history = []
with open('Prompts/message.txt') as f:
    chat_history.extend(f.readlines())

# create prompt

prompt = chat_template.invoke({'chat_history': chat_history, 'query': 'where is my refund'})

print(prompt)