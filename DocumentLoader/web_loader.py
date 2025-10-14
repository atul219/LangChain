from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

url = "https://python.langchain.com/docs/introduction/"

loader = WebBaseLoader(url)
docs = loader.load()

model = ChatOpenAI()
parser = StrOutputParser()

template = PromptTemplate(
    template = 'Give me a short of the following text \n {text}',
    input_variables= ['text']
)

chain = template | model | parser

result = chain.invoke({'text': docs})

print(result)