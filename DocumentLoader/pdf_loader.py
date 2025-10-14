from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader("path")

docs = loader.load()

model = ChatOpenAI()
parser = StrOutputParser()

template = PromptTemplate(
    template = 'Give me a short of the following text \n {text}',
    input_variables= ['text']
)

chain = template | model | parser

result = chain.invoke({'text': docs[0].page_content})

print(result)