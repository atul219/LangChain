from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

documents = [
    Document(page_content= "I had chocolate chip pancakes and scrambled eggs for breakfast this morning."),
    Document(page_content= "The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees."),
    Document(page_content= "Building an exciting new project with LangChain - come check it out!"),
    Document(page_content= "Robbers broke into the city bank and stole $1 million in cash.")
]

embedding_model = OpenAIEmbeddings()

vector_store = Chroma.from_documents(
    documents= documents,
    embedding= embedding_model,
    collection_name= "my_collection"
)

retriever = vector_store.as_retriever(search_kwargs = {"k":  2})

query = "money"
results = retriever.invoke(query)

print(results)