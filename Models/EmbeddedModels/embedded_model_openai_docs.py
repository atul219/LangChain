from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


embedding = OpenAIEmbeddings(
    model = 'text-embedding-3-large', 
    dimensions = 32)



docs = [
    "hello, my name is david",
    "what's your name",
    "Good Night"

]

vectors = embedding.embed_documents(docs)

print(vectors)