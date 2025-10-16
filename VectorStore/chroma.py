from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

doc1 = Document(
    page_content= "In 2013, Kohli was ranked number one in the ICC rankings for ODI batsmen. In 2015, he achieved the summit of T20I rankings. In 2020, the International Cricket Council named him the male cricketer of the decade.",
    meta_data = {"team": "Royal Challengers Banglore"}
)

doc2 = Document(
    page_content= "Rohit Gurunath Sharma (born 30 April 1987) is an Indian international cricketer and the former captain of the Indian national cricket team. He is a right-handed batsman who plays for Mumbai Indians in Indian Premier League and for Mumbai in domestic cricket",
    meta_data = {"team": "Mumbai Indians"}
)
doc3 = Document(
    page_content= "Mahendra Singh Dhoni born 7 July 1981 is an Indian professional cricketer who plays as a right-handed batter and a wicket-keeper. Widely regarded as one of the most prolific wicket-keeper batsmen and captains, he represented the Indian cricket team and was the captain of the side in limited overs formats from 2007 to 2017 and in test cricket from 2008 to 2014.",
    meta_data = {"team": "Chennai Super Kings"}
)
doc4 = Document(
    page_content= "Yuvraj Singh (born 12 December 1981) is a former Indian international cricketer who played in all formats of the game. An all-rounder who batted left-handed in the middle order and bowled slow left-arm orthodox, he has won 7 Player of the Series awards in One Day International cricket, which is the joint third-highest by an Indian cricketer",
    meta_data = {"team": "Punjab Kings"}
)

docs = [doc1, doc2, doc3, doc4]

vector_store = Chroma(
    embedding_function= OpenAIEmbeddings(),
    persist_directory= 'chroma_db',
    collection_name = 'sample'
) 

vector_store.add_documents(docs)

print(f"Data in vector: {vector_store.get(include= ['embeddings', 'documents', 'metadatas'])}")


# seach documents
result = vector_store.similarity_search(
    query= "Who among these is a all-rounder?",
    k =  1 # how many similar documents you want
)

print(f"Result from similarity search: {result}")

# seach documents with score
result_score = vector_store.similarity_search_with_score(
    query= "Who among these is a all-rounder?",
    k =  1 # how many similar documents you want
)

print(f"Result from similarity search with score: {result_score}")


# filter
result_filter = vector_store.similarity_search(
    query= "",
    filter= {'team': "Mumbai Indians"}
)
print(f"Result from filter: {result_filter}")
