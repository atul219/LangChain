from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embedding = OpenAIEmbeddings(
    model = 'text-embedding-3-large',
    dimensions= 300
)

documents = ["Virat Kohli is known as one of the best chasers in modern cricket with a remarkable record in ODIs.",
            "Sachin Tendulkar holds the record for the most runs in international cricket history.",
            "MS Dhoni is celebrated for his calm captaincy and finishing abilities under pressure.",
            "Jasprit Bumrah is famous for his deadly yorkers and unorthodox bowling action.",
            "Ben Stokes played a heroic innings in the 2019 World Cup final to help England lift the trophy."]


query = "tell me about dhoni"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key = lambda x:x[1])[-1]


print(f"The query: '{query}' is matching with document: '{documents[index]}' with score: {score}")