from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever()

query = "IPL winners list"

docs = retriever.invoke(query)

print(docs)