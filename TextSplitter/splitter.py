from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, Language
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings


text = """
LangChain is a framework for developing applications powered by large language models (LLMs).

LangChain simplifies every stage of the LLM application lifecycle:

Development: Build your applications using LangChain's open-source components and third-party integrations. Use LangGraph to build stateful agents with first-class streaming and human-in-the-loop support.
Productionization: Use LangSmith to inspect, monitor and evaluate your applications, so that you can continuously optimize and deploy with confidence.
Deployment: Turn your LangGraph applications into production-ready APIs and Assistants with LangGraph Platform.
"""

# ----------- Length Based Splitter ---------------#
splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap  = 0,
    separator= ''    
)
results = splitter.split_text(text)
# splitter.split_documents() -----> to split documents

# ----------- Text Based Splitter ---------------#

text2 = """
My name is Atul
I am 31 years old

I live in Cardiff
How are you
"""
splitter2 = RecursiveCharacterTextSplitter(
    chunk_size = 10,
    chunk_overlap = 0,
)
results2  = splitter2.split_text(text2)

# ----------- Document Based Splitter ---------------#
# we use different seprators

PYTHON_CODE = """
def hello_world():
    print("Hello, World!")

# Call the function
hello_world()
"""

splitter3 = RecursiveCharacterTextSplitter.from_language(
    language= Language.PYTHON,
    chunk_size = 10,
    chunk_overlap = 0,
)

results3  = splitter3.split_text(PYTHON_CODE)

# ----------- Semantic Based Splitter ---------------#
# experimental
splitter4 = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type ="standard_deviation",
    breakpoint_thrshold_amount = 1
)

result4 = splitter4.split_text(text)


print(results3)