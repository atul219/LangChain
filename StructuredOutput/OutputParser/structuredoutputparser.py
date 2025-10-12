# from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

# model = ChatOpenAI()
llm = HuggingFaceEndpoint(
    repo_id= "google/gemma-2-2b-it",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

schema = [
    ResponseSchema(name = 'fact_1', description= 'Fact 1 about the topic'),
    ResponseSchema(name = 'fact_2', description= 'Fact 2 about the topic'),
    ResponseSchema(name = 'fact_3', description= 'Fact 3 about the topic')
]

parser = StructuredOutputParser.from_response_schemas(schema)

template1 = PromptTemplate(
    template = "Give 3 facts about {topic} \n {format_instruction}",
    input_variables= ['topic'],
    partial_variables= {'format_instruction': parser.get_format_instructions()}
)

chain = template1 | model | parser

result = chain.invoke({'topic': 'black hole'})
print(result)