# from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import Field, BaseModel

load_dotenv()

# model = ChatOpenAI()
llm = HuggingFaceEndpoint(
    repo_id= "google/gemma-2-2b-it",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

class Person(BaseModel):

    name: str = Field(description= 'name of the person')
    age: int = Field(description= 'age of the person')
    city: str = Field(description= 'name of the city the person lives')


parser = PydanticOutputParser(pydantic_object= Person)


template1 = PromptTemplate(
    template = "Give me the name, age and city of a fictional {place} person \n {format_instruction}",
    input_variables= ['place'],
    partial_variables= {'format_instruction': parser.get_format_instructions()}
)

# prompt = template1.format()

# result = model.invoke(prompt)
# f_result = parser.parse(result.content)
# print(f_result)

# with chains
chain = template1 | model | parser

result = chain.invoke({'place': 'India'})
print(result)