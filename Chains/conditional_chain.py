from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatOpenAI()
parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description= 'feedback of the review')

parser2 = PydanticOutputParser(pydantic_object= Feedback)

prompt1 = PromptTemplate(
    template = 'Classify the sentiment of the following review text into positive or negative \n {feedback} \n {format_instructions}',
    input_variables= ['feedback'],
    partial_variables= {'format_instructions': parser2.get_format_instructions()}
)

classfiier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template = 'write an appropriate response to the positive feedback \n {feedback}',
    input_variables= ['feedback']
)

prompt3 = PromptTemplate(
    template = 'write an appropriate response to the negative feedback \n {feedback}',
    input_variables= ['feedback']
)


branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
    
)


chain = classfiier_chain | branch_chain

result = chain.invoke({'feedback': 'this is a terrible phone'})

print(result)