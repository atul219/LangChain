from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI()

# 1st prompt -> detail report
template1 = PromptTemplate(
    template = "Write a detail report on {topic}",
    input_variables= ['topic']
)

# 2nd promot -> summary
template2 = PromptTemplate(
    template= "Write a 5 line summary on the following text. /n {text}",
    input_variables= ['text']
)


# Without str output parser

# prompt1 = template1.invoke({'topic': 'black hole'})

# result = model.invoke(prompt1)

# prompt2 = template2.invoke({'text': result.content})

# result2 = model.invoke(prompt2)

# print(result2.content)


# with str output parser

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': 'black hole'})

print(result)
