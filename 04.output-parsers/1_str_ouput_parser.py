# Purpose: Simplest parser — just returns the raw string output from the LLM.
# How it works:
#  No schema, no validation.
#  Whatever the LLM outputs is passed through as a string.
# Practical Guidance:
#   Use StrOutputParser when you just want the raw text.

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 1st prompt
template1 = PromptTemplate(template="Write detailed report about {topic}.", input_variables=["topic"])
# prompt1 = template1.invoke({"topic": "English Premier League of 2023-24 season"})
# result1 = model.invoke(prompt1).content

# 2nd prompt
template2 = PromptTemplate(template="Summarize the following text in one sentence: {text}", input_variables=["text"])
# prompt2 = template2.invoke({"text": result1})
# result2 = model.invoke(prompt2)
# print(result2.content)

# chaining
parser = StrOutputParser()

chain = template1 | model | template2 | model | parser

result = chain.invoke({"topic": "English Premier League of 2023-24 season"})
print(result)