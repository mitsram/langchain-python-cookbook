# Definition: Chains that branch based on conditions.
# How it works:
#   You define a condition function (e.g., if input contains “math”, go to calculator chain; else go to summarizer chain).
#   The chain routes execution accordingly.
# Use Case: Decision-making pipelines:
#   Route customer queries to different LLMs/tools depending on intent.
#   Handle fallback logic (e.g., if parsing fails, retry with a different prompt).
# Practical Guidance:
#   Use ConditionalChain when you need routing logic (like intent classification).

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_classic.schema.runnable import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal


load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback as positive or negative: {feedback} and provide the response in the following format: {response_format}",
    input_variables=["feedback"],
    partial_variables={"response_format": parser2.get_format_instructions()}
)

classifer_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback: {feedback}",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback: {feedback}",
    input_variables=["feedback"]
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model | parser2),
    (lambda x: x.sentiment == "negative", prompt3 | model | parser2),
    RunnableLambda(lambda x: "Could not determine sentiment.")
)

chain = classifer_chain | branch_chain

result = chain.invoke({"feedback": "The phone is actually amazing"})
print(result)