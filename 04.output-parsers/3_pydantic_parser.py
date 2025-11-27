# Purpose: Enforces type-safe structured outputs using Pydantic models.
# How it works:
#   You define a Pydantic class with fields and types.
#   LangChain instructs the LLM to produce JSON that matches the model.
#   The parser validates and converts the output into a Pydantic object.
# Practical Guidance:
#   Use Pydantic Output Parser when you want robust, validated, type-safe objects — especially if the output feeds into downstream systems.

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class Person(BaseModel):
    name: str = Field(description="The full name of the person")
    age: int = Field(gt=18, lt=100, description="The age of the person in years")
    city: str = Field(description="The city where the person lives")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="""
    Give me the name, age and city of a fictional {place} person. Make sure the age is between 18 and 100. Return the response in the following format: {response_format}
    """,
    input_variables=["place"],
    partial_variables={"response_format": parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({"place": "New York"})
print(result)