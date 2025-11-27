# Definition: A minimal schema-driven approach where you ask the LLM to return output in a structured format (usually JSON or key-value pairs).
# How it works:
#   You define a schema (e.g., fields like title, summary, keywords).
#   LangChain adds instructions to the prompt so the LLM outputs in that format.
#   The result is parsed into a Python dictionary.


from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# schema
class TravelPlan(TypedDict):
    destination: str
    duration_days: int
    activities: list[str]
    budget_usd: float

structured_model = model.with_structured_output(TravelPlan)

prompt = "Create a travel plan for a 5-day trip to Japan with a budget of $2000."

result = structured_model.invoke(prompt)
print(result)

# new_prompt = f"Give me the must do activities. The number one activity is '{prompt}'"

# result = model.invoke(new_prompt)
# print(result)