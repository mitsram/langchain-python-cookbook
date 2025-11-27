from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
prompt = "Hi my name is Mitchell. What is your name?"
# prompt = "What is the capital city of USA"
# prompt = "Can you remember my name?"
result = model.invoke(prompt)

print(result.content)