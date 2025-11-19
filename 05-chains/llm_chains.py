from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt = PromptTemplate(
    template="Suggest a catchy blog title for a {topic}",
    input_variables=["topic"]
)

chain = LLMChain(prompt=prompt, llm=model)

topic = "3IU Atlas Interstellar Object"

response = chain.invoke({"topic": topic})
print(response)

