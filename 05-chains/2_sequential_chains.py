# Definition: Chains that run one after another in sequence.
# How it works:
#   Output of one chain becomes the input to the next.
#   Can be SimpleSequentialChain (just pass text) or SequentialChain (pass multiple variables).
# Use Case: Multi-step workflows like:
#   Generate an outline → expand into sections → polish into final essay.
#   Extract data → transform → store.
# Practical Guidance:
#   Use SequentialChain when building multi-step workflows (like your automation frameworks).


from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt1 = PromptTemplate(
    template="Generate 3 detailed report on a {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate 3 point summary on the following text {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = prompt1 | model | prompt2 | model | parser

result = chain.invoke({"topic": "3I Atlas Interstellar Probe"})

print(result)