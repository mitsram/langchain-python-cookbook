# Definition: Chains that run multiple subchains in parallel and combine results.
# How it works:
#   Each subchain executes independently.
#   Results are aggregated into a single output dictionary.
# Use Case: Efficiency and multi-perspective tasks:
#   Ask multiple LLMs for answers simultaneously.
#   Run summarization + sentiment analysis + keyword extraction in parallel on the same text.
# Practical Guidance:
#   Use ParallelChain when you want multiple analyses at once (e.g., log streaming + summarization + tagging).


from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableParallel

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following {topic}.",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answer from the following {text}.",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n notes: {notes} \n quiz: {quiz}"    
)

parser = StrOutputParser()

runnable_chain = RunnableParallel(
    {
        "notes": prompt1 | model | parser,
        "quiz": prompt2 | model | parser
    }
)

final_chain = prompt3 | model | parser

chains = runnable_chain | final_chain

text = """
Python is a high-level, interpreted programming language known for its simplicity and readability. 
Created by Guido van Rossum and first released in 1991, Python emphasizes code readability with its 
use of significant indentation. It supports multiple programming paradigms including procedural, 
object-oriented, and functional programming. Python has a comprehensive standard library and a vast 
ecosystem of third-party packages available through PyPI (Python Package Index). It's widely used in 
web development, data science, machine learning, automation, and scientific computing. Popular frameworks 
like Django and Flask for web development, and libraries like NumPy, Pandas, and TensorFlow for data 
science make Python one of the most versatile programming languages today.
"""

result = chains.invoke(text)
print(result)