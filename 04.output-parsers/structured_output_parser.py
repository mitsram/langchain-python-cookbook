from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema, OutputFixingParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

schema = [
    ResponseSchema(name="fact1", description="First face about blackholes"),
    ResponseSchema(name="fact2", description="Second fact about blackholes"),
    ResponseSchema(name="fact3", description="Third fact about blackholes")
]

parser = StructuredOutputParser.from_response_schemas(schema)

safe_parser = OutputFixingParser.from_llm(model, parser)

template = PromptTemplate(
    template="Give me 3 facts about {topic}. Return only valid json instructions that follows this format: {response_format}",
    input_variables=["topic", "response_format"],
    partial_variables={"response_format": parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({"topic": "blackholes"})
print(result)