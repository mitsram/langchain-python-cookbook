from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

chat_template = ChatPromptTemplate([
    ('system', "You are a helpful {domain} assistant."),
    ('human', "Please help me with {topic}.")
])

prompt = chat_template.invoke({
    'domain': 'travel',
    'topic': 'finding the best flight deals to Paris'
})

print(prompt)

# result = model.invoke(prompt)
# print(result)