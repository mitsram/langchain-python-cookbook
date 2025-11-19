from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ('system', "You are a helpful AI bot."),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human', '{query}')
])

chat_history = []

with open("chatbot_history.txt") as file:
    chat_history.extend(file.readlines())

prompt = chat_template.invoke({
    'chat_history': chat_history,
    'query': "I like tomatoes, what should I eat?"
})

print(prompt)