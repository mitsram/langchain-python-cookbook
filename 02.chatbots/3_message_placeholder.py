# Definition: A special placeholder inside a ChatPromptTemplate that lets you inject dynamic messages (like conversation history or tool outputs).
# How it works:
#   Instead of just {variable}, you can use MessagesPlaceholder("history").
#   At runtime, you pass in a list of messages (user/assistant turns).
# Practical Guidance:
#   Use message placeholders when building stateful bots with memory or multi-turn context.

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