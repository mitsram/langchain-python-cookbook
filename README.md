# LangChain Python Cookbook

A collection of practical examples and recipes for building applications with LangChain and Google's Gemini AI models.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Quickstart Guide](#quickstart-guide)
- [Usage Examples](#usage-examples)

## Prerequisites

- Python 3.8 or higher
- Google API key for Gemini models
- pip (Python package manager)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mitsram/langchain-python-cookbook.git
cd langchain-python-cookbook
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r dependencies.txt
```

Or install packages individually:
```bash
pip install langchain langchain_google_genai langchain_community langchain_core python-dotenv pydantic
```

## Environment Setup

### Setting Up API Keys

This project uses environment variables to securely store API keys and tokens. Follow these steps:

1. **Create a `.env` file** in the project root directory:
```bash
touch .env
```

2. **Add your API keys** to the `.env` file:
```env
GOOGLE_API_KEY=your_google_api_key_here
```

3. **Obtain your Google API Key**:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Sign in with your Google account
   - Click "Create API Key"
   - Copy the generated key and paste it in your `.env` file

4. **Important Security Notes**:
   - Never commit your `.env` file to version control
   - Add `.env` to your `.gitignore` file:
     ```bash
     echo ".env" >> .gitignore
     ```
   - Keep your API keys confidential and rotate them regularly

### Environment Variable Structure

Your `.env` file should look like this:

```env
# Google Gemini API Key
GOOGLE_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# Optional: Add other API keys as needed
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
```

## Project Structure

```
langchain-python-cookbook/
├── 01.llm-interaction/       # Basic LLM interactions
│   └── chatmodel_gemini.py   # Simple Gemini model usage
├── 02.chatbots/              # Chatbot implementations
│   ├── chatbot.py            # Interactive chatbot with history
│   ├── chat_prompt_template.py
│   └── message_placeholder.py
├── 03.structured-output/     # Structured data generation
│   ├── structured_output.py
│   ├── pydantic_structured_output.py
│   └── detailed_output_structure.py
├── 04.output-parsers/        # Output parsing examples
│   ├── str_ouput_parser.py
│   ├── pydantic_parser.py
│   └── structured_output_parser.py
├── dependencies.txt          # Required Python packages
├── .env                      # Environment variables (create this)
└── README.md                 # This file
```

## Quickstart Guide

### 1. Basic LLM Interaction

Run a simple query to test your setup:

```bash
python 01.llm-interaction/chatmodel_gemini.py
```

**What it does**: Sends a basic prompt to Gemini and displays the response.

### 2. Interactive Chatbot

Launch an interactive chatbot with conversation history:

```bash
python 02.chatbots/chatbot.py
```

**What it does**: 
- Maintains conversation context
- Allows multi-turn conversations
- Type `exit` or `quit` to end the session

### 3. Structured Output

Generate structured data from natural language:

```bash
python 03.structured-output/pydantic_structured_output.py
```

**What it does**: Demonstrates how to get JSON-structured responses using Pydantic models.

### 4. Output Parsers

Parse and format LLM outputs:

```bash
python 04.output-parsers/pydantic_parser.py
```

**What it does**: Shows different methods to parse and structure model outputs.

## Usage Examples

### Example 1: Simple Query

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the model (API key loaded from GOOGLE_API_KEY env var)
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Send a prompt
response = model.invoke("What is the capital of France?")
print(response.content)
```

### Example 2: Chatbot with History

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Initialize conversation with system message
chat_history = [
    SystemMessage(content="You are a helpful AI assistant")
]

# Add user message
chat_history.append(HumanMessage(content="Tell me about Python"))

# Get response
response = model.invoke(chat_history)
print(response.content)
```

## Troubleshooting

### Common Issues

1. **API Key Error**: 
   ```
   Error: GOOGLE_API_KEY not found
   ```
   - Ensure `.env` file exists in the project root
   - Verify the API key is correctly set in `.env`
   - Check that `python-dotenv` is installed

2. **Import Errors**:
   ```
   ModuleNotFoundError: No module named 'langchain_google_genai'
   ```
   - Install all dependencies: `pip install -r dependencies.txt`
   - Verify your virtual environment is activated

3. **Rate Limiting**:
   - Google's free tier has rate limits
   - Wait a few moments between requests
   - Consider upgrading your API plan for higher limits

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

MIT License

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Google Gemini API](https://ai.google.dev/)
- [Python dotenv Documentation](https://pypi.org/project/python-dotenv/)