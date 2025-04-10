import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import LLMChain
from langchain_core.runnables import RunnableSequence

from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Get the Google API key from the environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Check if the API key is available
if not GOOGLE_API_KEY:
    raise ValueError("Google API key not found. Please set the GOOGLE_API_KEY environment variable.")

# Initialize the Gemini Pro model
llm = GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)

print("Google Gemini Pro model initialized successfully.")
# Define a prompt template
prompt_template = PromptTemplate(
    input_variables=["input"],
    template="What is the capital of {input}?",
)
# Create a chain with the prompt template and the LLM
chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
)
# Run the chain with an example input
response = chain.invoke("France")
print(f"Response from Gemini Pro: {response}")
# Example output:
# Response from Gemini Pro: The capital of France is Paris.
# Note: The actual output may vary based on the model's response.
# This code initializes the Google Gemini Pro model using the Google API key,
# defines a prompt template to ask for the capital of a given country,
# creates a chain with the prompt template and the LLM, and runs the chain
# with an example input ("France"). The response is printed to the console.
# Ensure you have the required libraries installed:
# pip install langchain google-generative-ai python-dotenv
# Make sure to set the GOOGLE_API_KEY environment variable in your .env file
# before running the script. The .env file should contain:
# GOOGLE_API_KEY=your_google_api_key_here
# This code is a basic example of how to use the LangChain library