print("hello")
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Initialize the Gemini Pro model
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("human", "{topic}"),
])

# Create a chain
chain = prompt | llm

# Run the chain with a topic
response = chain.invoke({"topic": "Explain the concept of black holes in simple terms."})

# Print the response
print(response.content)