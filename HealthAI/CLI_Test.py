
import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType

from langchain_google_genai import GoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from LangChainWraper import symptom_tool, remedy_tool, medline_tool
load_dotenv()
 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")# Check if the API key is available
tools = [symptom_tool, remedy_tool, medline_tool]
llm = GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)
memory = ConversationBufferMemory(memory_key="chat_history")


# Create the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    memory=memory,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = agent.run(user_input)
    print("Health AI:", response)
