from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
#from langchain.chat_models import ChatOpenAI
import os

from langchain_google_genai import GoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = "GOOGLE_API_KEY"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# LLM setup
#llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
llm = GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key="GOOGLE_API_KEY")

# Tool: Suggest destinations
def suggest_places(query: str) -> str:
    # Dummy implementation for now
    return f"Based on your preferences, you might like Bali, Maldives, or Phuket!"

place_tool = Tool(
    name="Destination Suggestion Tool",
    func=suggest_places,
    description="Suggests travel destinations based on user preferences"
)

# Create the agent
agent = initialize_agent(
    tools=[place_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run the agent
user_input = input("What kind of vacation do you want? ")
response = agent.run(user_input)
print("\nAI Travel Planner:", response)
