from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain.memory import ConversationBufferMemory


# Load environment variables from .env file
load_dotenv()

# Get the Google API key from the environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Check if the API key is available
if not GOOGLE_API_KEY:
    raise ValueError("Google API key not found. Please set the GOOGLE_API_KEY environment variable.")

memory = ConversationBufferMemory(memory_key="chat_history")

# LLM setup
#llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
llm = GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)

# Tool: Suggest destinations
def suggest_places(query: str) -> str:
    # Dummy implementation for now
    return f"Based on your preferences, you might like Bali, Maldives, or Phuket!"

def generate_itinerary(destination: str) -> str:
    prompt = f"""Create a 3-day travel itinerary for a trip to {destination}.
    Include top attractions, local food, and relaxing activities."""
    return llm.predict(prompt)


place_tool = Tool(
    name="Destination Suggestion Tool",
    func=suggest_places,
    description="Suggests travel destinations based on user preferences"
)

itinerary_tool = Tool(
    name="Itinerary Generator",
    func=generate_itinerary,
    description="Creates a day-by-day travel itinerary for a given destination"
)

def estimate_budget(destination: str) -> str:
    cost_guide = {
        "Bali": 800,
        "Maldives": 2000,
        "Paris": 1500,
        "Thailand": 1000,
        "Japan": 1800
    }
    estimate = cost_guide.get(destination, 1200)
    return f"Estimated budget for a 3-day trip to {destination}: ${estimate}"

budget_tool = Tool(
    name="Budget Estimator",
    func=estimate_budget,
    description="Estimates cost for a short vacation to the destination"
)
tools = [place_tool, itinerary_tool, budget_tool]


# Create the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    memory=memory,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run the agent

"""user_input = input("Ask the Travel Agent something: ")
response = agent.run(user_input)
print("\nAI Travel Planner:", response) """
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = agent.run(user_input)
    print("AI Travel Planner:", response)

