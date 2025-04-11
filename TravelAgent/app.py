
import os
from dotenv import load_dotenv
import requests

import streamlit as st
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAI
load_dotenv()

# Get the Google API key from the environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Check if the API key is available
if not GOOGLE_API_KEY:
    raise ValueError("Google API key not found. Please set the GOOGLE_API_KEY environment variable.")

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST")


# Create the LLM
#llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
llm = GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)

# Define the tools
def suggest_places(query: str) -> str:
    return f"Based on your preferences, you might like Bali, Maldives, or Phuket!"

def generate_itinerary(destination: str) -> str:
    prompt = f"""Create a 3-day travel itinerary for a trip to {destination}.
    Include top attractions, local food, and relaxing activities."""
    return llm.predict(prompt)

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

def search_hotels(destination: str) -> str:
    url = "https://booking-com.p.rapidapi.com/v1/hotels/search"

    querystring = {
        "units": "metric",
        "room_number": "1",
        "checkout_date": "2025-05-05",
        "checkin_date": "2025-05-01",
        "adults_number": "2",
        "dest_type": "city",
        "locale": "en-us",
        "order_by": "popularity",
        "dest_id": get_destination_id(destination)
    }

    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": RAPIDAPI_HOST
    }

    response = requests.get(url, headers=headers, params=querystring)
    hotels = response.json().get("result", [])[:3]  # top 3 hotels

    if not hotels:
        return "No hotels found. Try another destination."

    result = "Top Hotels:\n"
    for hotel in hotels:
        result += f"- {hotel['hotel_name']} – ⭐ {hotel.get('review_score', 'N/A')} – ${hotel.get('min_total_price', 'N/A')}\n"
    return result

# Register tools
tools = [
    Tool(name="Destination Suggestion Tool", func=suggest_places,
         description="Suggests travel destinations based on user preferences"),
    Tool(name="Itinerary Generator", func=generate_itinerary,
         description="Creates a day-by-day travel itinerary for a given destination"),
    Tool(name="Budget Estimator", func=estimate_budget,
         description="Estimates cost for a short vacation to the destination")
]

# Memory for conversation
memory = ConversationBufferMemory(memory_key="chat_history")

hotel_tool = Tool(
    name="Hotel Finder",
    func=search_hotels,
    description="Finds top hotels in a destination using Booking.com"
)

# Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    hotel_tool=hotel_tool,
    verbose=False
)


def get_destination_image(query):
    url = f"https://source.unsplash.com/800x400/?{query},travel"
    return url  # Unsplash will return a random image based on the query

def get_destination_id(destination: str) -> str:
    url = "https://booking-com.p.rapidapi.com/v1/hotels/locations"

    querystring = {"name": destination, "locale": "en-us"}

    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": RAPIDAPI_HOST
    }

    response = requests.get(url, headers=headers, params=querystring)
    data = response.json()

    if data:
        return data[0]["dest_id"]
    return "0"

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="AI Travel Planner", page_icon="🧳")
st.title("🧳 AI Travel Planner")
st.write("Plan your perfect vacation with the help of AI!")
st.sidebar.header("Preferences")
travel_type = st.sidebar.selectbox("What kind of vacation?", [
    "Beach", "Adventure", "Cultural", "Romantic", "Family", "Solo"
])
# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.text_input(
    "Describe your trip or ask a question:",
    placeholder="e.g. I want a romantic beach vacation...",
    key="input"
)

if user_input:
    full_query = f"I want a {travel_type.lower()} vacation. {user_input}"
    with st.spinner("Planning your trip..."):
        response = agent.run(full_query)
        st.session_state.chat_history.append((user_input, response))

# Display conversation
for user_msg, ai_msg in st.session_state.chat_history:
    st.markdown(f"**You:** {user_msg}")
    # Try extracting destination name (very basic)
    for loc in ["Bali", "Maldives", "Paris", "Thailand", "Japan"]:
        if loc.lower() in ai_msg.lower():
            img_url = get_destination_image(loc)
            st.image(img_url, caption=loc)
            break
        if "Day 1" in ai_msg:
            for line in ai_msg.split("\n"):
                if line.strip().startswith("Day"):
                    st.markdown(f"### {line}")
                elif line.strip().startswith("-") or line.strip().startswith("•"):
                    st.markdown(f"- {line.strip('-• ')}")
                else:
                    st.markdown(line)
      #  else:
    #st.markdown(ai_msg)
        
    st.markdown(f"**Travel Agent:** {ai_msg}")

    
