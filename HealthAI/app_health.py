# app_health.py

import streamlit as st
import requests
import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType

from langchain_google_genai import GoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from LangChainWraper import symptom_tool, remedy_tool
load_dotenv()
 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")# Check if the API key is available
tools = [symptom_tool, remedy_tool]
llm = GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)
memory = ConversationBufferMemory(memory_key="chat_history")


agent = initialize_agent(
    tools=tools,
    llm=llm,
    memory=memory,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

st.set_page_config(page_title="AI Health Assistant", page_icon="🩺")
st.title("🩺 AI Health Assistant")
st.write("Enter symptoms or health questions. This is for informational purposes only.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Describe your symptoms or ask a health question:")

if user_input:
    full_query = f"{user_input}"
    with st.spinner("Thinking..."):
        response = agent.run(full_query)
        st.session_state.chat_history.append((user_input, response))

for user_msg, ai_msg in st.session_state.chat_history:
    st.markdown(f"**You:** {user_msg}")
    st.markdown(f"**Health AI:** {ai_msg}")
