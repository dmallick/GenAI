from langchain.agents import Tool
import requests
from langchain.agents.agent_types import AgentType
from langchain_google_genai import GoogleGenerativeAI
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv
import os

from langchain_google_genai import GoogleGenerativeAI
from langchain.memory import ConversationBufferMemory

load_dotenv()

# Get the Google API key from the environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")# Check if the API key is available
if not GOOGLE_API_KEY:
    raise ValueError("Google API key not found. Please set the GOOGLE_API_KEY environment variable.")
memory = ConversationBufferMemory(memory_key="chat_history")
# LLM setup

llm = GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)
# Tool: Suggest destinations




def home_remedies(condition: str) -> str:
    return llm.predict(f"What are safe and natural home remedies for {condition}?")

def explain_symptoms(symptoms: str) -> str:
    return llm.predict(f"Given the symptoms: {symptoms}, what are 2-3 possible causes? Be general and avoid making diagnoses.")

def search_medlineplus(query: str) -> str:
    url = "https://wsearch.nlm.nih.gov/ws/query"
    params = {
        "db": "healthTopics",
        "term": query,
        "rettype": "brief",
        "retmax": 3
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        return "Sorry, couldn't reach MedlinePlus."

    results = response.text
    if "<record>" not in results:
        return "No relevant topics found."

    # Simple parsing (not XML-safe but works for small use)
    titles = []
    links = []
    for line in results.split("\n"):
        if "<title>" in line:
            titles.append(line.strip().replace("<title>", "").replace("</title>", ""))
        elif "<url>" in line:
            links.append(line.strip().replace("<url>", "").replace("</url>", ""))

    output = "🧾 **Trusted Health Info:**\n"
    for t, l in zip(titles, links):
        output += f"- [{t}]({l})\n"
    return output


symptom_tool = Tool(
    name="Symptom Explainer",
    func=explain_symptoms,
    description="Explains potential causes for described symptoms"
)

remedy_tool = Tool(
    name="Home Remedy Advisor",
    func=home_remedies,
    description="Provides general wellness advice or natural remedies for a given condition"
)

medline_tool = Tool(
    name="Trusted Health Info Finder",
    func=search_medlineplus,
    description="Fetches official health info from MedlinePlus based on a health topic or symptom"
)

