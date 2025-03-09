import warnings
warnings.filterwarnings('ignore')


import streamlit as st
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool, WebsiteSearchTool
import requests

load_dotenv()

search_tool = SerperDevTool()
website_search_tool = WebsiteSearchTool()

os.environ["serper_api_key"] = os.getenv("SERPER_API_KEY")
os.environ['CREWAI_DISABLE_TELEMETRY'] = 'true'

def check_ollama_server():
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        if response.status_code == 200:
            st.title("Ollama server is running")
            return True
        else:
            st.title("Ollama server is not running")
            return False
    except Exception as e:
        st.title(f"Error checking Ollama server: {e}")
        return False
    
check_ollama_server()

def get_ollama_llm(llm_model="ollama/deepseek-r1:7b"):
    ollama_llm = LLM(
        model=llm_model,
        base_url='http://localhost:11434',
        api_key="",
    )
    return ollama_llm

ollama_llm = get_ollama_llm()

st.title("Ollama Agent Comparison")

