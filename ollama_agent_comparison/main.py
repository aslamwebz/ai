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

st.title("Ollama Agent Comparison")

