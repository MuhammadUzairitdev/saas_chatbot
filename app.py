"""
SaaS Automation AI Chatbot
A production-ready Streamlit application that acts as an AI assistant,
answers questions, suggests solutions, and captures leads seamlessly.
"""
from dotenv import load_dotenv
load_dotenv()   
import streamlit as st
import os
import json
import re
import time
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# --- Configuration ---
LEADS_FILE = "leads.json"
PAGE_TITLE = "SaaS Automation AI"
PAGE_ICON = "⚡"

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="centered")

# --- System Prompt ---
SYSTEM_PROMPT = """You are an AI assistant for a SaaS automation agency.

Your goal is to:
- Help users
- Understand their problem
- Suggest AI chatbot / AI agent solutions
- Collect lead info naturally (name, email, company)
- Guide toward booking a demo

Do not sound salesy.
Keep answers short and helpful (2-4 lines)."""

# --- Helper Functions ---
def save_lead(email: str, context: str):
    """Saves lead information to a local JSON file."""
    leads = []
    if os.path.exists(LEADS_FILE):
        try:
            with open(LEADS_FILE, "r") as f:
                leads = json.load(f)
        except json.JSONDecodeError:
            leads = []
            
    # Avoid duplicate emails, though context might be updated in a real app
    if not any(lead.get("email") == email for lead in leads):
        leads.append({
            "email": email,
            "timestamp": time.time(),
            "context": context
        })
        with open(LEADS_FILE, "w") as f:
            json.dump(leads, f, indent=4)

def get_llm(api_key: str):
    """Initializes the Mistral AI LLM."""
    return ChatMistralAI(
        temperature=0.7, 
        mistral_api_key=api_key, 
        model="mistral-large-latest",
        max_tokens=200
    )

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- UI Setup ---
st.title(f"{PAGE_ICON} {PAGE_TITLE}")
st.markdown("Welcome! How can we help you automate your business today?")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Mistral API Key", type="password", help="Required to run the chatbot")
    
    st.markdown("---")
    st.subheader("About")
    st.markdown("This chatbot assistant helps you find automation solutions and schedules demos.")
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if not api_key:
    st.info("👋 Please enter your Mistral API key in the sidebar to start chatting.")
    st.stop()

# Test API Key locally quickly
try:
    test_llm = get_llm(api_key)
except Exception as e:
    st.error(f"Failed to initialize chatbot: {str(e)}")
    st.stop()

# --- Chat Interface ---
# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Type your message here..."):
    # Render user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Detect leads using regex (Email)
    email_regex = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    emails = re.findall(email_regex, prompt)
    if emails:
        # Save the first detected email as a lead
        save_lead(emails[0], prompt)
        
    # Render assistant response with typing effect and spinner
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Assistant is typing..."):
            try:
                # Get response from LangChain
                llm = get_llm(api_key)
                
                # Build context
                chat_messages = [SystemMessage(content=SYSTEM_PROMPT)]
                for msg in st.session_state.messages: # st.session_state.messages already includes the latest prompt
                    if msg["role"] == "user":
                        chat_messages.append(HumanMessage(content=msg["content"]))
                    else:
                        chat_messages.append(AIMessage(content=msg["content"]))
                        
                response_ai = llm.invoke(chat_messages)
                response = response_ai.content
                
                # Simulate typing out the response
                full_response = ""
                # A simple word-level typing effect
                words = response.split()
                for i, word in enumerate(words):
                    full_response += word + " "
                    time.sleep(0.05)
                    # Add a blinking cursor for realism
                    message_placeholder.markdown(full_response + "▌")
                
                # Final output
                message_placeholder.markdown(full_response.strip())
                st.session_state.messages.append({"role": "assistant", "content": full_response.strip()})
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error. Please try again. ({str(e)})"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
