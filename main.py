#!/usr/bin/env python3
"""
Interactive RAG Agent for LinkedIn (Gemini Version).
- Scans folder for ANY file matching 'notes*.txt'
- Auto-generates a post on startup
- Stores data persistently in Pinecone across sessions
"""

import os
import glob
import sys
import time
from dotenv import load_dotenv

#CORE LANGCHAIN & GEMINI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore

#TOOLS & AGENTS
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import create_retriever_tool, tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

#PROMPTS & MESSAGES
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

#PINECONE
from pinecone import Pinecone, ServerlessSpec


def load_environment():
    """Load environment variables from .env file."""
    load_dotenv()
    required_vars = ["GOOGLE_API_KEY", "PINECONE_API_KEY", "TAVILY_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")
    print("Environment variables loaded successfully")


def load_all_notes():
    """Reads lines from ALL files starting with 'notes' ending in .txt."""
    files = glob.glob("notes*.txt")
    
    if not files:
        print("No 'notes*.txt' files found. Creating a sample 'notes.txt'...")
        with open("notes.txt", "w") as f:
            f.write("Tip: Authenticity is key on LinkedIn.\n")
            f.write("Tip: Post between 8am and 10am.\n")
            f.write("My thought: AI agents are the next big thing.\n")
        files = ["notes.txt"]

    all_lines = []
    print(f"\nFound {len(files)} note files: {files}")
    
    for filename in files:
        try:
            with open(filename, "r") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                all_lines.extend(lines)
                print(f"   - Loaded {len(lines)} lines from {filename}")
        except Exception as e:
            print(f"   x Error reading {filename}: {e}")
    
    if not all_lines:
        raise ValueError("All notes files were empty! Please add content.")
        
    return all_lines


def ingest_data(notes):
    """
    Ingest text data into Pinecone using Gemini Embeddings (3072 dim).
    """
    print(f"\nIngesting {len(notes)} total notes...")
    
    ### LIVE CODING SECTION 1: PINECONE VECTOR DB SETUP ###
    # Goal: Connect to Pinecone, create an index if needed, and store vectors.
    # ---------------------------------------------------------
    
    # [CODE REMOVED HERE]
    
    # ---------------------------------------------------------
    print("Notes stored in Pinecone (persistent across sessions)")
    return vector_store


def clean_agent_output(output):
    """Remove signature, extras, and other junk from agent output."""
    import re
    if isinstance(output, list):
        text = '\n'.join(str(item) for item in output)
    else:
        text = str(output)
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        if any(marker in line for marker in ['signature', 'extras', 'CiIB', 'gEBc']):
            continue
        if len(line) > 100 and all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in line.replace('\n', '')):
            continue
        cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines).strip()
    result = re.sub(r"'signature':\s*'[^']*'", '', result)
    result = re.sub(r'"signature":\s*"[^"]*"', '', result)
    return result


def setup_agent(vector_store):
    """Setup the Agent with Memory capabilities and CLEAN search."""
    print("\nInitializing Agent...")

    ### LIVE CODING SECTION 2: LANGCHAIN AGENT CONSTRUCTION ###
    # Goal: Define the LLM (Gemini), the Tools (Retriever + Search), and the Prompt.
    # ---------------------------------------------------------

    # [CODE REMOVED HERE]

    # ---------------------------------------------------------
    
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


def main():
    print("=" * 50)
    print("Interactive LinkedIn Agent (Gemini + Auto-Start)")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 50)
    
    try:
        load_environment()
        notes = load_all_notes()
        vector_store = ingest_data(notes)
        agent_executor = setup_agent(vector_store)
        
        #AUTO-START
        print("\nAUTOMATIC STARTUP: Generating proposal based on your notes...")
        chat_history = [] 
        initial_prompt = "Look through my notes and create a high-impact LinkedIn post proposal based on the most interesting idea you find."
        
        # Initial invocation
        response = agent_executor.invoke({
            "input": initial_prompt,
            "chat_history": chat_history
        })
        clean_output = clean_agent_output(response["output"])
        print(f"\nAGENT:\n{clean_output}\n")
        
        chat_history.append(HumanMessage(content=initial_prompt))
        chat_history.append(AIMessage(content=clean_output))
        
        ### LIVE CODING SECTION 3: INTERACTIVE EXECUTION LOOP ###
        # Goal: Create the 'while True' loop that takes user input and calls the agent.
        # ---------------------------------------------------------

        # [CODE REMOVED HERE]

        # ---------------------------------------------------------

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
