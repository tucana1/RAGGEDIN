#!/usr/bin/env python3
"""
Interactive RAG Agent for LinkedIn (Gemini Version).
WORKSHOP EDITION: Fill in the [TODO] sections to build the agent.
"""

import os
import glob
import sys
import time
from dotenv import load_dotenv

# CORE LANGCHAIN & GEMINI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore

# TOOLS & AGENTS
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import create_retriever_tool, tool
# PRESERVED USER IMPORT:
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

# PROMPTS & MESSAGES
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# PINECONE
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
    
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "linkedin-agent-gemini-v2"
    
    # Check if index exists
    existing_indexes = [i.name for i in pc.list_indexes()]
    
    # [TODO 6]: CREATE PINECONE INDEX
    # Check if index_name is NOT in existing_indexes.
    # If not, create it with dimension=3072, metric="cosine", and the ServerlessSpec.
    # if index_name not in existing_indexes:
        # print(f"Creating index: {index_name}")
        # pc.create_index(...)
    
    # Wait for index to be ready
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
    
    # [TODO 7]: INITIALIZE GEMINI EMBEDDINGS
    # Initialize GoogleGenerativeAIEmbeddings with model="models/gemini-embedding-001"
    # embeddings = ...
    embeddings = None
    
    # [TODO 8]: CREATE VECTOR STORE
    # Use PineconeVectorStore.from_texts to store the notes.
    # vector_store = ...
    vector_store = None

    print("Notes stored in Pinecone (persistent across sessions)")
    return vector_store


def clean_agent_output(output):
    """Remove signature, extras, and other junk from agent output."""
    import re
    if isinstance(output, list):
        text = '\n'.join(str(item) for item in output)
    else:
        text = str(output)
    
    text = re.sub(r"\{'type':\s*'text',\s*'text':\s*'([^']*)',\s*'extras':\s*\{[^}]*\},\s*'index':\s*\d+\}", r'\1', text)
    text = re.sub(r"'extras':\s*\{[^}]*signature[^}]*\}", '', text)
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        if len(line) > 200 and all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in line.replace('\n', '')):
            continue
        if any(marker in line for marker in ['signature', 'extras:', "'extras'"]):
            continue
        cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines).strip()
    result = re.sub(r"\},\s*\[", '[\n', result)
    result = re.sub(r"\{\s*'type'", '', result)
    result = re.sub(r"'index':\s*\d+\s*\}", '', result)
    
    return result


def setup_agent(vector_store):
    """Setup the Agent with Memory capabilities and CLEAN search."""
    print("\nInitializing Agent...")
    
    # [TODO 1]: INITIALIZE LLM
    # Initialize ChatGoogleGenerativeAI with model="gemini-2.5-flash" and temperature=0.7
    # llm = ...
    llm = None
    
    # [TODO 9]: CREATE RETRIEVER TOOL
    # Create the tool to search notes. Use vector_store.as_retriever with search_kwargs={"k": 5}
    # retriever_tool = ...
    retriever_tool = None
    
    @tool
    def search_web(query: str):
        """Search web for recent news. Checks Pinecone first with strict matching, then searches web."""
        
        # 1. DEFINE THRESHOLD
        SCORE_THRESHOLD = 0.80 

        # 2. SEARCH WITH SCORE
        results_with_scores = vector_store.similarity_search_with_score(query, k=1)
        
        if results_with_scores:
            best_doc, best_score = results_with_scores[0]
            print(f"   > Cache Check: '{query}' | Best Score: {best_score:.4f}")
            
            if best_score >= SCORE_THRESHOLD:
                print("   > High confidence match found in cache. Skipping web search.")
                return f"Cached (Score {best_score:.2f}): {best_doc.page_content}"

        # [TODO 2]: FALLBACK TO WEB SEARCH
        # Initialize TavilySearchResults(max_results=2) and invoke with the query
        print(f"   > Searching web for: {query}...")
        # tavily = ...
        # raw_results = ...
        
        # (Mock return for workshop if TODO is empty)
        return "Web search placeholder"

    tools = [retriever_tool, search_web]
    
    # [TODO 3]: CREATE PROMPT TEMPLATE
    # Define the ChatPromptTemplate with system message, chat_history, input, and agent_scratchpad
    # prompt = ...
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Placeholder System Prompt"),
        ("human", "{input}"),
    ])
    
    # [TODO 4]: CREATE AGENT EXECUTOR
    # Use create_tool_calling_agent and return AgentExecutor(..., verbose=True)
    # agent = ...
    # return ...
    return None


def main():
    print("=" * 50)
    print("Interactive LinkedIn Agent (Gemini + Auto-Start)")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 50)
    
    try:
        load_environment()
        
        # Load notes
        notes = load_all_notes()
        
        # Ingest
        vector_store = ingest_data(notes)
        
        # Setup Agent
        agent_executor = setup_agent(vector_store)
        
        # AUTO-START
        print("\nAUTOMATIC STARTUP: Generating proposal based on your notes...")
        chat_history = [] 
        
        initial_prompt = "Look through my notes and create a high-impact LinkedIn post proposal based on the most interesting idea you find."
        
        # [TODO 5]: INVOKE THE AGENT
        # response = agent_executor.invoke(...)
        # output_text = response["output"]
        
        output_text = "Agent not ready yet."
        
        clean_output = clean_agent_output(output_text)
        print(f"\nAGENT:\n{clean_output}\n")
        
        chat_history.append(HumanMessage(content=initial_prompt))
        chat_history.append(AIMessage(content=clean_output))
        
        # Interactive Loop
        while True:
            user_input = input("\nTopic or Feedback > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            print("\nThinking...")
            
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            
            output_text = response["output"]
            clean_output = clean_agent_output(output_text)
            print(f"\nAGENT:\n{clean_output}\n")
            
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=clean_output))

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()