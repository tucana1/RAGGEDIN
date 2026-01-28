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
#PRESERVED USER IMPORT:
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
    #Matches notes.txt, notes0.txt, notes1.txt, notes_draft.txt, etc.
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
    
    PERSISTENCE BEHAVIOR:
    1. Data is ADDED to the Pinecone index, not replaced.
    2. If the index exists from a previous run, new notes are appended.
    3. The index persists across sessions - old vectors remain unless manually deleted.
    4. Pinecone is a cloud vector database, so all stored data is persistent.
    """
    print(f"\nIngesting {len(notes)} total notes...")
    
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "linkedin-agent-gemini-v2"
    
    #Check if index exists
    existing_indexes = [i.name for i in pc.list_indexes()]
    
    if index_name in existing_indexes:
        #SAFETY CHECK: Verify dimensions
        try:
            index_description = pc.describe_index(index_name)
            if index_description.dimension != 3072:
                print(f"Found index with dimension {index_description.dimension}. Rebuilding as 3072...")
                pc.delete_index(index_name)
                time.sleep(5) 
                existing_indexes.remove(index_name)
        except Exception as e:
            print(f"Warning checking index: {e}")

    #Create index with 3072 dimensions if missing
    if index_name not in existing_indexes:
        print(f"Creating index: {index_name} (Dimension: 3072)")
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
    
    #Use Gemini embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    #Ingest or append to existing index
    vector_store = PineconeVectorStore.from_texts(
        texts=notes,
        embedding=embeddings,
        index_name=index_name
    )
    print("Notes stored in Pinecone (persistent across sessions)")
    return vector_store


def clean_agent_output(output):
    """Remove signature, extras, and other junk from agent output."""
    import re
    
    #Handle if output is a list (convert to string)
    if isinstance(output, list):
        text = '\n'.join(str(item) for item in output)
    else:
        text = str(output)
    
    #Split by common markers
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        #Skip lines containing signature indicators
        if any(marker in line for marker in ['signature', 'extras', 'CiIB', 'gEBc']):
            continue
        #Skip lines that are just base64-like strings
        if len(line) > 100 and all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in line.replace('\n', '')):
            continue
        cleaned_lines.append(line)
    
    #Join back and clean up excessive whitespace
    result = '\n'.join(cleaned_lines).strip()
    
    #Remove any remaining encoded signature blocks
    result = re.sub(r"'signature':\s*'[^']*'", '', result)
    result = re.sub(r'"signature":\s*"[^"]*"', '', result)
    
    return result


def setup_agent(vector_store):
    """Setup the Agent with Memory capabilities and CLEAN search."""
    print("\nInitializing Agent...")
    
    #Use the specific model requested
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=0.7,
        convert_system_message_to_human=True
    )
    
    #TOOL 1: Retriever
    retriever_tool = create_retriever_tool(
        vector_store.as_retriever(search_kwargs={"k": 5}),
        name="search_my_notes",
        description="Searches the user's personal notes and indexed web results. ALWAYS search this first."
    )
    
    #TOOL 2: CLEAN Search (Checks Pinecone first, then searches web and indexes results)
    @tool
    def search_web(query: str):
        """Search web for recent news. Checks Pinecone first, then searches web and indexes results."""
        #First check if this query exists in Pinecone
        existing_results = vector_store.similarity_search(query, k=3)
        
        if existing_results:
            print(f"Found cached results in Pinecone for: {query}")
            cached_text = "\n".join([f"Cached: {result.page_content}" for result in existing_results])
            return cached_text
        
        #If not in Pinecone, search the web
        tavily = TavilySearchResults(max_results=2)
        raw_results = tavily.invoke(query)
        
        #Strip all extraneous data (signature, extras, images, raw_content, etc.)
        clean_results = []
        texts_to_index = []
        
        if isinstance(raw_results, list):
            for item in raw_results:
                #Only extract url and content, discard everything else
                if isinstance(item, dict):
                    url = item.get("url", "")
                    content = item.get("content", "")
                    
                    #Only include if we have actual content
                    if url and content:
                        #Limit content to avoid overwhelming the LLM
                        content_limited = content[:500]
                        clean_results.append({
                            "url": url,
                            "content": content_limited
                        })
                        #Store for indexing with source attribution
                        texts_to_index.append(f"Web Search Result - Query: {query}\nSource: {url}\nContent: {content_limited}")
        
        #Index the web results to Pinecone for future searches
        if texts_to_index:
            try:
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-001",
                    google_api_key=os.getenv("GOOGLE_API_KEY")
                )
                vector_store.add_texts(texts_to_index)
                print(f"Indexed {len(texts_to_index)} web results to Pinecone")
            except Exception as e:
                print(f"Warning: Could not index web results: {e}")
        
        #Return as plain formatted text (not dict) to prevent metadata leakage
        if clean_results:
            output_lines = []
            for result in clean_results:
                output_lines.append(f"Source: {result['url']}")
                output_lines.append(f"Content: {result['content']}\n")
            return "\n".join(output_lines)
        
        return "No web results found."

    tools = [retriever_tool, search_web]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a viral LinkedIn ghostwriter. 
        1. Search the user's notes first to find their core idea.
        2. Search the web for recent news to back it up.
        3. Write a compelling, authentic LinkedIn post.
        
        If the user gives feedback, modify the previous post accordingly.
        
        IMPORTANT: Only present clean, formatted content to the user. Do not show raw data structures, signatures, or metadata."""),
        
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


def main():
    print("=" * 50)
    print("Interactive LinkedIn Agent (Gemini + Auto-Start)")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 50)
    
    try:
        load_environment()
        
        #Load notes from ALL matching files
        notes = load_all_notes()
        
        #Ingest
        vector_store = ingest_data(notes)
        
        #Setup Agent
        agent_executor = setup_agent(vector_store)
        
        #AUTO-START: Generate first post immediately
        print("\nAUTOMATIC STARTUP: Generating proposal based on your notes...")
        chat_history = [] 
        
        initial_prompt = "Look through my notes and create a high-impact LinkedIn post proposal based on the most interesting idea you find."
        
        response = agent_executor.invoke({
            "input": initial_prompt,
            "chat_history": chat_history
        })
        
        output_text = response["output"]
        
        #Strip signature and extras from output
        clean_output = clean_agent_output(output_text)
        print(f"\nAGENT:\n{clean_output}\n")
        
        #Save to history so the loop knows about it
        chat_history.append(HumanMessage(content=initial_prompt))
        chat_history.append(AIMessage(content=clean_output))
        
        #Interactive Loop
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
            
            #Strip signature and extras from output
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