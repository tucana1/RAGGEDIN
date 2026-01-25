#!/usr/bin/env python3
"""
RAG Agent for LinkedIn post brainstorming.
This script demonstrates how to create a LangChain agent that combines
Pinecone vector store retrieval with Tavily web search to generate viral LinkedIn posts.
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone, ServerlessSpec


def load_environment():
    """Load environment variables from .env file."""
    load_dotenv()
    
    # Verify required environment variables
    required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY", "TAVILY_API_KEY"]
    for var in required_vars:
        if not os.getenv(var):
            raise ValueError(f"Missing required environment variable: {var}")
    
    print("✓ Environment variables loaded successfully")


def ingest_sample_data():
    """Ingest sample text into Pinecone serverless index."""
    print("\n📝 Starting data ingestion...")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # Index configuration
    index_name = "linkedin-posts"
    dimension = 1536  # OpenAI embedding dimension
    
    # Create index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        print(f"Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"✓ Index '{index_name}' created")
    else:
        print(f"✓ Index '{index_name}' already exists")
    
    # Sample LinkedIn post ideas and content
    sample_texts = [
        "LinkedIn engagement tip: Start your posts with a hook that creates curiosity. Ask a question or share a surprising fact.",
        "Viral LinkedIn posts often include personal stories. Share your failures and lessons learned - authenticity drives engagement.",
        "The best time to post on LinkedIn is Tuesday through Thursday, between 8-10 AM and 12-2 PM when professionals check their feeds.",
        "Use short paragraphs and line breaks in LinkedIn posts. White space makes content more readable and increases engagement by 30%.",
        "Add a call-to-action at the end of your posts. Ask readers to share their thoughts or tag someone who needs to see this.",
        "LinkedIn carousel posts get 3x more engagement than regular posts. Break down complex topics into visual slides.",
        "Hashtags still matter on LinkedIn. Use 3-5 relevant hashtags to increase discoverability without looking spammy.",
        "Comment on others' posts before sharing your own. This warms up the algorithm and builds genuine connections.",
        "Video content on LinkedIn gets 5x more engagement. Don't worry about production quality - authenticity wins.",
        "Document your journey, don't just share the highlights. People connect with the process, not just the outcome.",
    ]
    
    # Initialize embeddings and vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    # Add documents to Pinecone
    vector_store = PineconeVectorStore.from_texts(
        texts=sample_texts,
        embedding=embeddings,
        index_name=index_name
    )
    
    print(f"✓ Ingested {len(sample_texts)} sample documents into Pinecone")
    
    return vector_store


def setup_agent(vector_store):
    """Setup LangChain agent with Pinecone retriever and Tavily search tools."""
    print("\n🤖 Setting up LangChain agent...")
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    # Create retriever tool from Pinecone
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    retriever_tool = create_retriever_tool(
        retriever,
        name="linkedin_notes_retriever",
        description="Search through curated LinkedIn post best practices and tips. Use this to find proven strategies for creating engaging LinkedIn content."
    )
    
    # Create Tavily search tool for web search
    tavily_search = TavilySearchResults(
        max_results=3,
        description="Search the web for current trends, news, and information. Use this to find trending topics and recent events for LinkedIn posts."
    )
    
    # Combine tools
    tools = [retriever_tool, tavily_search]
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a LinkedIn content expert specializing in creating viral posts.
        
Your goal is to help users create engaging LinkedIn posts by:
1. Searching through curated best practices using the linkedin_notes_retriever
2. Finding current trends and topics using web search
3. Combining both sources to craft compelling, authentic content

When creating posts:
- Use short paragraphs and line breaks for readability
- Start with a strong hook
- Include personal insights or stories
- End with a clear call-to-action
- Keep it professional yet authentic

Always cite your sources and explain your reasoning."""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    # Create agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    print("✓ Agent created with Retriever and TavilySearchResults tools")
    
    return agent_executor


def main():
    """Main execution function."""
    print("=" * 60)
    print("🚀 RAG Agent for LinkedIn Post Brainstorming")
    print("=" * 60)
    
    try:
        # Step 1: Load environment variables
        load_environment()
        
        # Step 2: Ingest sample data into Pinecone
        vector_store = ingest_sample_data()
        
        # Step 3: Setup agent with tools
        agent_executor = setup_agent(vector_store)
        
        # Step 4: Execute sample query
        print("\n" + "=" * 60)
        print("📱 Executing Sample Query")
        print("=" * 60)
        
        sample_query = """Write a viral LinkedIn post about the importance of AI skills in 2024. 
        Use both the curated best practices and current web trends to make it engaging."""
        
        print(f"\nQuery: {sample_query}\n")
        
        response = agent_executor.invoke({"input": sample_query})
        
        print("\n" + "=" * 60)
        print("✨ GENERATED LINKEDIN POST")
        print("=" * 60)
        print(response["output"])
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
