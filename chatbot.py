# chatbot.py
"""
Streamlit app for Wellsville Brewing Company (WBC) info and drink lookup.

Hybrid strategy:
  - Use SQL via Supabase for structured queries.
  - Fall back to semantic retrieval (Pinecone + HF embeddings) for descriptive info.

Required env:
  GEMINI_API_KEY
  PINECONE_API_KEY
  PINECONE_INDEX_NAME
  SUPABASE_DB_USER
  SUPABASE_DB_PASSWORD
  SUPABASE_DB_HOST
  SUPABASE_DB_PORT
  SUPABASE_DB_NAME
"""

import os
from typing import List

import streamlit as st
from dotenv import load_dotenv

# LangChain core
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# LangChain agents and SQL
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentExecutor, tool, create_tool_calling_agent

# LLM
from langchain_google_genai import ChatGoogleGenerativeAI

# Vector search
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings


# --- App setup ---------------------------------------------------------------

load_dotenv()
st.set_page_config(page_title="WBC BrewBot", layout="wide")


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def init_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        google_api_key=require_env("GEMINI_API_KEY"),
    )


def init_vector_store(model_name: str = "intfloat/e5-base") -> PineconeVectorStore:
    pc = Pinecone(api_key=require_env("PINECONE_API_KEY"))
    index = pc.Index(require_env("PINECONE_INDEX_NAME"))
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return PineconeVectorStore(index=index, embedding=embeddings)


def init_sql_db() -> SQLDatabase:
    # Provide concise schema help for the agent.
    custom_table_info = {
        "drinks": """
This table lists all beverages.
- Filter by category in 'type' (e.g., beer, cider).
- Filter by style in 'style' (use ILIKE for case-insensitive).
- Strength and bitterness are in 'abv' and 'ibu'.
"""
    }
    uri = (
        f"postgresql://{require_env('SUPABASE_DB_USER')}:{require_env('SUPABASE_DB_PASSWORD')}"
        f"@{require_env('SUPABASE_DB_HOST')}:{require_env('SUPABASE_DB_PORT')}/{require_env('SUPABASE_DB_NAME')}"
    )
    return SQLDatabase.from_uri(uri, custom_table_info=custom_table_info)


# Initialize services with clear failure path.
try:
    llm = init_llm()
    vector_store = init_vector_store()
    db = init_sql_db()
except Exception as e:
    st.error(f"Startup failed; check environment configuration. Details: {e}")
    st.stop()


# --- Tools ------------------------------------------------------------------

def format_docs_with_metadata(docs: List[Document]) -> str:
    """
    Render retrieved docs in a compact, human-readable block.
    """
    parts = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        fields = []
        for key in ("name", "type", "style", "abv", "ibu"):
            if key in meta:
                fields.append(f"{key.capitalize()}: {meta[key]}")
        detail = " | ".join(fields) if fields else "No details available"
        parts.append(f"Result {i}:\nDescription: {doc.page_content}\nDetails: {detail}")
    return "\n---\n".join(parts)


@tool
def semantic_brewery_search(query: str) -> str:
    """
    Use semantic search for general info: hours, location, history, or
    descriptive drink questions (e.g., “good summer beer”).
    """
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.5},
    )
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found in the brewery's general documents."
    return format_docs_with_metadata(docs)


# SQL Agent for structured queries
sql_agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="openai-tools",
    verbose=False,
    handle_parsing_errors=True,
    max_iterations=5,
)


@tool
def drink_database_tool(query: str) -> str:
    """
    Attempt to answer structured drink queries with SQL first. 
    If no meaningful results, fallback to semantic search.
    """
    try:
        result = sql_agent_executor.invoke({"input": query})
        output = (result or {}).get("output", "")
        negatives = [
            "no results",
            "no drinks",
            "no beers",
            "no ciders",
            "are no",
            "are not any",
            "don't have any",
            "do not have any",
            "doesn't have",
            "does not have",
            "I can tell you how many beers are offered. What is the name of the table that contains information about drinks?",
        ]
        meaningful = bool(output) and "[]" not in output and not any(
            kw in output.lower() for kw in negatives
        )
        if meaningful:
            return output
        # Explicit fallback to semantic search if SQL result is not useful
        return semantic_brewery_search(query)
    except Exception:
        # On error, also fallback to semantic search
        return semantic_brewery_search(query)


# --- Supervisor agent (tool-calling) ----------------------------------------

tools = [drink_database_tool, semantic_brewery_search]

tool_calling_template = """
You are BrewBot for Wellsville Brewing Company.

Rules:
1) For listing/counting/filtering drinks (e.g., “how many IPAs”, “strongest beer”, “list all ciders”),
   try drink_database_tool first.
2) If drink_database_tool returns no definitive results or fails, try semantic_brewery_search.
3) For general brewery questions or subjective recommendations, use semantic_brewery_search.

Question: {input}
Scratchpad:
{agent_scratchpad}
""".strip()

tc_prompt = ChatPromptTemplate.from_template(tool_calling_template)

supervisor_agent = create_tool_calling_agent(llm, tools, tc_prompt)
supervisor_agent_executor = AgentExecutor(
    agent=supervisor_agent,
    tools=tools,
    verbose=False,
    handle_parsing_errors=True,
    max_iterations=5,
)


# --- App logic --------------------------------------------------------------

def process_prompt(user_prompt: str) -> None:
    st.session_state.messages.append(HumanMessage(content=user_prompt))
    with st.spinner("Thinking..."):
        try:
            response = supervisor_agent_executor.invoke(
                {"input": user_prompt, "chat_history": st.session_state.messages}
            )
            result = response.get("output", "")
        except Exception:
            st.error("An error occurred while processing your request.")
            result = "Sorry, something went wrong. Please try again."
    st.session_state.messages.append(AIMessage(content=result))


# --- UI ---------------------------------------------------------------------

st.title("Wellsville Brewing Info BrewBot 🍺")
st.sidebar.title("Controls")

if st.sidebar.button("Restart Chat"):
    st.session_state.clear()
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Welcome to Wellsville Brewing Company! How can I help?")
    ]

# Display history
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Quick suggestions on first load
if len(st.session_state.messages) == 1:
    st.markdown("---")
    st.subheader("Try one of these:")
    cols = st.columns(3)
    suggestions = {
        "The Business 🏢": ["Tell me about the brewery", "Who are the owners?"],
        "Hours & Location 🕒": ["What are your hours?", "Where are you located?"],
        "The Brews 🍻": [
            "How many beers do you offer?",
            "Do you have any IPAs?",
            "Rank all beers by IBU",
            "What is the strongest drink?",
        ],
    }
    for i, (title, questions) in enumerate(suggestions.items()):
        with cols[i]:
            st.markdown(f"#### {title}")
            for q in questions:
                if st.button(q, use_container_width=True):
                    process_prompt(q)
                    st.rerun()

# Input
if prompt_text := st.chat_input("Ask me anything about the brewery!"):
    process_prompt(prompt_text)
    st.rerun()
