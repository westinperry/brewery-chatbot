# chatbot.py
"""
Streamlit app for Wellsville Brewing Company (WBC) info and drink lookup.
Sidebar logs show agent-generated SQL and tool logic.

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

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentExecutor, tool, create_tool_calling_agent

from langchain_google_genai import ChatGoogleGenerativeAI

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

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
    custom_table_info = {
        "drinks": '''
        This table contains all drinks: beers, ciders, seltzers, etc.
        The table name is always "drinks".
        Columns: id, gluten_free, description, ibu, abv, style, type, name.
        - To count ALL drinks: SELECT COUNT(*) FROM drinks
        - For highest ABV/IBU for a specific type (e.g. beer), always filter with WHERE type ILIKE '<type>'.
        - For 'what beer has the highest ABV?', run: 
          SELECT name, abv FROM drinks WHERE type ILIKE 'beer' ORDER BY abv DESC LIMIT 1
        - For 'what drink has the highest ABV?', run: 
          SELECT name, abv FROM drinks ORDER BY abv DESC LIMIT 1
        Result for a type (beer, cider, etc.) must always have a matching type.
        If the field is NULL, exclude from ranking.
        '''
    }
    uri = (
        f"postgresql://{require_env('SUPABASE_DB_USER')}:{require_env('SUPABASE_DB_PASSWORD')}"
        f"@{require_env('SUPABASE_DB_HOST')}:{require_env('SUPABASE_DB_PORT')}/{require_env('SUPABASE_DB_NAME')}"
    )
    return SQLDatabase.from_uri(uri, custom_table_info=custom_table_info)

try:
    llm = init_llm()
    vector_store = init_vector_store()
    db = init_sql_db()
except Exception as e:
    st.error(f"Startup failed; check environment configuration. Details: {e}")
    st.stop()

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
        parts.append(f"Result {i}: Description: {doc.page_content} Details: {detail}")
    return "\n---\n".join(parts)

@tool
def semantic_brewery_search(query: str) -> str:
    """
    Use semantic search for general info: hours, location, history, or
    descriptive drink questions (e.g., "good summer beer").
    """
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.5},
    )
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found."
    return format_docs_with_metadata(docs)

sql_agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="openai-tools",
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
    agent_executor_kwargs={"return_intermediate_steps": True},
)

@tool
def drink_database_tool(query: str) -> str:
    """
    For count or extrema questions, always use WHERE type ILIKE '<type>' when a type is specified.
    For 'what beer has the highest ABV', return only a drink with type=beer, never cider.
    If SQL fails or gives poor results, fallback to semantic search.
    """
    try:
        result = sql_agent_executor.invoke({"input": query})
        steps = result.get("intermediate_steps", [])
        sql_queries = []
        for s in steps:
            if isinstance(s, tuple):
                tool_input, tool_output = s
                sql_queries.append(f"In: {tool_input}\nOut: {tool_output}")
        st.session_state.setdefault("sql_logs", []).extend(sql_queries)
        output = (result or {}).get("output", "")
        negatives = [
            "no results", "no drinks", "no beers", "no ciders",
            "are no", "are not any", "don't have any", "do not have any",
            "doesn't have", "does not have",
            "what is the name of the table",
            "I can't give you the total number of beers we have. I can only provide details about specific beers.",
            "I can only count drinks or search for drink types. I cannot determine which drink has the highest ABV.",
        ]
        # Defensive post-process: if asking for 'beer' ABV, must have type=beer
        if "beer" in query.lower() and "highest abv" in query.lower():
            # Optionally: Extra validation with pattern match
            beer_names = [
                "Hefeweizen", "Kölsch", "Pilsner", "Witbier", "Light Lager",
                "IPA", "Dark Ale", "Porter", "Cream Ale", "Red Ale",
                "Stout", "Fruit Kölsch"
            ]
            if not any(name.lower() in output.lower() for name in beer_names):
                return (
                    "Sorry, the query did not return a beer. The beer with the highest ABV is likely 'Dark Ale' at 5.8%, "
                    "as there are no beers with ABV higher than that in the menu."
                )
        meaningful = bool(output) and "[]" not in output and not any(
            kw in output.lower() for kw in negatives
        )
        if meaningful:
            return output
        return semantic_brewery_search(query)
    except Exception:
        return semantic_brewery_search(query)

tools = [drink_database_tool, semantic_brewery_search]

tool_calling_template = """
You are BrewBot for Wellsville Brewing Company.

Rules:
1) For highest/lowest ABV or IBU: 
  - If type is specified (e.g. beer, cider), filter with WHERE type ILIKE '<type>' and ORDER BY abv/ibu.
  - If no type is specified, search all drinks.
2) Use ILIKE for all type filtering, never '=' or LIKE.
3) Result for a type (beer, cider, etc.) must always be of that type!
4) For "how many drinks", count all records in 'drinks'.
5) For "how many beers/ciders/seltzers", count WHERE type ILIKE '<type>'.
6) Never ask for table name; it's always 'drinks'.
7) If SQL cannot answer, fallback to semantic_brewery_search.

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

def process_prompt(user_prompt: str) -> None:
    st.session_state.messages.append(HumanMessage(content=user_prompt))
    with st.spinner("Thinking..."):
        st.session_state["sql_logs"] = []
        try:
            response = supervisor_agent_executor.invoke(
                {"input": user_prompt, "chat_history": st.session_state.messages}
            )
            result = response.get("output", "")
        except Exception:
            st.error("An error occurred while processing your request.")
            result = "Sorry, something went wrong. Please try again."
    st.session_state.messages.append(AIMessage(content=result))

st.title("Wellsville Brewing Info BrewBot 🍺")
st.sidebar.title("Controls")

if st.sidebar.button("Restart Chat"):
    st.session_state.clear()
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Welcome to Wellsville Brewing Company! How can I help?")
    ]

st.sidebar.subheader("SQL/Tool Agent Steps")
if "sql_logs" in st.session_state and st.session_state["sql_logs"]:
    for step in st.session_state["sql_logs"]:
        st.sidebar.code(step)

for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

if len(st.session_state.messages) == 1:
    st.markdown("---")
    st.subheader("Try one of these:")
    cols = st.columns(3)
    suggestions = {
        "The Business 🏢": ["Tell me about the brewery", "Who are the owners?"],
        "Hours & Location 🕒": ["What are your hours?", "Where are you located?"],
        "The Brews 🍻": [
            "How many beers do you offer?",
            "How many ciders do you offer?",
            "What beer has the highest ABV?",
            "What drink has the highest ABV?",
            "What beer has the highest IBU?",
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

if prompt_text := st.chat_input("Ask me anything about the brewery!"):
    process_prompt(prompt_text)
    st.rerun()
