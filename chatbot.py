import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="WBC BrewBot", layout="wide")

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentExecutor, create_react_agent, tool
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=os.environ["GEMINI_API_KEY"])

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ["PINECONE_INDEX_NAME"]
    index = pc.Index(index_name)
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-base")
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    # Tell the SQL agent how to use the drinks table effectively
    custom_table_info = {
        "drinks": """
        This table contains a complete list of all beverages.
        - To filter by type (e.g., beers, ciders), use the 'type' column.
        - To filter by style (e.g. IPA, Stout), use the 'style' column with a case-insensitive match (ILIKE).
        """
    }
    
    db_uri = (
        f"postgresql://{os.environ['SUPABASE_DB_USER']}:{os.environ['SUPABASE_DB_PASSWORD']}"
        f"@{os.environ['SUPABASE_DB_HOST']}:{os.environ['SUPABASE_DB_PORT']}/{os.environ['SUPABASE_DB_NAME']}"
    )
    db = SQLDatabase.from_uri(db_uri, custom_table_info=custom_table_info)

except (KeyError, Exception) as e:
    st.error(f"Failed to initialize one or more services. Please check your .env file and API keys, especially the SUPABASE_DB_* variables for the SQL agent. Error: {e}")
    st.stop()


# Format retrieved documents with their metadata for readability
def format_docs_with_metadata(docs: list[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        content = f"Result {i+1}:\nDescription: {doc.page_content}\n"
        
        meta_info = []
        if 'name' in doc.metadata:
            meta_info.append(f"Name: {doc.metadata['name']}")
        if 'type' in doc.metadata:
            meta_info.append(f"Type: {doc.metadata['type']}")
        if 'style' in doc.metadata:
            meta_info.append(f"Style: {doc.metadata['style']}")
        if 'abv' in doc.metadata:
            meta_info.append(f"ABV: {doc.metadata['abv']}")
        if 'ibu' in doc.metadata:
            meta_info.append(f"IBU: {doc.metadata['ibu']}")
        
        if meta_info:
            content += "Details: " + " | ".join(meta_info)

        formatted_docs.append(content)
        
    return "\n---\n".join(formatted_docs)


# Semantic search for general brewery info and subjective queries
@tool
def semantic_brewery_search(query: str) -> str:
    """
    Use this tool to search for general, descriptive information about the brewery, 
    its history, hours, location, atmosphere, and specific drinks when a precise database query is not possible.
    This tool is excellent for questions based on flavors, textures, or feelings like 'what is a good summer beer?' 
    or 'tell me about the brewery'. It should be used as a fallback if the drink_database_tool finds no results.
    """
    st.sidebar.info("Used Semantic Search for a descriptive answer.")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.5},
    )
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found in the brewery's general information."
    
    return format_docs_with_metadata(docs)


# SQL agent handles structured database queries
sql_agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="openai-tools",
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)

# Primary tool for drink lookups with fallback to semantic search
@tool
def drink_database_tool(query: str) -> str:
    """
    Use this primary tool for ANY question that requires finding, listing, counting, ranking, or filtering 
    the brewery's drinks based on specific criteria like ABV, IBU, style, or type.
    For example: 'What is the strongest beer?', 'Do you have any IPAs?', 'How many ciders are there?'
    This tool is NOT for subjective recommendations (e.g., 'what tastes good?').
    If this tool returns an empty or "no results" answer, you MUST use the `semantic_brewery_search` tool as a fallback.
    """
    try:
        st.sidebar.info("Attempting a precise database search (SQL)...")
        sql_result = sql_agent_executor.invoke({"input": query})
        output = sql_result.get("output", "")

        negative_keywords = [
            "no results", "no drinks", "no beers", "no ciders",
            "are no", "are not any", "don't have any", "do not have any",
            "doesn't have", "does not have"
        ]
        
        is_meaningful = output and "[]" not in output and not any(kw in output.lower() for kw in negative_keywords)

        if is_meaningful:
            st.sidebar.success("SQL Tool found a definitive answer.")
            return output
        else:
            st.sidebar.warning("SQL did not find a definitive answer.")
            return "No definitive results were found in the drink database. Try a different search."

    except Exception as e:
        print(f"SQL Agent failed. Error: {e}")
        return "The drink database could not be searched due to an error."


tools = [drink_database_tool, semantic_brewery_search]

supervisor_prompt_template = """
You are a helpful and friendly assistant for the Wellsville Brewing Company named 'BrewBot'.
Your primary goal is to provide accurate information from the brewery's database and documents.
Answer the user's questions as best as possible. You have access to the following tools:

{tools}

**IMPORTANT USAGE INSTRUCTIONS:**
1. For questions about listing, counting, or filtering drinks (e.g., "how many IPAs", "strongest beer", "list all ciders"), you MUST use the `drink_database_tool` first.
2. If the `drink_database_tool` returns an empty or "no results" answer, you MUST then use the `semantic_brewery_search` tool with the same query to find descriptive information as a fallback.
3. For general questions about the brewery, its hours, history, or for subjective recommendations (e.g., "what's a good summer beer?"), you should use the `semantic_brewery_search` tool directly.

Use the following format to answer the question:

Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer OR I need to ask the user for more information.
Final Answer: The final answer to the original input question OR a clarifying question to the user.

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

prompt = ChatPromptTemplate.from_template(supervisor_prompt_template)

supervisor_agent = create_react_agent(llm, tools, prompt)
supervisor_agent_executor = AgentExecutor(
    agent=supervisor_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)


def process_prompt(user_prompt: str):
    st.session_state.messages.append(HumanMessage(content=user_prompt))
    with st.spinner("Thinking..."):
        try:
            response = supervisor_agent_executor.invoke({
                "input": user_prompt,
                "chat_history": st.session_state.messages
            })
            result = response['output']
        except Exception as e:
            st.error(f"An error occurred: {e}")
            result = "I'm sorry, I encountered an error. Please try again."
    st.session_state.messages.append(AIMessage(content=result))

st.title("Wellsville Brewing Info BrewBot! 🍺")
st.sidebar.title("Controls")
if st.sidebar.button("Restart Chat"):
    st.session_state.clear()
    st.rerun()
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Welcome to the Wellsville Brewing Company! I'm the Info BrewBot. How can I help you today?")
    ]
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)
if len(st.session_state.messages) == 1:
    st.markdown("---")
    st.subheader("You can ask me about...")
    cols = st.columns(3)
    suggestions = {
        "The Business 🏢": ["Tell me about the brewery", "Who are the owners?"],
        "Hours & Location 🕒": ["What are your hours?", "Where are you located?"],
        "The Brews 🍻": ["How many beers do you offer?", "Do you have any IPAs?", "Rank all beers by IBU", "What is the strongest drink?"]
    }
    for i, (title, questions) in enumerate(suggestions.items()):
        with cols[i]:
            st.markdown(f"#### {title}")
            for q in questions:
                if st.button(q, use_container_width=True):
                    process_prompt(q)
                    st.rerun()
if prompt := st.chat_input("Ask me anything about the brewery!"):
    process_prompt(prompt)
    st.rerun()
