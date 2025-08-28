# chatbot.py
import streamlit as st
import os
from dotenv import load_dotenv

# Pinecone
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

# LangChain embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# LangChain messages
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document

# Gemini API
import google.generativeai as genai

# Supabase
from supabase import create_client, Client

load_dotenv()

st.set_page_config(page_title="WBC BrewBot", layout="wide")
st.title("Wellsville Brewing Info BrewBot! 🍻")

# -------------------------
# NEW: Restart Chat Button in Sidebar
# -------------------------
st.sidebar.title("Controls")
if st.sidebar.button("Restart Chat"):
    # Clear the chat history from session state
    st.session_state.messages = []
    # Rerun the app to reset the UI
    st.rerun()

# -------------------------
# Initialize API Clients
# -------------------------

# Configure Gemini API
genai.configure(api_key=os.environ.get("GEMINI_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_API_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE-API-KEY"))
index_name = os.environ.get("PINECONE-INDEX-NAME")
index = pc.Index(index_name)

# Initialize embeddings & vector store
embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-base")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)


# -------------------------
# Helper Functions
# -------------------------

def call_gemini(messages, context=""):
    gemini_messages = []
    if context:
        system_prompt = (
            "You are a helpful assistant for the Wellsville Brewing Company. "
            "Use the following pieces of context to answer the question. "
            "The context may include 'Facts' from a database and 'Semantic Context' from documents. "
            "If the 'Facts' section provides a direct answer, prioritize it. "
            "Keep the answer concise and conversational. If you don't know the answer, just say so."
        )
        full_context = f"{system_prompt}\n\nContext:\n{context}"
        gemini_messages.append({"role": "user", "parts": [full_context]})
        gemini_messages.append({"role": "model", "parts": ["Understood. I will prioritize facts from the database and use the semantic context to provide a complete, conversational answer."]})

    for message in messages:
        if isinstance(message, HumanMessage):
            gemini_messages.append({"role": "user", "parts": [message.content]})
        elif isinstance(message, AIMessage):
            gemini_messages.append({"role": "model", "parts": [message.content]})
        elif isinstance(message, SystemMessage):
            pass 
        else:
            print(f"Skipping unknown message type: {type(message)}")
            continue
    try:
        response = model.generate_content(gemini_messages)
        return response.text
    except Exception as e:
        print(f"Error calling Gemini: {e}")
        return "I encountered an error processing your request."

def query_supabase(prompt: str) -> str | None:
    prompt_lower = prompt.lower()
    if "highest abv" in prompt_lower or "strongest" in prompt_lower:
        query = supabase.table("drinks").select("name, abv, type").order("abv", desc=True).limit(1)
        if "beer" in prompt_lower:
            query = query.eq("type", "beer")
        response = query.execute()
        if response.data:
            drink = response.data[0]
            return f"Fact: The {drink['type']} with the highest ABV is {drink['name']} at {drink['abv']}%."
    if "highest ibu" in prompt_lower:
        response = supabase.table("drinks").select("name, ibu").eq("type", "beer").order("ibu", desc=True).limit(1).execute()
        if response.data:
            drink = response.data[0]
            return f"Fact: The beer with the highest IBU is {drink['name']} with an IBU of {drink['ibu']}."
    if "gluten free" in prompt_lower:
        response = supabase.table("drinks").select("name").eq("gluten_free", True).execute()
        if response.data:
            drink_names = ', '.join([d['name'] for d in response.data])
            return f"Fact: Gluten-free options are: {drink_names}."
        else:
            return "Fact: There are no gluten-free options available."
    return None

# -------------------------
# Refactored Core Logic
# -------------------------
def process_prompt(user_prompt: str):
    """
    This function takes a user's prompt, processes it, and adds the response to the chat history.
    """
    st.session_state.messages.append(HumanMessage(user_prompt))

    with st.spinner("Thinking..."):
        structured_context = query_supabase(user_prompt)
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": 0.5},
        )
        docs = retriever.invoke(user_prompt)
        semantic_context = "\n".join(
            f"- {d.metadata.get('name', 'N/A')}: {d.page_content}" for d in docs
        )
        combined_context = ""
        if structured_context:
            combined_context += f"Structured Data:\n{structured_context}\n\n"
        if semantic_context:
            combined_context += f"Semantic Context:\n{semantic_context}"

        result = call_gemini([st.session_state.messages[-1]], context=combined_context.strip())

    st.session_state.messages.append(AIMessage(result))


# -------------------------
# Initialize and Display Chat History
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# -------------------------
# Welcome Screen and Suggestion Buttons
# -------------------------
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("Welcome to the Wellsville Brewing Company! I'm the Info BrewBot. How can I help you today?")
        
    st.markdown("---")
    st.subheader("You can ask me about...")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 🏢 The Business")
        if st.button("Tell me about the brewery", use_container_width=True):
            process_prompt("Tell me about the brewery")
            st.rerun()
        if st.button("Who are the owners?", use_container_width=True):
            process_prompt("Who owns the brewery?")
            st.rerun()

    with col2:
        st.markdown("#### 🕒 Hours & Location")
        if st.button("What are your hours?", use_container_width=True):
            process_prompt("What are your hours?")
            st.rerun()
        if st.button("Where are you located?", use_container_width=True):
            process_prompt("Where are you located?")
            st.rerun()

    with col3:
        st.markdown("#### 🍺 The Brews")
        if st.button("What's the strongest beer?", use_container_width=True):
            process_prompt("What is the strongest beer on tap?")
            st.rerun()
        if st.button("Any gluten-free options?", use_container_width=True):
            process_prompt("Do you have any gluten-free options?")
            st.rerun()

# -------------------------
# Main Chat Input
# -------------------------
if prompt := st.chat_input("Ask me anything about the Wellsville Brewing Company!"):
    process_prompt(prompt)
    st.rerun()