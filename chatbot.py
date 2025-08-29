# chatbot.py
import streamlit as st
import os
import re
from dotenv import load_dotenv

# Pinecone
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

# LangChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Gemini API
import google.generativeai as genai

# Supabase
from supabase import create_client, Client

# --- Initialization ---
load_dotenv()
st.set_page_config(page_title="WBC BrewBot", layout="wide")

# --- API Clients & Model Configuration ---
try:
    # Configure Gemini API
    genai.configure(api_key=os.environ["GEMINI_KEY"])
    GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-flash")

    # Initialize Supabase client
    supabase_url = os.environ["SUPABASE_URL"]
    supabase_key = os.environ["SUPABASE_API_KEY"]
    SUPABASE_CLIENT: Client = create_client(supabase_url, supabase_key)

    # Initialize Pinecone
    pc = Pinecone(api_key=os.environ["PINECONE-API-KEY"])
    index_name = os.environ["PINECONE-INDEX-NAME"]
    index = pc.Index(index_name)

    # Initialize embeddings & vector store
    EMBEDDINGS = HuggingFaceEmbeddings(model_name="intfloat/e5-base")
    VECTOR_STORE = PineconeVectorStore(index=index, embedding=EMBEDDINGS)

except (KeyError, Exception) as e:
    st.error(f"Failed to initialize one or more services. Please check your .env file and API keys. Error: {e}")
    st.stop()


# --- Helper Functions ---

def rewrite_query_with_history(messages: list) -> str:
    """
    Uses an LLM to rewrite the user's latest query to be self-contained,
    incorporating context from the chat history. This is a robust way to
    handle conversational follow-up questions.
    """
    if not messages:
        return ""
    if len(messages) == 1:
        return messages[0].content

    formatted_history = ""
    # Format all messages except the very last one (the user's current question)
    for msg in messages[:-1]:
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        formatted_history += f"{role}: {msg.content}\n"
    
    last_user_question = messages[-1].content

    # --- START OF MODIFICATION: More specific instructions for the LLM ---
    # This new prompt explicitly tells the model how to handle general vs. specific questions,
    # preventing it from incorrectly narrowing the user's intent.
    rewrite_prompt = f"""Given the following chat history and a final user question, rewrite the user's question to be a single, standalone question.

Follow these strict rules:
1.  If the user's question is a follow-up that uses pronouns (e.g., "what is its ABV?", "tell me more about those"), incorporate the specific subjects from the chat history (e.g., "what is the ABV of Killer Lite?", "tell me more about the gluten-free options").
2.  If the user's question is a new, general question (e.g., "what drink has the highest ABV?", "what are your hours?"), return it exactly as is. Do not add context from the history.
3.  **Crucially, do not add restrictive keywords (like 'beer' or 'cider') to a general keyword (like 'drink')** unless the user's *latest* question uses a pronoun or is clearly a follow-up.
4.  Your output must only be the rewritten question and nothing else.

Chat History:
{formatted_history}
Final User Question: {last_user_question}

Standalone Question:"""
    # --- END OF MODIFICATION ---

    try:
        response = GEMINI_MODEL.generate_content(rewrite_prompt)
        return response.text.strip() if response.parts else last_user_question
    except Exception as e:
        print(f"Error during query rewriting: {e}")
        return last_user_question # Fallback to the original question


def call_gemini(messages: list, context: str = "") -> str:
    """Generates a response from the Gemini model with optional context."""
    system_prompt = (
        "You are a helpful and friendly assistant for the Wellsville Brewing Company. "
        "Use the following pieces of context to answer the question. "
        "The context may include 'Facts' from a database and 'Semantic Context' from documents. "
        "If the 'Facts' section provides a direct answer, prioritize it. "
        "When the context provides specific data like ABV or IBU values, you MUST include them in your answer. "
        "Keep the answer concise and conversational. If you don't know the answer from the context, just say so."
    )

    gemini_messages = [{"role": "user", "parts": [system_prompt]}]
    if context:
        gemini_messages[0]["parts"].append(f"\nContext:\n{context}")
    
    gemini_messages.append({"role": "model", "parts": ["Understood. I will act as the Wellsville Brewing Company assistant and use the provided context to answer questions."]})

    for message in messages:
        role = "user" if isinstance(message, HumanMessage) else "model"
        gemini_messages.append({"role": role, "parts": [message.content]})

    try:
        response = GEMINI_MODEL.generate_content(gemini_messages)
        return response.text
    except Exception as e:
        print(f"Error calling Gemini: {e}")
        return "I'm sorry, I encountered an error while processing your request."


def query_supabase(prompt: str) -> str | None:
    """
    Queries the Supabase database for structured information based on a standalone prompt.
    """
    prompt_lower = prompt.lower()

    drink_type_filter = None
    if "beer" in prompt_lower: drink_type_filter = "beer"
    elif "cider" in prompt_lower: drink_type_filter = "cider"
    elif "seltzer" in prompt_lower: drink_type_filter = "seltzer"

    # --- Ranking Queries ---
    if any(word in prompt_lower for word in ["rank", "list", "order", "sort", "top", "most"]):
        limit_match = re.search(r'(\d+)', prompt_lower)
        limit = int(limit_match.group(1)) if limit_match else None

        if "abv" in prompt_lower or "alcohol" in prompt_lower:
            query = SUPABASE_CLIENT.table("drinks").select("name, abv, type")
            if drink_type_filter: query = query.eq("type", drink_type_filter)
            
            is_ascending = any(word in prompt_lower for word in ["lowest", "ascending", "weakest"])
            query = query.order("abv", desc=not is_ascending)
            order_desc = "lowest to highest" if is_ascending else "highest to lowest"
            if limit: query = query.limit(limit)
            
            response = query.execute()
            if response.data:
                ranked_list = "\n".join([f"- {d['name']} ({d['abv']}% ABV)" for d in response.data])
                return f"Fact: Here are the drinks ranked by ABV ({order_desc}):\n{ranked_list}"

        if "ibu" in prompt_lower or "bitter" in prompt_lower:
            query = SUPABASE_CLIENT.table("drinks").select("name, ibu").eq("type", "beer").not_.is_("ibu", "null")
            is_ascending = any(word in prompt_lower for word in ["lowest", "least bitter", "ascending"])
            query = query.order("ibu", desc=not is_ascending)
            order_desc = "least to most bitter" if is_ascending else "most to least bitter"
            if limit: query = query.limit(limit)

            response = query.execute()
            if response.data:
                ranked_list = "\n".join([f"- {d['name']} ({d['ibu']} IBU)" for d in response.data])
                return f"Fact: Here are the beers ranked by IBU ({order_desc}):\n{ranked_list}"

    # --- Specific Fact Queries ---
    if any(word in prompt_lower for word in ["highest abv", "strongest", "most alcohol"]):
        query = SUPABASE_CLIENT.table("drinks").select("name, abv, type").order("abv", desc=True).limit(1)
        if drink_type_filter: query = query.eq("type", drink_type_filter)
        response = query.execute()
        if response.data:
            drink = response.data[0]
            return f"Fact: The {drink['type']} with the highest ABV is {drink['name']} at {drink['abv']}%."

    if any(word in prompt_lower for word in ["lowest abv", "weakest", "least alcohol"]):
        query = SUPABASE_CLIENT.table("drinks").select("name, abv, type").order("abv", desc=False).limit(1)
        if drink_type_filter: query = query.eq("type", drink_type_filter)
        response = query.execute()
        if response.data:
            drink = response.data[0]
            return f"Fact: The {drink['type']} with the lowest ABV is {drink['name']} at {drink['abv']}%."
    
    if "gluten free" in prompt_lower or "gluten-free" in prompt_lower:
        response = SUPABASE_CLIENT.table("drinks").select("name, type").eq("gluten_free", True).execute()
        if response.data:
            names = ', '.join([f"{d['name']} ({d['type']})" for d in response.data])
            return f"Fact: Gluten-free options are: {names}."
        return "Fact: There are no gluten-free options available."

    if any(p in prompt_lower for p in ["all drinks", "what do you have", "list all"]):
        query = SUPABASE_CLIENT.table("drinks").select("name, type, abv")
        if drink_type_filter: query = query.eq("type", drink_type_filter)
        response = query.execute()
        if response.data:
            drinks_by_type = {}
            for drink in response.data:
                drinks_by_type.setdefault(drink['type'], []).append(f"- {drink['name']} ({drink['abv']}% ABV)")
            result = "Fact: Here are all available drinks:\n"
            for dtype, dlist in drinks_by_type.items():
                result += f"\n{dtype.capitalize()}s:\n" + "\n".join(dlist)
            return result

    # --- Fallback: Check for specific drink names mentioned in the prompt ---
    all_drinks_resp = SUPABASE_CLIENT.table("drinks").select("name, abv, ibu").execute()
    if all_drinks_resp.data:
        matched = []
        for drink in all_drinks_resp.data:
            if drink["name"].lower() in prompt_lower:
                details = f"{drink['name']}: {drink['abv']}% ABV"
                if drink.get('ibu') is not None:
                    details += f", {drink['ibu']} IBU"
                matched.append(details)
        if matched:
            return "Fact: " + ", ".join(matched) + "."

    return None


# --- Core Application Logic ---

def process_prompt(user_prompt: str):
    """
    Main function to handle user input, orchestrate retrieval and generation,
    and manage session state.
    """
    st.session_state.messages.append(HumanMessage(content=user_prompt))

    with st.spinner("Thinking..."):
        # 1. Rewrite the user's query to be self-contained using chat history
        standalone_query = rewrite_query_with_history(st.session_state.messages)
        st.sidebar.info(f"**Rewritten Query:**\n\n{standalone_query}") # Debugging

        # 2. Retrieve context using the standalone query
        structured_context = query_supabase(standalone_query)
        semantic_context = ""
        
        # 3. If no structured data is found, perform a semantic search
        if not structured_context:
            retriever = VECTOR_STORE.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 3, "score_threshold": 0.6},
            )
            docs = retriever.invoke(standalone_query)
            semantic_context = "\n".join([d.page_content for d in docs])
        
        # 4. Combine context and generate the final response
        context_parts = []
        if structured_context:
            context_parts.append(f"Structured Facts:\n{structured_context}")
        if semantic_context:
            context_parts.append(f"Relevant Information:\n{semantic_context}")
        
        combined_context = "\n\n".join(context_parts)
        
        # The main LLM call uses the original message history for conversational context
        result = call_gemini(st.session_state.messages, context=combined_context)
    
    st.session_state.messages.append(AIMessage(content=result))


# --- Streamlit UI ---

st.title("Wellsville Brewing Info BrewBot! 🍺")

# Sidebar for controls
st.sidebar.title("Controls")
if st.sidebar.button("Restart Chat"):
    st.session_state.clear()
    st.rerun()

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Welcome to the Wellsville Brewing Company! I'm the Info BrewBot. How can I help you today?")
    ]

# Display chat history
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Display suggestion buttons only on first load
if len(st.session_state.messages) == 1:
    st.markdown("---")
    st.subheader("You can ask me about...")
    cols = st.columns(3)
    suggestions = {
        "The Business 🏢": ["Tell me about the brewery", "Who are the owners?"],
        "Hours & Location 🕒": ["What are your hours?", "Where are you located?"],
        "The Brews 🍻": ["What's the strongest drink?", "Any gluten-free options?", "Rank beers by IBU", "What IPAs do you have?"]
    }
    for i, (title, questions) in enumerate(suggestions.items()):
        with cols[i]:
            st.markdown(f"#### {title}")
            for q in questions:
                if st.button(q, use_container_width=True):
                    process_prompt(q)
                    st.rerun()

# Main chat input
if prompt := st.chat_input("Ask me anything about the brewery!"):
    process_prompt(prompt)
    st.rerun()