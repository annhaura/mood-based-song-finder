# RAG + Agentic AI version of Mood Song Finder

import streamlit as st
import os
import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langdetect import detect

# --- Streamlit UI Setup ---
st.set_page_config(page_title="🎵 Mood Song Finder RAG", page_icon="🎶")
st.title("🎶 Mood Song Finder RAG")
st.markdown("Find songs that match your mood. Powered by LangChain + Gemini RAG agent.")

# --- API Key Input ---
api_key = st.secrets.get("GOOGLE_API_KEY") or st.text_input("Enter your **Google API Key**", type="password")
if not api_key:
    st.warning("Please enter your API Key to continue.", icon="🔑")
    st.stop()
os.environ["GOOGLE_API_KEY"] = api_key

# --- Load Dataset + Embedding + Vectorstore ---
csv_url = "https://raw.githubusercontent.com/annhaura/mood-based-song-finder/main/spotify_songs.csv"
df = pd.read_csv(csv_url).head(300)
df["combined_text"] = df.apply(lambda row: f"{row['track_name']} by {row['track_artist']}", axis=1)
documents = [Document(page_content=text, metadata=row.to_dict()) for text, (_, row) in zip(df["combined_text"], df.iterrows())]

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

@st.cache_resource
def load_vectorstore():
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embedding)
    else:
        vs = FAISS.from_documents(documents, embedding)
        vs.save_local("faiss_index")
        return vs

vectorstore = load_vectorstore()

# --- Tools ---
@tool
def detect_language_tool(text: str) -> str:
    """Detect language code from text, like 'en' or 'id'."""
    return detect(text)

@tool
def classify_mood_tool(text: str) -> str:
    """Classify emotional mood from user text input (e.g. happy, sad, energetic)."""
    return llm.invoke(f"What is the user's mood in 1-3 words?\n{text}").content.strip().lower()

@tool
def infer_genre_tool(text: str) -> str:
    """Infer a suitable music genre based on user input."""
    return llm.invoke(f"Suggest a suitable music genre: {text}").content.strip()

@tool
def retrieve_similar_songs_tool(query: str) -> list:
    """Retrieve similar songs to the query from the dataset."""
    results = vectorstore.similarity_search(query, k=5)
    return [doc.page_content for doc in results]

@tool
def explain_recommendation_tool(song_title: str, mood: str, user_input: str, lang: str) -> str:
    """Explain why the song matches the user's mood/input."""
    if lang == "id":
        prompt = f"Kenapa lagu '{song_title}' cocok untuk mood '{mood}' berdasarkan: '{user_input}'? 1-2 kalimat. Gaya santai."
    else:
        prompt = f"Explain in 1-2 short sentences why '{song_title}' fits mood '{mood}' from input: '{user_input}'."
    return llm.invoke(prompt).content.strip()

# --- LLM & Prompt Template ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful music recommendation agent."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# --- Agent Setup ---
tools = [
    detect_language_tool,
    classify_mood_tool,
    infer_genre_tool,
    retrieve_similar_songs_tool,
    explain_recommendation_tool,
]

agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- Memory ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Chat Input ---
user_input = st.chat_input("Apa yang ingin kamu dengar hari ini?")
if user_input:
    st.chat_message("user").markdown(user_input)

    with st.spinner("🤖 Agent is thinking..."):
        result = agent_executor.invoke({
            "input": user_input,
            "chat_history": st.session_state.chat_history
        })
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=result["output"]))
        st.chat_message("assistant").markdown(result["output"])

# --- Display Previous Chat History ---
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)
