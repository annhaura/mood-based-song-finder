import streamlit as st
import pandas as pd
import os
from random import shuffle
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# --- Streamlit Setup ---
st.set_page_config(page_title="🎵 Mood Song Finder", page_icon="🎶")
st.title("🎶 Mood Song Finder")
st.markdown("Find songs that match your mood!")

# --- API Key Input  ---
api_key = st.secrets.get("GOOGLE_API_KEY") or st.text_input("Enter your **Google API Key**", type="password")
if not api_key:
    st.warning("Please enter your API Key to continue.", icon="🔑")
    st.stop()
os.environ["GOOGLE_API_KEY"] = api_key

# --- Load LLM & Memory ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- Load Dataset ---
csv_url = "https://raw.githubusercontent.com/annhaura/mood-based-song-finder/main/spotify_songs.csv"
with st.spinner("📥 Loading dataset..."):
    df = pd.read_csv(csv_url).head(300)
    df["combined_text"] = df.apply(lambda row: f"{row['track_name']} by {row['track_artist']}", axis=1)
    documents = [Document(page_content=text, metadata={"index": i}) for i, text in enumerate(df["combined_text"])]

# --- Embedding & Vectorstore ---
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

@st.cache_resource
def load_vectorstore():
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embedding_model)
    else:
        vectorstore = FAISS.from_documents(documents, embedding_model)
        vectorstore.save_local("faiss_index")
        return vectorstore

vectorstore = load_vectorstore()

# --- Tool Functions ---
def classify_mood(query: str) -> str:
    prompt = f"Classify the emotional mood of this text (examples: happy, sad, nostalgic, energetic, romantic):\n\n{query}"
    return llm.invoke(prompt).content.strip().lower()

def infer_genre(query: str) -> str:
    prompt = f"Suggest a suitable music genre for this mood or query:\n\n{query}"
    return llm.invoke(prompt).content.strip()

def retrieve_similar_songs(query: str, k=3) -> str:
    results = vectorstore.similarity_search(query, k=k)
    songs = [f"🎵 {doc.page_content}" for doc in results]
    return "\n".join(songs)

def randomize_list(text_block: str) -> str:
    lines = text_block.strip().splitlines()
    shuffle(lines)
    return "\n".join(lines)

def explain_recommendation(query: str) -> str:
    prompt = f"Explain why these kinds of songs would match this mood or situation:\n\n{query}"
    return llm.invoke(prompt).content.strip()

# --- Tools List ---
tools = [
    Tool(name="MoodClassifier", func=classify_mood, description="Detects emotional mood from user input."),
    Tool(name="InferGenre", func=infer_genre, description="Suggests a suitable music genre."),
    Tool(name="RetrieveSimilarSongs", func=retrieve_similar_songs, description="Finds matching songs from the dataset."),
    Tool(name="Randomizer", func=randomize_list, description="Randomizes the order of song recommendations."),
    Tool(name="ExplainRecommendation", func=explain_recommendation, description="Explains why a song fits the input mood.")
]

# --- Initialize Agent ---
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=False,
)

# --- Streamlit Chat UI ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("What kind of music do you want to hear today?")
if user_input:
    with st.spinner("🤖 Thinking..."):
        mood = classify_mood(user_input)
        genre = infer_genre(user_input)
        response = agent.run(user_input)

        summary = f"**Detected Mood:** `{mood}`\n**Inferred Genre:** `{genre}`\n\n"
        full_response = summary + response

        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("AI", full_response))

# --- Display Chat History ---
for speaker, text in st.session_state.chat_history:
    with st.chat_message(speaker):
        st.markdown(text)

# --- Limit History Length ---
MAX_HISTORY = 10
st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY:]
