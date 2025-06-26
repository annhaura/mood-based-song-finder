import streamlit as st
import pandas as pd
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langdetect import detect

# --- Streamlit Setup ---
st.set_page_config(page_title="🎵 Mood Song Finder", page_icon="🎶")
st.title("🎶 Mood Song Finder")
st.markdown("Find songs that match your mood. Powered by LangChain + Gemini LLM.")

# --- API Key Input / Secret ---
api_key = st.secrets.get("GOOGLE_API_KEY") or st.text_input("Enter your **Google API Key**", type="password")
if not api_key:
    st.warning("Please enter your API Key to continue.", icon="🔑")
    st.stop()
os.environ["GOOGLE_API_KEY"] = api_key

# --- Load LLM & Memory ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- Load Dataset ---
csv_url = "https://raw.githubusercontent.com/annhaura/mood-song-finder/main/spotify_songs.csv"
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
def detect_language(text: str) -> str:
    try:
        return detect(text)
    except:
        return "en"

def classify_mood(query: str) -> str:
    prompt = f"Describe the user's emotional mood in 1–3 words (e.g. happy, sad, nostalgic, energetic, romantic, etc):\n\n{query}"
    return llm.invoke(prompt).content.strip().lower()

def infer_genre(query: str) -> str:
    prompt = f"Suggest a suitable music genre based on this input: {query}"
    return llm.invoke(prompt).content.strip()

def retrieve_similar_songs(query: str, k=2, exclude=set()) -> list:
    results = vectorstore.similarity_search(query, k=10)
    filtered = [doc for doc in results if doc.page_content not in exclude]
    return filtered[:k]

def explain_recommendation(song_title: str, mood: str, lang: str, user_input: str = "") -> str:
    try:
        if lang == "id":
            prompt = f"Singkat saja ya. Jelaskan dalam 1–2 kalimat kenapa lagu '{song_title}' cocok untuk suasana hati '{mood}', berdasarkan pernyataan: '{user_input}', dalam gaya bicara santai dan empatik."
        else:
            prompt = f"Briefly explain in 1–2 sentences in a warm and friendly tone why the song '{song_title}' fits the mood '{mood}', based on what the user said: '{user_input}'."
        return llm.invoke(prompt).content.strip()
    except:
        return "❗ Gagal mengambil penjelasan."

def generate_intro(user_input: str, mood: str, lang: str) -> str:
    try:
        if lang == "id":
            prompt = f"Buat satu paragraf singkat dan empatik sebagai respons ke seseorang yang berkata: '{user_input}'\nMood-nya adalah: '{mood}'.\nTulis dengan gaya manusiawi, seperti teman curhat."
        else:
            prompt = f"Write a short, empathetic paragraph responding to someone who says: '{user_input}'\nTheir mood is: '{mood}'. Write like a caring friend."
        return llm.invoke(prompt).content.strip()
    except:
        return ""

def suggest_artists(genre: str, lang: str) -> str:
    try:
        if lang == "id":
            prompt = f"Sebutkan 3–5 artis musik populer yang cocok dengan genre '{genre}' dan sedang relevan."
        else:
            prompt = f"Name 3–5 relevant and popular music artists in the '{genre}' genre."
        return llm.invoke(prompt).content.strip()
    except:
        return ""

# --- Memory Init ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "seen_songs" not in st.session_state:
    st.session_state.seen_songs = set()

# --- Chat Input ---
user_input = st.chat_input("What kind of music do you want to hear today?")
if user_input:
    with st.spinner("🤖 Thinking..."):
        lang = detect_language(user_input)
        mood = classify_mood(user_input)
        genre = infer_genre(user_input)
        semantic_input = f"User said: '{user_input}'. Interpreted mood: {mood}. Genre suggestion: {genre}. Suggest fitting songs."

        songs = retrieve_similar_songs(semantic_input, k=2, exclude=st.session_state.seen_songs)

        if not songs:
            result = (
                "😕 Belum nemu lagu yang pas. Mau coba mood atau genre lain?"
                if lang == "id"
                else "Hmm, I can't find more songs right now. Want to try another mood or genre?"
            )
        else:
            intro = generate_intro(user_input, mood, lang)
            artist_info = suggest_artists(genre, lang)
            response_lines = [intro, ""]
            for song in songs:
                st.session_state.seen_songs.add(song.page_content)
                reason = explain_recommendation(song.page_content, mood, lang, user_input)
                line = f"🎵 {song.page_content} 👉 {reason}"
                response_lines.append(line)
            if artist_info:
                response_lines.append("\n**🎤 Recommended Artists:**\n" + artist_info)
            result = "\n\n".join(response_lines)

        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("AI", result))

# --- Display Chat History ---
for speaker, text in st.session_state.chat_history:
    with st.chat_message(speaker):
        st.markdown(text)

# --- Limit History Length ---
MAX_HISTORY = 10
st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY:]
