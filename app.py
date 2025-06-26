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
            prompt = (
                f"Singkat saja. Jelaskan dalam 1–2 kalimat kenapa lagu '{song_title}' cocok untuk mood '{mood}', "
                f"berdasarkan pernyataan: '{user_input}', tanpa basa-basi, langsung ke poin."
            )
        else:
            prompt = (
                f"Briefly explain in 1–2 sentences why the song '{song_title}' fits the mood '{mood}', "
                f"based on: '{user_input}', without repeating the mood or feelings already acknowledged. Be clear and to the point."
            )
        return llm.invoke(prompt).content.strip()
    except:
        return "❗ Gagal mengambil penjelasan."

def generate_intro(user_input: str, mood: str, lang: str) -> str:
    try:
        if lang == "id":
            prompt = f"Tulis satu kalimat pendek dan netral menanggapi: '{user_input}'\nMood-nya: '{mood}'. Jangan terlalu panjang atau berlebihan."
        else:
            prompt = f"Write a short and neutral sentence responding to: '{user_input}'\nMood: '{mood}'. Avoid dramatic tone."
        return llm.invoke(prompt).content.strip()
    except:
        return ""

def is_followup_input(user_input: str) -> bool:
    prompt = (
        "Is this message a follow-up in a conversation about music recommendation?\n"
        "If it's continuing a previous mood or vibe, or asking for more music in a similar context, reply only 'yes'.\n"
        "If it's a new topic or a different mood, reply only 'no'.\n\n"
        f"Message: {user_input}"
    )
    try:
        result = llm.invoke(prompt).content.strip().lower()
        return result == "yes"
    except:
        return False

# --- Memory Init ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "seen_songs" not in st.session_state:
    st.session_state.seen_songs = set()
if "last_lang" not in st.session_state:
    st.session_state.last_lang = "en"
if "last_input" not in st.session_state:
    st.session_state.last_input = ""
if "last_mood" not in st.session_state:
    st.session_state.last_mood = ""
if "last_genre" not in st.session_state:
    st.session_state.last_genre = ""

# --- Chat Input ---
user_input = st.chat_input("What kind of music do you want to hear today?")
if user_input:
    with st.spinner("🤖 Thinking..."):
        lang = detect_language(user_input)
        if lang != st.session_state.last_lang:
            st.session_state.last_lang = lang

        is_followup = is_followup_input(user_input)

        if is_followup:
            mood = st.session_state.last_mood
            genre = st.session_state.last_genre
            semantic_input = f"User said: '{user_input}' (context: '{st.session_state.last_input}'). Mood: {mood}. Genre: {genre}. Suggest fitting songs."
        else:
            mood = classify_mood(user_input)
            genre = infer_genre(f"The user said: '{user_input}'. Mood: {mood}.")
            semantic_input = f"User said: '{user_input}'. Interpreted mood: {mood}. Genre suggestion: {genre}. Suggest fitting songs."
            st.session_state.last_mood = mood
            st.session_state.last_genre = genre

        songs = retrieve_similar_songs(semantic_input, k=2, exclude=st.session_state.seen_songs)

        if not songs:
            result = (
                "😕 Belum nemu lagu yang pas. Mau coba mood atau genre lain?"
                if lang == "id"
                else "Hmm, I can't find more songs right now. Want to try another mood or genre?"
            )
        else:
            intro = generate_intro(user_input, mood, lang)
            response_lines = [intro, ""]
            for song in songs:
                if song.page_content not in st.session_state.seen_songs:
                    st.session_state.seen_songs.add(song.page_content)
                    reason = explain_recommendation(song.page_content, mood, lang, user_input)
                    line = f"🎵 {song.page_content} 👉 {reason}"
                    if line not in response_lines:
                        response_lines.append(line)
            result = "\n\n".join(response_lines)

        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("AI", result))
        st.session_state.last_input = user_input

# --- Display Chat History ---
for speaker, text in st.session_state.chat_history:
    with st.chat_message(speaker):
        st.markdown(text)

# --- Limit History Length ---
MAX_HISTORY = 10
st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY:]
