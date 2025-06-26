import streamlit as st
import pandas as pd
import os
import difflib
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langdetect import detect

# --- Streamlit Setup ---
st.set_page_config(page_title="🎵 Mood Song Finder", page_icon="🎶")
st.title("🎶 Mood Song Finder")
st.markdown("Find songs that match your mood. Powered by LangChain + Gemini LLM.")

# --- API Key Input ---
if "GOOGLE_API_KEY" not in st.session_state:
    st.session_state.GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or st.text_input("Enter your **Google API Key**", type="password")

if not st.session_state.GOOGLE_API_KEY:
    st.warning("Please enter your API Key to continue.", icon="🔑")
    st.stop()

os.environ["GOOGLE_API_KEY"] = st.session_state.GOOGLE_API_KEY

# --- Load LLM & Memory ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- Load Dataset ---
csv_url = "https://raw.githubusercontent.com/annhaura/mood-song-finder/main/spotify_songs.csv"
with st.spinner("📥 Loading dataset..."):
    df = pd.read_csv(csv_url).head(300)
    df["combined_text"] = df.apply(lambda row:
        f"{row['track_name']} by {row['track_artist']}. "
        f"Genre: {row['playlist_genre']} ({row['playlist_subgenre']}). "
        f"Valence: {row['valence']:.2f}, Energy: {row['energy']:.2f}, Danceability: {row['danceability']:.2f}.", axis=1)

ALL_GENRES = sorted(set(df["playlist_genre"].dropna().unique()))

# --- Functions ---
def detect_requested_genre(user_input: str, all_genres: list[str]) -> str:
    lowered = user_input.lower()
    for genre in all_genres:
        if genre.lower() in lowered:
            return genre
    return ""

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

def explain_recommendation(song_title: str, mood: str, lang: str, user_input: str = "") -> str:
    try:
        if lang == "id":
            prompt = f"Singkat saja ya. Jelaskan dalam 1–2 kalimat kenapa lagu '{song_title}' cocok untuk mood '{mood}', berdasarkan pernyataan: '{user_input}'. Gaya bicara santai dan langsung ke inti."
        else:
            prompt = f"In 1–2 short sentences, explain why '{song_title}' fits the mood '{mood}' based on: '{user_input}'. Be direct, warm, and avoid repeating emotions already mentioned."
        return llm.invoke(prompt).content.strip()
    except:
        return "❗ Gagal mengambil penjelasan."

def generate_intro(user_input: str, mood: str, lang: str) -> str:
    try:
        if lang == "id":
            prompt = f"Respons singkat dan empatik untuk seseorang yang bilang: '{user_input}' (mood: '{mood}'). Hindari basa-basi berlebihan. 2–3 kalimat cukup."
        else:
            prompt = f"Write a short, warm, and natural response to someone who says: '{user_input}' (mood: '{mood}'). Avoid excessive intro or dramatic tone. 2–3 short sentences only."
        return llm.invoke(prompt).content.strip()
    except:
        return ""

def is_followup_input(user_input: str) -> bool:
    prompt = (
        "This is part of an AI conversation for music recommendation.\n"
        "Decide whether the following message is a follow-up to the previous conversation about mood/music.\n"
        "If YES, reply exactly with 'yes'. If it's a new topic, reply 'no'.\n\n"
        f"Previous conversation: {st.session_state.last_input}\n"
        f"Current message: {user_input}"
    )
    try:
        result = llm.invoke(prompt).content.strip().lower()
        return result == "yes"
    except:
        return False

def recommend_song_by_genre_via_llm(mood: str, genre: str, lang: str) -> str:
    prompt = (
        f"Beri 1–2 rekomendasi lagu yang cocok untuk mood '{mood}' dan genre '{genre}'. Tampilkan nama lagu dan artis, lalu berikan penjelasan singkat."
        if lang == "id" else
        f"Recommend 1–2 songs that fit the mood '{mood}' in the '{genre}' genre. Include title, artist, and a short explanation."
    )
    try:
        return llm.invoke(prompt).content.strip()
    except:
        return "❗ Gagal mendapatkan lagu dari LLM fallback."

def retrieve_songs_from_dataset(query: str, genre: str = "", k=2) -> list[Document]:
    filtered_df = df
    if genre:
        filtered_df = df[df["playlist_genre"].str.contains(genre, case=False, na=False)]
    docs = [Document(page_content=row["combined_text"], metadata={"index": i}) for i, row in filtered_df.iterrows()]
    vectorstore = FAISS.from_documents(docs, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
    results = vectorstore.similarity_search(query, k=10)
    return [r for r in results if r.page_content not in st.session_state.seen_songs][:k]

# --- Session States ---
for key in ["chat_history", "seen_songs", "last_lang", "last_input", "last_mood", "last_genre"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "chat_history" else set() if key == "seen_songs" else ""

# --- Chat Input ---
user_input = st.chat_input("What kind of music do you want to hear today?")
if user_input:
    with st.spinner("🤖 Thinking..."):
        lang = detect_language(user_input)
        if lang != st.session_state.last_lang:
            st.session_state.last_lang = lang

        is_followup = is_followup_input(user_input)

        if is_followup and st.session_state.last_mood:
            mood = st.session_state.last_mood
            genre = st.session_state.last_genre
        else:
            mood = classify_mood(user_input)
            genre = infer_genre(f"The user said: '{user_input}'. Mood: {mood}.")
            st.session_state.last_mood = mood
            st.session_state.last_genre = genre

        requested_genre = detect_requested_genre(user_input, ALL_GENRES)
        query_context = f"User said: '{user_input}'. Mood: {mood}. Genre: {genre}."
        if requested_genre:
            query_context += f" Explicitly requested: {requested_genre}."

        results = retrieve_songs_from_dataset(query_context, genre=requested_genre, k=2)

        if not results:
            result = recommend_song_by_genre_via_llm(mood, requested_genre or genre, lang)
        else:
            intro = generate_intro(user_input, mood, lang)
            response_lines = [intro, ""]
            for song in results:
                st.session_state.seen_songs.add(song.page_content)
                reason = explain_recommendation(song.page_content, mood, lang, user_input)
                response_lines.append(f"🎵 {song.page_content} 👉 {reason}")
            result = "\n\n".join(response_lines)

        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("AI", result))
        st.session_state.last_input = user_input

# --- Display Chat History ---
for speaker, text in st.session_state.chat_history:
    with st.chat_message(speaker):
        st.markdown(text)

# --- Limit History Length ---
st.session_state.chat_history = st.session_state.chat_history[-10:]
