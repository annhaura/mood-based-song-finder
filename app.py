import streamlit as st
import os
import pandas as pd
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langdetect import detect

# --- Streamlit Setup ---
st.set_page_config(page_title="Mood-Based Song Finder", page_icon="💿")
st.title("🎧 Mood-Based Song Recommender 🎶")
st.markdown("Find songs that match your mood. Powered by LangChain + Gemini.")

# --- API Key Input ---
api_key = st.secrets.get("GOOGLE_API_KEY") or st.text_input("Enter your **Google API Key**", type="password")
if not api_key:
    st.warning("Please enter your API Key to continue.", icon="🔑")
    st.stop()
os.environ["GOOGLE_API_KEY"] = api_key


# --- LLM Setup ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# --- Load Dataset + Vectorstore ---
csv_url = "https://raw.githubusercontent.com/annhaura/mood-based-song-finder/main/spotify_songs.csv"

@st.cache_data
def load_and_sample_dataset(csv_url, max_per_subgenre=80):
    df_full = pd.read_csv(csv_url)
    df_sampled = df_full.groupby('playlist_subgenre', group_keys=False)\
                        .apply(lambda x: x.sample(min(len(x), max_per_subgenre), random_state=42))\
                        .reset_index(drop=True)
    df_sampled["combined_text"] = df_sampled.apply(
        lambda row: f"{row['track_name']} by {row['track_artist']}", axis=1)
    return df_sampled

df = load_and_sample_dataset(csv_url)
documents = [Document(page_content=row["combined_text"], metadata={"index": i}) for i, row in df.iterrows()]


@st.cache_resource
def load_vectorstore():
    try:
        # Coba load dari local
        if os.path.exists("faiss_index") and os.path.exists("faiss_index/index.faiss"):
            return FAISS.load_local("faiss_index", embedding_model)
        else:
            raise FileNotFoundError
    except:
        # Kalau gagal (karena file nggak ada atau corrupt), bangun ulang
        vectorstore = FAISS.from_documents(documents, embedding_model)
        try:
            vectorstore.save_local("faiss_index")  # Hanya berhasil kalau running lokal
        except:
            pass  # Ignore error kalau di Streamlit Cloud
        return vectorstore

vectorstore = load_vectorstore()

# --- Language Detection ---
def detect_language(text: str) -> str:
    try:
        return detect(text)
    except:
        return "en"

# --- TOOL: Detect Mood ---
def detect_mood_tool(user_input: str) -> str:
    if not user_input.strip():
        return "Mood tidak terdeteksi. Coba masukkan perasaanmu."
    prompt = f"Describe the user's emotional mood in 1–3 words:\n\n{user_input}"
    return llm.invoke(prompt).content.strip()


# --- TOOL: Infer Genre ---
def infer_genre_tool(user_input: str) -> str:
    if not user_input.strip():
        return "Pop"  # fallback genre
    prompt = f"Suggest a suitable music genre for: {user_input}"
    return llm.invoke(prompt).content.strip()


# --- TOOL: Retrieve Songs from Vectorstore ---
def retrieve_songs_tool(query: str) -> str:
    if not query.strip():
        return "Tidak ada input lagu yang valid."
    results = vectorstore.similarity_search(query, k=3)
    if not results:
        return "Tidak ditemukan lagu yang cocok."
    return "\n".join([f"🎵 {doc.page_content}" for doc in results])


# --- TOOL: Explain Why the Song Fits ---
def explain_choice_tool(song_and_mood: str) -> str:
    parts = song_and_mood.split("|")
    if len(parts) != 3:
        return "Input tidak valid. Format: judul | mood | input user"
    title, mood, user_input = parts
    lang = detect(user_input)
    prompt = (
        f"Jelaskan kenapa lagu '{title}' cocok untuk mood '{mood}' berdasarkan input '{user_input}'. Singkat ya."
        if lang == "id"
        else f"Why does the song '{title}' fit the mood '{mood}' from input '{user_input}'? Keep it short."
    )
    return llm.invoke(prompt).content.strip()


# --- TOOL: Translate to Bahasa Indonesia if needed ---
def translate_output_tool(text: str) -> str:
    try:
        lang = detect(text)
        if lang == "id":
            return text
        prompt = f"Terjemahkan ini ke Bahasa Indonesia secara alami:\n\n{text}"
        return llm.invoke(prompt).content.strip()
    except:
        return text


# --- TOOL: Get Similar Songs based on a reference song ---
def get_similar_song_tool(song_title: str) -> str:
    if not song_title.strip():
        return "Masukkan judul lagu untuk mencari yang mirip."
    results = vectorstore.similarity_search(song_title, k=4)
    if not results:
        return "Tidak ada lagu mirip ditemukan."
    return "\n".join([f"🎶 {doc.page_content}" for doc in results[1:]])
# --- LangChain Tools ---
tools = [
    Tool.from_function(detect_mood_tool, name="DetectMood", description="Detect user's mood."),
    Tool.from_function(infer_genre_tool, name="InferGenre", description="Suggest music genre."),
    Tool.from_function(retrieve_songs_tool, name="RetrieveSongs", description="Find songs matching mood/genre."),
    Tool.from_function(explain_choice_tool, name="ExplainChoice", description="Explain why a song fits the mood."),
    Tool.from_function(translate_output_tool, name="TranslateOutput", description="Translate output to Bahasa Indonesia."),
    Tool.from_function(get_similar_song_tool, name="GetSimilarSongs", description="Find songs similar to a given title.")
]
# --- Agent Initialization ---
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)
# --- Chat Input ---
user_input = st.chat_input("Apa yang ingin kamu dengar hari ini?")
if user_input:
    # Tampilkan pesan user di UI chat
    st.chat_message("user").markdown(user_input)

    with st.spinner("🤖 Agent is thinking..."):
        result = agent_executor.invoke({"input": user_input})
        st.chat_message("AI").markdown(result)
