# streamlit_app.py (Refactored for agentic song recommender)
import streamlit as st
import os
import pandas as pd
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 42

# --- Streamlit UI ---
st.set_page_config(page_title="🎵 Mood Song Agent", page_icon="🎧")
st.title("Mood-Based Song Recommender")
st.markdown("Powered by LangChain + Gemini + FAISS. With follow-up questions and dynamic genre support.")

# --- API Key ---
api_key = st.secrets.get("GOOGLE_API_KEY") or st.text_input("🔑 Masukkan Google API Key kamu:", type="password")
if not api_key:
    st.stop()
os.environ["GOOGLE_API_KEY"] = api_key

# --- LLM Setup ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7, google_api_key=api_key)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- Load Dataset + Vectorstore ---
csv_url = "https://raw.githubusercontent.com/annhaura/mood-based-song-finder/main/spotify_songs.csv"

@st.cache_data
def load_dataset():
    df = pd.read_csv(csv_url)
    df = df.groupby('playlist_subgenre', group_keys=False).apply(lambda x: x.sample(min(80, len(x)), random_state=42)).reset_index(drop=True)
    df["combined_text"] = df.apply(lambda row: f"{row['track_name']} by {row['track_artist']}", axis=1)
    return df

df = load_dataset()
documents = [Document(page_content=row["combined_text"], metadata={"index": i}) for i, row in df.iterrows()]

@st.cache_resource
def load_vectorstore():
    index_path = "faiss_index"
    if not os.path.exists(index_path):
        vectorstore = FAISS.from_documents(documents, embedding_model)
        vectorstore.save_local(index_path)
        return vectorstore
    else:
        return FAISS.load_local(index_path, embedding_model)

vectorstore = load_vectorstore()

# --- Helper Tools ---
def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        return lang if lang in ["en", "id"] else "id"
    except:
        return "id"

def detect_mood(user_input: str) -> str:
    return llm.invoke(f"Describe the user's emotional state in 1–3 words: {user_input}").content.strip()

def infer_genre(user_input: str) -> str:
    return llm.invoke(f"Suggest a music genre based on: {user_input}").content.strip()

def count_songs(user_input: str) -> str:
    return llm.invoke(f"How many songs does the user want (just the number, default 3)? Input: {user_input}").content.strip()

def detect_genre_switch(user_input: str) -> str:
    return llm.invoke(f"Does the user want to change genre? (yes/no). Input: {user_input}").content.strip().lower()

def retrieve_songs(query: str) -> str:
    try:
        q, count = query.rsplit("|", 1)
        count = int(count.strip())
    except:
        q, count = query, 3
    results = vectorstore.similarity_search(q.strip(), k=count)
    return "\n".join([f"🎵 {doc.page_content}" for doc in results])

def explain_choice(input_str: str) -> str:
    try:
        song, mood, user_input = input_str.split("|")
    except:
        return "Invalid format."
    lang = detect_language(user_input)
    if lang == "id":
        prompt = f"Jelaskan kenapa lagu '{song}' cocok untuk mood '{mood}', dari input: '{user_input}'"
    else:
        prompt = f"Why does the song '{song}' match the mood '{mood}' from input: '{user_input}'?"
    return llm.invoke(prompt).content.strip()

def translate_if_needed(text_lang: str) -> str:
    text, lang = text_lang.rsplit("|", 1)
    if lang == "id":
        return llm.invoke(f"Terjemahkan ini ke Bahasa Indonesia dengan alami: {text}").content.strip()
    return text

# --- Register Tools ---
tools = [
    Tool.from_function(detect_mood, name="DetectMood", description="Analyze user's mood"),
    Tool.from_function(infer_genre, name="InferGenre", description="Predict music genre"),
    Tool.from_function(count_songs, name="CountRequestedSongs", description="How many songs user wants"),
    Tool.from_function(detect_genre_switch, name="DetectGenreChange", description="Detect if user wants to switch genre"),
    Tool.from_function(retrieve_songs, name="RetrieveSongs", description="Search for similar songs. Use format: 'query | number'"),
    Tool.from_function(explain_choice, name="ExplainChoice", description="Explain why a song fits"),
    Tool.from_function(translate_if_needed, name="TranslateOutput", description="Translate result to user's language if needed"),
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    memory=memory
)

# --- Session & UI ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Apa yang kamu ingin dengar hari ini?")
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append(("user", user_input))

    lang = detect_language(user_input)
    full_prompt = (
        "You are a sensitive and intelligent mood-based song recommender agent."
        " Given the user's emotional input, you will:"
        "\n1. Detect their mood"
        "\n2. Infer a fitting music genre"
        "\n3. Detect if they ask for a new genre"
        "\n4. Detect how many songs they want"
        "\n5. Retrieve songs from vector database"
        "\n6. Explain the fit of each song"
        "\n7. Translate the result if needed"
        f"\n\nUser Input: {user_input}"
    )

    with st.spinner("🎧 Mencari lagu yang cocok untukmu..."):
        result = agent.run({"input": full_prompt})
        st.chat_message("AI").markdown(result)
        st.session_state.chat_history.append(("AI", result))

for speaker, message in st.session_state.chat_history:
    with st.chat_message(speaker):
        st.markdown(message)
