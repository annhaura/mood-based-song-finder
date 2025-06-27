# mood_based_agent.py
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
st.set_page_config(page_title="Mood Song Agent", page_icon="🎶")
st.title("Mood-Based Song Recommender")
st.markdown("Powered by LangChain + Gemini + FAISS. With follow-up questions and dynamic genre support.")

# --- API Key ---
api_key = st.secrets.get("GOOGLE_API_KEY") or st.text_input("\ud83d\udd11 Masukkan Google API Key kamu:", type="password")
if not api_key:
    st.stop()
os.environ["GOOGLE_API_KEY"] = api_key

# --- LLM Setup ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7, google_api_key=api_key)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

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
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embedding_model)
    else:
        vectorstore = FAISS.from_documents(documents, embedding_model)
        vectorstore.save_local("faiss_index")
        return vectorstore

vectorstore = load_vectorstore()

# --- Utility Functions ---
def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        return lang if lang in ["en", "id"] else "id"
    except:
        return "id"

# --- Tools ---
def detect_mood_tool(user_input: str) -> str:
    return llm.invoke(f"Describe the user's emotional mood in 1–3 words based on this input:\n\n{user_input}").content.strip()

def infer_genre_tool(user_input: str) -> str:
    return llm.invoke(f"Suggest a suitable music genre for this input: {user_input}").content.strip()

def retrieve_songs_tool(query: str) -> str:
    count = 3
    if "|" in query:
        query, count_str = query.rsplit("|", 1)
        try:
            count = int(count_str.strip())
        except:
            count = 3
    results = vectorstore.similarity_search(query, k=count)
    return "\n".join([f"\ud83c\udfb5 {doc.page_content}" for doc in results])

def explain_choice_tool(song_and_mood: str) -> str:
    try:
        title, mood, user_input = song_and_mood.split("|")
    except:
        return "Invalid input format."
    lang = detect_language(user_input)
    if lang == "id":
        prompt = f"Jelaskan kenapa lagu '{title}' cocok untuk mood '{mood}', berdasarkan: '{user_input}'. Singkat saja ya."
    else:
        prompt = f"Why does the song '{title}' fit mood '{mood}' from input: '{user_input}'? Short and to the point."
    return llm.invoke(prompt).content.strip()

def translate_output_tool(text_and_lang: str) -> str:
    text, lang = text_and_lang.rsplit("|", 1)
    if lang == "id":
        return llm.invoke(f"Terjemahkan ini ke Bahasa Indonesia secara alami dan singkat:\n\n{text}").content.strip()
    return text

def ask_song_count_tool(user_input: str) -> str:
    return llm.invoke(f"From this input, how many songs does the user want (return only a number)? If unclear, return 3:\n\n{user_input}").content.strip()

def genre_switch_detector_tool(user_input: str) -> str:
    return llm.invoke(f"Is the user asking to change the genre or mood of music in this message? Reply only 'yes' or 'no'.\n\n{user_input}").content.strip().lower()

# --- LangChain Tools ---
tools = [
    Tool.from_function(detect_mood_tool, name="DetectMood", description="Detect user's mood from their input."),
    Tool.from_function(infer_genre_tool, name="InferGenre", description="Suggest music genre based on user input."),
    Tool.from_function(retrieve_songs_tool, name="RetrieveSongs", description="Retrieve songs. Input format: 'query | number'"),
    Tool.from_function(explain_choice_tool, name="ExplainChoice", description="Explain why a song fits user's mood."),
    Tool.from_function(translate_output_tool, name="TranslateOutput", description="Translate output to user's language."),
    Tool.from_function(ask_song_count_tool, name="CountRequestedSongs", description="Detect how many songs user wants."),
    Tool.from_function(genre_switch_detector_tool, name="DetectGenreChange", description="Detect if user wants to switch genre."),
]

# --- Agent Initialization ---
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    memory=memory
)

# --- Chat State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Chat Input ---
user_input = st.chat_input("Apa yang ingin kamu dengar hari ini?")
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append(("user", user_input))

    lang = detect_language(user_input)
    full_prompt = (
        f"You are a smart mood-based song recommender.\n"
        f"User said: '{user_input}'\n"
        f"Steps to follow:\n"
        f"1. Detect user's mood\n"
        f"2. Suggest music genre\n"
        f"3. Ask how many songs\n"
        f"4. Retrieve that many songs from vectorstore\n"
        f"5. Explain why each song fits\n"
        f"6. Detect if user wants to switch genre\n"
        f"7. Translate final output to '{lang}' if needed"
    )
    with st.spinner("\ud83e\udd16 Agent is thinking..."):
        result = agent_executor.run({"input": full_prompt, "chat_history": memory.chat_memory.messages})
        st.chat_message("AI").markdown(result)
        st.session_state.chat_history.append(("AI", result))

# --- Display Chat History ---
for speaker, msg in st.session_state.chat_history:
    with st.chat_message(speaker):
        st.markdown(msg)
