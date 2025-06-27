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

# --- Streamlit UI ---
st.set_page_config(page_title="🎵 Mood Song Agent", page_icon="🎧")
st.title("Mood-Based Song Recommender")
st.markdown("Powered by LangChain + Gemini + FAISS. Minimal 5 smart tools 🤖")

# --- API Key ---
api_key = st.secrets.get("GOOGLE_API_KEY") or st.text_input("🔑 Masukkan Google API Key kamu:", type="password")
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

def retrieve_songs_tool(query: str) -> str:
    if not query or len(query.strip()) < 3:
        return "Query too short."
    results = vectorstore.similarity_search(query, k=3)
    return "\n".join([f"🎵 {doc.page_content}" for doc in results])

# --- Language Detection ---
def detect_language(text: str) -> str:
    try:
        return detect(text)
    except:
        return "en"

# --- Tool Definitions ---
def detect_mood_tool(user_input: str) -> str:
    prompt = f"Describe the user's emotional mood in 1–3 words based on this input:\n\n{user_input}"
    return llm.invoke(prompt).content.strip()

def infer_genre_tool(user_input: str) -> str:
    prompt = f"Suggest a suitable music genre for this input: {user_input}"
    return llm.invoke(prompt).content.strip()

def retrieve_songs_tool(query: str) -> str:
    results = vectorstore.similarity_search(query, k=3)
    return "\n".join([f"🎵 {doc.page_content}" for doc in results])

def explain_choice_tool(song_and_mood: str) -> str:
    # Expected input: "song_title | mood | original_input"
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
        prompt = f"Terjemahkan ini ke Bahasa Indonesia secara alami dan singkat:\n\n{text}"
        return llm.invoke(prompt).content.strip()
    return text  # return original if already English

# --- LangChain Tools ---
tools = [
    Tool.from_function(func=detect_mood_tool, name="DetectMood", description="Detect user's mood from their input."),
    Tool.from_function(func=infer_genre_tool, name="InferGenre", description="Suggest music genre based on user input."),
    Tool.from_function(func=retrieve_songs_tool, name="RetrieveSongs", description="Retrieve matching songs from the vectorstore."),
    Tool.from_function(func=explain_choice_tool, name="ExplainChoice", description="Explain why a song fits user's mood."),
    Tool.from_function(func=translate_output_tool, name="TranslateOutput", description="Translate output to user's language."),
]

# --- Agent Initialization ---
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    memory=memory
)

# --- Chat Input ---
user_input = st.chat_input("Apa yang ingin kamu dengar hari ini?")
if user_input:
    with st.spinner("🤖 Agent is thinking..."):
        lang = detect_language(user_input)
        full_prompt = (
    f"You are a mood-based song recommendation agent.\n"
    f"User said: '{user_input}'\n"
    f"- Step 1: Detect the user's mood\n"
    f"- Step 2: Infer music genre\n"
    f"- Step 3: Retrieve 3 songs from vectorstore\n"
    f"- Step 4: Explain why each song fits the user's mood\n"
    f"- Step 5: Translate output to '{lang}' if needed\n"
)
        result = agent_executor.run({"input": user_input})
        st.chat_message("AI").markdown(result)
