import os
# Suppress TensorFlow logging & oneDNN noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings("ignore")

import re
import torch
import whisper
import tempfile
import numpy as np
import soundfile as sf
import streamlit as st
from streamlit_mic_recorder import mic_recorder
from streamlit_TTS import text_to_speech  
from rag_core import setup_qa_chain, create_vector_db, DATA_DIR

# --- OPTIMIZATIONS ---
torch.backends.cuda.enable_math_sdp(True)

# --- BROKER NORMALIZATION ---
MONTH_MAP = {"january": "Jan", "february": "Feb", "march": "Mar", "april": "Apr", "may": "May", "june": "Jun", "july": "Jul", "august": "Aug", "september": "Sep", "october": "Oct", "november": "Nov", "december": "Dec", "jan": "Jan", "feb": "Feb", "mar": "Mar"}

def normalize_broker_text(text: str) -> str:
    if not text: return text
    t = text.lower().strip()
    t = re.sub(r"\b[s]\s*[x]\s*(?:5|five)\s*[e]\b", "SX5E", t)
    def repl_exp(m):
        mon = MONTH_MAP.get(m.group(1))
        return f"{mon}{m.group(2)[-2:]}" if mon else m.group(0)
    t = re.sub(r"\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+(\d{2,4})\b", repl_exp, t)
    for acronym in ["trf", "sx5e"]: t = re.sub(rf"\b{acronym}\b", acronym.upper(), t)
    return t

# --- UI SETUP ---
# BRANDING: Browser Tab
st.set_page_config(page_title="Financial Document Analysing Assistant", layout="wide", page_icon="🎙️")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR: VOICE & ADMIN ---
with st.sidebar:
    # BRANDING: Sidebar Header
    st.title("📊 Finance Analyzing Assistant")
    
    st.divider()
    st.title("⚙️ Voice Settings")
    auto_speak = st.toggle("Auto-Speak Answers", value=False, help="Play audio automatically for every new answer.")
    
    st.divider()
    st.title("📂 Admin Portal")
    uploaded_file = st.file_uploader("Upload PDF Profile", type="pdf")
    if uploaded_file:
        if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
        with open(os.path.join(DATA_DIR, uploaded_file.name), "wb") as f: 
            f.write(uploaded_file.getbuffer())
        st.success("PDF Saved!")
    if st.button("🔄 Re-Index Database"):
        with st.spinner("Indexing..."):
            if create_vector_db(): 
                st.success("Done!")
                st.rerun()
    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

@st.cache_resource
def load_all_engines():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stt = whisper.load_model("base", device=device)
    try: 
        rag = setup_qa_chain() 
    except: 
        rag = None
    return stt, rag

stt_model, rag_chain = load_all_engines()

# --- MAIN CHAT INTERFACE ---
# UPDATED TITLE: Removed model name
st.title("🎙️ Financial Document Analysing Assistant")

# 1. DISPLAY CHAT HISTORY
chat_container = st.container()
with chat_container:
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "sources" in msg:
                st.caption(f"Sources: {', '.join(msg['sources'])}")
            
            # Button to speak this specific answer
            if msg["role"] == "assistant":
                if st.button(f"🔊 Speak Answer", key=f"speak_btn_{i}"):
                    text_to_speech(text=msg["content"], language='en')

st.divider()

# 2. INPUT AREA
tab_voice, tab_text = st.tabs(["🎤 Voice Query", "⌨️ Text Query"])

with tab_voice:
    st.write("Record your question, then click 'Analyze My Speech'.")
    voice_data = mic_recorder(start_prompt="Record", stop_prompt="Stop", format="wav", key="voice_input")
    
    if voice_data and voice_data.get("bytes"):
        if st.button("Analyze My Speech", key="analyze_btn"):
            with st.spinner("Transcribing..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(voice_data["bytes"])
                    stt_result = stt_model.transcribe(tmp.name)
                    user_query = normalize_broker_text(stt_result['text']).strip()
                
                if user_query:
                    st.session_state.messages.append({"role": "user", "content": user_query})
                    if rag_chain:
                        with st.spinner("Thinking..."):
                            output = rag_chain.invoke(user_query)
                            answer_text = output["answer"]
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": answer_text, 
                                "sources": output["sources"]
                            })
                            # Speak out if auto-speak is enabled
                            if auto_speak:
                                text_to_speech(text=answer_text, language='en')
                    st.rerun()

with tab_text:
    if text_query := st.chat_input("Type a question and press Enter..."):
        st.session_state.messages.append({"role": "user", "content": text_query})
        if rag_chain:
            with st.spinner("Analyzing Documents..."):
                output = rag_chain.invoke(text_query)
                answer_text = output["answer"]
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer_text, 
                    "sources": output["sources"]
                })
                # Speak out if auto-speak is enabled
                if auto_speak:
                    text_to_speech(text=answer_text, language='en')
        st.rerun()