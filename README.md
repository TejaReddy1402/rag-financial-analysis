# 🎙️ Financial Document Analysing Assistant

A high-performance, multi-modal **Retrieval-Augmented Generation (RAG)** platform designed for professional financial analysts. This system allows for seamless interaction with complex financial PDFs through **Voice-to-Text**, **Gemini 2.5 Flash reasoning**, and **Text-to-Speech** feedback.

---

## 🚀 Professional Project Narrative

In the high-stakes world of finance, analysts are frequently overwhelmed by the sheer volume of quarterly reports, CEO transcripts, and broker profiles. The primary challenge lies in the inefficiency of manual search—traditional tools often fail to connect strategic "visionary" statements with hard numerical data across disparate, lengthy documents.

To address this, I developed an intelligent assistant capable of ingesting, indexing, and standardizing specialized financial terminology (e.g., automated mapping for tickers like SX5E and TRF). The objective was to create a hands-free research experience where an analyst could query documents naturally via voice and receive cited, synthesized professional responses.



The solution involved engineering a custom RAG pipeline using **LangChain** and **FAISS**. By implementing a **1500-character chunking strategy** with an increased retrieval depth ($k=8$), the system successfully captures the nuance of executive commentary that smaller snippets often lose. I integrated **Google’s Gemini 2.5 Flash** for its state-of-the-art reasoning and massive context window. The interface was rounded out with **OpenAI Whisper** for high-accuracy speech transcription and **gTTS** for real-time audio synthesis.

The resulting platform delivers accurate, cited summaries in seconds. It bridges the gap between raw data and actionable insights, allowing users to move from question to audible answer without a single keystroke. This significantly reduces the time spent on document discovery and enhances the quality of strategic research.

---

## 🛠️ Technical Stack

| Component | Technology |
| :--- | :--- |
| **LLM (Reasoning)** | **Google Gemini 2.5 Flash** |
| **Vector Database** | **FAISS** (Facebook AI Similarity Search) |
| **Orchestration** | **LangChain** & **LangChain-HuggingFace** |
| **Speech-to-Text** | **OpenAI Whisper** (Base Model) |
| **Text-to-Speech** | **gTTS** & **Streamlit-TTS** |
| **Embeddings** | **Sentence-Transformers** (`all-MiniLM-L6-v2`) |
| **Frontend UI** | **Streamlit** |
| **Audio Processing** | **Soundfile**, **FFmpeg**, **Streamlit-Mic-Recorder** |

---

## 💎 Key Features

* **Financial Normalization:** Custom regex logic to standardize tickers like `SX5E`, `NVIDIA`, and specific broker date formats.
* **Context-Aware Retrieval:** A deep $k=8$ retrieval window ensures strategic, non-numerical insights are synthesized alongside data.
* **Multi-Modal Querying:** Seamless switching between a professional chat interface and a hands-free voice interface.
* **Audio Synthesis:** "Auto-Speak" mode for continuous conversation or manual "Speak Answer" buttons for selective feedback.
* **Admin Control:** Real-time PDF uploading and database re-indexing directly from the sidebar.

---

## 📦 Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/TejaReddy1402/rag-financial-analysis.git](https://github.com/TejaReddy1402/rag-financial-analysis.git)
   cd rag-financial-analysis
