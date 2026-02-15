# RAG Pipeline Demo

A complete Retrieval-Augmented Generation (RAG) pipeline demo built with Gradio and LangChain.

**Pipeline:** Web Scraping → Text Chunking (LangChain) → Vector Embedding → FAISS Indexing (LangChain) → Semantic Retrieval → LLM Generation (OpenAI)

**Data Sources:**
- [ajithayasmin.com](https://ajithayasmin.com/)
- [psychxgalore.wordpress.com](https://psychxgalore.wordpress.com/)

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Run the app

```bash
python app.py
```

The app will be available at **http://localhost:7860**.

## How to Use

1. **Build Pipeline** tab — Click "Build RAG Pipeline" to scrape the websites, chunk the text, generate embeddings, and build the FAISS index.
2. **With RAG** tab — Ask questions and get answers grounded in the scraped documents.
3. **Without RAG** tab — Ask the same questions using only the LLM's general knowledge (no retrieval).
4. **Compare** tab — See RAG vs. non-RAG answers side by side.
5. **How It Works** tab — Learn about each stage of the pipeline.

## Project Structure

```
├── app.py           # Main app: RAG pipeline, LLM generation, Gradio UI
├── scraper.py       # Web scraping utilities (fetch, parse, cache)
├── requirements.txt
├── README.md
└── cache/           # Auto-created: scraped pages + FAISS index
```

## Tech Stack

| Component | Tool |
|-----------|------|
| Web Scraping | requests + BeautifulSoup |
| Text Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | LangChain HuggingFaceEmbeddings (all-MiniLM-L6-v2) |
| Vector Store | LangChain FAISS |
| LLM | OpenAI GPT-4o-mini |
| UI | Gradio |
