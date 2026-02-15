# RAG Pipeline Demo

A complete Retrieval-Augmented Generation pipeline demo using LangChain, ChromaDB, and Gradio.

**Pipeline:** Web Scraping -> Chunking (LangChain) -> Embedding -> ChromaDB -> Retrieval -> OpenAI

**Data Sources:**
- [ajithayasmin.com](https://ajithayasmin.com/)
- [ajithayasmin.wordpress.com](https://ajithayasmin.wordpress.com/)

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
python app.py
```

Open **http://localhost:7860**.

## How to Use

1. **Build Pipeline** — Scrape, chunk, embed, and index into ChromaDB.
2. **With RAG** — Ask questions answered using retrieved context.
3. **Without RAG** — Same questions using only LLM knowledge.
4. **Compare** — Side-by-side RAG vs non-RAG.

## Project Structure

```
app.py           # RAG pipeline, LLM calls, Gradio UI
scraper.py       # Web scraping utilities
requirements.txt
cache/           # Auto-created: scraped pages + ChromaDB
```

## Tech Stack

| Component | Tool |
|-----------|------|
| Scraping | requests + BeautifulSoup |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | LangChain HuggingFaceEmbeddings (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB |
| LLM | OpenAI GPT-4o-mini |
| UI | Gradio |
