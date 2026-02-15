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

## Screenshots

<img width="1465" height="805" alt="Screenshot 2026-02-15 at 4 17 15 PM" src="https://github.com/user-attachments/assets/54b5bd6b-8184-424f-a474-273861b649cd" />
<img width="1443" height="799" alt="Screenshot 2026-02-15 at 4 18 58 PM" src="https://github.com/user-attachments/assets/ec8b6782-f9e6-4f30-94bf-cd34fbf3ef31" />
<img width="1438" height="798" alt="Screenshot 2026-02-15 at 4 19 12 PM" src="https://github.com/user-attachments/assets/c55c4979-902a-402e-be93-45e0a795796a" />

