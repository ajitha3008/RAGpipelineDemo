"""
RAG Pipeline Demo – Complete Retrieval-Augmented Generation Pipeline
=====================================================================
Demonstrates: Web Scraping → Chunking → Embedding → Vector Store → Retrieval → Generation
Data Sources:
  - https://ajithayasmin.com/
  - https://psychxgalore.wordpress.com/
"""

import os
from pathlib import Path

import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from scraper import scrape_all_sources

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
FAISS_INDEX_PATH = str(CACHE_DIR / "faiss_index")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """End-to-end RAG pipeline: scrape → chunk → embed → store → retrieve."""

    def __init__(self):
        self.vectorstore = None
        self.all_chunks = []
        self.is_built = False
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
        )

    def build_index(self) -> str:
        """Run the full pipeline and return a build log."""
        log = []

        # Step 1: Scrape
        log.append("=" * 60)
        log.append("STEP 1: WEB SCRAPING")
        log.append("=" * 60)
        sources = scrape_all_sources()
        for src, text in sources.items():
            log.append(f"  > {src}: {len(text):,} characters scraped")

        # Step 2: Chunk with LangChain
        log.append("\n" + "=" * 60)
        log.append("STEP 2: TEXT CHUNKING (LangChain RecursiveCharacterTextSplitter)")
        log.append("  Chunk size: 500 chars | Overlap: 100 chars")
        log.append("=" * 60)
        all_texts = []
        all_metadatas = []
        for src, text in sources.items():
            chunks = self.splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                all_texts.append(chunk)
                all_metadatas.append({"source": src, "chunk_id": f"{src}_{i}"})
            log.append(f"  > {src}: {len(chunks)} chunks created")
        log.append(f"  Total chunks: {len(all_texts)}")
        self.all_chunks = [
            {"text": t, **m} for t, m in zip(all_texts, all_metadatas)
        ]

        # Step 3 & 4: Embed + Build FAISS index via LangChain
        log.append("\n" + "=" * 60)
        log.append("STEP 3: EMBEDDING + FAISS INDEX (LangChain)")
        log.append(f"  Embedding model: {EMBED_MODEL_NAME} (384-dim)")
        log.append("=" * 60)
        self.vectorstore = FAISS.from_texts(
            all_texts, self.embeddings, metadatas=all_metadatas
        )
        self.vectorstore.save_local(FAISS_INDEX_PATH)
        log.append(f"  > Indexed {len(all_texts)} chunks into FAISS")
        log.append(f"  > Index saved to {FAISS_INDEX_PATH}")

        self.is_built = True
        log.append("\nRAG Pipeline built successfully!")
        return "\n".join(log)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve relevant chunks for a query."""
        if not self.is_built:
            return []
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        return [
            {"text": doc.page_content, "score": float(score), **doc.metadata}
            for doc, score in results
        ]

    def get_pipeline_stats(self) -> str:
        """Return human-readable pipeline statistics."""
        if not self.is_built:
            return "Pipeline not built yet. Click 'Build RAG Pipeline' first."

        stats = [
            "RAG Pipeline Statistics",
            "-" * 40,
            f"Total chunks indexed: {len(self.all_chunks)}",
            f"Vector dimensions: 384",
            f"Embedding model: {EMBED_MODEL_NAME}",
            f"Index type: FAISS (LangChain)",
        ]
        src_counts = {}
        for c in self.all_chunks:
            src_counts[c["source"]] = src_counts.get(c["source"], 0) + 1
        stats.append("\nChunks by source:")
        for src, count in src_counts.items():
            stats.append(f"  {src}: {count}")
        return "\n".join(stats)


# ---------------------------------------------------------------------------
# LLM Generation (OpenAI only)
# ---------------------------------------------------------------------------

def call_llm(prompt: str, system_msg: str = "") -> str:
    """Call OpenAI API to generate an answer."""
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        return (
            "No OPENAI_API_KEY found. Set it as an environment variable.\n\n"
            f"**Your question:** {prompt[:200]}...\n\n"
            "Without an API key, I can't generate a natural language answer, "
            "but the retrieved context (shown on the right) demonstrates the "
            "retrieval step of the RAG pipeline."
        )

    from openai import OpenAI
    client = OpenAI(api_key=openai_key)
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=1024,
    )
    return resp.choices[0].message.content


def answer_with_rag(query: str, pipeline: RAGPipeline) -> tuple[str, str, str]:
    """Answer using RAG: retrieve context, build prompt, call LLM."""
    if not pipeline.is_built:
        return ("Pipeline not built. Click 'Build RAG Pipeline' first.", "", "")

    results = pipeline.retrieve(query, top_k=5)

    context_parts = []
    detail_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(f"[{i}] {r['text']}")
        detail_parts.append(
            f"**Chunk {i}** (score: {r['score']:.4f}, source: {r['source']})\n"
            f"```\n{r['text'][:300]}{'...' if len(r['text']) > 300 else ''}\n```"
        )

    context = "\n\n".join(context_parts)
    details = "\n\n".join(detail_parts)

    system_msg = (
        "You are a helpful assistant. Answer questions using ONLY the provided context. "
        "If the context doesn't contain enough information, say so. "
        "Cite the chunk numbers [1], [2], etc. when referencing information."
    )
    prompt = f"""Context from retrieved documents:

{context}

Question: {query}

Answer based on the context above:"""

    answer = call_llm(prompt, system_msg)
    return answer, details, context


def answer_without_rag(query: str) -> str:
    """Answer WITHOUT RAG – pure LLM knowledge."""
    system_msg = (
        "You are a helpful assistant. Answer the question using only your general knowledge. "
        "You do NOT have access to any specific documents or websites."
    )
    return call_llm(query, system_msg)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

rag_pipeline = RAGPipeline()


def build_pipeline():
    return rag_pipeline.build_index()


def get_stats():
    return rag_pipeline.get_pipeline_stats()


def query_with_rag(query: str):
    if not query.strip():
        return "Please enter a question.", ""
    answer, details, _ = answer_with_rag(query, rag_pipeline)
    return answer, details


def query_without_rag(query: str):
    if not query.strip():
        return "Please enter a question."
    return answer_without_rag(query)


def compare_query(query: str):
    if not query.strip():
        return "Please enter a question.", "Please enter a question.", ""
    answer_rag, details, _ = answer_with_rag(query, rag_pipeline)
    answer_plain = answer_without_rag(query)
    return answer_rag, answer_plain, details


def show_chunks():
    if not rag_pipeline.is_built:
        return "Pipeline not built yet."
    output = []
    for i, c in enumerate(rag_pipeline.all_chunks):
        output.append(
            f"**Chunk {i+1}** (source: {c['source']})\n"
            f"```\n{c['text'][:200]}{'...' if len(c['text']) > 200 else ''}\n```\n"
        )
    return "\n".join(output)


SAMPLE_QUESTIONS = [
    "Who is Ajitha Yasmin and what does she do?",
    "What are the soft skills mentioned in the blog?",
    "What AI tools does Ajitha work with?",
    "Tell me about Ted Bundy's psychology analysis",
    "What is the psychology of soldiers?",
    "What are signs of confidence in body language?",
    "What is Ajitha's philosophy about AI?",
    "What does the Psych Galore blog cover?",
]


def create_ui():
    with gr.Blocks(title="RAG Pipeline Demo") as demo:
        gr.Markdown(
            """
            # RAG Pipeline Demo
            ### Complete Retrieval-Augmented Generation Pipeline
            **Data Sources:** [ajithayasmin.com](https://ajithayasmin.com/) | [psychxgalore.wordpress.com](https://psychxgalore.wordpress.com/)

            This app demonstrates the complete RAG pipeline:
            **Web Scraping -> Text Chunking (LangChain) -> Vector Embedding -> FAISS Indexing (LangChain) -> Semantic Retrieval -> LLM Generation (OpenAI)**
            """,
        )

        with gr.Tab("Build Pipeline"):
            gr.Markdown("### Step-by-step RAG pipeline construction")
            build_btn = gr.Button("Build RAG Pipeline", variant="primary", size="lg")
            build_log = gr.Textbox(
                label="Pipeline Build Log", lines=25, interactive=False,
            )
            with gr.Row():
                stats_btn = gr.Button("Show Stats")
                chunks_btn = gr.Button("Show All Chunks")
            stats_output = gr.Textbox(label="Pipeline Statistics", lines=12, interactive=False)
            chunks_output = gr.Markdown(label="Indexed Chunks")
            build_btn.click(fn=build_pipeline, outputs=build_log)
            stats_btn.click(fn=get_stats, outputs=stats_output)
            chunks_btn.click(fn=show_chunks, outputs=chunks_output)

        with gr.Tab("With RAG"):
            gr.Markdown(
                "### Query with RAG-augmented answers\n"
                "The LLM receives **retrieved context** from the indexed documents before answering."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    rag_query = gr.Textbox(label="Your Question", placeholder="e.g., What AI tools does Ajitha use?", lines=2)
                    gr.Examples(examples=[[q] for q in SAMPLE_QUESTIONS], inputs=rag_query)
                    rag_btn = gr.Button("Search & Answer", variant="primary")
                with gr.Column(scale=1):
                    rag_answer = gr.Markdown(label="RAG Answer")
            rag_context = gr.Markdown(label="Retrieved Chunks (Context fed to LLM)")
            rag_btn.click(fn=query_with_rag, inputs=rag_query, outputs=[rag_answer, rag_context])

        with gr.Tab("Without RAG"):
            gr.Markdown(
                "### Query WITHOUT RAG\n"
                "The LLM answers using **only its general knowledge** — no document retrieval."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    no_rag_query = gr.Textbox(label="Your Question", placeholder="e.g., Who is Ajitha Yasmin?", lines=2)
                    gr.Examples(examples=[[q] for q in SAMPLE_QUESTIONS], inputs=no_rag_query)
                    no_rag_btn = gr.Button("Ask (No RAG)", variant="secondary")
                with gr.Column(scale=1):
                    no_rag_answer = gr.Markdown(label="Answer (General Knowledge Only)")
            no_rag_btn.click(fn=query_without_rag, inputs=no_rag_query, outputs=no_rag_answer)

        with gr.Tab("Compare"):
            gr.Markdown(
                "### Side-by-Side Comparison\n"
                "Ask the same question and see how RAG vs. no-RAG answers differ."
            )
            compare_query_input = gr.Textbox(label="Your Question", placeholder="Ask anything about Ajitha or psychology topics...", lines=2)
            gr.Examples(examples=[[q] for q in SAMPLE_QUESTIONS], inputs=compare_query_input)
            compare_btn = gr.Button("Compare Both", variant="primary", size="lg")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### With RAG")
                    compare_rag = gr.Markdown(label="RAG Answer")
                with gr.Column():
                    gr.Markdown("### Without RAG")
                    compare_no_rag = gr.Markdown(label="Plain LLM Answer")
            compare_context = gr.Markdown(label="Retrieved Chunks (used by RAG)")
            compare_btn.click(
                fn=compare_query, inputs=compare_query_input,
                outputs=[compare_rag, compare_no_rag, compare_context],
            )

        with gr.Tab("How It Works"):
            gr.Markdown(
                """
                ## RAG Pipeline Architecture

                ```
                ┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
                │  Web Scrape  │────>│  Text Chunks  │────>│  Embeddings     │
                │  (requests + │     │  (LangChain   │     │  (all-MiniLM-   │
                │   BS4)       │     │   splitter)   │     │   L6-v2, 384d)  │
                └─────────────┘     └──────────────┘     └────────┬────────┘
                                                                   │
                                                                   v
                ┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
                │  LLM Answer  │<────│  RAG Prompt   │<────│  FAISS Index    │
                │  (GPT-4o     │     │  (query +     │     │  (LangChain     │
                │   mini)      │     │   context)    │     │   FAISS)        │
                └─────────────┘     └──────────────┘     └─────────────────┘
                ```

                ### Pipeline Steps

                **1. Web Scraping** — Fetches HTML using `requests`, parses with `BeautifulSoup`, caches locally.

                **2. Text Chunking** — Uses LangChain `RecursiveCharacterTextSplitter` (500 char chunks, 100 char overlap).

                **3. Vector Embedding** — Uses `all-MiniLM-L6-v2` via LangChain `HuggingFaceEmbeddings` (384-dim vectors).

                **4. FAISS Vector Store** — Built and queried via LangChain's `FAISS` wrapper. Persisted to disk.

                **5. Retrieval** — Query is embedded and top-k similar chunks are retrieved with similarity scores.

                **6. Generation** — Retrieved chunks are injected into the prompt. OpenAI GPT-4o-mini generates the answer.

                ### Data Sources

                | Source | Content |
                |--------|---------|
                | [ajithayasmin.com](https://ajithayasmin.com/) | Portfolio, projects, blog posts, professional experience |
                | [psychxgalore.wordpress.com](https://psychxgalore.wordpress.com/) | Psychology blog: soft skills, criminal psychology, body language |

                ### Why RAG Matters

                - **Without RAG:** LLM may hallucinate or lack specific knowledge
                - **With RAG:** LLM answers are grounded in actual source documents
                - **Comparison tab** lets you see the difference side-by-side
                """
            )

    return demo


if __name__ == "__main__":
    print("=" * 60)
    print("  RAG Pipeline Demo")
    print("  Data Sources:")
    print("    - https://ajithayasmin.com/")
    print("    - https://psychxgalore.wordpress.com/")
    print("=" * 60)

    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
