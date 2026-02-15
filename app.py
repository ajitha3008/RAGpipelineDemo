"""
RAG Pipeline Demo
=================
Web Scraping -> Chunking -> Embedding -> ChromaDB -> Retrieval -> LLM Generation
Data Sources: ajithayasmin.com, ajithayasmin.wordpress.com
"""

import asyncio
import os
import warnings
warnings.filterwarnings("ignore", message=".*google.generativeai.*")

import gradio as gr
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from scraper import scrape_all_sources

EMBED_MODEL = "all-MiniLM-L6-v2"
CHROMA_DIR = "cache/chroma_db"


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    def __init__(self):
        self.vectorstore = None
        self.all_chunks = []
        self.is_built = False
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    def build_index(self) -> str:
        log = []

        # Scrape
        log.append("STEP 1: WEB SCRAPING\n" + "=" * 50)
        sources = scrape_all_sources()
        for src, text in sources.items():
            log.append(f"  {src}: {len(text):,} chars")

        # Chunk
        log.append("\nSTEP 2: CHUNKING (LangChain)\n" + "=" * 50)
        all_texts, all_metas = [], []
        for src, text in sources.items():
            chunks = self.splitter.split_text(text)
            all_texts.extend(chunks)
            all_metas.extend({"source": src, "chunk_id": f"{src}_{i}"} for i in range(len(chunks)))
            log.append(f"  {src}: {len(chunks)} chunks")
        self.all_chunks = [{"text": t, **m} for t, m in zip(all_texts, all_metas)]
        log.append(f"  Total: {len(all_texts)} chunks")

        # Embed + Store in Chroma
        log.append(f"\nSTEP 3: EMBED + STORE (ChromaDB)\n" + "=" * 50)
        self.vectorstore = Chroma.from_texts(
            all_texts, self.embeddings, metadatas=all_metas,
            persist_directory=CHROMA_DIR,
        )
        log.append(f"  Indexed {len(all_texts)} chunks into ChromaDB")

        self.is_built = True
        log.append("\nPipeline built successfully!")
        return "\n".join(log)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.is_built:
            return []
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        return [{"text": doc.page_content, "score": float(score), **doc.metadata} for doc, score in results]

    def get_stats(self) -> str:
        if not self.is_built:
            return "Pipeline not built yet."
        src_counts = {}
        for c in self.all_chunks:
            src_counts[c["source"]] = src_counts.get(c["source"], 0) + 1
        lines = [f"Chunks: {len(self.all_chunks)} | Model: {EMBED_MODEL} | Store: ChromaDB", ""]
        for src, count in src_counts.items():
            lines.append(f"  {src}: {count} chunks")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM (OpenAI)
# ---------------------------------------------------------------------------

def call_llm(prompt: str, system_msg: str = "") -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return "No OPENAI_API_KEY set. Export it and restart."

    from openai import OpenAI
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": prompt})
    resp = OpenAI(api_key=api_key).chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=1024)
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# RAGAS Evaluation
# ---------------------------------------------------------------------------

def evaluate_rag_response(question: str, answer: str, contexts: list[str]) -> dict:
    """Evaluate a RAG response using RAGAS metrics (Faithfulness & Answer Relevancy)."""
    try:
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmbeddings
        from ragas.metrics.collections import Faithfulness, AnswerRelevancy

        client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        llm = llm_factory("gpt-4o-mini", client=client)
        embeddings = RagasHFEmbeddings(model=EMBED_MODEL)

        faithfulness = Faithfulness(llm=llm)
        relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)

        async def _eval():
            faith_score = await faithfulness.ascore(
                user_input=question, response=answer, retrieved_contexts=contexts,
            )
            rel_score = await relevancy.ascore(
                user_input=question, response=answer,
            )
            return float(faith_score), float(rel_score)

        loop = asyncio.new_event_loop()
        faith, rel = loop.run_until_complete(_eval())
        loop.close()

        overall = round((faith + rel) / 2, 4)
        return {"faithfulness": round(faith, 4), "answer_relevancy": round(rel, 4), "overall": overall}
    except Exception as e:
        return {"faithfulness": 0, "answer_relevancy": 0, "overall": 0, "error": str(e)}


def format_eval_scores(scores: dict) -> str:
    if scores.get("error"):
        return f"**RAGAS Eval failed:** {scores['error']}"
    bar = lambda v: "█" * round(v * 10) + "░" * (10 - round(v * 10))
    return (
        f"### RAGAS Eval Scores\n"
        f"| Metric | Score | |\n"
        f"|--------|-------|---|\n"
        f"| Faithfulness | {bar(scores['faithfulness'])} | **{scores['faithfulness']:.2f}** |\n"
        f"| Answer Relevancy | {bar(scores['answer_relevancy'])} | **{scores['answer_relevancy']:.2f}** |\n"
        f"| **Overall** | | **{scores['overall']:.2f}** |"
    )


RAG_SYSTEM = (
    "Answer using ONLY the provided context. If insufficient, say so. "
    "Cite chunk numbers [1], [2], etc."
)

NO_RAG_SYSTEM = "Answer using only your general knowledge. You have no access to specific documents."


def answer_with_rag(query: str, pipeline: RAGPipeline, progress=gr.Progress()):
    if not pipeline.is_built:
        yield "Pipeline not built yet.", "", ""
        return

    progress(0.1, desc="Retrieving relevant chunks...")
    results = pipeline.retrieve(query, top_k=5)
    contexts = [r["text"] for r in results]
    context_str = "\n\n".join(f"[{i}] {r['text']}" for i, r in enumerate(results, 1))
    details = "\n\n".join(
        f"**Chunk {i}** (score: {r['score']:.4f}, source: {r['source']})\n```\n{r['text'][:300]}\n```"
        for i, r in enumerate(results, 1)
    )

    progress(0.3, desc="Generating answer from LLM...")
    yield "Generating answer...", details, ""
    answer = call_llm(f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:", RAG_SYSTEM)

    progress(0.7, desc="Running RAGAS evaluation...")
    yield answer, details, "Running RAGAS evaluation..."
    scores = evaluate_rag_response(query, answer, contexts)
    eval_display = format_eval_scores(scores)

    progress(1.0, desc="Done!")
    yield answer, details, eval_display


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

pipeline = RAGPipeline()

SAMPLES = [
    "Who is Ajitha Yasmin and what does she do?",
    "What AI tools does Ajitha work with?",
    "What is Ajitha's philosophy about AI?",
    "What topics does the WordPress blog cover?",
    "What are Ajitha's professional skills?",
]


def create_ui():
    with gr.Blocks(title="RAG Pipeline Demo") as demo:
        gr.Markdown(
            "# RAG Pipeline Demo\n"
            "**Sources:** [ajithayasmin.com](https://ajithayasmin.com/) | "
            "[ajithayasmin.wordpress.com](https://ajithayasmin.wordpress.com/)\n\n"
            "**Pipeline:** Web Scraping -> Chunking (LangChain) -> Embedding -> ChromaDB -> Retrieval -> OpenAI"
        )

        with gr.Tab("Build Pipeline"):
            build_btn = gr.Button("Build RAG Pipeline", variant="primary", size="lg")
            build_log = gr.Textbox(label="Build Log", lines=20, interactive=False)
            with gr.Row():
                stats_btn = gr.Button("Stats")
                chunks_btn = gr.Button("Show Chunks")
            stats_out = gr.Textbox(label="Stats", lines=6, interactive=False)
            chunks_out = gr.Markdown()
            build_btn.click(fn=pipeline.build_index, outputs=build_log)
            stats_btn.click(fn=pipeline.get_stats, outputs=stats_out)
            chunks_btn.click(
                fn=lambda: "\n".join(
                    f"**{i+1}.** ({c['source']}): {c['text'][:150]}..." for i, c in enumerate(pipeline.all_chunks)
                ) if pipeline.is_built else "Pipeline not built yet.",
                outputs=chunks_out,
            )

        with gr.Tab("With RAG"):
            gr.Markdown("### Ask with retrieved context")
            rag_q = gr.Textbox(label="Question", lines=2)
            gr.Examples([[q] for q in SAMPLES], inputs=rag_q)
            rag_btn = gr.Button("Search & Answer", variant="primary")
            rag_ans = gr.Markdown(label="Answer")
            rag_eval = gr.Markdown(label="LLM Eval Scores")
            rag_ctx = gr.Markdown(label="Retrieved Chunks")
            def do_rag(q):
                if not q.strip():
                    yield "Enter a question.", "", ""
                    return
                yield from answer_with_rag(q, pipeline)

            rag_btn.click(fn=do_rag, inputs=rag_q, outputs=[rag_ans, rag_ctx, rag_eval])

        with gr.Tab("Without RAG"):
            gr.Markdown("### Ask without retrieval (LLM knowledge only)")
            norag_q = gr.Textbox(label="Question", lines=2)
            gr.Examples([[q] for q in SAMPLES], inputs=norag_q)
            norag_btn = gr.Button("Ask", variant="secondary")
            norag_ans = gr.Markdown(label="Answer")
            norag_btn.click(fn=lambda q: call_llm(q, NO_RAG_SYSTEM) if q.strip() else "Enter a question.", inputs=norag_q, outputs=norag_ans)

        with gr.Tab("Compare"):
            gr.Markdown("### Side-by-side: RAG vs No RAG")
            cmp_q = gr.Textbox(label="Question", lines=2)
            gr.Examples([[q] for q in SAMPLES], inputs=cmp_q)
            cmp_btn = gr.Button("Compare", variant="primary", size="lg")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### With RAG")
                    cmp_rag = gr.Markdown()
                with gr.Column():
                    gr.Markdown("### Without RAG")
                    cmp_norag = gr.Markdown()
            cmp_eval = gr.Markdown(label="LLM Eval Scores")
            cmp_ctx = gr.Markdown(label="Retrieved Chunks")

            def do_compare(q, progress=gr.Progress()):
                if not q.strip():
                    yield "Enter a question.", "Enter a question.", "", ""
                    return

                progress(0.1, desc="Retrieving relevant chunks...")
                results = pipeline.retrieve(q, top_k=5)
                contexts = [r["text"] for r in results]
                context_str = "\n\n".join(f"[{i}] {r['text']}" for i, r in enumerate(results, 1))
                details = "\n\n".join(
                    f"**Chunk {i}** (score: {r['score']:.4f}, source: {r['source']})\n```\n{r['text'][:300]}\n```"
                    for i, r in enumerate(results, 1)
                )

                progress(0.25, desc="Generating RAG answer...")
                yield "Generating answer...", "", "", details
                rag_a = call_llm(f"Context:\n{context_str}\n\nQuestion: {q}\n\nAnswer:", RAG_SYSTEM)

                progress(0.5, desc="Generating non-RAG answer...")
                yield rag_a, "Generating answer...", "", details
                norag_a = call_llm(q, NO_RAG_SYSTEM)

                progress(0.75, desc="Running RAGAS evaluation...")
                yield rag_a, norag_a, "Running RAGAS evaluation...", details
                scores = evaluate_rag_response(q, rag_a, contexts)
                eval_display = format_eval_scores(scores)

                progress(1.0, desc="Done!")
                yield rag_a, norag_a, eval_display, details

            cmp_btn.click(fn=do_compare, inputs=cmp_q, outputs=[cmp_rag, cmp_norag, cmp_eval, cmp_ctx])

        with gr.Tab("How It Works"):
            gr.Markdown("""
## Architecture

```
Web Scrape (requests+BS4) -> Chunk (LangChain) -> Embed (all-MiniLM-L6-v2) -> ChromaDB -> Retrieve -> LLM (GPT-4o-mini)
```

**1. Scraping** — Fetch and parse HTML, cache locally.\n
**2. Chunking** — LangChain `RecursiveCharacterTextSplitter` (500 chars, 100 overlap).\n
**3. Embedding** — `all-MiniLM-L6-v2` via LangChain (384-dim vectors).\n
**4. Vector Store** — ChromaDB with cosine similarity, persisted to disk.\n
**5. Retrieval** — Embed query, find top-k similar chunks.\n
**6. Generation** — Inject context into prompt, generate with OpenAI GPT-4o-mini.\n

| Source | Content |
|--------|---------|
| [ajithayasmin.com](https://ajithayasmin.com/) | Portfolio, projects, professional experience |
| [ajithayasmin.wordpress.com](https://ajithayasmin.wordpress.com/) | Blog posts and articles |
""")

    return demo


if __name__ == "__main__":
    create_ui().launch(server_name="0.0.0.0", server_port=7860, share=False)
