"""
EarningsIQ: Multi-Quarter Earnings Intelligence Assistant
A RAG-powered app that answers cross-quarter questions over earnings call transcripts.
"""

import os
import json
import math
import tempfile
import numpy as np
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from anthropic import Anthropic
from pypdf import PdfReader

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EarningsIQ",
    page_icon="📊",
    layout="wide",
)

# ── Anthropic client ──────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()
import dotenv
dotenv.load_dotenv(override=True)
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_text(file) -> str:
    """Extract plain text from a PDF or TXT file."""
    name = file.name.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        return file.read().decode("utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 30) -> list[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using Anthropic-compatible OpenAI embeddings via a helper."""
    # Use OpenAI embeddings if available, otherwise fall back to simple TF-IDF-like hashing
    try:
        from openai import OpenAI
        oc = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        response = oc.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [r.embedding for r in response.data]
    except Exception:
        # Fallback: simple hash-based pseudo-embedding (for demo without OpenAI key)
        import hashlib
        result = []
        for text in texts:
            vec = [0.0] * 128
            for i, word in enumerate(text.lower().split()):
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                vec[h % 128] += 1.0
            mag = math.sqrt(sum(v * v for v in vec)) or 1.0
            result.append([v / mag for v in vec])
        return result


def build_index(transcripts: list[dict]) -> list[dict]:
    """
    Chunk and embed all transcripts.
    Returns list of dicts: {quarter, chunk_text, embedding}
    """
    all_chunks = []
    for t in transcripts:
        chunks = chunk_text(t["text"])
        for c in chunks:
            all_chunks.append({"quarter": t["quarter"], "chunk_text": c})

    texts = [c["chunk_text"] for c in all_chunks]
    embeddings = embed(texts)
    for chunk, emb in zip(all_chunks, embeddings):
        chunk["embedding"] = emb

    return all_chunks


def retrieve(query: str, index: list[dict], top_k: int = 5) -> list[dict]:
    """Retrieve top-k most relevant chunks for a query."""
    q_emb = embed([query])[0]
    scored = []
    for chunk in index:
        sim = cosine_similarity(q_emb, chunk["embedding"])
        scored.append({**chunk, "score": sim})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ── Calculator tool ────────────────────────────────────────────────────────────

def calculator_tool(expression: str) -> str:
    """Safely evaluate a math expression."""
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return f"Error: invalid characters in expression"
        result = eval(expression, {"__builtins__": {}})
        return str(round(float(result), 4))
    except Exception as e:
        return f"Error: {e}"


TOOLS = [
    {
        "name": "calculator_tool",
        "description": (
            "Evaluate a mathematical expression and return the result. "
            "Use this for computing revenue growth %, margin changes, "
            "earnings surprises, or any precise financial calculation. "
            "Examples: '(2.4 - 2.1) / 2.1 * 100', '(89 - 92) / 92 * 100'"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A valid math expression using +, -, *, /, (), and numbers only.",
                }
            },
            "required": ["expression"],
        },
    }
]


def run_agent(question: str, chunks: list[dict]) -> tuple[str, list[dict]]:
    """
    Run the ReAct agent with RAG context and calculator tool.
    Returns (answer, retrieved_chunks).
    """
    # Build context from retrieved chunks, sorted by quarter
    chunks_sorted = sorted(chunks, key=lambda x: x["quarter"])
    context_parts = []
    for i, c in enumerate(chunks_sorted, 1):
        context_parts.append(
            f"[SOURCE {i} — {c['quarter']} | relevance: {c['score']:.2f}]\n{c['chunk_text']}"
        )
    context = "\n\n".join(context_parts)

    system_prompt = f"""You are EarningsIQ, a financial research assistant that answers questions about earnings call transcripts.

RETRIEVED CONTEXT FROM TRANSCRIPTS:
{context}

INSTRUCTIONS:
1. Answer the user's question using ONLY the retrieved context above.
2. Always cite which quarter each piece of information comes from (e.g., "In Q1 2024, management stated...").
3. For cross-quarter questions, explicitly compare what changed between quarters.
4. If a question requires computing a percentage, growth rate, or financial metric, use the calculator_tool — do NOT estimate math in your head.
5. If the answer is not in the retrieved context, say: "I could not find this information in the uploaded transcripts."
6. Never make up information. Every claim must trace to a source above.
7. End your answer with a brief "Sources:" section listing which quarters you cited.

DISCLAIMER: This is a research aid only. Not financial advice."""

    messages = [{"role": "user", "content": question}]

    # Agentic loop (ReAct)
    for _ in range(10):  # max iterations
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1500,
            system=system_prompt,
            tools=TOOLS,
            messages=messages,
        )

        # Check if we need to handle tool calls
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        if not tool_use_blocks:
            # No tool calls — extract final text answer
            text_blocks = [b for b in response.content if b.type == "text"]
            answer = " ".join(b.text for b in text_blocks)
            return answer, chunks_sorted

        # Process tool calls
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for tool_use in tool_use_blocks:
            if tool_use.name == "calculator_tool":
                result = calculator_tool(tool_use.input.get("expression", ""))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result,
                })

        messages.append({"role": "user", "content": tool_results})

    return "The agent reached the maximum number of steps without completing the answer.", chunks_sorted


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.title("📊 EarningsIQ")
st.caption("Multi-Quarter Earnings Intelligence Assistant — ask cross-quarter questions across earnings call transcripts")

# ── Sidebar: Upload transcripts ───────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Upload Transcripts")
    st.caption("Upload 2–4 earnings call transcripts (PDF or TXT). Label each by quarter.")

    num_transcripts = st.selectbox("How many transcripts?", [2, 3, 4], index=1)

    transcripts_raw = []
    quarters_used = []
    quarter_options = [
        "Q1 2023", "Q2 2023", "Q3 2023", "Q4 2023",
        "Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024",
        "Q1 2025", "Q2 2025", "Q3 2025", "Q4 2025",
    ]

    for i in range(num_transcripts):
        st.divider()
        col1, col2 = st.columns([1, 1])
        with col1:
            quarter = st.selectbox(
                f"Quarter {i+1}",
                [q for q in quarter_options if q not in quarters_used],
                key=f"quarter_{i}",
            )
            quarters_used.append(quarter)
        with col2:
            uploaded = st.file_uploader(
                f"Transcript {i+1}",
                type=["pdf", "txt"],
                key=f"file_{i}",
            )
        if uploaded:
            transcripts_raw.append({"quarter": quarter, "file": uploaded})

    st.divider()
    build_btn = st.button("🔍 Build Index", type="primary", use_container_width=True)

    if build_btn:
        if len(transcripts_raw) < 2:
            st.error("Please upload at least 2 transcripts.")
        else:
            with st.spinner("Extracting text and building index..."):
                transcripts = []
                for t in transcripts_raw:
                    text = extract_text(t["file"])
                    transcripts.append({"quarter": t["quarter"], "text": text})
                    st.success(f"✅ {t['quarter']} — {len(text.split())} words")

                index = build_index(transcripts)
                st.session_state["index"] = index
                st.session_state["quarters"] = [t["quarter"] for t in transcripts]
                st.success(f"Index built! {len(index)} chunks across {len(transcripts)} transcripts.")

# ── Main: Q&A ─────────────────────────────────────────────────────────────────

if "index" not in st.session_state:
    st.info("👈 Upload your earnings call transcripts in the sidebar and click **Build Index** to get started.")

    st.subheader("💡 Example questions you can ask:")
    examples = [
        "How has management's tone on gross margins changed across quarters?",
        "When did the company first mention AI investment, and how has that evolved?",
        "What revenue guidance did management give each quarter?",
        "What was the quarter-over-quarter revenue growth rate between Q1 and Q2?",
        "Did the CFO sound more or less confident about the services segment over time?",
    ]
    for ex in examples:
        st.markdown(f"- *{ex}*")
else:
    quarters = st.session_state["quarters"]
    st.success(f"✅ Index ready — {len(quarters)} quarters loaded: {', '.join(quarters)}")

    # Question input
    question = st.text_input(
        "Ask a question across your earnings transcripts:",
        placeholder="e.g. How has management's tone on margins changed across quarters?",
    )

    if st.button("🔎 Ask EarningsIQ", type="primary") and question:
        with st.spinner("Retrieving relevant passages and generating answer..."):
            index = st.session_state["index"]
            chunks = retrieve(question, index, top_k=5)
            answer, sorted_chunks = run_agent(question, chunks)

        # Display answer
        st.subheader("📝 Answer")
        st.markdown(answer)

        # Display retrieved sources
        with st.expander("📎 Retrieved Sources (click to expand)", expanded=False):
            for i, c in enumerate(sorted_chunks, 1):
                st.markdown(f"**Source {i} — {c['quarter']}** (relevance: {c['score']:.2f})")
                st.text(c["chunk_text"][:500] + ("..." if len(c["chunk_text"]) > 500 else ""))
                st.divider()

    # Disclaimer
    st.divider()
    st.caption("⚠️ EarningsIQ answers are based solely on the uploaded transcripts. This is not financial advice.")
