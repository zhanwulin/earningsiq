# EarningsIQ — Multi-Quarter Earnings Intelligence Assistant

**Student:** Zhanwu Lin
**Course:** Generative AI — JHU Spring II 2026

## 🎥 Walkthrough Video
https://www.youtube.com/watch?v=bPpY83tmmCI

---

## 1. Context, User, and Problem

**Who the user is:**
A junior financial analyst or business student who regularly reads public company earnings call transcripts to track financial performance, management commentary, and forward guidance across multiple quarters.

**What workflow I am improving:**
The analyst uploads 2–4 earnings call transcripts and asks cross-quarter questions such as "How has management's tone on gross margins changed from Q2 to Q4?" or "What was the revenue growth rate between Q2 and Q3?" Today, answering these questions requires manually Ctrl+F searching through several long PDFs, reading surrounding context, and mentally synthesizing differences across documents — a process that takes 30–90 minutes per question.

**Why it matters:**
Keyword search fails when the analyst does not know the exact word used in the transcript. It cannot synthesize trends across quarters or compute financial metrics automatically. EarningsIQ reduces cross-quarter analysis from 30–90 minutes to under 30 seconds while grounding every answer in the actual source documents.

---

## 2. Solution and Design

**What I built:**
A Streamlit app that accepts 2–4 earnings call transcripts (PDF or TXT), labels each by quarter, builds a multi-document RAG index, and answers cross-quarter questions in plain English with per-quarter citations and computed financial metrics.

**How it works:**
1. User uploads 2–4 transcript files and labels each by quarter
2. Each transcript is chunked into ~300-word segments with 10% overlap
3. All chunks are embedded and stored in a single in-memory index with quarter metadata
4. User asks a question; the top-5 most relevant chunks are retrieved across all quarters
5. Retrieved chunks are passed to Claude with a structured system prompt
6. If the question involves financial metrics, the agent calls a calculator tool for precise computation
7. The model returns a cited answer with per-quarter sources

**Key design choices:**

- **Multi-document RAG:** All transcripts are indexed together with quarter tags so the system can retrieve relevant passages from any quarter simultaneously. This is harder than single-document RAG because retrieval must balance relevance against temporal coverage.

- **Calculator tool:** When the user asks about revenue growth, margin changes, or other financial metrics, the agent calls a Python calculator tool rather than estimating math in its head. This ensures numerical claims are always exact.

- **Refusal design:** The system prompt instructs the model to say "I could not find this in the uploaded transcripts" rather than guess when no relevant chunk is retrieved. This prevents hallucination on out-of-scope questions.

**Why RAG is justified:**
A single prompt cannot hold 3–4 full transcripts in context. RAG allows the system to selectively retrieve only the relevant passages, keeping the context focused and the answers grounded.

---

## 3. Evaluation and Results

**Baseline:**
Manual Ctrl+F keyword search across 3 separate PDF files. The analyst picks a keyword, searches each PDF, reads surrounding context, and manually synthesizes differences. Timed for the same 10 test questions.

**Test set:**
10 questions drawn from Apple Q2, Q3, and Q4 2025 earnings call transcripts (all publicly available):
- 4 factual lookups (single-quarter facts)
- 4 cross-quarter trend questions
- 2 out-of-scope questions (answer not in transcripts)

**What counted as a good answer:**
- Correct factual content matching the transcript
- At least one citation per quarter referenced
- Correct refusal on out-of-scope questions

**Results:**

| Metric | Baseline (Ctrl+F) | EarningsIQ |
|---|---|---|
| Time per question | 15–30 minutes | < 30 seconds |
| Accuracy (10 questions) | ~60% | 80% |
| Cross-quarter synthesis | Manual | Automatic with citations |
| Out-of-scope refusal | No | Yes |
| Financial metric computation | Manual | Calculator tool (exact) |

**What worked:**
- Factual lookups from a single quarter were consistently accurate
- Cross-quarter tone questions correctly identified directional changes (e.g. margin tone shifted from defensive in Q2 to confident in Q4)
- The system correctly refused both out-of-scope questions

**Where it broke down:**
- Questions requiring external data (analyst consensus estimates) not present in the transcripts
- Tone questions where the relevant hedging language was not captured in the top-5 retrieved chunks
- Annualized return calculations on very short windows (same edge case identified in the stock-return-calculator skill)

---

## 4. Artifact Snapshot

**Sample question:** *"How has Apple's gross margin changed from Q2 to Q4 2025?"*

**Sample answer:**
- Q2 2025: Gross margin down 70 bps year-over-year, driven by mix and foreign exchange. Tone: neutral/defensive.
- Q3 2025: Gross margin 46.5%, at the high end of guidance range. Tone: cautiously optimistic.
- Q4 2025: Gross margin 47.2%, above the high end of guidance range. CFO stated they "landed in a pretty good spot." Tone: confident.

**Trend summary:** Management's tone on margins shifted from defensive explanations in Q2 to confident operational execution in Q4, with gross margin improving 70 basis points over the period.

The walkthrough video above shows the app running on Apple's Q2, Q3, and Q4 2025 transcripts with live questions and answers.

---

## Setup and Usage

### Requirements
- Python 3.10+
- An Anthropic API key (get one at https://console.anthropic.com/settings/keys)
- Optional: An OpenAI API key for higher quality embeddings (https://platform.openai.com/api-keys). The app runs without it using a built-in fallback.

### Installation

**1. Clone the repo**
```bash
git clone https://github.com/zhanwulin/earningsiq.git
cd earningsiq
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up API keys**

Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your-anthropic-key-here
OPENAI_API_KEY=your-openai-key-here
```

Or set the environment variable directly:
```bash
export ANTHROPIC_API_KEY=your-anthropic-key-here   # Mac/Linux
$env:ANTHROPIC_API_KEY="your-anthropic-key-here"   # Windows PowerShell
```

**4. Run the app**
```bash
streamlit run app.py
```

The app opens at http://localhost:8501

### Usage

1. In the sidebar, select how many transcripts to upload (2–4)
2. Label each transcript by quarter (e.g. Q1 2024, Q2 2024)
3. Upload each transcript file (PDF or TXT)
4. Click **Build Index**
5. Type a question and click **Ask EarningsIQ**

### Sample transcripts
Sample Apple earnings call transcripts (Q1–Q4 2025) are included in the `earnings_transcript_samples/` folder for testing.

---

## Limitations

- Works with uploaded transcript files only — no live or real-time data
- Does not include analyst consensus estimates (external data)
- Answers are based solely on the uploaded transcripts
- Not for investment decisions — research aid only
