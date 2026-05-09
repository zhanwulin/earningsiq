# EarningsIQ — Multi-Quarter Earnings Intelligence Assistant

**Student:** Zhanwu Lin
**Course:** Generative AI — JHU Spring II 2026

## Walkthrough Video
[Add your video link here]

---

## What It Does

EarningsIQ is a RAG-powered Streamlit app that lets a junior financial analyst ask cross-quarter questions across 2–4 earnings call transcripts. Instead of manually Ctrl+F-ing across multiple PDFs, the analyst uploads the transcripts, labels each by quarter, and asks questions in plain English.

The system:
1. Chunks and embeds all transcripts into a single FAISS-style index with quarter metadata
2. Retrieves the top-5 most relevant passages across all quarters
3. Uses a **calculator tool** (from HW4) to compute precise financial metrics
4. Generates a grounded, cited answer that tracks trends across quarters

## Course Concepts Integrated

- **RAG** — multi-document chunking, embedding, and similarity retrieval with quarter metadata
- **Tool use** — `calculator_tool` computes revenue growth %, margins, earnings surprises
- **Evaluation design** — 15-question test set with model-as-judge scoring

## How to Run

### 1. Clone the repo
```bash
git clone https://github.com/zhanwulin/earningsiq.git
cd earningsiq
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your API keys
```bash
cp .env.example .env
```
Edit `.env` and add your Anthropic API key (required) and OpenAI API key (optional, for better embeddings).

### 4. Run the app
```bash
streamlit run app.py
```

### 5. Use the app
1. Upload 2–4 earnings call transcripts (PDF or TXT) in the sidebar
2. Label each by quarter (e.g., Q1 2024, Q2 2024)
3. Click **Build Index**
4. Type a question and click **Ask EarningsIQ**

## Example Questions

- "How has management's tone on gross margins changed across quarters?"
- "When did the company first mention AI investment, and how has that evolved?"
- "What was the quarter-over-quarter revenue growth rate between Q1 and Q2?"
- "Did the CFO sound more or less confident about the services segment over time?"

## Baseline Comparison

The baseline is manual Ctrl+F keyword search across 3 separate PDF files. The same 15 test questions are timed against both the system and the baseline.

## Project Structure

```
earningsiq/
├── app.py              # Main Streamlit app
├── requirements.txt    # Python dependencies
├── .env.example        # API key template
├── .gitignore          # Never commit .env
└── README.md           # This file
```

## Limitations

- Does not account for analyst consensus estimates (external data not in transcripts)
- Embedding quality depends on whether OpenAI key is provided (falls back to hash-based pseudo-embedding)
- Cross-quarter retrieval may favor the most recent transcript if question uses recent terminology
- Not suitable for investment decisions — research aid only
