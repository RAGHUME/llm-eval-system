# 🧪 LLM Evaluation & Prompt Optimization System

> **Evaluate. Compare. Optimize.** — Test your LLM prompts against 7 advanced metrics and improve them automatically.

A production-grade web application for evaluating Large Language Model (LLM) responses across multiple prompts, comparing outputs using measurable criteria, and improving prompt quality through systematic iteration.

---

## 🎯 Problem Statement

When working with LLMs, **prompt quality dramatically affects output quality** — but there's no easy way to:
- Objectively compare different prompt strategies
- Measure response quality with multiple metrics
- Systematically improve prompts based on data

This system solves all three problems with a **100% local, open-source** toolchain.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Web Browser (UI)                      │
│         Tailwind CSS + Chart.js + Jinja2 Templates       │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP
┌─────────────────────▼───────────────────────────────────┐
│                  FastAPI Server                          │
│   ┌──────────┐  ┌──────────┐  ┌────────────────────┐   │
│   │  Routes   │  │ Evaluator│  │    Optimizer        │   │
│   └────┬─────┘  └────┬─────┘  └────────┬───────────┘   │
│        │              │                  │               │
│   ┌────▼─────┐  ┌────▼─────┐  ┌────────▼───────────┐   │
│   │  Prompt   │  │ 7 Metric │  │ Weakness Analyzer   │   │
│   │  Engine   │  │ Scorers  │  │ + Auto-Improver     │   │
│   └────┬─────┘  └──────────┘  └─────────────────────┘   │
│        │                                                 │
│   ┌────▼──────────────────┐  ┌──────────────────────┐   │
│   │  Ollama Interface     │  │   SQLite Database     │   │
│   │  (Async Parallel)     │  │   (SQLAlchemy ORM)    │   │
│   └────┬──────────────────┘  └──────────────────────┘   │
└────────┼────────────────────────────────────────────────┘
         │ HTTP (localhost:11434)
┌────────▼────────────────────────────────────────────────┐
│              Ollama (Local LLM Runner)                    │
│              Model: phi3:mini (2.3GB RAM)                 │
└─────────────────────────────────────────────────────────┘
```

---

## ⚡ Tech Stack

| Component        | Technology                          | Why                                    |
|------------------|-------------------------------------|----------------------------------------|
| Backend          | FastAPI + Uvicorn                   | Modern async Python framework          |
| Frontend         | Jinja2 + Tailwind CSS + Chart.js    | Server-rendered, beautiful dashboards  |
| LLM Runner       | Ollama (phi3:mini)                  | Local, free, fits 8GB RAM             |
| Database         | SQLite + SQLAlchemy                 | Zero setup, reliable storage           |
| Async HTTP       | httpx                               | Non-blocking Ollama calls              |
| NLP Metrics      | nltk, rouge-score                   | BLEU + ROUGE scoring                   |
| Semantic Scoring | sentence-transformers (MiniLM-L6)   | Cosine similarity, small model         |
| Entity Analysis  | nltk (POS tagging)                  | Keyword extraction                     |

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- 8GB RAM minimum

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/llm-eval-system.git
cd llm-eval-system

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('stopwords')"

# 5. Pull the LLM model
ollama pull phi3:mini

# 6. Start the application
uvicorn main:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## 📊 Evaluation Metrics

| #  | Metric       | Score Range | What It Measures                              |
|----|-------------|-------------|-----------------------------------------------|
| 1  | BLEU        | 0.0 – 1.0  | N-gram overlap with reference answer          |
| 2  | ROUGE-L     | 0.0 – 1.0  | Longest common subsequence similarity         |
| 3  | Relevance   | 0.0 – 1.0  | Semantic similarity via embeddings            |
| 4  | Entity      | 0.0 – 1.0  | Key concept coverage                          |
| 5  | Structure   | 0.0 – 1.0  | Response formatting quality                   |
| 6  | Consistency | 0.0 – 1.0  | Stability across multiple runs                |
| 7  | LLM Judge   | 0.0 – 1.0  | AI-evaluated accuracy, clarity, completeness  |

### Scoring Formula
```
total = 0.10×BLEU + 0.10×ROUGE + 0.20×Relevance + 0.15×Entity
      + 0.10×Structure + 0.10×Consistency + 0.25×LLM_Judge
```

---

## 🛠️ API Endpoints

| Method | Endpoint           | Description                    |
|--------|-------------------|--------------------------------|
| GET    | `/`               | Main evaluation dashboard      |
| POST   | `/evaluate`       | Run evaluation on prompts      |
| POST   | `/optimize`       | Optimize worst-performing prompt|
| GET    | `/history`        | View all past evaluation runs  |
| GET    | `/history/{id}`   | View specific run details      |
| GET    | `/health`         | System health check            |

---

## 📁 Project Structure

```
llm-eval-system/
├── main.py                     # FastAPI app entry point
├── requirements.txt            # Pinned dependencies
├── .gitignore
├── README.md
├── core/                       # Core engine modules
│   ├── database.py             # SQLite schema + CRUD
│   ├── ollama_interface.py     # Async Ollama communication
│   └── prompt_engine.py        # Prompt template strategies
├── evaluation/                 # Scoring pipeline
│   ├── evaluator.py            # Combined evaluator
│   ├── ranker.py               # Rank by total score
│   └── metrics/                # Individual metric scorers
│       ├── bleu_score.py
│       ├── rouge_score.py
│       ├── relevance_score.py
│       ├── entity_score.py
│       ├── structure_score.py
│       ├── consistency_score.py
│       └── llm_judge.py
├── analysis/
│   └── error_analyzer.py       # Error detection
├── optimization/
│   └── optimizer.py            # Prompt improvement engine
├── api/
│   └── routes.py               # FastAPI route handlers
├── templates/                  # Jinja2 HTML pages
│   ├── base.html
│   ├── index.html
│   ├── results.html
│   ├── history.html
│   └── optimize.html
├── static/                     # Static assets
├── data/                       # SQLite database (auto-created)
├── tests/                      # Unit tests
└── docs/                       # Documentation
    ├── RESEARCH.md
    └── ITERATIONS.md
```

---

## 🚀 Status

🔨 **Under active development** — Building core modules.

---

*Built with ❤️ using 100% open-source tools. No paid APIs required.*
