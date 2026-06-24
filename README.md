# SujudSense

SujudSense is an AI-powered coaching assistant designed to help Muslims safely adapt their prayer postures (like Ruku and Sujud) when dealing with physical injuries, joint pain, or mobility limitations.

When modifying prayer movements, worshippers face a unique challenge: they must balance physical safety (sports biomechanics) with canonical validity (Islamic jurisprudence or Fiqh). Standard AI chatbots are dangerous for this—they frequently hallucinate medical diagnoses or invent religious exceptions that do not exist.

SujudSense solves this by restricting the AI to a strict, verified knowledge base. It will not guess, and it will not act like a doctor. If a user asks a valid question, it provides safe, ergonomically sound postural adjustments backed by established Fiqh. If a user asks an off-topic or purely medical question, the system's safety firewalls instantly block the request.

---

## 🚀 High-Level Overview

Modifying prayer positions (such as *Ruku* and *Sujud*) due to injury requires a highly delicate balance between joint ergonomics and canonical alignment. Standard LLMs frequently hallucinate unsafe medical adjustments or invalid jurisprudential exceptions.

**SujudSense** solves this by enforcing a strictly bounded knowledge environment. Operating under a closed-world assumption, the system pairs high-speed vector retrieval with programmatic guardrails to act as a reliable, safety-first assistant. It is engineered defensively to mitigate common production RAG failures: token bleeding, event-loop blocking, and prompt injection attacks.

### Key Engineering Highlights

* **Decoupled Service-Layer Pattern:** Business logic (`engine.py`) is completely separated from the presentation layer (`app.py`), allowing seamless porting to alternative interfaces (e.g., FastAPI) without rewriting core RAG logic.
* **Non-Blocking Asynchronous I/O:** Every stage of the data pipeline utilizes native Python `asyncio` (`ainvoke`, `asimilarity_search_with_score`) to prevent database and LLM queries from blocking the server's single-threaded event loop under concurrent user loads.
* **Cost-Defensive "Zero-Token Firewall":** Intercepts off-topic queries locally on the host CPU using an embedding L2 distance check. Unrelated inputs are rejected immediately, protecting cloud API endpoints from infinite generation loops and off-topic billings.
* **Adversarial Hardening:** Uses tagged context enclosure markers and recency-biased security instructions to help mitigate prompt injection and persona-switching jailbreak attempts.

---

## 🛠️ Technology Stack

* **Orchestration & UI:** LangChain Ecosystem, Chainlit (Asynchronous WebSockets)
* **Vector Compute & Database:** ChromaDB (Persistent Disk Storage Layout)
* **Local Embeddings:** Hugging Face `sentence-transformers/all-MiniLM-L6-v2` (CPU-Optimized)
* **Cloud Inference:** Groq API Cloud Engine (`llama-3.3-70b-versatile`)
* **Observability:** Centralized Native Python Logging (`logging` module wrapper)

---

## 📐 System Architecture

The workflow follows a deterministic, sequential security and retrieval pipeline:

```
[ User Input ]
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ Phase 1: Zero-Token Firewall (Local CPU Check)         │
│ - Computes L2 Vector Distance against local embeddings │
│ - IF best_score > threshold ──► [ Local Refusal Exit ]  │
└─────────────────────────┬──────────────────────────────┘
                          │ (Passes Threshold)
                          ▼
┌────────────────────────────────────────────────────────┐
│ Phase 2: Asynchronous Diverse Retrieval                │
│ - Executes non-blocking MMR search against ChromaDB     │
│ - Isolates k=3 unique context chunks across data paths │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ Phase 3: Hardened Prompt & Enclosure Assembly          │
│ - Encloses raw source data within tagged context blocks (`<context>...</context>`) │
│ - Attaches immutable persona & security directives     │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ Phase 4: Cloud Inference & Execution                  │
│ - Dispatches execution payload to Groq (Llama 3.3 70B) │
│ - Enforces max_tokens output caps for budget security  │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
                [ Streaming UI Response ]

```

---

## 📂 Repository Structure

```
├── app.py              # Presentation Layer: Manages WebSocket sessions and UI rendering
├── engine.py           # Service Layer: Handles RAG orchestration, database queries, and firewalling
├── config.py           # Configuration Layer: Environment variable resolution and type-safe defaults
├── logger.py           # Telemetry Layer: Centralized, environment-toggled application logger
├── chainlit.md         # Application entry view and user welcome screen
├── data/               # Ground-Truth Knowledge Base (Immutable Source Text)
│   ├── biomechanics.txt# Structured orthopedic and athletic movement constraints
│   └── fiqh.txt        # Canonical jurisprudential modification rulings
└── pyproject.toml      # Deterministic project metadata and dependencies

```

---

## ⚙️ Configuration & Environment Variables

The system architecture relies entirely on twelve-factor configuration practices. All system limits, paths, and behaviors are set using environment defaults:

| Variable | Default Value | Purpose |
| --- | --- | --- |
| `LOG_LEVEL` | `INFO` | Toggles telemetry output verbosity (`DEBUG` or `INFO`) |
| `FIREWALL_THRESHOLD` | `1.4` | Calibrated semantic distance cap for off-topic query rejection |
| `LLM_MAX_TOKENS` | `512` | Output response limit protecting against token drain |
| `LLM_TEMPERATURE` | `0.1` | Low temperature variable ensuring highly deterministic generation |
| `RETRIEVAL_K` | `3` | Number of context documents passed to the compilation prompt |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Disk directory mapping for the "Init-Once, Load-Many" persistent layer |

---

## ⚡ Setup & Execution

### 1. Environment Initialization

Ensure Python 3.11+ is active. Set your cloud inference token via your terminal profile or environment manager:

```bash
export GROQ_API_KEY="your-production-api-key-here"
export LOG_LEVEL="INFO"

```

### 2. Install Project Core Dependencies

```bash
pip install chainlit chromadb langchain langchain-classic langchain-community langchain-groq langchain-huggingface sentence-transformers

```

### 3. Launching the App

Run the local serving loop with the hot-reload watch flag active:

```bash
chainlit run app.py -w

```

### 4. Codebase Telemetry & Performance Auditing

To inspect the internal database inventory, execute diagnostic runs, or audit individual vector search distances on the fly, pass the runtime debug switch prior to boot:

```bash
export LOG_LEVEL=DEBUG
chainlit run app.py
```

## 🚀 Hugging Face Spaces Deployment

This project can be deployed as a Hugging Face Space using Docker.

### Recommended Hugging Face Space setup

1. Create a new Space on Hugging Face using the Docker SDK type.
2. Add the Space repository as a git remote in your local repo:

```bash
git remote add hf https://huggingface.co/spaces/<your-username>/sujudsense
```

3. Push your current branch to the HF remote:

```bash
git push hf main
```

4. Configure required secrets in the Space settings:

- `GROQ_API_KEY`
- `HF_TOKEN` (optional, only if you need authenticated HF Hub access)

### Spaces configuration

You can provide deployment metadata either through README frontmatter or using a `spaces.yaml` file.

Example `spaces.yaml`:

```yaml
sdk: docker
title: SujudSense
emoji: "📉"
colorFrom: "red"
colorTo: "pink"
pinned: false
```

For more configuration options, see:
https://huggingface.co/docs/hub/spaces-config-reference
