---
sdk: docker
title: SujudSense
emoji: "🕋"
colorFrom: "gray"
colorTo: "blue"
pinned: false
---

# SujudSense

SujudSense is an AI-powered coaching assistant designed to help Muslims safely adapt their prayer postures (like Ruku and Sujud) when dealing with physical injuries, joint pain, or mobility limitations.

When modifying prayer movements, worshippers face a unique challenge: they must balance physical safety (sports biomechanics) with canonical validity (Islamic jurisprudence or Fiqh). Standard AI chatbots are dangerous for this—they frequently hallucinate medical diagnoses or invent religious exceptions that do not exist.

SujudSense solves this by restricting the AI to a strict, verified knowledge base. It will not guess, and it will not act like a doctor. If a user asks a valid question, it provides safe, ergonomically sound postural adjustments backed by established Fiqh. If a user asks an off-topic or purely medical question, the system's safety firewalls instantly block the request.

---

## 🚀 High-Level Overview

Modifying prayer positions due to injury requires a highly delicate balance between joint ergonomics and canonical alignment. Standard LLMs frequently hallucinate unsafe medical adjustments or invalid jurisprudential exceptions.

**SujudSense** solves this by enforcing a strictly bounded knowledge environment. Operating under a closed-world assumption, the system pairs high-speed vector retrieval with programmatic guardrails to act as a reliable, safety-first assistant. It is engineered defensively to mitigate common production RAG failures: token bleeding, conversational context loss, and prompt injection attacks.

### Key Engineering Highlights

* **Decoupled Service-Layer Pattern:** Business logic (`engine.py`) is completely separated from the presentation layer (`app.py`), allowing seamless porting to alternative interfaces (e.g., FastAPI) without rewriting core RAG logic.
* **Conversational Memory Condensing:** Implements a pre-processing LLM chain to rewrite ambiguous follow-up questions using session history, ensuring semantic search and firewalls never lose context.
* **Cost-Defensive Firewalls:** Intercepts off-topic queries locally using structured output Intent Classification and L2 embedding distance checks. Unrelated inputs are rejected immediately, protecting cloud endpoints from infinite generation loops and off-topic billings.
* **Non-Blocking Asynchronous I/O:** Every stage of the data pipeline utilizes native Python `asyncio` (`ainvoke`, `asimilarity_search_with_score`) to prevent database and LLM queries from blocking the server's single-threaded event loop under concurrent user loads.

---

## 🛠️ Technology Stack

* **Orchestration & UI:** LangChain Ecosystem, Chainlit (Asynchronous WebSockets)
* **Vector Compute & Database:** ChromaDB (Persistent Disk Storage Layout)
* **Local Embeddings:** Hugging Face `sentence-transformers/all-MiniLM-L6-v2` (CPU-Optimized)
* **Cloud Inference:** Groq API Cloud Engine 
  * *Routing, Memory, & Synthesis:* `llama-3.3-70b-versatile`
* **Telemetry & Analytics:** Google Analytics (GA4 client injection via `public/analytics.js`)
* **Observability:** Centralized Native Python Logging (`logging` module wrapper)

---

## 📐 System Architecture

The workflow follows a deterministic, sequential security and retrieval pipeline:

```text
[ User Input + Session History ]
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ Phase 1: Context Condenser (Memory State)              │
│ - Analyzes history via LLM                             │
│ - Rewrites ambiguous inputs into a Standalone Query    │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ Phase 2: Dual-Layer Safety Firewall                    │
│ - LLM checks for medical/prayer overlap                │
│ - CPU computes L2 vector distance against valid corpus │
│ - IF invalid ──► [ Local Refusal Exit ]                │
└─────────────────────────┬──────────────────────────────┘
                          │ (Passes Threshold)
                          ▼
┌────────────────────────────────────────────────────────┐
│ Phase 3: Asynchronous Diverse Retrieval                │
│ - Executes non-blocking similarity search via ChromaDB │
│ - Isolates k=3 unique context chunks across data paths │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ Phase 4: Cloud Inference & Execution                   │
│ - Dispatches execution payload to LLM                  │
│ - Enforces strict 512 token headrooms to block loops   │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ Phase 5: Formatting & Output Guardrails                │
│ - Validates terminal punctuation (prevents cutoff)     │
│ - Appends mandatory Medical Disclaimer                 │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
                [ Streaming UI Response ]
```

---

## 🛡️ Safety and Response Rules

SujudSense applies explicit programmatic safety rules before and after LLM generation:

- **Intent-Based Filtering:** Uses a structured-output LLM pass to classify the user's intent. If a query does not contain *both* a prayer-related context AND a medical/mobility context, it is safely rejected.
- **Off-Topic/Jailbreak Blocking:** Queries containing blacklisted patterns (e.g., `python script`, `hack`, `ignore previous instructions`) are intercepted programmatically before reaching the reasoning chains.
- **Capability Queries:** Questions like `what can you do?` or `how can you help?` bypass the LLM and return a hardcoded, safe scope description to save API compute costs.
- **Completeness Enforcement:** Output payloads are systematically formatted with structural headings (`Anatomical Cue` and `Fiqh Validation`). Generation lengths are explicitly capped at `512` tokens.
- **Medical Safety Notice:** When the response suggests physical prayer adjustments, the system explicitly appends a medical caution advising consultation with a healthcare professional for severe or worsening pain.

---

## 📂 Repository Structure

```text
├── app.py              # Presentation Layer: Manages WebSocket sessions and UI rendering
├── engine.py           # Service Layer: Handles conversational state, RAG orchestration, and firewalls
├── config.py           # Configuration Layer: Environment variable resolution and type-safe defaults
├── safety.py           # Security Layer: Hardcoded policies, intent schema, and blocklists
├── logger.py           # Telemetry Layer: Centralized, environment-toggled application logger
├── chainlit.md         # Application entry view and user welcome screen
├── public/
│   └── analytics.js    # Client-side Google Analytics (GA4) injection script
├── .chainlit/
│   └── config.toml     # Chainlit UI setup linking custom JS assets
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
| `HEAVY_LLM_MODEL` | `llama-3.3-70b-versatile` | Model used for final RAG synthesis and reasoning |
| `FAST_LLM_MODEL` | `llama-3.3-70b-versatile` | Model used for low-latency routing and context condensing |
| `LLM_MAX_TOKENS` | `512` | Output response limit protecting against token drain |
| `LLM_TEMPERATURE` | `0.1` | Low temperature variable ensuring highly deterministic generation |
| `RETRIEVAL_K` | `3` | Number of context documents passed to the compilation prompt |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Disk directory mapping for the persistent storage layer |

---

## ⚡ Setup & Execution

### 1. Environment Initialization

Ensure Python 3.12+ is active. Set your cloud inference token via your terminal profile or environment manager:

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

---

## 🚀 Hugging Face Spaces Deployment

This project is fully containerized and can be deployed natively as a Hugging Face Space using the Docker SDK.

1. Create a new Space on Hugging Face using the **Docker** SDK type (Choose the **Blank** template).
2. Add the Space repository as a git remote in your local repo:
   ```bash
   git remote add hf [https://huggingface.co/spaces/](https://huggingface.co/spaces/)<your-username>/sujudsense
   ```
3. Push your current branch to the HF remote:
   ```bash
   git push hf main
   ```
4. Configure required secrets in the Space settings UI:
   - `GROQ_API_KEY`
