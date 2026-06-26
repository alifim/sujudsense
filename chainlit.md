# Welcome to SujudSense 🕋

**SujudSense** is an AI-powered coaching assistant designed to help you safely adapt your prayer postures (like *Ruku* and *Sujud*) when dealing with physical injuries, joint pain, or mobility limitations.

> ⚠️ **Disclaimer:** SujudSense is a V1 technology proof-of-concept. It is not a substitute for a licensed medical professional or a qualified Islamic scholar. Always consult a doctor for severe pain.

## How to Use This App

Simply describe your physical limitation or pain, and the assistant will provide an ergonomically sound adjustment backed by established Fiqh. The app features conversational memory, so you can ask follow-up questions naturally!

**Try asking:**
* *"I have sharp knee pain when bending fully. How should I modify my Sujud?"*
* *"My lower back hurts. I can't do ruku properly."*
* *(Follow-up)*: *"What if I also feel it in my shoulders?"*

*(Note: The system is strictly scoped. If you ask a purely medical diagnosis question or a general off-topic question, the safety firewall will respectfully decline to answer.)*

---

## 🛠️ For Developers

SujudSense demonstrates an enterprise-grade, deterministic **Retrieval-Augmented Generation (RAG)** architecture designed to prevent hallucinations, token-bleeding, and context-loss.

* **The Engine:** Built with LangChain, utilizing Groq for high-speed inference (`llama-3.3-70b-versatile`).
* **Conversational Memory:** Implements a Context Condenser chain to rewrite ambiguous follow-up queries using session history, ensuring firewalls never lose context.
* **Dual-Layer Safety Firewall:** Uses a structured-output Intent Classifier (`llama-3.3-70b-versatile`) paired with a local CPU vector distance check (L2) to drop off-topic or jailbreak queries *before* they reach the main reasoning chain.
* **The Vector Store:** Persistent ChromaDB with Hugging Face local CPU embeddings (`sentence-transformers/all-MiniLM-L6-v2`).
* **Telemetry & Analytics:** Integrated client-side Google Analytics (GA4) telemetry injected natively via `.chainlit/config.toml` and custom JavaScript to track live user interactions seamlessly across Hugging Face iframe boundaries.
* **The UI:** Asynchronous WebSocket streaming via Chainlit. 

💻 **[View the full decoupled architecture on GitHub](https://github.com/alifim/sujudsense)**
