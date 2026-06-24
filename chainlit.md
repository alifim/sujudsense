# Welcome to SujudSense 🕋

**SujudSense** is an AI-powered coaching assistant designed to help you safely adapt your prayer postures (like *Ruku* and *Sujud*) when dealing with physical injuries, joint pain, or mobility limitations.

> ⚠️ **Disclaimer:** SujudSense is a V1 technology proof-of-concept. It is not a substitute for a licensed medical professional or a qualified Islamic scholar. Always consult a doctor for severe pain.

## How to Use This App

Simply describe your physical limitation or pain, and the assistant will provide an ergonomically sound adjustment backed by established Fiqh. 

**Try asking:**
* *"I have sharp knee pain when bending fully. How should I modify my Sujud?"*
* *"My lower back hurts. I can't do ruku properly. What should I do?"*

*(Note: The system is strictly scoped. If you ask a purely medical diagnosis question or a general off-topic question, the safety firewall will respectfully decline to answer.)*

---

## 🛠️ For Developers & Tech Recruiters

SujudSense demonstrates an enterprise-grade, deterministic **Retrieval-Augmented Generation (RAG)** architecture designed to prevent hallucinations and API token-bleeding.

* **The Engine:** Built with LangChain, using a Groq inference backend (`llama-3.3-70b-versatile`).
* **The Vector Store:** ChromaDB with Hugging Face local CPU embeddings (`sentence-transformers/all-MiniLM-L6-v2`).
* **The Safety Layer:** Implements a programmatic "Zero-Token Firewall" that calculates L2 vector distance to intercept and drop off-topic or jailbreak queries locally *before* they reach the cloud LLM.
* **The UI:** Asynchronous WebSocket streaming via Chainlit. 

💻 **[View the full decoupled architecture on GitHub](https://github.com/alifim/sujudsense)**