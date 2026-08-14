import logging
import os
import json
from typing import Optional, Any, Dict, List
from collections import deque
from datetime import datetime

from groq import AsyncGroq
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_core.documents import Document
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from config import config
from logger import logger
from safety import SafetyPolicy, QueryIntent


class SujudSenseEngine:
    def __init__(self):
        self.vector_store: Optional[Chroma] = None
        self.rag_chain: Optional[Runnable] = None
        self.retriever: Optional[Runnable] = None
        self.base_retriever: Optional[Runnable] = None  # Store pre-reranking retriever for testing
        self._chunks: List[Document] = []
        self.use_hybrid = getattr(config, "use_hybrid", False)
        self.hybrid_weights = getattr(config, "hybrid_weights", [0.5, 0.5])
        
        # LLM call tracking
        self.llm_call_counts = {"condense": 0, "classify": 0, "generate": 0, "test": 0}
        self.llm_call_log: deque = deque(maxlen=1000)  # last 1000 calls only

    
    def load_clean_sources(self) -> List[Document]:
        """Load from cleaned sources (verified PDF extractions)."""
        docs = []
        
        for path, domain, tier in [
            ("sources/v3_clean/biomechanics_clean.md", "biomechanics", "verified_peer_reviewed"),
            ("sources/v3_clean/fiqh_clean.md", "fiqh", "verified_official_fatwa"),
        ]:
            if os.path.exists(path):
                loaded = TextLoader(path, encoding="utf-8").load()
                for doc in loaded:
                    doc.metadata["domain"] = domain
                    doc.metadata["source_tier"] = tier
                docs.extend(loaded)
                logger.info(f"Loaded {domain}: {path}")
            else:
                logger.warning(f"Source not found: {path}")
        
        # Fallback to legacy TXT files
        if not docs:
            logger.warning("Cleaned sources not found, falling back to legacy TXT files")
            for path in ["data/biomechanics.txt", "data/fiqh.txt"]:
                if os.path.exists(path):
                    docs.extend(TextLoader(path, encoding="utf-8").load())
        
        return docs

    def _chunks_file_path(self) -> str:
        return os.path.join(config.persist_directory, "chunks.jsonl")

    def _save_chunks(self, chunks: List[Document]) -> None:
        try:
            os.makedirs(config.persist_directory, exist_ok=True)
            path = self._chunks_file_path()
            with open(path, "w", encoding="utf-8") as f:
                for doc in chunks:
                    json.dump({"page_content": doc.page_content, "metadata": doc.metadata}, f, ensure_ascii=False)
                    f.write("\n")
            logger.info(f"Saved {len(chunks)} chunks to {path}")
        except Exception as e:
            logger.warning(f"Failed to save chunks: {e}")

    def _load_chunks(self) -> List[Document]:
        path = self._chunks_file_path()
        docs: List[Document] = []
        if not os.path.exists(path):
            return docs
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    docs.append(Document(page_content=obj.get("page_content", ""), metadata=obj.get("metadata", {})))
            logger.info(f"Loaded {len(docs)} chunks from {path}")
        except Exception as e:
            logger.warning(f"Failed to load chunks: {e}")
        return docs

    async def initialize(self):
        """Asynchronously sets up the RAG assets."""
        logger.info("Bootstrapping SujudSenseEngine...")
        embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)

        if os.path.exists(config.persist_directory) and os.listdir(config.persist_directory):
            logger.info("Loading existing vector store from disk.")
            self.vector_store = Chroma(
                persist_directory=config.persist_directory,
                embedding_function=embeddings,
            )
            # Create chroma retriever first
            chroma_retriever = self.vector_store.as_retriever(search_kwargs={"k": config.retrieval_k})

            # Attempt to load persisted chunks so BM25/hybrid can be reconstructed
            chunks = self._load_chunks()
            self._chunks = chunks

            if self.use_hybrid:
                if chunks:
                    logger.info("Persisted chunks found; building hybrid BM25+Chroma retriever.")
                    bm25_retriever = BM25Retriever.from_documents(chunks)
                    bm25_retriever.k = config.retrieval_k

                    self.retriever = EnsembleRetriever(
                        retrievers=[bm25_retriever, chroma_retriever],
                        weights=self.hybrid_weights,
                    )
                else:
                    logger.warning(
                        "Hybrid search requested but persisted chunks not found; falling back to dense-only retriever. "
                        "Delete persist_directory and restart to rebuild with hybrid support."
                    )
                    self.retriever = chroma_retriever
            else:
                self.retriever = chroma_retriever
        else:
            logger.info("Building new vector store...")
            docs = self.load_clean_sources()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                separators=["\\n\\n", "\\n", ". ", " "],
            )
            chunks = text_splitter.split_documents(docs)
            self._chunks = chunks
            
            logger.info(f"Split into {len(chunks)} chunks")

            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=config.persist_directory,
            )
            # Persist chunk snapshot so we can rebuild BM25 on subsequent loads
            try:
                self._save_chunks(chunks)
            except Exception:
                logger.warning("Failed to persist chunks snapshot; hybrid rebuild may require re-splitting sources.")
            
            # Build retriever: dense-only or hybrid ensemble
            chroma_retriever = self.vector_store.as_retriever(search_kwargs={"k": config.retrieval_k})
            
            if self.use_hybrid:
                logger.info(f"Building hybrid ensemble (weights={self.hybrid_weights})...")
                bm25_retriever = BM25Retriever.from_documents(chunks)
                bm25_retriever.k = config.retrieval_k
                
                self.retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, chroma_retriever],
                    weights=self.hybrid_weights,
                )
            else:
                self.retriever = chroma_retriever

        # Store base retriever before adding reranker (for pytest comparison)
        self.base_retriever = self.retriever

        if config.use_reranker:
            cross_encoder = HuggingFaceCrossEncoder(model_name=config.reranker_model)
            reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)
            self.retriever = ContextualCompressionRetriever(
                        base_compressor=reranker,
                        base_retriever=self.retriever,
                    )

        self._build_chain()
        logger.info("Engine initialization complete.")

    def _ensure_initialized(self) -> None:
        if self.retriever is None or self.rag_chain is None or self.vector_store is None:
            logger.error("Attempted execution before engine assets were initialized.")
            raise RuntimeError("Engine assets are not fully initialized")

    def _track_llm(self, method: str, model: str, query_preview: str) -> None:
        """Track LLM calls with timestamp, model, and query preview."""
        self.llm_call_counts[method] += 1
        self.llm_call_log.append({
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "model": model,
            "query": query_preview[:80]
        })

    def get_llm_stats(self) -> dict:
        """Safe for production — bounded, never grows unbounded."""
        return {
            "totals": dict(self.llm_call_counts),
            "recent_calls": list(self.llm_call_log),
            "recent_count": len(self.llm_call_log)
        }

    def reset_llm_counts(self) -> None:
        """Use in tests only. In production, counts are cumulative by design."""
        self.llm_call_counts = {"condense": 0, "classify": 0, "generate": 0}
        self.llm_call_log.clear()

    def is_blocked_by_hardcoded_policy(self, query: str) -> bool:
        return SafetyPolicy.should_block(query)

    def is_capability_query(self, query: str) -> bool:
        return SafetyPolicy.should_provide_capability_response(query)

    def _rule_based_condense(self, query: str, chat_history: list) -> str:
        """Deterministic fallback for query condensation. Handles common correction patterns."""
        import re

        query_lower = query.lower()
        positions = ["ruku", "sujud", "julus", "tashahhud", "qiyam", "salam", "sitting between", "sajdah"]

        # Detect correction patterns: "I mean X", "actually X", "not Y, I want X"
        correction_patterns = [
            r'i mean\s+(\w+)',
            r'actually\s+i?\s*meant?\s+(\w+)',
            r'switch(?:ing)?\s+to\s+(\w+)',
            r'what about\s+(\w+)\s+instead',
        ]

        current_pos = None
        for pattern in correction_patterns:
            match = re.search(pattern, query_lower)
            if match:
                candidate = match.group(1)
                if candidate in positions:
                    current_pos = candidate
                    break

        # If no correction detected, find first position in query
        if not current_pos:
            current_pos = next((p for p in positions if p in query_lower), None)

        # Extract body part/pain from last human message
        if len(chat_history) >= 2:
            last_human_msg = chat_history[-2]
            last_human = str(getattr(last_human_msg, 'content', last_human_msg)).lower()
        else:
            last_human = ""

        body_parts = ["lower back", "back", "knee", "shoulder", "hip", "wrist", "elbow", "neck", "ankle", "leg", "arm"]
        mentioned_body = [b for b in body_parts if b in last_human]

        pain_terms = ["pain", "hurt", "surgery", "injury", "cannot", "unable", "recovery", "sore", "ache"]
        has_pain = any(t in last_human for t in pain_terms)

        # Build standalone query
        if current_pos and has_pain and mentioned_body:
            body_str = mentioned_body[0]
            result = f"What adjustments for {body_str} pain during {current_pos}?"
        elif current_pos and has_pain:
            result = f"What adjustments for pain during {current_pos}?"
        elif current_pos:
            result = f"What adjustments during {current_pos}?"
        else:
            result = query

        # Preserve meta-instructions like "simplify", "clarify", "explain", etc.
        meta_instructions = ["simplify", "simpler", "easy", "explain", "clarify", "clarification", "more detail", "details"]
        if any(meta in query_lower for meta in meta_instructions):
            # Find which meta-instruction is present
            found_meta = [meta for meta in meta_instructions if meta in query_lower]
            if found_meta:
                result = f"{result} Please {found_meta[0]} your language."

        logger.info(f"Memory Condenser (Rule-based) | Rewrote to: '{result}'")
        return result

    async def condense_query(self, query: str, chat_history: list) -> str:
        if not chat_history:
            return query

        # Track if original query has meta-instructions
        query_lower = query.lower()
        meta_instructions = ["simplify", "simpler", "easy", "explain", "clarify", "clarification", "more detail", "details"]
        has_meta = any(meta in query_lower for meta in meta_instructions)
        found_meta = [meta for meta in meta_instructions if meta in query_lower][0] if has_meta else None

        try:
            llm_result = await self.condenser_chain.ainvoke({
                "chat_history": chat_history,
                "input": query,
            })
            llm_result = llm_result.strip()
            
            # Track LLM call
            self._track_llm("condense", config.fast_llm_model, query)

            # Validate LLM output
            bad_patterns = ["yoga", "pose", "movement", "(prostration)", "(seated)"]
            if llm_result and not any(bp in llm_result.lower() for bp in bad_patterns):
                # If original query had meta-instructions but result doesn't, add them back
                if has_meta and found_meta and found_meta not in llm_result.lower():
                    llm_result = f"{llm_result} Please {found_meta} your language."
                
                if llm_result != query:
                    logger.info(f"Memory Condenser | Rewrote to: '{llm_result}'")
                return llm_result

            logger.warning(f"LLM condenser output failed validation: '{llm_result}'. Falling back to rule-based.")
        except Exception as e:
            logger.warning(f"LLM condenser failed: {e}. Falling back to rule-based.")

        return self._rule_based_condense(query, chat_history)

    async def vector_firewall_score(self, standalone_query: str) -> Optional[float]:
        self._ensure_initialized()
        assert self.vector_store is not None
        raw_results = await self.vector_store.asimilarity_search_with_score(standalone_query, k=1)
        if not raw_results:
            return None
        _, best_score = raw_results[0]
        return best_score

    async def classify_intent(self, standalone_query: str) -> QueryIntent:
        self._ensure_initialized()
        
        # Build strict JSON schema from Pydantic model
        schema = QueryIntent.model_json_schema()
        schema["additionalProperties"] = False  # required for strict mode
        schema["required"] = list(schema.get("properties", {}).keys())  # all fields required
        
        system_prompt = """You are an intent classifier for SujudSense, an Islamic prayer guidance app.
    Analyze the user's query and classify it according to the schema."""
        
        response = await self.groq_client.chat.completions.create(
            model=config.fast_llm_model, 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": standalone_query}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "query_intent",
                    "strict": True,
                    "schema": schema
                }
            }
        )
        
        content = response.choices[0].message.content
        assert content is not None, "Groq strict mode should always return content"
        
        # Track LLM call
        self._track_llm("classify", config.fast_llm_model, standalone_query)
        
        parsed = json.loads(content)
        return QueryIntent(**parsed)

    async def intent_allows_query(self, standalone_query: str) -> tuple[bool, QueryIntent]:
        if SafetyPolicy.is_obvious_mobility_adaptation(standalone_query):
            # Synthetic intent for logging consistency
            bypass_intent = QueryIntent(
                reasoning="Hardcoded bypass: obvious mobility+prayer pattern detected",
                is_prayer_related=True,
                is_valid_mobility_adaptation_request=True,
            )
            logger.info(f"Intent Classification | Hardcoded bypass triggered for query: '{standalone_query}'")
            return True, bypass_intent
        
        intent = await self.classify_intent(standalone_query)
        is_allowed = intent.is_prayer_related and intent.is_valid_mobility_adaptation_request
        return is_allowed, intent

    async def evaluate_stages(self, query: str, chat_history: list) -> Dict[str, Any]:
        self._ensure_initialized()
        hardcoded_block = self.is_blocked_by_hardcoded_policy(query)
        capability_trigger = self.is_capability_query(query)
        standalone_query = await self.condense_query(query, chat_history)
        vector_score = await self.vector_firewall_score(standalone_query)
        vector_pass = vector_score is None or vector_score <= config.firewall_threshold
        intent = await self.classify_intent(standalone_query)
        intent_pass = intent.is_prayer_related and intent.is_valid_mobility_adaptation_request

        return {
            "raw_query": query,
            "standalone_query": standalone_query,
            "hardcoded_block": hardcoded_block,
            "capability_trigger": capability_trigger,
            "vector_score": vector_score,
            "vector_pass": vector_pass,
            "intent": intent.model_dump(),
            "intent_pass": intent_pass,
        }

    def _build_chain(self):
        deterministic_llm = ChatGroq(
            model=config.fast_llm_model, 
            temperature=0,
            max_tokens=config.fast_llm_model_max_tokens
        )
        
        # Build intent classifier using Groq client directly since GPT OSS doesn't support tool-calling
        self.groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

        # Build FEW-SHOT condenser (replaces the old generic condenser)
        condense_system = (
            "You are a strict query rewriter for a prayer posture assistant.\n\n"
            "Your job: combine chat history + latest user message into ONE standalone question.\n\n"
            "CRITICAL RULES:\n"
            "1. Output ONLY the rewritten question. No advice, no answers, no explanations.\n"
            "2. If the user corrects a position ('I mean X, not Y', 'actually X', 'switch to X', 'what about X instead'), use ONLY the corrected position X.\n"
            "3. Use exact prayer position names: Ruku, Sujud, Julus, Qiyam, Tashahhud, Salam. Do NOT add labels like '(prostration)', '(seated)', 'movement', 'pose', or 'yoga'.\n"
            "4. Carry forward pain/injury/body part context from history ONLY if the user hasn't explicitly changed the body part or dismissed it.\n"
            "5. Keep it concise (under 15 words if possible).\n"
            "6. Never reframe static positions as 'movements'.\n\n"
            "7. **META-REQUEST RULE**: If the user asks to simplify, rephrase, explain more, or give more detail about a PREVIOUS answer, "
            "   output the EXACT form: 'Simplify/rephrase/explain the previous answer about [body part] pain during [position] using plain language.' "
            "   Do NOT drop the body part or position. Do NOT output generic phrases like 'Please simplify your language.'\n\n"
            "EXAMPLES:\n\n"
            "History: [User: 'I have knee pain in Sujud']\n"
            "Latest: 'What about Ruku?'\n"
            "Output: What adjustments for knee pain during Ruku?\n\n"
            "History: [User: 'Lower back pain during Ruku']\n"
            "Latest: 'I mean julus, not ruku'\n"
            "Output: What adjustments for lower back pain during Julus?\n\n"
            "History: [User: 'Shoulder hurts in Ruku']\n"
            "Latest: 'Actually I meant Sujud, my shoulder is fine in Ruku'\n"
            "Output: What adjustments for shoulder pain during Sujud?\n\n"
            "History: [User: 'I feel lower back pain during Ruku; what adjustments are safe?']\n"
            "Latest: 'can you simplify your language? i don't understand intradiscal pressure, lumbar herniation'\n"
            "Output: Simplify the previous answer about lower back pain during Ruku using plain language.\n\n"
            "Now rewrite this conversation:"
        )
        condense_prompt = ChatPromptTemplate.from_messages([
            ("system", condense_system),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        self.condenser_chain = condense_prompt | deterministic_llm | StrOutputParser()

        system_prompt = (
            "You are SujudSense, an AI coaching agent specializing in prayer posture adjustments for physical ailments.\\n"
            "You can assume the user has already mentioned a physical limitation. "
            "Your task is to resolve their posture issue using ONLY the provided Context.\\n\\n"
            "<context>\\n{context}\\n</context>\\n\\n"
            "Synthesize your answer with the anatomical cue first, then the Fiqh validation.\\n"
            "If the context lacks specific advice for their ailment, state you do not have enough context.\\n"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        llm = ChatGroq(
            model=config.heavy_llm_model,
            temperature=config.heavy_llm_temperature,
            max_tokens=config.heavy_llm_max_tokens,
        )

        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        self.rag_chain = create_retrieval_chain(self.retriever, combine_docs_chain)

    async def generate_response(self, query: str, chat_history: list) -> str:
        if self.retriever is None or self.rag_chain is None or self.vector_store is None:
            logger.error("Attempted generation before engine assets were initialized.")
            raise RuntimeError("Engine assets are not fully initialized")

        logger.info(f"Incoming Request | History Depth: {len(chat_history)} | Raw Input: \'{query}\'")

        if self.is_blocked_by_hardcoded_policy(query):
            logger.warning(f"Security Alert | Hardcoded Policy Triggered | Blocked pattern in raw input: \'{query}\'")
            return SafetyPolicy.JAILBREAK_PHRASE
            
        if self.is_capability_query(query):
            logger.info("System Route | Capability request handled locally.")
            return SafetyPolicy.GENERAL_CAPABILITY_RESPONSE

        standalone_query = await self.condense_query(query, chat_history)
        if standalone_query != query:
            logger.info(f"Memory Condenser | Rewrote to Standalone Query: \'{standalone_query}\'")

        score = await self.vector_firewall_score(standalone_query)
        if score is not None and score > config.firewall_threshold:
            logger.warning(
                f"Firewall Block | L2 Distance Exceeded | Score: {score:.4f} > "
                f"Threshold: {config.firewall_threshold} | Standalone Query: \'{standalone_query}\'"
            )
            return SafetyPolicy.REFUSAL_PHRASE

        try:
            intent_pass, intent = await self.intent_allows_query(standalone_query)
            if not intent_pass:
                logger.debug(f"Intent Classification Metrics: {intent.model_dump()}")
                logger.warning(
                    f"Firewall Block | Intent Mismatch | Prayer: {intent.is_prayer_related} | "
                    f"Medical: {intent.is_valid_mobility_adaptation_request} | Standalone Query: '{standalone_query}'"
                )
                return SafetyPolicy.REFUSAL_PHRASE
        except Exception as e:
            logger.error(f"Firewall System Failure | Intent classification raised exception: {e}", exc_info=True)
            return SafetyPolicy.ERROR_PHRASE

        logger.info(f"Execution Pipeline | Dispatching valid query to Heavy Synthesis Chain.")
        
        if logger.isEnabledFor(logging.DEBUG):
            docs = await self.retriever.ainvoke(standalone_query)
            for i, doc in enumerate(docs):
                src = doc.metadata.get("source_tier", "unknown")
                logger.debug(f"Retrieved Chunk {i+1} Source: {src} | Preview: {doc.page_content[:100]}...")

        response = await self.rag_chain.ainvoke({"input": standalone_query})
        
        # Track LLM call
        self._track_llm("generate", config.heavy_llm_model, standalone_query)
        
        answer = (response.get("answer") or "").strip()

        truncated_indicators = ("adjust your", "you may need to adjust", "adjust", "to adjust")
        if not answer or answer[-1] not in ".!?" or answer.lower().endswith(truncated_indicators):
            try:
                logger.info("Output Guardrail | Potential truncation detected. Invoking completion sequence.")
                cont_prompt = f"Please continue the previous answer concisely. Previous: {answer}"
                cont_resp = await self.rag_chain.ainvoke({"input": cont_prompt})
                cont = (cont_resp.get("answer") or "").strip()
                if cont:
                    answer = f"{answer} {cont}".strip()
            except Exception as e:
                logger.warning(f"Output Guardrail | Continuation pass failed execution: {e}")

        if answer and answer[-1] not in ".!?":
            answer += "."

        physical_terms = ("knee", "back", "sujud", "ruku", "shoulder", "pain", "injury")
        if any(term in answer.lower() for term in physical_terms):
            if SafetyPolicy.MEDICAL_NOTICE.lower() not in answer.lower():
                answer = f"{answer} {SafetyPolicy.MEDICAL_NOTICE}"
                logger.debug("Output Guardrail | Appended standard medical safety notice to response payload.")

        logger.info("Request Cycle Complete | Successfully returned synchronized response.")
        return answer
