import logging
import os
from typing import Optional, cast

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from config import config
from logger import logger
from safety import SafetyPolicy, QueryIntent

class SujudSenseEngine:
    def __init__(self):
        self.vector_store: Optional[Chroma] = None
        self.rag_chain: Optional[Runnable] = None
        self.retriever: Optional[Runnable] = None

    async def initialize(self):
        """Asynchronously sets up the RAG assets."""
        logger.info("Bootstrapping SujudSenseEngine...")
        embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)

        if os.path.exists(config.persist_directory) and os.listdir(config.persist_directory):
            logger.info("Loading existing vector store from disk.")
            self.vector_store = Chroma(
                persist_directory=config.persist_directory, 
                embedding_function=embeddings
            )
        else:
            logger.info("Building new vector store...")
            docs = []
            for path in ["data/biomechanics.txt", "data/fiqh.txt"]:
                docs.extend(TextLoader(path).load())

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size, 
                chunk_overlap=config.chunk_overlap
            )
            chunks = text_splitter.split_documents(docs)

            self.vector_store = Chroma.from_documents(
                documents=chunks, 
                embedding=embeddings, 
                persist_directory=config.persist_directory
            )

        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": config.retrieval_k})
        self._build_chain()
        logger.info("Engine initialization complete.")

    def _build_chain(self):
        # Temperature 0 is crucial for deterministic classification and rewriting
        deterministic_llm = ChatGroq(
            model=config.fast_llm_model, 
            temperature=0,
            max_tokens=config.fast_llm_model_max_tokens
        )
        
        # 1. The Intent Classifier
        self.intent_classifier = deterministic_llm.with_structured_output(QueryIntent, method="json_schema")

        # 2. The Contextual Condenser (Query Rewriter)
        condense_system = (
            "Given a chat history and the latest user question which might reference context in the chat history, "
            "formulate a standalone question which can be understood without the chat history. "
            "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
        )
        condense_prompt = ChatPromptTemplate.from_messages([
            ("system", condense_system),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        self.condenser_chain = condense_prompt | deterministic_llm | StrOutputParser()

        # 3. Main Generator (No history placeholder needed here because the query is already condensed)
        system_prompt = (
            "You are SujudSense, an AI coaching agent specializing in prayer posture adjustments for physical ailments.\n"
            "You can assume the user has already mentioned a physical limitation. "
            "Your task is to resolve their posture issue using ONLY the provided Context.\n\n"
            "<context>\n{context}\n</context>\n\n"
            "Synthesize your answer with the anatomical cue first, then the Fiqh validation.\n"
            "If the context lacks specific advice for their ailment, state you do not have enough context.\n"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        llm = ChatGroq(
            model=config.heavy_llm_model, 
            temperature=config.heavy_llm_temperature, 
            max_tokens=config.heavy_llm_max_tokens
        )

        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        self.rag_chain = create_retrieval_chain(self.retriever, combine_docs_chain)


    async def generate_response(self, query: str, chat_history: list) -> str:
        """Fully asynchronous execution integrating rewriting, firewalls, and generation with structured logging."""
        if self.retriever is None or self.rag_chain is None or self.vector_store is None:
            logger.error("Attempted generation before engine assets were initialized.")
            raise RuntimeError("Engine assets are not fully initialized")

        logger.info(f"Incoming Request | History Depth: {len(chat_history)} | Raw Input: '{query}'")

        # 1. Hardcoded Fast-Pass Checks (On Raw Query)
        if SafetyPolicy.should_block(query):
            logger.warning(f"Security Alert | Hardcoded Policy Triggered | Blocked pattern in raw input: '{query}'")
            return SafetyPolicy.JAILBREAK_PHRASE
            
        if SafetyPolicy.should_provide_capability_response(query):
            logger.info("System Route | Capability request handled locally.")
            return SafetyPolicy.GENERAL_CAPABILITY_RESPONSE

        # 2. Condense the Query (Memory Injection)
        if chat_history:
            standalone_query = await self.condenser_chain.ainvoke({
                "chat_history": chat_history,
                "input": query
            })
            logger.info(f"Memory Condenser | Rewrote to Standalone Query: '{standalone_query}'")
        else:
            standalone_query = query

        # 3. The L2 Vector Firewall (On Standalone Query)
        raw_results = await self.vector_store.asimilarity_search_with_score(standalone_query, k=1)
        if raw_results:
            _, best_score = raw_results[0]
            logger.debug(f"Firewall Check | Vector L2 Distance Score: {best_score:.4f}")
            if best_score > config.firewall_threshold:
                logger.warning(
                    f"Firewall Block | L2 Distance Exceeded | Score: {best_score:.4f} > "
                    f"Threshold: {config.firewall_threshold} | Standalone Query: '{standalone_query}'"
                )
                return SafetyPolicy.REFUSAL_PHRASE

        # 4. The Intent Classifier Firewall (On Standalone Query)
        try:
            intent = cast(QueryIntent, await self.intent_classifier.ainvoke(standalone_query))
            logger.debug(f"Intent Classification Metrics: {intent.model_dump()}")
            
            if not intent.is_prayer_related or not intent.has_medical_or_mobility_context:
                logger.warning(
                    f"Firewall Block | Intent Mismatch | Prayer: {intent.is_prayer_related} | "
                    f"Medical: {intent.has_medical_or_mobility_context} | Standalone Query: '{standalone_query}'"
                )
                return SafetyPolicy.REFUSAL_PHRASE
        except Exception as e:
            logger.error(f"Firewall System Failure | Intent classification raised exception: {e}", exc_info=True)
            return SafetyPolicy.REFUSAL_PHRASE

        # 5. RAG Execution
        logger.info(f"Execution Pipeline | Dispatching valid query to Heavy Synthesis Chain.")
        if logger.isEnabledFor(logging.DEBUG):
            docs = await self.retriever.ainvoke(standalone_query)
            for i, doc in enumerate(docs):
                logger.debug(f"Retrieved Chunk {i+1} Source: {doc.metadata.get('source')} | Preview: {doc.page_content[:100]}...")

        response = await self.rag_chain.ainvoke({"input": standalone_query})
        answer = (response.get("answer") or "").strip()

        # Continuation check for abruptly cut off outputs
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

        # Append medical notice
        physical_terms = ("knee", "back", "sujud", "ruku", "shoulder", "pain", "injury")
        if any(term in answer.lower() for term in physical_terms):
            if SafetyPolicy.MEDICAL_NOTICE.lower() not in answer.lower():
                answer = f"{answer} {SafetyPolicy.MEDICAL_NOTICE}"
                logger.debug("Output Guardrail | Appended standard medical safety notice to response payload.")

        logger.info("Request Cycle Complete | Successfully returned synchronized response.")
        return answer
