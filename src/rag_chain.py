import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List

from .document_manager import DocumentManager

load_dotenv()

class ResponseFormat(BaseModel):
    """
    Strict output schema enforced via Gemini's Structured Output API.
    
    Why Pydantic + with_structured_output()?
    ──────────────────────────────────────────
    Instead of asking the LLM to "return JSON" in the prompt (which fails ~20% of the
    time with complex answers), we use LangChain's native `with_structured_output()`.
    This leverages the provider's function-calling / constrained decoding API to
    GUARANTEE the output matches this schema on every single call.
    """
    answer: str = Field(description="The final answer to the user's question. If the information is not in the context or if you have to abstain, state that clearly.")
    sources: List[str] = Field(description="A list of source document IDs (e.g., kb_001) used to formulate the answer. Empty list if none used.")

def format_docs(docs):
    """
    Format retrieved documents into a structured context string for the LLM.
    
    KEY DESIGN DECISIONS:
    ─────────────────────
    1. METADATA INJECTION: We explicitly include Status, Last Updated, and Title
       so the LLM can reason about document freshness and trustworthiness.
    
    2. DEPRECATION WARNING: If a document's status is 'deprecated', we inject a
       prominent warning tag. This is BETTER than filtering deprecated docs out
       entirely because:
         - Users might ask "What was the OLD rate?" → we need the deprecated data
         - But we never want the LLM to cite deprecated data as current truth
         - The warning lets the LLM use its reasoning to decide contextually
    
    3. SEPARATOR: We use '---' between documents so the LLM clearly sees
       document boundaries and doesn't accidentally merge facts across sources.
    """
    formatted = []
    for doc in docs:
        status_note = f" [WARNING: THIS DOCUMENT IS DEPRECATED. DO NOT USE IT FOR CURRENT ANSWERS UNLESS EXPLICITLY RELEVANT.]" if doc.metadata.get('status') == 'deprecated' else ""
        text = f"Source ID: {doc.metadata['id']}\nTitle: {doc.metadata['title']}\nStatus: {doc.metadata['status']}{status_note}\nLast Updated: {doc.metadata['last_updated']}\nContent: {doc.page_content}\n---"
        formatted.append(text)
    return "\n".join(formatted)

def get_rag_chain():
    """
    Build the complete RAG chain using LangChain Expression Language (LCEL).
    
    PIPELINE ARCHITECTURE:
    ──────────────────────
    User Question
         │
         ├──→ [Retriever] → Top-5 documents → format_docs() → Context string
         │
         └──→ [Passthrough] → Original question text
                  │
                  ▼
         [ChatPromptTemplate] → System prompt + context + question
                  │
                  ▼
         [Gemini 2.5 Flash + Structured Output] → ResponseFormat(answer, sources)
    
    WHY K=5 FOR RETRIEVAL?
    ──────────────────────
    - Too few (k=1-2): Misses multi-hop queries that need 2+ articles
    - Too many (k=10+): Floods the context with noise, diluting signal
    - k=5 is the sweet spot: covers multi-hop while keeping context focused
    - With 24 total KB articles, k=5 retrieves ~20% of the corpus per query
    
    QUERY AUGMENTATION (Interview Talking Point):
    ──────────────────────────────────────────────
    We chose NOT to implement query augmentation (HyDE, query decomposition) here
    because our KB is small and well-structured. However, in production:
      - HyDE: Generate a hypothetical answer, embed THAT, and retrieve. Boosts
        recall for vague or short queries.
      - Query Decomposition: Break complex multi-hop questions into sub-queries,
        retrieve for each, then synthesize. We handle multi-hop via k=5 instead.
    
    RE-RANKING (Interview Talking Point):
    ─────────────────────────────────────
    We retrieve k=5 and pass all 5 to the LLM. In a larger KB, we would:
      1. Retrieve k=20 candidates with the fast bi-encoder (MiniLM)
      2. Re-rank with a cross-encoder (e.g., ms-marco-MiniLM-L-6-v2) to k=5
      3. Pass only the top-5 re-ranked documents to the LLM
    Cross-encoders are ~100x slower but dramatically more accurate for ranking.
    
    HYBRID SEARCH (Interview Talking Point):
    ────────────────────────────────────────
    Our current retrieval is dense-only (vector similarity). For production:
      - Add BM25 sparse retrieval for exact keyword matching (error codes, IDs)
      - Use Reciprocal Rank Fusion (RRF) to merge dense + sparse results
      - This is called "Hybrid Search" and catches queries that vectors miss
    """
    manager = DocumentManager()
    vector_store = manager.initialize_vector_store()
    
    # K=5 to ensure we fetch enough context for multi-hop queries
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Using Gemini 2.5 Flash for low latency and high accuracy
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

    system_prompt = """You are a highly accurate, production-grade technical support assistant for OmniComm, a telecom and cloud communications platform.
Your primary objective is to answer user questions based STRICTLY on the provided Knowledge Base context.

CRITICAL RULES:
1. NO HALLUCINATIONS: You must ONLY use the facts provided in the "Context" below. If the context does not contain enough information to answer the question, you MUST ABSTAIN. Say something like: "I don't have enough information to answer this based on the current knowledge base."
2. MULTI-HOP REASONING: Some questions require combining facts from multiple context documents. Connect the dots carefully to provide a comprehensive answer.
3. DEPRECATED CONTENT: Pay close attention to the `Status` and `Last Updated` metadata of each document. If a document is `deprecated`, you MUST NOT use its information as current truth. You may reference it ONLY if the user is specifically asking about historical or old data, and you must clearly note it is deprecated.
4. POLICY AND RESTRICTIONS: If a user asks for workarounds to policies (e.g., calling emergency numbers), firmly state the policy and refuse to suggest workarounds.
5. CITED SOURCES: You must return the EXACT Source IDs of the documents you used to generate the answer in the `sources` array.
6. IN-LINE CITATIONS: Your `answer` string MUST include inline citations referring to the original source ID whenever you state a fact, like this: [kb_001]. Every sentence that uses information from the KB must have a citation at the end.

Context:
{context}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    structured_llm = llm.with_structured_output(ResponseFormat)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | structured_llm
    )

    return chain
