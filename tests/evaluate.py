import sys
import os
import json

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from src.rag_chain import get_rag_chain
from src.document_manager import DocumentManager
import warnings
warnings.filterwarnings("ignore")

def load_evaluation_queries(path="data/evaluation_queries.json"):
    with open(path, "r") as f:
        return json.load(f)

def run_evaluation():
    queries = load_evaluation_queries()
    rag_chain = get_rag_chain()
    
    data = {"question": [], "answer": [], "contexts": []}
    
    print("Generating answers for evaluation queries...")
    manager = DocumentManager()
    vector_store = manager.initialize_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    for i, q in enumerate(queries):
        question_text = q["question"]
        print(f"[{i+1}/{len(queries)}] Processing: {question_text}")
        
        try:
            docs = retriever.invoke(question_text)
            contexts = [doc.page_content for doc in docs]
            
            result = rag_chain.invoke(question_text)
            result_dict = result.model_dump() if hasattr(result, 'model_dump') else result
            answer = result_dict.get("answer", "")
            
            data["question"].append(question_text)
            data["answer"].append(answer)
            data["contexts"].append(contexts)
        except Exception as e:
            print(f"Error processing question '{question_text}': {e}")
            continue
            
    dataset = Dataset.from_dict(data)
    
    print("\nRunning Ragas evaluation...")
    
    # Configure Ragas to use Gemini 2.5 Flash for evaluation
    evaluator_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model="gemini-2.5-flash"))
    evaluator_embeddings = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"))
    
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )
    
    df = result.to_pandas()
    df.to_csv("evaluation_results.csv", index=False)
    
    print("\n=== Evaluation Results ===")
    print(result)
    print("Detailed results saved to evaluation_results.csv")

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    run_evaluation()
