import os
import sys
import time
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

from datasets import load_dataset

# Ensure root directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, INITIAL_TOP_K, RERANKED_TOP_K
from pipeline.cleaner import DocumentCleaner
from pipeline.chunker import DocumentChunker
from vectorstore.embeddings import EmbeddingManager
from vectorstore.vector_db import VectorDatabase
from rag.reranker import HybridReranker
from rag.llm import GeminiLLM
from rag.chain import RAGChain
from evaluation.grounding_eval import DocumentGroundingEvaluator


def run_ragbench_evaluation(
    subset: str = "covidqa",
    max_samples: int = 20,
    output_dir: str = "ragbench eval score"
):
    """
    Evaluates the RAG pipeline on RAGBench dataset samples.
    Performs dynamic document indexing per sample (no hardcoded index).
    Saves evaluation scores to the specified output folder.
    """
    out_path = BASE_DIR / output_dir
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading RAGBench dataset '{subset}' (test split)...")
    ds = load_dataset("rungalileo/ragbench", subset, split="test")
    
    total_samples = min(max_samples, len(ds)) if max_samples else len(ds)
    print(f"Running evaluation on {total_samples} samples...")

    # Initialize shared models
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    embedding_mgr = EmbeddingManager(use_gemini=False)
    llm = GeminiLLM(api_key=api_key)
    reranker = HybridReranker()
    evaluator = DocumentGroundingEvaluator(llm=llm)

    chunker = DocumentChunker(chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)

    results = []
    total_grounding = 0.0
    total_faithfulness = 0.0
    total_numerical = 0.0
    passed_count = 0

    for i in range(total_samples):
        sample = ds[i]
        sample_id = sample.get("id", f"sample_{i+1}")
        question = sample.get("question", "")
        documents = sample.get("documents", [])
        
        # Ground truth RAGBench metrics if available
        gt_adherence = sample.get("adherence_score", None)
        gt_relevance = sample.get("relevance_score", None)
        gt_completeness = sample.get("completeness_score", None)

        print(f"\n[{i+1}/{total_samples}] Processing sample ID: {sample_id}", flush=True)
        print(f"Question: {question[:80]}...", flush=True)

        # 1. Dynamic Document Ingestion & Chunking (No hardcoded index)
        vdb = VectorDatabase(vector_dir=BASE_DIR / "data" / "temp_eval_vdb")
        vdb.clear()

        doc_chunks = []
        for doc_idx, raw_doc in enumerate(documents):
            cleaned_doc = DocumentCleaner.clean_document_text(raw_doc)
            parsed_mock = {
                "filename": f"ragbench_doc_{doc_idx+1}",
                "pages": [{"page_number": 1, "cleaned_text": cleaned_doc}]
            }
            chunks = chunker.chunk_parsed_document(parsed_mock)
            doc_chunks.extend(chunks)

        if doc_chunks:
            chunk_texts = [c["text"] for c in doc_chunks]
            vecs = embedding_mgr.embed_texts(chunk_texts)
            vdb.add_chunks(doc_chunks, vecs)

        # 2. RAG Execution
        rag_chain = RAGChain(
            vector_db=vdb,
            embedding_manager=embedding_mgr,
            llm=llm,
            reranker=reranker
        )
        rag_output = rag_chain.run(query=question, initial_top_k=INITIAL_TOP_K, rerank_top_k=RERANKED_TOP_K)

        # 3. Document Grounding Audit
        eval_output = evaluator.evaluate_grounding(
            query=question,
            answer=rag_output["answer"],
            retrieved_chunks=rag_output["reranked_chunks"]
        )

        is_passed = eval_output["is_passed"]
        if is_passed:
            passed_count += 1

        total_grounding += eval_output["overall_grounding_score"]
        total_faithfulness += eval_output["faithfulness_score"]
        total_numerical += eval_output["numerical_accuracy_score"]

        results.append({
            "sample_id": sample_id,
            "question": question,
            "num_source_docs": len(documents),
            "generated_answer": rag_output["answer"],
            "citations": "; ".join(rag_output["citations"]),
            "overall_grounding_score": eval_output["overall_grounding_score"],
            "faithfulness_score": eval_output["faithfulness_score"],
            "numerical_accuracy_score": eval_output["numerical_accuracy_score"],
            "is_passed": is_passed,
            "unsupported_numbers": "; ".join(eval_output["unsupported_numbers"]),
            "gt_adherence_score": gt_adherence,
            "gt_relevance_score": gt_relevance,
            "gt_completeness_score": gt_completeness
        })

        # Cleanup temp vector DB & rate-limit pause
        vdb.clear()
        time.sleep(5)

    # 4. Save Evaluation Results
    df_results = pd.DataFrame(results)
    csv_filename = out_path / f"ragbench_{subset}_eval_results.csv"
    df_results.to_csv(csv_filename, index=False, encoding="utf-8")

    summary = {
        "dataset_name": "rungalileo/ragbench",
        "subset": subset,
        "total_eval_samples": total_samples,
        "passed_samples": passed_count,
        "pass_rate_pct": round((passed_count / total_samples) * 100, 2) if total_samples > 0 else 0,
        "avg_overall_grounding_score": round(total_grounding / total_samples, 4) if total_samples > 0 else 0,
        "avg_faithfulness_score": round(total_faithfulness / total_samples, 4) if total_samples > 0 else 0,
        "avg_numerical_accuracy_score": round(total_numerical / total_samples, 4) if total_samples > 0 else 0
    }

    json_filename = out_path / f"ragbench_{subset}_summary.json"
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60, flush=True)
    print("RAGBench Evaluation Completed!", flush=True)
    print(f"Pass Rate: {summary['pass_rate_pct']}%")
    print(f"Average Grounding Score: {summary['avg_overall_grounding_score']}")
    print(f"Average Faithfulness Score: {summary['avg_faithfulness_score']}")
    print(f"Results saved to folder: '{output_dir}'")
    print(f"  - CSV: {csv_filename.resolve()}")
    print(f"  - JSON: {json_filename.resolve()}")
    print("="*60)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline on RAGBench dataset")
    parser.add_argument("--subset", type=str, default="covidqa", help="RAGBench dataset subset (e.g. covidqa, finqa, hotpotqa)")
    parser.add_argument("--max_samples", type=int, default=10, help="Maximum samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="ragbench eval score", help="Output directory name")

    args = parser.parse_args()
    
    from dotenv import load_dotenv
    load_dotenv()

    run_ragbench_evaluation(subset=args.subset, max_samples=args.max_samples, output_dir=args.output_dir)
