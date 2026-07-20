import re
import json
from typing import List, Dict, Any, Tuple
from rag.llm import GeminiLLM
from config import GROUNDING_PASS_THRESHOLD

class DocumentGroundingEvaluator:
    """
    Evaluates RAG generation outputs for Document Grounding, Faithfulness,
    and Numerical Accuracy against retrieved context chunks.
    """

    CLAIM_EVAL_PROMPT = """You are an impartial, strict LLM evaluator auditing RAG responses for Grounding and Faithfulness.

CONTEXT CHUNKS:
{context}

GENERATED ANSWER:
{answer}

Task:
1. Break down the GENERATED ANSWER into individual claims or factual statements.
2. For each claim, evaluate if it is FULLY SUPPORTED by the CONTEXT CHUNKS.
3. Output a JSON object matching this schema:
{{
  "claims": [
    {{
      "claim": "string",
      "is_grounded": true/false,
      "explanation": "string explaining why or identifying missing evidence"
    }}
  ],
  "faithfulness_score": float (0.0 to 1.0, fraction of grounded claims),
  "summary": "brief summary of grounding assessment"
}}

Output ONLY valid JSON:"""

    def __init__(self, llm: GeminiLLM = None):
        self.llm = llm or GeminiLLM()

    def evaluate_grounding(
        self,
        query: str,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Runs comprehensive grounding evaluation returning claim checks, numerical accuracy, and faithfulness scores.
        """
        combined_context = "\n\n".join([f"[{c.get('filename','doc')} Page {c.get('page_number',1)}]\n{c.get('text','')}" for c in retrieved_chunks])
        
        # 1. Numerical Accuracy Check
        num_check = self._audit_numerical_accuracy(answer, combined_context)
        
        # 2. Claim-level Faithfulness via Gemini LLM Judge (or fallback)
        if self.llm and self.llm.is_available() and answer and not answer.startswith("[LLM Offline"):
            llm_eval = self._evaluate_claims_with_llm(answer, combined_context)
        else:
            llm_eval = self._rule_based_claim_eval(answer, combined_context)

        faithfulness_score = llm_eval.get("faithfulness_score", 0.0)
        numerical_score = num_check["numerical_accuracy_score"]
        
        # Combined overall grounding score
        overall_grounding_score = round(0.6 * faithfulness_score + 0.4 * numerical_score, 3)
        is_passed = overall_grounding_score >= GROUNDING_PASS_THRESHOLD

        return {
            "query": query,
            "overall_grounding_score": overall_grounding_score,
            "is_passed": is_passed,
            "faithfulness_score": round(faithfulness_score, 3),
            "numerical_accuracy_score": round(numerical_score, 3),
            "claims": llm_eval.get("claims", []),
            "unsupported_numbers": num_check["unsupported_numbers"],
            "supported_numbers": num_check["supported_numbers"],
            "summary": llm_eval.get("summary", "Grounding evaluation complete.")
        }

    def _audit_numerical_accuracy(self, answer: str, context: str) -> Dict[str, Any]:
        """
        Extracts numbers, percentages, currency figures from answer and verifies presence in context.
        """
        # Match figures like $45.2M, 12.5%, 2024, 100,000, 45.2
        pattern = r'\b\$?\d+(?:,\d{3})*(?:\.\d+)?%?(?:[M|B|k|K])?\b'
        
        answer_nums = set(re.findall(pattern, answer))
        context_nums = set(re.findall(pattern, context))

        # Filter out trivial page numbers or single digits under 5 unless decimal
        meaningful_answer_nums = {n for n in answer_nums if len(n) > 1 or '.' in n}
        
        if not meaningful_answer_nums:
            return {
                "numerical_accuracy_score": 1.0,
                "supported_numbers": [],
                "unsupported_numbers": []
            }

        supported = []
        unsupported = []

        for num in meaningful_answer_nums:
            # Clean num for soft matching (remove $ or %)
            clean_num = num.strip("$%,").lower()
            # Check direct match or normalized float match
            matched = False
            for c_num in context_nums:
                clean_c = c_num.strip("$%,").lower()
                if clean_num == clean_c:
                    matched = True
                    break
                try:
                    if abs(float(clean_num) - float(clean_c)) < 1e-4:
                        matched = True
                        break
                except ValueError:
                    pass

            if matched:
                supported.append(num)
            else:
                unsupported.append(num)

        score = len(supported) / len(meaningful_answer_nums) if meaningful_answer_nums else 1.0

        return {
            "numerical_accuracy_score": score,
            "supported_numbers": supported,
            "unsupported_numbers": unsupported
        }

    def _evaluate_claims_with_llm(self, answer: str, context: str) -> Dict[str, Any]:
        """Queries Gemini LLM to audit sentence-level claims."""
        prompt = self.CLAIM_EVAL_PROMPT.format(context=context, answer=answer)
        try:
            res_text = self.llm.generate(prompt=prompt, temperature=0.0)
            # Find JSON block
            json_match = re.search(r'\{.*\}', res_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return data
        except Exception:
            pass
        return self._rule_based_claim_eval(answer, context)

    def _rule_based_claim_eval(self, answer: str, context: str) -> Dict[str, Any]:
        """Fallback rule-based claim evaluator using n-gram containment."""
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if s.strip()]
        if not sentences:
            return {"claims": [], "faithfulness_score": 1.0, "summary": "No claims detected."}

        context_lower = context.lower()
        claims = []
        grounded_count = 0

        for s in sentences:
            s_words = [w.lower() for w in re.findall(r'\b\w+\b', s) if len(w) > 3]
            if not s_words:
                claims.append({"claim": s, "is_grounded": True, "explanation": "Trivial claim"})
                grounded_count += 1
                continue
            
            matched_words = [w for w in s_words if w in context_lower]
            ratio = len(matched_words) / len(s_words)
            is_grounded = ratio >= 0.6
            if is_grounded:
                grounded_count += 1
            
            claims.append({
                "claim": s,
                "is_grounded": is_grounded,
                "explanation": f"Keyword match ratio: {round(ratio, 2)}"
            })

        faithfulness = grounded_count / len(sentences)
        return {
            "claims": claims,
            "faithfulness_score": faithfulness,
            "summary": f"{grounded_count}/{len(sentences)} claims verified via fallback matcher."
        }
