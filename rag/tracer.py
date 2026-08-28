"""
rag/tracer.py
─────────────
Comprehensive tracing and telemetry system for LLMs, Agents, Retrieval, and Memory.

Captures granular step-by-step execution traces:
  - Query planning & sub-query decomposition
  - Dense + Sparse retrieval & BM25/RRF reranking
  - Cognitive memory tier activations (Working, Episodic, Semantic, Procedural)
  - LLM model routing, prompts, completion tokens, latency, and fallback attempts
  - Merge agent synthesis & claim verification
"""

import time
import uuid
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class TraceSpan:
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    component: str = ""  # 'query_planner', 'retrieval', 'reranker', 'memory', 'llm', 'merge_agent', 'eval'
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: float = 0.0
    status: str = "running"  # 'running', 'success', 'warning', 'error'
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def finish(self, status: str = "success", error: Optional[str] = None):
        self.end_time = time.time()
        self.duration_ms = round((self.end_time - self.start_time) * 1000, 2)
        self.status = status
        if error:
            self.error = error
            self.status = "error"


class ExecutionTracer:
    """
    Session and query-level execution tracer.
    Thread-safe context collector for tracking pipeline stages and troubleshooting bottlenecks.
    """

    def __init__(self, query_id: Optional[str] = None):
        self.query_id = query_id or str(uuid.uuid4())[:8]
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.total_duration_ms: float = 0.0
        self.spans: List[TraceSpan] = []

    def start_span(
        self,
        name: str,
        component: str,
        inputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TraceSpan:
        span = TraceSpan(
            name=name,
            component=component,
            inputs=inputs or {},
            metadata=metadata or {},
        )
        self.spans.append(span)
        return span

    def finish_span(
        self,
        span: TraceSpan,
        outputs: Optional[Dict[str, Any]] = None,
        status: str = "success",
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if outputs:
            span.outputs = outputs
        if metadata:
            span.metadata.update(metadata)
        span.finish(status=status, error=error)

    def finish(self):
        self.end_time = time.time()
        self.total_duration_ms = round((self.end_time - self.start_time) * 1000, 2)

    def get_summary(self) -> Dict[str, Any]:
        """Returns a high-level summary of all executed spans."""
        llm_calls = [s for s in self.spans if s.component == "llm"]
        total_llm_time = sum(s.duration_ms for s in llm_calls)
        total_tokens = sum(
            s.metadata.get("prompt_tokens", 0) + s.metadata.get("completion_tokens", 0)
            for s in llm_calls
        )

        return {
            "query_id": self.query_id,
            "total_duration_ms": self.total_duration_ms,
            "span_count": len(self.spans),
            "llm_calls_count": len(llm_calls),
            "total_llm_time_ms": round(total_llm_time, 2),
            "total_tokens_estimated": total_tokens,
            "models_used": list(set(s.metadata.get("model", "unknown") for s in llm_calls if "model" in s.metadata)),
            "errors": [s.error for s in self.spans if s.error],
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration_ms": self.total_duration_ms,
            "summary": self.get_summary(),
            "spans": [asdict(s) for s in self.spans],
        }
