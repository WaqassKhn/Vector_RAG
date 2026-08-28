"""
rag/error_handler.py
────────────────────
Centralized Diagnostic & Actionable Error Handling System.

Transforms raw Python exceptions, HTTP errors, API errors, and database locks
into structured, human-readable diagnostics with:
  - Error Title & Category
  - What Happened (Plain English summary)
  - Root Cause Analysis (Technical diagnosis)
  - Actionable Remediation Steps (Step-by-step instructions to fix)
  - Raw Traceback for developer inspection
"""

import sys
import traceback
from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class DiagnosticError:
    category: str
    title: str
    message: str
    root_cause: str
    remedy_steps: list[str]
    raw_error: str = ""
    severity: str = "error"  # 'error', 'warning', 'critical'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "title": self.title,
            "message": self.message,
            "root_cause": self.root_cause,
            "remedy_steps": self.remedy_steps,
            "raw_error": self.raw_error,
            "severity": self.severity,
        }

    def format_markdown(self) -> str:
        badge = "[ERROR]" if self.severity == "error" else ("[WARNING]" if self.severity == "warning" else "[CRITICAL]")
        steps_md = "\n".join(f"- {step}" for step in self.remedy_steps)
        
        md = f"""### {badge} {self.title} (`{self.category}`)

**What Happened:**
{self.message}

**Root Cause Analysis:**
> {self.root_cause}

**Recommended Action Steps:**
{steps_md}
"""
        if self.raw_error:
            md += f"""
<details>
<summary><b>Technical Details & Raw Traceback</b></summary>

```python
{self.raw_error}
```
</details>
"""
        return md


class ErrorDiagnosticManager:
    """
    Analyzes any exception or error condition across LLMs, Vector DBs,
    Document Parsers, and SQLite to produce an actionable DiagnosticError.
    """

    @staticmethod
    def diagnose(exc: Exception, context: Optional[str] = None) -> DiagnosticError:
        exc_str = str(exc)
        exc_type = type(exc).__name__
        raw_tb = traceback.format_exc()

        # ── 1. OpenRouter / LLM Authentication & Keys ──
        if "401" in exc_str or "Unauthorized" in exc_str or "Invalid API Key" in exc_str or "api_key" in exc_str.lower() and "missing" in exc_str.lower():
            return DiagnosticError(
                category="LLM_AUTHENTICATION_ERROR",
                title="Invalid or Missing LLM API Key",
                message="The language model request was rejected due to missing or invalid authentication credentials.",
                root_cause="Your `OPENROUTER_API_KEY` or `GEMINI_API_KEY` in `.env` is either blank, expired, or incorrect.",
                remedy_steps=[
                    "Open your local `.env` file in the project root.",
                    "Verify `OPENROUTER_API_KEY=sk-or-v1-...` is present and active (obtain a free key from https://openrouter.ai/keys).",
                    "If using Gemini fallback, verify `GEMINI_API_KEY=...` from https://aistudio.google.com/.",
                    "Restart the application to reload the environment variables.",
                ],
                raw_error=raw_tb,
            )

        # ── 2. OpenRouter Rate Limit (429) ──
        if "429" in exc_str or "rate limit" in exc_str.lower() or "too many requests" in exc_str.lower():
            return DiagnosticError(
                category="LLM_RATE_LIMIT_ERROR",
                title="LLM Provider Rate Limit Exceeded (HTTP 429)",
                message="The free-tier model provider has temporarily throttled incoming requests due to traffic volume.",
                root_cause="OpenRouter's `:free` models enforce per-minute and daily burst request limits across shared public slots.",
                remedy_steps=[
                    "Wait 10-20 seconds for the burst window to reset and re-submit your query.",
                    "Ensure your `GEMINI_API_KEY` is configured in `.env` so the automatic fallback can seamlessly take over.",
                    "In Tab 3 (Settings), verify the active model routing priorities.",
                ],
                raw_error=raw_tb,
            )

        # ── 3. Connection / Network Timeout ──
        if "timeout" in exc_str.lower() or "connecterror" in exc_str.lower() or "connection error" in exc_str.lower() or "unreachable" in exc_str.lower():
            return DiagnosticError(
                category="NETWORK_CONNECTION_TIMEOUT",
                title="Network Connection Failed or Timed Out",
                message="The system could not establish a connection with the remote cloud API server (Pinecone or OpenRouter).",
                root_cause="An active firewall, proxy, VPN, or intermittent internet connectivity issue interrupted the SSL/HTTP handshake.",
                remedy_steps=[
                    "Check your internet connection and verify you can access https://openrouter.ai and https://pinecone.io.",
                    "If you are behind a corporate VPN or proxy, ensure outbound HTTPS traffic on port 443 is permitted.",
                    "Retry the request — transient packet drops will auto-recover.",
                ],
                raw_error=raw_tb,
            )

        # ── 4. Pinecone Dimension Mismatch ──
        if "dimension" in exc_str.lower() and ("mismatch" in exc_str.lower() or "expected" in exc_str.lower()):
            return DiagnosticError(
                category="PINECONE_DIMENSION_MISMATCH",
                title="Vector Database Dimension Mismatch",
                message="The query vector dimension does not match the configured Pinecone index dimension.",
                root_cause="The local embedding model produces 384-dimensional vectors (`all-MiniLM-L6-v2`), but your Pinecone index was created with a different dimension (e.g. 1536 or 768).",
                remedy_steps=[
                    "Log in to https://app.pinecone.io.",
                    "Create or recreate a Serverless index named `rag-ntpc` with **Dimensions: 384** and **Metric: cosine**.",
                    "Update `PINECONE_INDEX_NAME` in `.env` if using a different index name.",
                ],
                raw_error=raw_tb,
            )

        # ── 5. Pinecone Index Not Found or Key Invalid ──
        if "pinecone" in exc_str.lower() and ("not found" in exc_str.lower() or "404" in exc_str or "forbidden" in exc_str.lower()):
            return DiagnosticError(
                category="PINECONE_INDEX_ERROR",
                title="Pinecone Index Not Found or Unauthorized",
                message="The system was unable to locate or authenticate against the configured Pinecone index.",
                root_cause="`PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, or `PINECONE_REGION` in your `.env` does not match your Pinecone console settings.",
                remedy_steps=[
                    "Verify your `PINECONE_API_KEY` from https://app.pinecone.io/keys.",
                    "Verify that an index with `PINECONE_INDEX_NAME` exists under region `PINECONE_REGION` (e.g. `us-east-1`).",
                    "If using purely local development without Pinecone, the app will automatically operate using local SQLite metadata caches.",
                ],
                raw_error=raw_tb,
            )

        # ── 6. SQLite Database Locked / Busy ──
        if "database is locked" in exc_str.lower() or "operationalerror: database is locked" in exc_str.lower():
            return DiagnosticError(
                category="DATABASE_LOCK_ERROR",
                title="SQLite Database File is Locked",
                message="Another process or background thread is holding an exclusive write lock on `data/rag_app.db`.",
                root_cause="SQLite allows concurrent readers in WAL mode, but concurrent writes from multiple threads require brief sequential queuing.",
                remedy_steps=[
                    "Wait 2-3 seconds as the built-in WAL busy handler automatically retries the transaction.",
                    "Ensure no other external SQLite browser (e.g. DB Browser for SQLite) has an uncommitted write transaction open on `data/rag_app.db`.",
                ],
                raw_error=raw_tb,
            )

        # ── 7. Document Parser Corrupt or Password-Protected File ──
        if "password" in exc_str.lower() or "encrypted" in exc_str.lower() or "pdf" in exc_str.lower() and ("corrupt" in exc_str.lower() or "syntaxerror" in exc_str.lower()):
            return DiagnosticError(
                category="DOCUMENT_PARSING_ERROR",
                title="Encrypted, Password-Protected, or Corrupted File",
                message="The uploaded document could not be decoded or extracted into text chunks.",
                root_cause="The document is either protected with a password/encryption, has a malformed PDF header, or is an unsupported binary format.",
                remedy_steps=[
                    "Ensure the PDF/Word document is not password-protected or restricted by DRM.",
                    "Re-save or export the document as a standard PDF or text file before uploading.",
                    "Supported formats: PDF, DOCX, CSV, XLSX, TXT.",
                ],
                raw_error=raw_tb,
            )

        # ── 8. Empty Document Text (Scanned images without OCR) ──
        if "empty" in exc_str.lower() or "0 characters" in exc_str.lower() or "no text extracted" in exc_str.lower():
            return DiagnosticError(
                category="DOCUMENT_EMPTY_TEXT_WARNING",
                title="Zero Extractable Text in Document",
                message="The uploaded file parsed successfully, but contained zero machine-readable text characters.",
                root_cause="The file is likely a scanned image PDF (photocopy) without an embedded OCR text layer.",
                remedy_steps=[
                    "Run an OCR tool (e.g. Adobe Acrobat OCR, Apple Preview text scan, or Tesseract) on the PDF to embed a readable text layer.",
                    "Re-upload the searchable PDF or copy/paste the contents into a `.txt` file.",
                ],
                raw_error=raw_tb,
                severity="warning",
            )

        # ── 9. Generic Fallback ──
        return DiagnosticError(
            category=f"INTERNAL_{exc_type.upper()}",
            title=f"Execution Error: {exc_type}",
            message=f"An unexpected error occurred during {context or 'pipeline execution'}: {exc_str}",
            root_cause="An unhandled exception was intercepted by the diagnostic manager.",
            remedy_steps=[
                "Review the raw traceback details below for the exact offending line.",
                "Check that all environment variables in `.env` are set correctly.",
                "Verify file permissions in `./data` and `./vectorstore` directories.",
            ],
            raw_error=raw_tb,
        )
