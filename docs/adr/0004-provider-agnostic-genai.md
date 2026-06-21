# 4. Provider-agnostic GenAI layer

**Status:** Accepted

**Context.** The project should "use GenAI like Claude" and be future-proof, while staying at
£0. The Claude API is metered (not free); model names and providers change frequently.

**Decision.** Build a thin **provider-agnostic** LLM client (`src/mmi/ai/llm.py`) with a single
`complete()` interface and a swappable backend chosen by `LLM_PROVIDER` (gemini | groq | claude).
Default to a **free** provider (Gemini/Groq). If no key is set, callers fall back to a
deterministic template so the feature always works.

**Consequences.**
- Zero marginal cost by default; Claude is a one-line switch when desired.
- Resilient to model churn — only a constant/env changes.
- Slightly more code than calling one SDK directly, but far more flexible and honest about cost.
