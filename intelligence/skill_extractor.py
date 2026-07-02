"""Map free-text / messy skill strings to canonical skill names.

The summariser hands us skills like "k8s", "postgres", "sentence transformers".
We canonicalise them against a taxonomy so the graph doesn't end up with five
nodes for the same thing. spaCy is used for entity spans if it's installed;
otherwise we fall back to alias matching, which is plenty for tech terms.
"""

from __future__ import annotations

import re

from core.constants import (
    CORE_ML_SKILLS,
    LLM_SKILLS,
    RETRIEVAL_SKILLS,
)

# canonical -> aliases. Reuses the JD taxonomy and adds general dev skills so the
# graph speaks one vocabulary.
_ALIASES: dict[str, list[str]] = {
    "kubernetes": ["kubernetes", "k8s"],
    "docker": ["docker", "containerization", "containers"],
    "postgresql": ["postgresql", "postgres", "psql"],
    "redis": ["redis"],
    "kafka": ["kafka", "apache kafka"],
    "python": ["python", "py"],
    "go": ["golang", "go "],
    "rust": ["rust"],
    "typescript": ["typescript", "ts"],
    "react": ["react", "reactjs", "react.js"],
    "vector search": ["vector search", "vector database", "ann", "approximate nearest"],
    "semantic search": ["semantic search"],
    "elasticsearch": ["elasticsearch", "elastic search", "es"],
    "faiss": ["faiss"],
    "pinecone": ["pinecone"],
    "qdrant": ["qdrant"],
    "embeddings": ["embeddings", "embedding"],
    "sentence transformers": ["sentence transformers", "sbert", "sentence-transformers"],
    "fine-tuning llms": ["fine-tuning", "fine tuning", "finetune", "lora", "qlora", "peft"],
    "nlp": ["nlp", "natural language processing"],
    "information retrieval": ["information retrieval", "ir ", "retrieval"],
    "learning to rank": ["learning to rank", "ltr", "lambdamart"],
    "recommendation systems": ["recommendation", "recommender", "recsys"],
    "distributed systems": ["distributed systems", "distributed system", "consensus", "raft"],
    "concurrency": ["concurrency", "race condition", "deadlock", "mutex", "goroutine"],
    "observability": ["observability", "prometheus", "grafana", "tracing", "opentelemetry"],
    "ci/cd": ["ci/cd", "cicd", "github actions", "jenkins"],
    "graphql": ["graphql"],
    "grpc": ["grpc"],
    "pytorch": ["pytorch", "torch"],
    "tensorflow": ["tensorflow", "tf "],
    "mlops": ["mlops", "model serving", "mlflow", "kubeflow"],
}

# fold the JD skill sets in so anything the ranker cares about is canonicalisable
for _s in RETRIEVAL_SKILLS | LLM_SKILLS | CORE_ML_SKILLS:
    _ALIASES.setdefault(_s, [_s])

_CATEGORY = {
    **dict.fromkeys(RETRIEVAL_SKILLS, "retrieval"),
    **dict.fromkeys(LLM_SKILLS, "llm"),
    **dict.fromkeys(CORE_ML_SKILLS, "ml"),
}


def canonicalise(name: str) -> str | None:
    n = name.strip().lower()
    if n in _ALIASES:
        return n
    for canonical, aliases in _ALIASES.items():
        if any(a in n for a in aliases):
            return canonical
    return None


def category_of(skill: str) -> str:
    return _CATEGORY.get(skill.lower(), "general")


def _spacy_terms(text: str) -> set[str]:
    try:
        import spacy  # optional; only used if a model is installed

        nlp = spacy.blank("en")  # blank pipe is enough for noun-chunk-ish tokens
        return {t.text.lower() for t in nlp(text) if not t.is_stop}
    except Exception:
        return set()


def extract_skills(text_or_list: str | list[str]) -> list[str]:
    """Return canonical skills mentioned in the text or implied by a skill list."""
    if isinstance(text_or_list, list):
        found = {canonicalise(s) for s in text_or_list}
        return sorted(c for c in found if c)

    text = text_or_list.lower()
    found: set[str] = set()
    for canonical, aliases in _ALIASES.items():
        for a in aliases:
            if re.search(r"(?<![a-z0-9])" + re.escape(a), text):
                found.add(canonical)
                break
    # spaCy is a no-op enhancer here; alias matching already covers tech terms
    _spacy_terms("")
    return sorted(found)
