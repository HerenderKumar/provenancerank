"""Single source of truth for domain vocabularies and scoring constants.

Every other module imports skill taxonomies, consulting-firm names and weight
tables from here. Nothing in this list is duplicated elsewhere in the codebase
(enforced by ``tests/test_no_duplicate_constants`` intent).

The taxonomies were calibrated against the *actual* Redrob ``candidates.jsonl``
skill vocabulary (e.g. "Vector Search", "Qdrant", "Learning to Rank",
"Fine-tuning LLMs", "Sentence Transformers") rather than guessed, so JD matching
and honeypot detection operate on strings that really occur in the data.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Skill taxonomy (all lower-cased for case-insensitive matching)
# ---------------------------------------------------------------------------

# The decisive JD signal: retrieval / ranking / vector-search / IR experience.
RETRIEVAL_SKILLS: frozenset[str] = frozenset(
    {
        "vector search",
        "semantic search",
        "hybrid search",
        "faiss",
        "pinecone",
        "qdrant",
        "weaviate",
        "milvus",
        "pgvector",
        "elasticsearch",
        "opensearch",
        "bm25",
        "haystack",
        "information retrieval",
        "retrieval",
        "recommendation systems",
        "recommendation",
        "learning to rank",
        "ltr",
        "ranking",
        "search relevance",
        "embeddings",
        "sentence transformers",
    }
)

# Embeddings + LLM tooling (relevant, but weaker than retrieval on its own).
LLM_SKILLS: frozenset[str] = frozenset(
    {
        "llms",
        "large language models",
        "fine-tuning llms",
        "fine-tuning",
        "lora",
        "qlora",
        "peft",
        "rag",
        "hugging face transformers",
        "transformers",
        "prompt engineering",
        "langchain",
        "llamaindex",
    }
)

# Core ML / NLP foundations the JD wants ("pre-LLM-era ML production").
CORE_ML_SKILLS: frozenset[str] = frozenset(
    {
        "nlp",
        "natural language processing",
        "machine learning",
        "deep learning",
        "pytorch",
        "tensorflow",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "feature engineering",
        "mlops",
        "mlflow",
        "kubeflow",
        "statistical modeling",
        "data science",
        "model serving",
        "bentoml",
        "weights & biases",
    }
)

# Canonical JD-relevant skill set (union of the three above + python).
JD_CORE_SKILLS: frozenset[str] = RETRIEVAL_SKILLS | LLM_SKILLS | CORE_ML_SKILLS | {"python"}

# Computer-vision / speech / robotics. The JD disqualifies candidates whose
# *primary* expertise is here WITHOUT NLP/IR exposure, so these never count as
# positive evidence and contribute to the "wrong specialisation" penalty.
CV_SPEECH_ROBOTICS_SKILLS: frozenset[str] = frozenset(
    {
        "computer vision",
        "opencv",
        "cnn",
        "object detection",
        "yolo",
        "image classification",
        "image segmentation",
        "diffusion models",
        "gans",
        "asr",
        "tts",
        "speech recognition",
        "speech synthesis",
        "robotics",
        "slam",
        "point cloud",
        "optical flow",
    }
)

# Skills whose presence is irrelevant-to-negative for this JD. Used to detect
# keyword stuffers (many AI skills next to a non-technical skill base).
NON_RELEVANT_SKILLS: frozenset[str] = frozenset(
    {
        "photoshop",
        "illustrator",
        "figma",
        "solidworks",
        "creo",
        "ansys",
        "autocad",
        "marketing",
        "seo",
        "excel",
        "powerpoint",
        "accounting",
        "tally",
        "sap",
        "hr",
        "human resources",
        "recruitment",
        "salesforce crm",
        "salesforce",
        "crm",
        "erp",
        "content writing",
        "copywriting",
        "sales",
        "six sigma",
        "customer support",
        "project management",
    }
)

# "Buzzword-lite" skills that title-chasers / framework enthusiasts attach to a
# profile without deeper systems experience. Down-weighted relative to the
# operational retrieval skills above.
BUZZWORD_LITE_SKILLS: frozenset[str] = frozenset(
    {
        "langchain",
        "prompt engineering",
        "llms",
        "rag",
    }
)

PROFICIENCY_WEIGHTS: dict[str, float] = {
    "beginner": 0.25,
    "intermediate": 0.50,
    "advanced": 0.80,
    "expert": 1.00,
}

# ---------------------------------------------------------------------------
# Company / career vocabularies
# ---------------------------------------------------------------------------

# Pure consulting/services firms (lower-cased substrings, matched on company).
CONSULTING_FIRMS: frozenset[str] = frozenset(
    {
        "tcs",
        "tata consultancy",
        "infosys",
        "wipro",
        "accenture",
        "cognizant",
        "capgemini",
        "hcl",
        "tech mahindra",
        "mphasis",
        "ltimindtree",
        "mindtree",
        "l&t infotech",
        "lti",
        "igate",
        "syntel",
        "hexaware",
        "birlasoft",
        "cybage",
        "persistent systems",
        "ust global",
        "ust",
        "nttdata",
        "ntt data",
        "dxc technology",
        "dxc",
        "atos",
        "ibm services",
    }
)

# Title tokens that indicate a genuinely technical / ML role (positive signal).
TECHNICAL_TITLE_TOKENS: tuple[str, ...] = (
    "engineer",
    "developer",
    "scientist",
    "ml ",
    "ai ",
    "machine learning",
    "data",
    "programmer",
    "architect",
    "sde",
    "researcher",
    "research",
    "applied scientist",
    "nlp",
    "devops",
    "sre",
    "backend",
    "platform",
)

# Title tokens that indicate a non-technical role (negative for AI-skill spam).
NON_TECHNICAL_TITLE_TOKENS: tuple[str, ...] = (
    "marketing",
    "hr",
    "human resource",
    "sales",
    "account",
    "graphic",
    "content writer",
    "content writing",
    "copywriter",
    "operations",
    "recruit",
    "talent",
    "customer support",
    "business development",
    "civil engineer",
    "mechanical engineer",
    "electrical engineer",
    "designer",
    "project manager",
    "program manager",
)

# Description tokens that evidence real production ML/IR work.
PRODUCTION_KEYWORDS: tuple[str, ...] = (
    "production",
    "deployed",
    "deploy",
    "users",
    "scale",
    "scaled",
    "latency",
    "pipeline",
    "real-time",
    "realtime",
    "a/b",
    "experiment",
    "metric",
    "ndcg",
    "mrr",
    "recall",
    "precision",
    "retrieval",
    "ranking",
    "rank",
    "embedding",
    "vector",
    "fine-tun",
    "recommendation",
    "recommender",
    "search",
    "throughput",
    "sla",
    "p99",
    "inference",
    "serving",
    "index",
    "relevance",
    "click-through",
    "ctr",
)

# ---------------------------------------------------------------------------
# Location (JD: Pune/Noida hybrid, India-only, Tier-1 metros)
# ---------------------------------------------------------------------------

PREFERRED_LOCATIONS: frozenset[str] = frozenset(
    {
        "pune",
        "noida",
        "delhi",
        "new delhi",
        "gurgaon",
        "gurugram",
        "ncr",
        "hyderabad",
        "mumbai",
        "bangalore",
        "bengaluru",
    }
)
INDIA_TOKENS: frozenset[str] = frozenset({"india"})

# ---------------------------------------------------------------------------
# Feature matrix column order - the contract between feature engineering and ML.
# Keeping this explicit makes the materialised view reproducible and the model
# input deterministic across runs.
# ---------------------------------------------------------------------------

NUMERIC_FEATURE_COLUMNS: tuple[str, ...] = (
    # career
    "role_count",
    "total_experience_years",
    "avg_tenure_months",
    "scope_progression_score",
    "has_recent_career_gap",
    "product_company_ratio",
    "all_consulting_career",
    "last_role_start_recency_months",
    "avg_title_tenure_months",
    "title_chaser_flag",
    "avg_description_words",
    "has_production_keywords",
    # skill
    "jd_skill_match_count",
    "jd_skill_match_ratio",
    "retrieval_skill_count",
    "llm_skill_count",
    "cv_speech_dominance",
    "non_relevant_skill_count",
    "skill_trust_score",
    "keyword_stuffer_flag",
    "assessment_verified_skills",
    "assessment_mean_score",
    "top_assessment_score",
    "max_jd_skill_duration_months",
    "recent_jd_skill_flag",
    # jd-fit
    "jd_fit_score",
    "location_fit",
    "wrong_specialisation_penalty",
    # signal
    "days_since_active",
    "recency_score",
    "availability_score",
    "engagement_score",
    "trust_score",
    "github_activity_score",
    "github_linked",
    "notice_period_days",
    "open_to_work",
    "offer_acceptance_rate",
    "behavioural_composite",
    # honeypot
    "is_honeypot",
)

# Default final-merge weights (mirrors config; duplicated here only as the
# documented canonical default referenced by docs/tests).
DEFAULT_MERGE_WEIGHTS = {"ml": 0.40, "retrieval": 0.35, "signal": 0.25}
