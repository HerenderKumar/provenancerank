// Typed API client. One place that knows the backend contract; pages call these
// functions and never touch fetch directly.

const BASE = import.meta.env.VITE_API_BASE || "/api";

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}
export interface User {
  id: string;
  email: string;
  role: "recruiter" | "admin";
  created_at: string;
}
export interface ApiKey {
  id: string;
  name: string;
  last4: string;
  scopes: string;
  is_active: boolean;
  created_at: string;
  last_used_at: string | null;
}
export interface RankRow {
  candidate_id: string;
  rank: number;
  score: number;
  reasoning: string;
}
export interface JobStatus {
  job_id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  source: string | null;
  cache_hit: boolean;
  candidates_count: number;
  top_k: number;
  elapsed_ms: number | null;
  gate_stats: Record<string, number>;
  coverage: Record<string, number>;
  error: string | null;
  created_at: string | null;
  finished_at: string | null;
}
export interface ArtifactStatus {
  ready: boolean;
  candidates: number;
  has_embeddings: boolean;
  has_bm25: boolean;
  jd_source: string;
  embedding_backend: string | null;
  built_at: string | null;
}

let accessToken: string | null = localStorage.getItem("pr_token");
export function setToken(t: string | null) {
  accessToken = t;
  if (t) localStorage.setItem("pr_token", t);
  else localStorage.removeItem("pr_token");
}
export function getToken() {
  return accessToken;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
  ) {
    super(message);
  }
}

async function req<T>(path: string, opts: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(opts.headers as Record<string, string>),
  };
  if (accessToken) headers["Authorization"] = `Bearer ${accessToken}`;
  const res = await fetch(`${BASE}${path}`, { ...opts, headers });
  if (res.status === 204) return undefined as T;
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new ApiError(res.status, body.error || "error", body.message || res.statusText);
  }
  return body as T;
}

export const api = {
  // auth
  login: (email: string, password: string) =>
    req<TokenResponse>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),
  me: () => req<User>("/auth/me"),
  listApiKeys: () => req<ApiKey[]>("/auth/api-keys"),
  createApiKey: (name: string) =>
    req<{ id: string; name: string; api_key: string; last4: string }>("/auth/api-keys", {
      method: "POST",
      body: JSON.stringify({ name }),
    }),
  revokeApiKey: (id: string) => req<void>(`/auth/api-keys/${id}`, { method: "DELETE" }),

  // ranking
  submitRank: (payload: { jd_text?: string; top_k?: number }) =>
    req<{ job_id: string; status: string; stream_url: string; results_url: string }>("/rank", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  getJob: (id: string) => req<JobStatus>(`/rank/${id}`),
  getResults: (id: string) =>
    req<{ job_id: string; status: string; count: number; results: RankRow[] }>(
      `/rank/${id}/results`,
    ),
  listJobs: () => req<JobStatus[]>("/rank"),
  csvUrl: (id: string) => `${BASE}/rank/${id}/submission.csv`,

  // admin
  artifacts: () => req<ArtifactStatus>("/admin/artifacts"),
  triggerPrecompute: () => req<{ status: string }>("/admin/precompute", { method: "POST" }),

  // live ingestion layer
  connectGithub: (github_username: string, display_name?: string) =>
    req<{ developer_id: string; github_username: string; status: string }>(
      "/developers/connect-github",
      { method: "POST", body: JSON.stringify({ github_username, display_name }) },
    ),
  syncStatus: (id: string) => req<SyncStatus>(`/developers/${id}/sync-status`),
  evidenceGraph: (id: string) => req<EvidenceGraph>(`/developers/${id}/evidence-graph`),
  nlSearch: (query: string) =>
    req<NLSearchResult>("/search/natural-language", {
      method: "POST",
      body: JSON.stringify({ query }),
    }),
  bySkill: (skill: string, min_confidence = 0.6) =>
    req<{ skill: string; count: number; candidates: SearchCandidate[] }>(
      `/search/by-skill/${encodeURIComponent(skill)}?min_confidence=${min_confidence}`,
    ),
};

export interface Evidence {
  id?: string;
  artifact_id?: string;
  source_type?: string;
  summary?: string;
  complexity?: number;
  production_signal?: boolean;
  problem_category?: string;
  url?: string;
  date?: string;
  content_hash?: string;
}
export interface SearchCandidate {
  developer_id: string;
  github_username?: string | null;
  display_name?: string | null;
  confidence: number;
  evidence_count: number;
  evidence: Evidence[];
}
export interface NLSearchResult {
  query_interpreted: Record<string, unknown>;
  candidates: SearchCandidate[];
}
export interface SyncStatus {
  developer_id: string;
  github_username?: string | null;
  status: string;
  last_synced_at?: string | null;
  artifact_count: number;
  skill_count: number;
}
export interface EvidenceGraph {
  developer: { id: string; github_username?: string; display_name?: string; sync_status?: string };
  graph: { skills: { skill: string; confidence: number; evidence_count: number }[]; artifacts: Evidence[] };
  artifacts: Evidence[];
}
