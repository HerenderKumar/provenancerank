import { Github, Search, Sparkles } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import ProofOfWorkCard from "../components/developer/ProofOfWorkCard";
import { Spinner } from "../components/ui";
import { api, type NLSearchResult } from "../lib/api";

const EXAMPLES = [
  "who debugged a race condition in production",
  "who shipped a vector search system to real users",
  "who has fine-tuned an LLM and deployed it",
  "who built a recommendation system at scale",
];

export default function NaturalLanguageSearch() {
  const [query, setQuery] = useState("");
  const [placeholder, setPlaceholder] = useState(EXAMPLES[0]);
  const [result, setResult] = useState<NLSearchResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [gh, setGh] = useState("");
  const [ghMsg, setGhMsg] = useState<string | null>(null);
  const [devId, setDevId] = useState<string | null>(null);
  const debounce = useRef<number>();

  // cycle the placeholder so the input advertises what it can do
  useEffect(() => {
    let i = 0;
    const t = window.setInterval(() => {
      i = (i + 1) % EXAMPLES.length;
      setPlaceholder(EXAMPLES[i]);
    }, 3000);
    return () => window.clearInterval(t);
  }, []);

  // debounced live search
  useEffect(() => {
    if (debounce.current) window.clearTimeout(debounce.current);
    if (query.trim().length < 3) {
      setResult(null);
      return;
    }
    setLoading(true);
    debounce.current = window.setTimeout(async () => {
      try {
        setResult(await api.nlSearch(query.trim()));
      } catch {
        setResult(null);
      } finally {
        setLoading(false);
      }
    }, 500);
  }, [query]);

  async function connect(e: React.FormEvent) {
    e.preventDefault();
    if (!gh.trim()) return;
    try {
      const r = await api.connectGithub(gh.trim());
      setGhMsg(`Syncing @${r.github_username} (id ${r.developer_id.slice(0, 8)})…`);
      setGh("");
      setDevId(r.developer_id); // kick off real-status polling (effect below)
    } catch (err) {
      setDevId(null);
      setGhMsg((err as Error).message);
    }
  }

  // Poll the REAL backend status until the sync actually finishes, so the message
  // reflects reality (synced / error) instead of a permanent "Syncing…".
  useEffect(() => {
    if (!devId) return;
    let tries = 0;
    const tick = async () => {
      try {
        const s = await api.syncStatus(devId);
        if (s.status === "synced") {
          setGhMsg(`✓ Synced @${s.github_username ?? ""} — ${s.artifact_count} artifacts, ${s.skill_count} skills.`);
          setDevId(null);
          return;
        }
        if (s.status === "error") {
          setGhMsg(`✗ Sync failed for @${s.github_username ?? ""}. Check GITHUB_TOKEN on the server and the username.`);
          setDevId(null);
          return;
        }
        setGhMsg(`Syncing @${s.github_username ?? ""}… (${s.artifact_count} artifacts so far)`);
      } catch {
        /* transient — keep polling */
      }
      if (++tries > 40) {
        // ~100s cap: stop polling but don't claim success
        setGhMsg("Sync is taking a while — check the server logs (developer.sync_failed / sync.complete).");
        setDevId(null);
      }
    };
    tick();
    const iv = window.setInterval(tick, 2500);
    return () => window.clearInterval(iv);
  }, [devId]);

  const interp = result?.query_interpreted as
    | { skill_names?: string[]; require_production_signal?: boolean; relaxed?: boolean }
    | undefined;

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div className="text-center">
        <div className="mb-2 inline-flex items-center gap-2 text-brand">
          <Sparkles size={18} /> <span className="text-sm uppercase tracking-widest">evidence search</span>
        </div>
        <h1 className="text-2xl font-semibold">Ask anything about candidates</h1>
        <p className="text-sm text-muted">
          Backed by real commits, PRs and reviews — not self-reported skills.
        </p>
      </div>

      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-muted" size={18} />
        <input
          autoFocus
          className="input !py-3 pl-10 text-base"
          placeholder={placeholder}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </div>

      {interp && (
        <div className="flex flex-wrap items-center gap-2 text-xs text-muted">
          <span>interpreted as:</span>
          {(interp.skill_names || []).map((s) => (
            <span key={s} className="rounded bg-brand/15 px-2 py-0.5 text-brand">{s}</span>
          ))}
          {interp.require_production_signal && (
            <span className="rounded bg-good/15 px-2 py-0.5 text-good">production signal</span>
          )}
          {interp.relaxed && (
            <span className="rounded bg-edge px-2 py-0.5">broadened — few exact matches</span>
          )}
        </div>
      )}

      <div className="space-y-3">
        {loading && <Spinner label="Searching evidence…" />}
        {!loading && result && result.candidates.length === 0 && (
          <div className="card text-sm text-muted">
            No developers with verified evidence yet. Connect a GitHub account below to build a graph.
          </div>
        )}
        {result?.candidates.map((c) => (
          <ProofOfWorkCard key={c.developer_id} candidate={c} />
        ))}
      </div>

      <form onSubmit={connect} className="card">
        <div className="mb-2 flex items-center gap-2 font-medium">
          <Github size={16} /> Connect a developer
        </div>
        <div className="flex gap-2">
          <input
            className="input"
            placeholder="github username"
            value={gh}
            onChange={(e) => setGh(e.target.value)}
          />
          <button className="btn">Connect</button>
        </div>
        {ghMsg && <div className="mt-2 text-xs text-muted">{ghMsg}</div>}
        <p className="mt-2 text-xs text-muted">
          Ingestion runs in the background (needs a GITHUB_TOKEN on the server). Their commits, PRs
          and reviews become verifiable, cryptographically-anchored evidence.
        </p>
      </form>
    </div>
  );
}
