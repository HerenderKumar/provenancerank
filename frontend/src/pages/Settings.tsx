import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Copy, KeyRound, RefreshCw, Trash2 } from "lucide-react";
import { useState } from "react";
import { Spinner, Stat } from "../components/ui";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";

export default function Settings() {
  const { user } = useAuth();
  const qc = useQueryClient();
  const [name, setName] = useState("");
  const [fresh, setFresh] = useState<string | null>(null);

  const keys = useQuery({ queryKey: ["keys"], queryFn: api.listApiKeys });
  const artifacts = useQuery({ queryKey: ["artifacts"], queryFn: api.artifacts, retry: false });

  const create = useMutation({
    mutationFn: () => api.createApiKey(name || "api key"),
    onSuccess: (r) => {
      setFresh(r.api_key);
      setName("");
      qc.invalidateQueries({ queryKey: ["keys"] });
    },
  });
  const revoke = useMutation({
    mutationFn: (id: string) => api.revokeApiKey(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["keys"] }),
  });
  const precompute = useMutation({ mutationFn: () => api.triggerPrecompute() });

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">Settings</h1>

      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <Stat label="Account" value={user?.role} sub={user?.email} />
        <Stat label="Candidates" value={artifacts.data?.candidates?.toLocaleString() ?? "—"} />
        <Stat label="Embeddings" value={artifacts.data?.has_embeddings ? "yes" : "no"} />
        <Stat label="BM25 index" value={artifacts.data?.has_bm25 ? "yes" : "no"} />
      </div>

      <div className="card">
        <div className="mb-3 flex items-center gap-2 font-medium">
          <KeyRound size={16} /> API keys
        </div>
        <div className="flex gap-2">
          <input
            className="input"
            placeholder="key name (e.g. ci-pipeline)"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          <button className="btn" onClick={() => create.mutate()} disabled={create.isPending}>
            {create.isPending ? <Spinner /> : "Create"}
          </button>
        </div>
        {fresh && (
          <div className="mt-3 flex items-center gap-2 rounded-lg border border-warn/40 bg-warn/10 p-2 text-sm">
            <span className="font-mono text-warn">{fresh}</span>
            <button className="btn-ghost ml-auto" onClick={() => navigator.clipboard.writeText(fresh)}>
              <Copy size={14} /> copy
            </button>
            <span className="text-xs text-muted">shown once</span>
          </div>
        )}
        <div className="mt-4 divide-y divide-edge">
          {keys.data?.map((k) => (
            <div key={k.id} className="flex items-center gap-3 py-2 text-sm">
              <span className="font-mono text-muted">…{k.last4}</span>
              <span>{k.name}</span>
              <span className="text-xs text-muted">{k.scopes}</span>
              {!k.is_active && <span className="text-xs text-bad">revoked</span>}
              {k.is_active && (
                <button
                  className="btn-ghost ml-auto !px-2 !py-1"
                  onClick={() => revoke.mutate(k.id)}
                >
                  <Trash2 size={14} />
                </button>
              )}
            </div>
          ))}
          {!keys.data?.length && <div className="py-4 text-sm text-muted">No keys yet.</div>}
        </div>
      </div>

      {user?.role === "admin" && (
        <div className="card">
          <div className="mb-2 font-medium">Artifacts</div>
          <p className="text-sm text-muted">
            Rebuild the feature matrix, embeddings, BM25 index and model from the candidate file.
            Runs offline in a worker; the ranker hot-reloads when done.
          </p>
          <button
            className="btn mt-3"
            onClick={() => precompute.mutate()}
            disabled={precompute.isPending}
          >
            <RefreshCw size={14} /> Re-run precompute
          </button>
          {precompute.isSuccess && <span className="ml-3 text-sm text-good">started</span>}
        </div>
      )}
    </div>
  );
}
