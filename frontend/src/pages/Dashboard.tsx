import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Play } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Spinner, Stat, StatusDot } from "../components/ui";
import { api } from "../lib/api";

export default function Dashboard() {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [jd, setJd] = useState("");
  const [topK, setTopK] = useState(100);

  const artifacts = useQuery({ queryKey: ["artifacts"], queryFn: api.artifacts, retry: false });
  const jobs = useQuery({ queryKey: ["jobs"], queryFn: api.listJobs, refetchInterval: 4000 });

  const submit = useMutation({
    mutationFn: () => api.submitRank({ jd_text: jd.trim() || undefined, top_k: topK }),
    onSuccess: (r) => {
      qc.invalidateQueries({ queryKey: ["jobs"] });
      navigate(`/results?job=${r.job_id}`);
    },
  });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Dashboard</h1>
        <p className="text-sm text-muted">Rank the candidate pool against the Senior AI Engineer JD.</p>
      </div>

      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <Stat label="Candidates loaded" value={artifacts.data?.candidates?.toLocaleString() ?? "—"} />
        <Stat
          label="Ranker"
          value={artifacts.data?.ready ? "Ready" : "Not ready"}
          sub={artifacts.data?.embedding_backend ?? ""}
        />
        <Stat label="Index built" value={artifacts.data?.built_at?.slice(0, 10) ?? "—"} />
        <Stat label="Recent jobs" value={jobs.data?.length ?? 0} />
      </div>

      <div className="card">
        <div className="label mb-2">Custom job description (optional — blank uses the released JD)</div>
        <textarea
          className="input h-28 resize-none"
          placeholder="Paste a JD to rank against it instead…"
          value={jd}
          onChange={(e) => setJd(e.target.value)}
        />
        <div className="mt-3 flex items-center gap-3">
          <label className="label">Top K</label>
          <input
            type="number"
            min={1}
            max={100}
            className="input w-24"
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
          />
          <button className="btn ml-auto" onClick={() => submit.mutate()} disabled={submit.isPending}>
            {submit.isPending ? <Spinner /> : <Play size={16} />} Run ranking
          </button>
        </div>
        {submit.isError && <div className="mt-2 text-sm text-bad">{(submit.error as Error).message}</div>}
      </div>

      <div className="card">
        <div className="mb-3 font-medium">Recent runs</div>
        <div className="divide-y divide-edge">
          {jobs.data?.length ? (
            jobs.data.map((j) => (
              <button
                key={j.job_id}
                onClick={() => navigate(`/results?job=${j.job_id}`)}
                className="flex w-full items-center gap-3 py-2 text-left text-sm hover:text-white"
              >
                <StatusDot status={j.status} />
                <span className="font-mono text-xs text-muted">{j.job_id.slice(0, 8)}</span>
                <span className="text-muted">{j.source}</span>
                {j.cache_hit && <span className="rounded bg-edge px-1.5 text-[10px]">cached</span>}
                <span className="ml-auto text-muted">
                  {j.elapsed_ms ? `${j.elapsed_ms.toFixed(0)} ms` : j.status}
                </span>
              </button>
            ))
          ) : (
            <div className="py-6 text-center text-sm text-muted">No runs yet.</div>
          )}
        </div>
      </div>
    </div>
  );
}
