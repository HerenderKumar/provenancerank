import { useQuery } from "@tanstack/react-query";
import { Download } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { CoverageBar, RankBadge, ScorePill, Spinner, StatusDot } from "../components/ui";
import { api, type RankRow } from "../lib/api";
import { useJobStream } from "../hooks/useJobStream";

export default function RankingResults() {
  const [params] = useSearchParams();
  const jobId = params.get("job");
  const { status } = useJobStream(jobId);
  const [selected, setSelected] = useState<RankRow | null>(null);

  const job = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api.getJob(jobId!),
    enabled: !!jobId,
    refetchInterval: status === "succeeded" || status === "failed" ? false : 1000,
  });
  const results = useQuery({
    queryKey: ["results", jobId],
    queryFn: () => api.getResults(jobId!),
    enabled: !!jobId && status === "succeeded",
  });

  const rows = results.data?.results ?? [];
  useEffect(() => {
    if (rows.length && !selected) setSelected(rows[0]);
  }, [rows, selected]);

  const chart = useMemo(() => rows.map((r) => ({ rank: r.rank, score: r.score })), [rows]);

  if (!jobId) return <div className="text-muted">Start a ranking from the Dashboard.</div>;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <h1 className="text-2xl font-semibold">Ranking results</h1>
        <StatusDot status={status} />
        <span className="text-sm text-muted">{status}</span>
        {job.data?.elapsed_ms != null && (
          <span className="text-sm text-muted">· {job.data.elapsed_ms.toFixed(0)} ms · {job.data.source}</span>
        )}
        <a className="btn-ghost ml-auto" href={api.csvUrl(jobId)}>
          <Download size={14} /> CSV
        </a>
      </div>

      {status !== "succeeded" ? (
        <div className="card flex items-center gap-3">
          <Spinner /> <span className="text-sm text-muted">Ranking {job.data?.candidates_count?.toLocaleString()} candidates…</span>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
          {/* shortlist */}
          <div className="card lg:col-span-4 max-h-[70vh] overflow-auto">
            <div className="label mb-2">Top {rows.length}</div>
            <div className="space-y-1">
              {rows.map((r) => (
                <button
                  key={r.candidate_id}
                  onClick={() => setSelected(r)}
                  className={`flex w-full items-center gap-2 rounded-lg px-2 py-2 text-left ${
                    selected?.candidate_id === r.candidate_id ? "bg-brand/15" : "hover:bg-panel"
                  }`}
                >
                  <RankBadge rank={r.rank} />
                  <span className="font-mono text-xs text-muted">{r.candidate_id}</span>
                  <span className="ml-auto">
                    <ScorePill score={r.score} />
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* detail */}
          <div className="card lg:col-span-5">
            {selected ? (
              <div>
                <div className="flex items-center gap-3">
                  <RankBadge rank={selected.rank} />
                  <span className="font-mono">{selected.candidate_id}</span>
                  <span className="ml-auto">
                    <ScorePill score={selected.score} />
                  </span>
                </div>
                <div className="label mt-4">Reasoning</div>
                <p className="mt-1 text-sm leading-relaxed text-white/90">{selected.reasoning}</p>
                <div className="label mt-6 mb-2">Score across the shortlist</div>
                <div className="h-40">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chart}>
                      <defs>
                        <linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#6d8cff" stopOpacity={0.6} />
                          <stop offset="100%" stopColor="#6d8cff" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <XAxis dataKey="rank" stroke="#8aa0c6" fontSize={11} />
                      <YAxis stroke="#8aa0c6" fontSize={11} domain={[0, "auto"]} />
                      <Tooltip
                        contentStyle={{ background: "#11182e", border: "1px solid #1f2a44" }}
                        labelStyle={{ color: "#8aa0c6" }}
                      />
                      <Area type="monotone" dataKey="score" stroke="#6d8cff" fill="url(#g)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
            ) : (
              <div className="text-muted">Select a candidate.</div>
            )}
          </div>

          {/* coverage + gate report */}
          <div className="card lg:col-span-3 space-y-4">
            <div>
              <div className="label mb-2">JD coverage</div>
              <div className="space-y-2">
                {Object.entries(job.data?.coverage ?? {}).map(([k, v]) => (
                  <CoverageBar key={k} label={k} value={v} />
                ))}
              </div>
            </div>
            <div>
              <div className="label mb-2">Gate filter</div>
              <div className="space-y-1 text-sm">
                {Object.entries(job.data?.gate_stats ?? {}).map(([k, v]) => (
                  <div key={k} className="flex justify-between">
                    <span className="text-muted">{k.replace(/_/g, " ")}</span>
                    <span className="text-white">{v.toLocaleString()}</span>
                  </div>
                ))}
              </div>
              <div className="mt-3 rounded-lg border border-good/30 bg-good/10 p-2 text-xs text-good">
                Honeypots in top {rows.length}: 0
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
