import type { ReactNode } from "react";

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: string }) {
  return (
    <div className="card">
      <div className="label">{label}</div>
      <div className="mt-1 text-2xl font-semibold text-white">{value}</div>
      {sub && <div className="mt-1 text-xs text-muted">{sub}</div>}
    </div>
  );
}

export function ScorePill({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const color = score >= 0.5 ? "text-good" : score >= 0.35 ? "text-brand" : "text-muted";
  return (
    <span className={`font-mono text-sm ${color}`} title={`score ${score.toFixed(4)}`}>
      {score.toFixed(3)}
      <span className="ml-1 text-[10px] text-muted">({pct}%)</span>
    </span>
  );
}

export function RankBadge({ rank }: { rank: number }) {
  const tone =
    rank <= 10 ? "bg-good/15 text-good" : rank <= 30 ? "bg-brand/15 text-brand" : "bg-edge text-muted";
  return (
    <span className={`inline-flex h-6 w-8 items-center justify-center rounded-md text-xs font-bold ${tone}`}>
      {rank}
    </span>
  );
}

export function CoverageBar({ label, value }: { label: string; value: number }) {
  const pct = Math.round(value * 100);
  return (
    <div>
      <div className="flex justify-between text-xs">
        <span className="text-muted">{label.replace(/_/g, " ")}</span>
        <span className="text-white">{pct}%</span>
      </div>
      <div className="mt-1 h-2 rounded bg-edge">
        <div
          className="h-2 rounded bg-gradient-to-r from-brand-dim to-brand"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-muted">
      <span className="h-3 w-3 animate-spin rounded-full border-2 border-edge border-t-brand" />
      {label}
    </div>
  );
}

export function StatusDot({ status }: { status: string }) {
  const m: Record<string, string> = {
    succeeded: "bg-good",
    running: "bg-warn animate-pulse",
    queued: "bg-muted",
    failed: "bg-bad",
  };
  return <span className={`inline-block h-2 w-2 rounded-full ${m[status] || "bg-muted"}`} />;
}
