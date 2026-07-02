import { ExternalLink, GitCommit, GitPullRequest, MessageSquare, Star } from "lucide-react";
import type { Evidence } from "../../lib/api";

const ICON: Record<string, typeof GitCommit> = {
  commit_diff: GitCommit,
  pr_review: GitPullRequest,
  issue_thread: MessageSquare,
  so_answer: MessageSquare,
};

function Complexity({ n }: { n?: number }) {
  const c = n ?? 0;
  return (
    <span className="inline-flex">
      {[1, 2, 3, 4, 5].map((i) => (
        <Star
          key={i}
          size={11}
          className={i <= c ? "fill-warn text-warn" : "text-edge"}
        />
      ))}
    </span>
  );
}

export default function EvidenceTimeline({ items }: { items: Evidence[] }) {
  if (!items.length) return <div className="text-sm text-muted">No evidence yet.</div>;
  return (
    <ol className="relative ml-3 border-l border-edge">
      {items.map((e, i) => {
        const Icon = ICON[e.source_type || ""] || GitCommit;
        return (
          <li key={e.id || e.artifact_id || i} className="mb-4 ml-4">
            <span className="absolute -left-[9px] mt-1 grid h-4 w-4 place-items-center rounded-full bg-panel ring-2 ring-edge">
              <Icon size={10} className="text-brand" />
            </span>
            <div className="flex items-center gap-2 text-xs text-muted">
              <span>{(e.source_type || "").replace("_", " ")}</span>
              {e.date && <span>· {e.date.slice(0, 10)}</span>}
              <Complexity n={e.complexity} />
              {e.production_signal && (
                <span className="rounded bg-good/15 px-1.5 text-[10px] text-good">production</span>
              )}
            </div>
            <p className="mt-1 text-sm text-white/90">{e.summary}</p>
            {e.url && (
              <a
                href={e.url}
                target="_blank"
                rel="noreferrer"
                className="mt-0.5 inline-flex items-center gap-1 text-xs text-brand hover:underline"
              >
                <ExternalLink size={11} /> view artifact
              </a>
            )}
          </li>
        );
      })}
    </ol>
  );
}
