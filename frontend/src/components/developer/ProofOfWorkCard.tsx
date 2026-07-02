import { BadgeCheck, Share2 } from "lucide-react";
import { useState } from "react";
import type { SearchCandidate } from "../../lib/api";
import EvidenceTimeline from "./EvidenceTimeline";

// Shareable "proof of work" card. The whole growth loop: a developer shares this
// on their README/LinkedIn, recruiters see verified evidence, they connect too.
export default function ProofOfWorkCard({ candidate }: { candidate: SearchCandidate }) {
  const [copied, setCopied] = useState(false);
  const name = candidate.display_name || candidate.github_username || candidate.developer_id.slice(0, 8);
  const pct = Math.round(candidate.confidence * 100);

  function share() {
    const url = `${window.location.origin}/developers/${candidate.developer_id}`;
    navigator.clipboard?.writeText(url);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <div className="card">
      <div className="flex items-center gap-3">
        <div className="grid h-11 w-11 place-items-center rounded-full bg-brand/15 text-brand">
          <BadgeCheck size={22} />
        </div>
        <div className="min-w-0">
          <div className="flex items-center gap-2 font-medium">
            {name}
            {candidate.github_username && (
              <a
                href={`https://github.com/${candidate.github_username}`}
                target="_blank"
                rel="noreferrer"
                className="text-xs text-muted hover:text-brand"
              >
                @{candidate.github_username}
              </a>
            )}
          </div>
          <div className="text-xs text-muted">
            verified evidence · {candidate.evidence_count} artifacts
          </div>
        </div>
        <div className="ml-auto text-right">
          <div className="text-lg font-semibold text-good">{pct}%</div>
          <div className="text-[10px] uppercase tracking-wide text-muted">confidence</div>
        </div>
        <button className="btn-ghost !px-2 !py-1" onClick={share} title="copy profile link">
          <Share2 size={14} /> {copied ? "copied" : "share"}
        </button>
      </div>

      <div className="mt-3 h-1.5 rounded bg-edge">
        <div className="h-1.5 rounded bg-gradient-to-r from-good/70 to-good" style={{ width: `${pct}%` }} />
      </div>

      <div className="mt-4">
        <div className="label mb-2">Proof of work</div>
        <EvidenceTimeline items={candidate.evidence} />
      </div>
    </div>
  );
}
