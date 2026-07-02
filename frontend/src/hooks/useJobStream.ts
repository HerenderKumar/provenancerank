import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api";

export interface JobEvent {
  event: string;
  stage?: string;
  status?: string;
  summary?: { elapsed_ms: number; source: string; cache_hit: boolean };
  error?: string;
}

// Polls job status until it reaches a terminal state. Polling (not raw
// EventSource) because EventSource can't send the Authorization header; the
// backend's SSE endpoint is still there for same-origin/cookie auth setups.
export function useJobStream(jobId: string | null) {
  const [events, setEvents] = useState<JobEvent[]>([]);
  const [status, setStatus] = useState<string>("queued");
  const timer = useRef<number>();

  useEffect(() => {
    if (!jobId) return;
    setEvents([]);
    setStatus("queued");
    let stop = false;

    async function poll() {
      try {
        const job = await api.getJob(jobId!);
        setStatus(job.status);
        setEvents((e) => [...e, { event: "status", status: job.status }]);
        if (job.status === "succeeded" || job.status === "failed") return;
      } catch {
        /* transient — keep polling */
      }
      if (!stop) timer.current = window.setTimeout(poll, 600);
    }
    poll();
    return () => {
      stop = true;
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, [jobId]);

  return { events, status };
}
