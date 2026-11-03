import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import type { JobSnapshot } from "./types";

/** Poll the single active job; the backend exposes only one at a time. */
export function useJob(pollMs = 1200) {
  const [snapshot, setSnapshot] = useState<JobSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const alive = useRef(true);

  const refresh = useCallback(async () => {
    try {
      const next = await api<JobSnapshot>("/api/jobs/current");
      if (alive.current) {
        setSnapshot(next);
        setError(null);
      }
    } catch (caught) {
      if (alive.current) {
        setError(caught instanceof Error ? caught.message : String(caught));
      }
    }
  }, []);

  useEffect(() => {
    alive.current = true;
    void refresh();
    const timer = window.setInterval(() => void refresh(), pollMs);
    return () => {
      alive.current = false;
      window.clearInterval(timer);
    };
  }, [pollMs, refresh]);

  return { snapshot, error, refresh };
}
