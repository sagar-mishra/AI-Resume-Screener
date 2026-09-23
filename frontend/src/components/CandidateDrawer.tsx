"use client";

import type { ScreeningResult } from "@/lib/types";

type Props = {
  result: ScreeningResult | null;
  onClose: () => void;
};

export function CandidateDrawer({ result, onClose }: Props) {
  if (!result) return null;

  return (
    <aside className="rounded-2xl border border-white/10 bg-ink-800/90 p-5 shadow-xl shadow-black/40">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
            Deep dive
          </p>
          <h3 className="text-xl font-semibold text-white">{result.candidate_name}</h3>
          <p className="mt-1 text-sm text-slate-400">
            {result.score} · {result.recommendation}
            {result.flag_for_human ? " · flagged for human review" : ""}
          </p>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="rounded-md px-2 py-1 text-slate-400 hover:bg-white/5 hover:text-white"
        >
          Close
        </button>
      </div>

      <div className="mt-5 grid gap-4 md:grid-cols-2">
        <div className="rounded-xl border border-emerald-400/20 bg-emerald-500/10 p-4">
          <p className="text-xs font-semibold uppercase tracking-wider text-emerald-300">
            Green flags
          </p>
          <p className="mt-2 text-sm leading-6 text-emerald-50/90">{result.strengths}</p>
        </div>
        <div className="rounded-xl border border-rose-400/20 bg-rose-500/10 p-4">
          <p className="text-xs font-semibold uppercase tracking-wider text-rose-300">
            Red flags
          </p>
          <p className="mt-2 text-sm leading-6 text-rose-50/90">{result.weaknesses}</p>
        </div>
      </div>

      <div className="mt-4">
        <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
          Raw extracted JSON
        </p>
        <pre className="mt-2 max-h-72 overflow-auto rounded-xl border border-white/10 bg-ink-950 p-4 text-xs leading-5 text-slate-300">
          {JSON.stringify(result.raw_json, null, 2)}
        </pre>
      </div>
    </aside>
  );
}
