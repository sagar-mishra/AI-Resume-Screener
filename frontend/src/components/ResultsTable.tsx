"use client";

import type { ScreeningResult } from "@/lib/types";

function scoreTone(score: number): string {
  if (score >= 85) return "text-emerald-300";
  if (score >= 65) return "text-sky-300";
  if (score >= 45) return "text-amber-300";
  return "text-rose-300";
}

type Props = {
  results: ScreeningResult[];
  selected: string | null;
  onSelect: (filename: string) => void;
};

export function ResultsTable({ results, selected, onSelect }: Props) {
  const sorted = [...results].sort((a, b) => b.score - a.score);

  if (sorted.length === 0) return null;

  return (
    <div className="overflow-hidden rounded-2xl border border-white/10 bg-ink-800/80 shadow-xl shadow-black/30">
      <div className="border-b border-white/10 px-5 py-4">
        <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-emerald-400/90">
          Phase 3
        </p>
        <h2 className="text-lg font-semibold text-white">
          Leaderboard
          <span className="ml-2 text-sm font-normal text-slate-500">
            {sorted.length} candidate{sorted.length === 1 ? "" : "s"}
          </span>
        </h2>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[720px] text-left text-sm">
          <thead className="bg-ink-950/60 text-[11px] uppercase tracking-wider text-slate-500">
            <tr>
              <th className="px-5 py-3 font-medium">Candidate</th>
              <th className="px-3 py-3 font-medium">Score</th>
              <th className="px-3 py-3 font-medium">Recommendation</th>
              <th className="px-3 py-3 font-medium">Quick summary</th>
              <th className="px-5 py-3 font-medium">Action</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((row) => {
              const open = selected === row.filename;
              return (
                <tr
                  key={row.filename}
                  onClick={() => onSelect(row.filename)}
                  className={`cursor-pointer border-t border-white/5 transition hover:bg-white/[0.03] ${
                    open ? "bg-emerald-500/[0.06]" : ""
                  }`}
                >
                  <td className="px-5 py-3.5">
                    <p className="font-medium text-slate-100">{row.candidate_name}</p>
                    <p className="text-[11px] text-slate-500">{row.filename}</p>
                  </td>
                  <td className={`px-3 py-3.5 text-lg font-semibold tabular-nums ${scoreTone(row.score)}`}>
                    {row.score}
                  </td>
                  <td className="px-3 py-3.5 text-slate-200">{row.recommendation}</td>
                  <td className="max-w-xs px-3 py-3.5 text-slate-400">
                    <span className="line-clamp-2">{row.quick_summary}</span>
                  </td>
                  <td className="px-5 py-3.5 text-xs font-medium text-emerald-300">
                    {open ? "Hide details" : "Deep dive"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
