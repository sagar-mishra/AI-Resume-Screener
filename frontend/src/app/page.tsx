"use client";

import { useMemo, useState } from "react";
import { CandidateDrawer } from "@/components/CandidateDrawer";
import { JobDescriptionPanel } from "@/components/JobDescriptionPanel";
import { ResumeListPanel } from "@/components/ResumeListPanel";
import { ResultsTable } from "@/components/ResultsTable";
import { runScreening } from "@/lib/api";
import type { ScreeningResult } from "@/lib/types";

export default function HomePage() {
  const [jdTab, setJdTab] = useState<"text" | "file">("text");
  const [jdText, setJdText] = useState("");
  const [jdFile, setJdFile] = useState<File | null>(null);
  const [resumes, setResumes] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<ScreeningResult[]>([]);
  const [selected, setSelected] = useState<string | null>(null);

  const selectedResult = useMemo(
    () => results.find((r) => r.filename === selected) ?? null,
    [results, selected],
  );

  const canRun =
    !busy &&
    resumes.length > 0 &&
    (jdTab === "text" ? jdText.trim().length > 0 : Boolean(jdFile));

  function addResumes(list: FileList | null) {
    if (!list) return;
    const incoming = Array.from(list);
    setResumes((prev) => {
      const names = new Set(prev.map((f) => `${f.name}:${f.size}`));
      const extra = incoming.filter((f) => !names.has(`${f.name}:${f.size}`));
      return [...prev, ...extra];
    });
  }

  async function onRun() {
    setBusy(true);
    setError(null);
    setResults([]);
    setSelected(null);
    setProgress("Preparing sequential queue…");
    try {
      await runScreening({
        jdText: jdTab === "text" ? jdText : "",
        jdFile: jdTab === "file" ? jdFile : null,
        resumes,
        onEvent: (event) => {
          if (event.type === "progress") {
            setProgress(event.message);
          } else if (event.type === "result") {
            setResults((prev) => [...prev, event.result]);
          } else if (event.type === "error") {
            setError((prev) =>
              prev ? `${prev}\n${event.message}` : event.message,
            );
          }
        },
      });
      setProgress(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Screening failed");
      setProgress(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="mx-auto min-h-screen max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      <header className="mb-8 flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.22em] text-emerald-400">
            Local ATS
          </p>
          <h1 className="font-display mt-1 text-4xl text-white sm:text-5xl">
            AI Resume Screener
          </h1>
          <p className="mt-2 max-w-xl text-sm leading-6 text-slate-400">
            Sequential screening on a 6GB GPU. FastAPI parses PDFs; Ollama scores
            locally; recommendation tiers are computed on the server.
          </p>
        </div>
        <button
          type="button"
          disabled={!canRun}
          onClick={onRun}
          className="rounded-xl bg-emerald-500 px-6 py-3 text-sm font-semibold text-ink-950 shadow-lg shadow-emerald-500/20 transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400 disabled:shadow-none"
        >
          {busy ? "Screening…" : "Run AI Screening"}
        </button>
      </header>

      <div className="grid min-h-[420px] gap-5 lg:grid-cols-2 lg:items-stretch">
        <JobDescriptionPanel
          tab={jdTab}
          onTab={setJdTab}
          text={jdText}
          onText={setJdText}
          file={jdFile}
          onFile={setJdFile}
        />
        <ResumeListPanel
          files={resumes}
          onAdd={addResumes}
          onRemove={(i) => setResumes((prev) => prev.filter((_, idx) => idx !== i))}
        />
      </div>

      {progress ? (
        <div className="mt-6 flex items-center gap-3 rounded-2xl border border-sky-400/20 bg-sky-500/10 px-5 py-4 text-sm text-sky-100">
          <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-sky-400" />
          {progress}
        </div>
      ) : null}

      {error ? (
        <div className="mt-4 whitespace-pre-wrap rounded-2xl border border-rose-400/25 bg-rose-500/10 px-5 py-4 text-sm text-rose-100">
          {error}
        </div>
      ) : null}

      <div className="mt-6 space-y-5">
        <ResultsTable
          results={results}
          selected={selected}
          onSelect={(name) => setSelected((cur) => (cur === name ? null : name))}
        />
        <CandidateDrawer result={selectedResult} onClose={() => setSelected(null)} />
      </div>
    </main>
  );
}
