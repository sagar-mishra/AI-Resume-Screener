"use client";

type Props = {
  tab: "text" | "file";
  onTab: (tab: "text" | "file") => void;
  text: string;
  onText: (value: string) => void;
  file: File | null;
  onFile: (file: File | null) => void;
};

export function JobDescriptionPanel({
  tab,
  onTab,
  text,
  onText,
  file,
  onFile,
}: Props) {
  return (
    <section className="flex h-full min-h-0 flex-col rounded-2xl border border-white/10 bg-ink-800/80 shadow-xl shadow-black/30 backdrop-blur">
      <header className="flex items-center justify-between border-b border-white/10 px-5 py-4">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-emerald-400/90">
            Left panel
          </p>
          <h2 className="text-lg font-semibold text-white">Job description</h2>
        </div>
        <div className="flex rounded-lg bg-ink-950 p-1 ring-1 ring-white/10">
          <button
            type="button"
            onClick={() => onTab("text")}
            className={`rounded-md px-3 py-1.5 text-xs font-medium transition ${
              tab === "text"
                ? "bg-emerald-500/20 text-emerald-200"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Paste text
          </button>
          <button
            type="button"
            onClick={() => onTab("file")}
            className={`rounded-md px-3 py-1.5 text-xs font-medium transition ${
              tab === "file"
                ? "bg-emerald-500/20 text-emerald-200"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Upload file
          </button>
        </div>
      </header>

      <div className="min-h-0 flex-1 p-5">
        {tab === "text" ? (
          <textarea
            value={text}
            onChange={(e) => onText(e.target.value)}
            placeholder="Paste the full job description here…"
            className="h-full min-h-[280px] w-full resize-none rounded-xl border border-white/10 bg-ink-950/80 px-4 py-3 text-sm leading-6 text-slate-200 outline-none ring-emerald-400/0 transition placeholder:text-slate-600 focus:border-emerald-400/40 focus:ring-2 focus:ring-emerald-400/20"
          />
        ) : (
          <label className="flex h-full min-h-[280px] cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-white/15 bg-ink-950/50 px-6 text-center transition hover:border-emerald-400/40 hover:bg-ink-950">
            <input
              type="file"
              accept=".pdf,.txt,.text,.md,application/pdf,text/plain"
              className="hidden"
              onChange={(e) => onFile(e.target.files?.[0] ?? null)}
            />
            <span className="rounded-full bg-emerald-500/15 px-3 py-1 text-xs font-medium text-emerald-300">
              PDF or TXT
            </span>
            <p className="mt-3 text-sm text-slate-300">
              {file ? file.name : "Click to choose a JD document"}
            </p>
            {file ? (
              <button
                type="button"
                className="mt-3 text-xs text-slate-500 underline hover:text-slate-300"
                onClick={(e) => {
                  e.preventDefault();
                  onFile(null);
                }}
              >
                Remove file
              </button>
            ) : (
              <p className="mt-1 text-xs text-slate-500">
                Parsed locally with PyMuPDF on the backend
              </p>
            )}
          </label>
        )}
      </div>
    </section>
  );
}
