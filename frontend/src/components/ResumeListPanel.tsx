"use client";

type Props = {
  files: File[];
  onAdd: (files: FileList | null) => void;
  onRemove: (index: number) => void;
};

export function ResumeListPanel({ files, onAdd, onRemove }: Props) {
  return (
    <section className="flex h-full min-h-0 flex-col rounded-2xl border border-white/10 bg-ink-800/80 shadow-xl shadow-black/30 backdrop-blur">
      <header className="flex items-center justify-between border-b border-white/10 px-5 py-4">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-sky-300/90">
            Right panel
          </p>
          <h2 className="text-lg font-semibold text-white">Resumes</h2>
        </div>
        <label className="cursor-pointer rounded-lg bg-sky-500/15 px-3.5 py-2 text-sm font-medium text-sky-200 ring-1 ring-sky-400/30 transition hover:bg-sky-500/25">
          Select resumes
          <input
            type="file"
            accept="application/pdf,.pdf"
            multiple
            className="hidden"
            onChange={(e) => {
              onAdd(e.target.files);
              e.target.value = "";
            }}
          />
        </label>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
        {files.length === 0 ? (
          <p className="px-2 py-10 text-center text-sm text-slate-500">
            No resumes selected yet. Choose one or more PDF files.
          </p>
        ) : (
          <ul className="space-y-2">
            {files.map((file, index) => (
              <li
                key={`${file.name}-${file.size}-${index}`}
                className="flex items-center justify-between gap-3 rounded-xl border border-white/8 bg-ink-950/70 px-3 py-2.5"
              >
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium text-slate-100">
                    {file.name}
                  </p>
                  <p className="text-[11px] text-slate-500">
                    {(file.size / 1024).toFixed(1)} KB
                  </p>
                </div>
                <button
                  type="button"
                  aria-label={`Remove ${file.name}`}
                  onClick={() => onRemove(index)}
                  className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-slate-400 transition hover:bg-rose-500/15 hover:text-rose-300"
                >
                  ×
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}
