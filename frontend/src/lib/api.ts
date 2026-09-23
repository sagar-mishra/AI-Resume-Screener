import type { ScreeningResult, StreamEvent } from "./types";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8080";

export { API_BASE };

export async function checkHealth(): Promise<{
  ok: boolean;
  llm?: { reachable: boolean; models?: string[]; error?: string };
}> {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) return { ok: false };
    return { ok: true, ...(await res.json()) };
  } catch {
    return { ok: false };
  }
}

export async function runScreening(opts: {
  jdText: string;
  jdFile: File | null;
  resumes: File[];
  onEvent: (event: StreamEvent) => void;
}): Promise<ScreeningResult[]> {
  const form = new FormData();
  if (opts.jdText.trim()) form.append("jd_text", opts.jdText.trim());
  if (opts.jdFile) form.append("jd_file", opts.jdFile);
  for (const file of opts.resumes) {
    form.append("resumes", file);
  }

  const res = await fetch(`${API_BASE}/api/screen`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }

  if (!res.body) throw new Error("No response stream from backend");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const collected: ScreeningResult[] = [];

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() ?? "";

    for (const chunk of chunks) {
      const line = chunk
        .split("\n")
        .find((l) => l.startsWith("data: "));
      if (!line) continue;
      const event = JSON.parse(line.slice(6)) as StreamEvent;
      opts.onEvent(event);
      if (event.type === "result") collected.push(event.result);
    }
  }

  return collected;
}
