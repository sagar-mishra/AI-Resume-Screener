export type ScreeningResult = {
  filename: string;
  candidate_name: string;
  score: number;
  recommendation: string;
  confidence_level: string;
  flag_for_human: boolean;
  quick_summary: string;
  strengths: string;
  weaknesses: string;
  raw_json: Record<string, unknown>;
};

export type ProgressEvent = {
  type: "progress";
  current: number;
  total: number;
  filename: string;
  message: string;
};

export type ResultEvent = {
  type: "result";
  result: ScreeningResult;
};

export type ErrorEvent = {
  type: "error";
  filename?: string;
  message: string;
};

export type DoneEvent = {
  type: "done";
};

export type StreamEvent = ProgressEvent | ResultEvent | ErrorEvent | DoneEvent;
