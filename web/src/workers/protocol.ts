// Worker protocol of docs/CONTRACTS.md (web reducer section), plus the SVG path that replaces the bitmap by default.
export interface ParseRequest {
  type: 'parse';
  url: string;
  role: string;
  noun: string;
  // Fill color for the optional ImageBitmap masks; the main thread colors paths itself.
  color?: string;
  // The contract's ImageBitmap per object. Off by default: a 600-frame door lane would hold about 160 MB of bitmaps.
  bitmaps?: boolean;
}

export interface SnapshotObject {
  id: string;
  box: [number, number, number, number];
  c: number | null;
  mask: ImageBitmap | null;
  // decodeMaskToSVGPath of the mask, in raster pixels; translate to box[0], box[1] when drawing.
  path: string;
}

export interface SnapshotMessage {
  type: 'snapshot';
  role: string;
  noun: string;
  revision: number;
  frame: number;
  objects: SnapshotObject[];
}

export interface DoneMessage {
  type: 'done';
  role: string;
  noun: string;
  frames: number;
  outcome: string;
  diagnostics: number;
  parseMs: number;
}

export interface ErrorMessage {
  type: 'error';
  role: string;
  noun: string;
  message: string;
}

export type WorkerMessage = SnapshotMessage | DoneMessage | ErrorMessage;

export interface HashStart {
  type: 'start';
  id: string;
}
export interface HashChunk {
  type: 'chunk';
  id: string;
  blob: Blob;
  index: number;
}
export interface HashFinish {
  type: 'finish';
  id: string;
}
export type HashRequest = HashStart | HashChunk | HashFinish;
export type HashMessage =
  | { type: 'progress'; id: string; index: number }
  | { type: 'digest'; id: string; hex: string }
  | { type: 'error'; id: string; message: string };
