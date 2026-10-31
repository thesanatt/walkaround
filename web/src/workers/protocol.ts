// Messages between the main thread and the two workers (docs/CONTRACTS.md, web reducer section).

// Lane worker. Deviation from the contract, on purpose: each object carries its mask as an SVG path (exact
// pixel-edge outline from @meta-sam/parser's decodeMaskToSVGPath) instead of an ImageBitmap, because a 600-frame
// door lane of bitmaps is about 160 MB; `bitmaps: true` still posts the contract's ImageBitmap as well.
export interface ParseRequest {
  type: 'parse';
  url: string;
  role: string;
  noun: string; // the lane id, e.g. car_door
  bitmaps?: boolean;
  color?: string; // fill for the optional bitmaps
}

export interface SnapshotObject {
  id: string;
  box: [number, number, number, number];
  c: number | null;
  path: string; // raster pixels, (0, 0) at the box's top-left; '' for a box-only record
  rw: number; // raster width and height, so the outline scales to the box exactly as @meta-sam/graphics does
  rh: number;
  mask: ImageBitmap | null;
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
  outcome: 'completed' | 'incomplete';
  diagnostics: number;
  parseMs: number;
}

export interface ErrorMessage {
  type: 'error';
  role: string;
  noun: string;
  message: string;
}

export type LaneWorkerMessage = SnapshotMessage | DoneMessage | ErrorMessage;

// Hash worker: SHA-256 with hash-wasm over the same 8 MB slices the uploader sends, in index order.
export type HashRequest = { type: 'start'; id: string } | { type: 'chunk'; id: string; index: number; blob: Blob } | { type: 'finish'; id: string };

export type HashMessage =
  | { type: 'progress'; id: string; index: number }
  | { type: 'digest'; id: string; hex: string }
  | { type: 'error'; id: string; message: string };
