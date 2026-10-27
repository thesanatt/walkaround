// SHA-256 over the same 8 MB slices the uploader sends, with hash-wasm, off the main thread.
import { createSHA256, type IHasher } from 'hash-wasm';

import type { HashMessage, HashRequest } from './protocol.ts';

interface WorkerScope {
  postMessage(message: HashMessage): void;
  onmessage: ((event: MessageEvent<HashRequest>) => void) | null;
}

const scope = self as unknown as WorkerScope;
const hashers = new Map<string, Promise<IHasher>>();
let queue: Promise<void> = Promise.resolve();

function enqueue(task: () => Promise<void>): void {
  queue = queue.then(task).catch(() => undefined);
}

scope.onmessage = (event: MessageEvent<HashRequest>) => {
  const message = event.data;
  if (message.type === 'start') {
    hashers.set(
      message.id,
      createSHA256().then((h) => {
        h.init();
        return h;
      }),
    );
    return;
  }
  enqueue(async () => {
    const pending = hashers.get(message.id);
    if (!pending) {
      scope.postMessage({ type: 'error', id: message.id, message: 'hash not started' });
      return;
    }
    try {
      const hasher = await pending;
      if (message.type === 'chunk') {
        hasher.update(new Uint8Array(await message.blob.arrayBuffer()));
        scope.postMessage({ type: 'progress', id: message.id, index: message.index });
      } else {
        const hex = hasher.digest('hex');
        hashers.delete(message.id);
        scope.postMessage({ type: 'digest', id: message.id, hex });
      }
    } catch (err) {
      scope.postMessage({ type: 'error', id: message.id, message: err instanceof Error ? err.message : String(err) });
    }
  });
};
