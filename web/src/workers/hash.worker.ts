// SHA-256 with hash-wasm over the same 8 MB slices the uploader sends, in index order, off the main thread.
// WebCrypto's digest cannot stream, so a 400 MB walk would otherwise sit in memory whole.
import { createSHA256, type IHasher } from 'hash-wasm';

import type { HashMessage, HashRequest } from './protocol.ts';

interface WorkerScope {
  postMessage(message: HashMessage): void;
  onmessage: ((event: MessageEvent<HashRequest>) => void) | null;
}

const scope = self as unknown as WorkerScope;
const hashers = new Map<string, Promise<IHasher>>();
let queue: Promise<void> = Promise.resolve();

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
  queue = queue.then(async () => {
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
        hashers.delete(message.id);
        scope.postMessage({ type: 'digest', id: message.id, hex: hasher.digest('hex') });
      }
    } catch (err) {
      hashers.delete(message.id);
      scope.postMessage({ type: 'error', id: message.id, message: err instanceof Error ? err.message : String(err) });
    }
  });
};
