import { useEffect, useRef, useState } from 'react';

// A narration chip: the renter's own words, cut per turn. Play and pause, the length, and a plain error when the
// file is gone.
export function AudioChip({ src, text, seconds }: { src: string; text: string; seconds: number }) {
  const ref = useRef<HTMLAudioElement>(null);
  const [state, setState] = useState<'idle' | 'playing' | 'error'>('idle');

  useEffect(() => {
    setState('idle');
  }, [src]);

  const toggle = () => {
    const a = ref.current;
    if (!a) return;
    if (!a.paused) {
      a.pause();
      return;
    }
    void a.play().catch(() => setState('error'));
  };

  if (state === 'error') return <span className="audio-chip error small">the audio chip could not be played</span>;
  return (
    <span className="audio-chip">
      <button type="button" onClick={toggle} aria-label={`${state === 'playing' ? 'pause' : 'play'} the renter saying "${text}"`}>
        <svg viewBox="0 0 12 12" width="12" height="12" aria-hidden="true">
          {state === 'playing' ? (
            <g fill="currentColor">
              <rect x="2" y="1.5" width="3" height="9" />
              <rect x="7" y="1.5" width="3" height="9" />
            </g>
          ) : (
            <path d="M3 1.5 L10.5 6 L3 10.5 Z" fill="currentColor" />
          )}
        </svg>
        <span className="mono">{seconds.toFixed(1)} s</span>
      </button>
      <audio ref={ref} src={src} preload="none" onPlay={() => setState('playing')} onPause={() => setState('idle')} onEnded={() => setState('idle')} onError={() => setState('error')} />
    </span>
  );
}
