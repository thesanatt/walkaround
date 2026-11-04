import { Bar } from '../components/Bar.tsx';

// Appendix C of the sheet: anyone holding the bundle can check it with tools that are not Walkaround.
export function VerifyPage() {
  return (
    <div className="verify">
      <Bar />
      <h1>Check a sheet yourself</h1>
      <p className="verify-lead">
        A run's bundle holds the originals' hashes, the OpenTimestamps proofs and the RFC 3161 token. Steps 1 to 3 check them with standard tools; step 4
        needs the whole run directory, which holds the event log.
      </p>
      <h2>1. The file you kept is the file that was sealed</h2>
      <pre>{`sha256sum pickup.mp4 return.mp4
# compare with bundle/manifest.json and the hash prefix in every sheet footer`}</pre>
      <h2>2. The Bitcoin attestation</h2>
      <pre>{`ots info bundle/manifest.json.ots
ots verify bundle/manifest.json.ots   # or drop the .ots file on opentimestamps.org`}</pre>
      <h2>3. The RFC 3161 token</h2>
      <pre>{`openssl ts -verify -data bundle/manifest.json -in bundle/manifest.json.tsr \\
  -CAfile bundle/cacert.pem -untrusted bundle/tsa.crt`}</pre>
      <h2>4. The event chain and the rebuild</h2>
      <pre>{`python -m walkaround verify <run_id>
# from the repo, with runs/<run_id>/ present (events.ndjson is there, not in the bundle)
# prints "rebuild identical" or the first file that differs; no network call`}</pre>
      <h2>What the seal does not prove</h2>
      <p className="verify-lead">
        Which car it is, unless the plate was kept at the renter's request. That the walk was not staged before upload. Anything about pickup when the seal came
        after the return. Page 1 prints three clocks for this reason: when the file was sealed, when the first model call was made, and how many minutes after
        the camera's own time the seal came.
      </p>
    </div>
  );
}
