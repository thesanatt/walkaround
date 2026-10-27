// The five commands of appendix C: anyone with the bundle can check the sheet without this app.
export function VerifyPage() {
  return (
    <div className="verify">
      <h1>Verify a sheet yourself</h1>
      <p className="muted" style={{ maxWidth: '70ch' }}>
        A run's bundle holds the originals' hashes, the OpenTimestamps proofs, the RFC 3161 token and the event log. Each line below checks one of them with a tool
        that is not Walkaround.
      </p>
      <h2>1. The file you kept is the file that was sealed</h2>
      <pre>{`sha256sum pickup.mp4 return.mp4
# compare with bundle/manifest.json and the footer of every sheet page`}</pre>
      <h2>2. The Bitcoin attestation</h2>
      <pre>{`ots info bundle/manifest.json.ots
ots verify bundle/manifest.json.ots   # or paste the .ots at opentimestamps.org`}</pre>
      <h2>3. The RFC 3161 token</h2>
      <pre>{`openssl ts -verify -data bundle/manifest.json -in bundle/manifest.json.tsr \\
  -CAfile bundle/cacert.pem -untrusted bundle/tsa.crt`}</pre>
      <h2>4. The event chain and the rebuild</h2>
      <pre>{`walkaround verify <run_id>
# prints "rebuild identical" or the first file that differs; makes no network call`}</pre>
      <h2>What the seal does not prove</h2>
      <p className="muted" style={{ maxWidth: '70ch' }}>
        Which car (unless the plate was kept at the renter's request), that the walk was not staged before upload, or anything about pickup when the seal came
        after return. The three clocks on page 1 say when the file was sealed, when the first model call was made, and how long after the camera time the seal came.
      </p>
    </div>
  );
}
