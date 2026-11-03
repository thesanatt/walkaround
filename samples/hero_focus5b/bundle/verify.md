Verify it yourself (5 steps; none needs Walkaround's server)

This run's verdicts recomputed on Nov 2, 2026 with this version's rules (code d44edde) from the model answers stored on Nov 2, 2026 by run hero_focus5; no new model calls. The originals' hashes and stamps are that run's seal.

1. Hash the video file on your phone or computer and compare it with page 1:
   shasum -a 256 <your video file>     (Linux: sha256sum <your video file>)
   pickup: 0b1d3a373eeccd030eb8cc1eeadfa408d929a48cbabcf2f90481923bf4eca2cf
   return: cc683f02ad13d63d148fb5568724560f5c35e3d15202edcf7a21609cc23024f3
2. Read the OpenTimestamps proof (pip install opentimestamps-client):
   ots info bundle/manifest.json.ots
3. Verify it against the Bitcoin block, or drop the .ots file on https://opentimestamps.org :
   ots verify bundle/manifest.json.ots
4. Verify the RFC 3161 token from freetsa (2026-11-02T23:42:57Z) with freetsa's certificates (download https://freetsa.org/files/cacert.pem and https://freetsa.org/files/tsa.crt):
   openssl ts -verify -data bundle/manifest.json -in bundle/manifest.json.tsr -CAfile cacert.pem -untrusted tsa.crt
5. Re-hash every file in the bundle and rebuild the derived files from the stored model answers:
   walkaround verify hero_focus5b
