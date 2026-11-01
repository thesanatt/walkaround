Verify it yourself (5 steps; none needs Walkaround's server)

1. Hash the video file on your phone or computer and compare it with page 1:
   shasum -a 256 <your video file>     (Linux: sha256sum <your video file>)
   pickup: 0f612417615726cf0bc5d0662a5373cdf92cd2defc7329e063685580a2fd04ed
   return: a157a4a7483b7540d1f4e133c1b69d7f51deeca9360336e96004e64fea2a7fed
2. Read the OpenTimestamps proof (pip install opentimestamps-client):
   ots info bundle/manifest.json.ots
3. Verify it against the Bitcoin block, or drop the .ots file on https://opentimestamps.org :
   ots verify bundle/manifest.json.ots
4. Verify the RFC 3161 token from freetsa (2026-11-01T07:15:41Z) with freetsa's certificates (download https://freetsa.org/files/cacert.pem and https://freetsa.org/files/tsa.crt):
   openssl ts -verify -data bundle/manifest.json -in bundle/manifest.json.tsr -CAfile cacert.pem -untrusted tsa.crt
5. Re-hash every file in the bundle and rebuild the derived files from the stored model answers:
   walkaround verify eval_nhtsa15759_r6
