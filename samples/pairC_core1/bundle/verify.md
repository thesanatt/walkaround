Verify it yourself (5 steps; none needs Walkaround's server)

1. Hash the video file on your phone or computer and compare it with page 1:
   shasum -a 256 <your video file>     (Linux: sha256sum <your video file>)
   pickup: 25f68a8992c980976df1352c0d2894eca301a13b576bcac397e61952e1440efb
   return: f49b432f2932d4a9ffeb59881c992c5054e09a2d7d6b816a41d5dd1df9c8646b
2. Read the OpenTimestamps proof (pip install opentimestamps-client):
   ots info bundle/manifest.json.ots
3. Verify it against the Bitcoin block, or drop the .ots file on https://opentimestamps.org :
   ots verify bundle/manifest.json.ots
4. Verify the RFC 3161 token from freetsa (2026-10-31T22:50:54Z) with freetsa's certificates (download https://freetsa.org/files/cacert.pem and https://freetsa.org/files/tsa.crt):
   openssl ts -verify -data bundle/manifest.json -in bundle/manifest.json.tsr -CAfile cacert.pem -untrusted tsa.crt
5. Re-hash every file in the bundle and rebuild the derived files from the stored model answers:
   walkaround verify pairC_core1
