// Shown whenever /api/status says engine "mock": the data on screen is the dev fixture, not a car run.
export function MockBanner() {
  return (
    <div className="mock-banner" role="note">
      mock server: the SAM lanes are the 10/26 ping pong fixture and the panels, marks and verdicts are hand-written dev data. Not a car, not a real run.
    </div>
  );
}
