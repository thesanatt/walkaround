// Shown whenever /api/status reports engine "mock": everything on screen is dev data, not a car and not a real run.
export function MockBanner() {
  return (
    <div className="mock-banner" role="note">
      Mock server. The walk is the cookbook ping pong clip and its SAM 3.1 lanes are the saved 10/26 streams; the panels, crops, marks and verdicts are
      hand-made dev data. No car, no model call.
    </div>
  );
}
