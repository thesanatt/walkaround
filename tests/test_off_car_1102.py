"""R26 (11/2): panel tracks mostly off the target car's own mask. panels.derive drops them before the vote under
thresholds panels.off_car_tracks (absent from frozen runs); for a run frozen before it the walk overlay hides them
through GET /api/runs/{id}/off_car, a display filter only (orchestrator.off_car_view)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from tests.test_orchestrator import make_orch, runs_dir, th, write_run  # noqa: F401
from walkaround import contracts as c
from walkaround import panels, paths
from walkaround.app import create_app
from walkaround.contracts import Noun

SIZE = (100, 100)
CAR_BOX = [0, 0, 50, 100]  # the target car fills the left half of the frame
DOOR_BOXES = {"1": [10, 10, 40, 60], "2": [60, 10, 90, 60], "3": [35, 10, 65, 60], "4": [45, 10, 75, 60]}  # inside 1.0, 0.0, 0.5, 1/6
FRAMES = list(range(0, 31, 5))


def ref(box: list[int]) -> c.MaskRef:
    return c.MaskRef(h=box[3] - box[1], w=box[2] - box[0], encoding=c.MaskEncoding.ONE_BIT, payload_offset=0, payload_len=0)


def lane(boxes: dict[str, list[int]]) -> list[c.IndexFrame]:
    return [c.IndexFrame(frame=f, objects=[c.IndexObject(id=oid, box=box, mask=ref(box)) for oid, box in boxes.items()]) for f in FRAMES]


def door(n: int, oid: str) -> c.PanelTrack:
    box = DOOR_BOXES[oid]
    rows = [c.FrameObs(frame=f, pts=f / 15, box=[float(v) for v in box], box_area_frac=0.15, in_car_box=True) for f in FRAMES]
    return c.PanelTrack(
        track_id=f"pickup_car_door_{n:02d}", role="pickup", noun=Noun.CAR_DOOR, sam_object_ids=[oid],
        fragments=[c.TrackFragment(object_id=oid, first_frame=FRAMES[0], last_frame=FRAMES[-1])], per_frame=rows,
        panel_id=c.PanelId.LEFT_REAR_DOOR,
    )


class FakeMasks:
    """_Masks.lane on whole-box rasters: every mask fills its box."""

    usable = True

    def __init__(self) -> None:
        self.decoded: list[tuple[str, int, str]] = []

    def lane(self, lane_path: Path | None, pairs: list[tuple[int, str]], lookup: dict, *, rows: int | None = None) -> dict:
        out = {}
        for key in pairs:
            obj = lookup.get(key)
            if obj is None or lane_path is None:
                continue
            self.decoded.append((lane_path.name, *key))
            out[key] = np.ones((obj.box[3] - obj.box[1], obj.box[2] - obj.box[0]), dtype=bool)
        return out


def shares_of(tracks: list[c.PanelTrack], masks: FakeMasks | None = None) -> dict[str, float]:
    frames = {Noun.CAR: lane({"0": CAR_BOX}), Noun.CAR_DOOR: lane(DOOR_BOXES)}
    lane_paths = {Noun.CAR: Path("car.ndjson.gz"), Noun.CAR_DOOR: Path("car_door.ndjson.gz")}
    return panels.off_car_shares(tracks, frames, lane_paths, masks or FakeMasks(), SIZE, dict(panels.OFF_CAR_DEFAULTS))


def test_the_switch_is_on_for_new_runs_and_absent_from_frozen_ones(th: dict[str, Any]) -> None:
    cfg = panels.off_car_settings(th)
    assert cfg is not None and cfg["min_inside"] == 0.25
    frozen = {**th, "panels": {k: v for k, v in th["panels"].items() if k != "off_car_tracks"}}
    assert panels.off_car_settings(frozen) is None
    off = {**th, "panels": {**th["panels"], "off_car_tracks": {**th["panels"]["off_car_tracks"], "enabled": False}}}
    assert panels.off_car_settings(off) is None


def test_frames_come_from_the_grid_spread_over_the_track() -> None:
    t = door(1, "1")
    cfg = dict(panels.OFF_CAR_DEFAULTS)
    assert panels.off_car_frames(t, set(FRAMES), cfg) == [(0, "1"), (20, "1"), (30, "1")]  # grid frames 0, 10, 20, 30
    assert panels.off_car_frames(t, {5, 15}, cfg) == [(5, "1"), (15, "1")]  # no grid frame with a car: any frame
    assert panels.off_car_frames(t, set(), cfg) == []


def test_a_door_of_another_car_is_off_the_car_mask_and_a_door_on_its_edge_stays() -> None:
    tracks = [door(1, "1"), door(2, "2"), door(3, "3"), door(4, "4")]
    masks = FakeMasks()
    shares = shares_of(tracks, masks)
    assert shares == {"pickup_car_door_01": 1.0, "pickup_car_door_02": 0.0, "pickup_car_door_03": 0.5, "pickup_car_door_04": round(5 / 30, 6)}
    assert panels.off_car_ids(shares, dict(panels.OFF_CAR_DEFAULTS)) == ["pickup_car_door_02", "pickup_car_door_04"]
    # three frames a track, one car decode a frame for every track
    assert sum(1 for d in masks.decoded if d[0] == "car.ndjson.gz") == 3


def test_plates_faces_the_car_and_built_fenders_are_not_measured() -> None:
    plate = door(5, "1").model_copy(update={"track_id": "pickup_license_plate_00", "noun": Noun.LICENSE_PLATE})
    car = door(6, "1").model_copy(update={"track_id": "pickup_car_00", "noun": Noun.CAR})
    assert shares_of([plate, car]) == {}


def test_derive_drops_the_tracks_under_min_inside(monkeypatch: Any, th: dict[str, Any]) -> None:
    monkeypatch.setattr(panels, "_Masks", lambda decode, th: SimpleNamespace(usable=True))
    monkeypatch.setattr(panels, "off_car_shares", lambda *a, **k: {"pickup_car_door_01": 0.97, "pickup_car_door_02": 0.0})
    tracks = [door(1, "1"), door(2, "2"), door(3, "3")]
    kept = panels._drop_off_car("run", "pickup", tracks, {}, {}, SIZE, None, dict(panels.OFF_CAR_DEFAULTS), th)
    assert [t.track_id for t in kept] == ["pickup_car_door_01", "pickup_car_door_03"]


def test_the_overlay_view_measures_a_frozen_run_once_and_names_nothing_for_a_new_one(runs_dir: Path, th: dict[str, Any], monkeypatch: Any) -> None:
    run_id = "01TESTOFFCAR00000000000000"
    write_run(run_id, created_at="2026-11-02T20:00:00Z", kind=c.RunKind.PAIR, status=c.RunStatus.DONE, live=False)
    c.write_json(paths.tracks_json(run_id), [door(1, "1"), door(2, "2")])
    calls: list[str] = []

    def measured(rid: str, role: str, tracks: list, lanes: dict, size: tuple, th_run: dict) -> dict[str, float]:
        calls.append(role)
        return {"pickup_car_door_01": 1.0, "pickup_car_door_02": 0.0}

    frozen = {**th, "panels": {k: v for k, v in th["panels"].items() if k != "off_car_tracks"}}
    orch = make_orch(frozen)
    orch.overrides["panels.stored_off_car"] = measured
    monkeypatch.setattr(orch, "load_video", lambda rid, role: SimpleNamespace(proxy=SimpleNamespace(width=100, height=100)))
    run = orch.load_run(run_id)
    view = orch.off_car_view(run)
    assert view == {"in_core": False, "min_inside": 0.25, "shares": {"pickup_car_door_01": 1.0, "pickup_car_door_02": 0.0}, "track_ids": ["pickup_car_door_02"]}
    assert orch.off_car_view(run) == view and calls == ["pickup"]  # cached; the return walk has no track
    with TestClient(create_app(orch, background=False)) as client:
        assert client.get(f"/api/runs/{run_id}/off_car").json()["track_ids"] == ["pickup_car_door_02"]
        assert client.get("/api/runs/01TESTNOSUCHRUN00000000000/off_car").status_code == 404
    new = run.model_copy(update={"thresholds": th})
    assert orch.off_car_view(new) == {"in_core": True, "min_inside": None, "shares": {}, "track_ids": []}
