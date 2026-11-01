"""Overlay images for the box study: truth green, Spark's box orange, the SAM box cyan, Spark's point a dot."""

from __future__ import annotations

import cv2
import numpy as np

GREEN, ORANGE, CYAN, WHITE = (60, 220, 60), (255, 150, 0), (0, 230, 255), (255, 255, 255)


def read_rgb(path: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def draw(rgb: np.ndarray, truth: list[list[float]] = (), spark: list[list[float]] = (), sam: list[list[float]] = (),
         points: list[list[float]] = (), targets: list[list[float]] = (), window: list[float] | None = None,
         zoom: int = 3, mask: np.ndarray | None = None) -> np.ndarray:
    """Draws at zoom x on the window [x1, y1, x2, y2] of the crop (whole crop when None), so 1 px lines stay exact."""
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = window or [0, 0, w, h]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
    img = rgb[y1:y2, x1:x2].copy()
    if mask is not None:
        m = mask[y1:y2, x1:x2].astype(bool)
        img[m] = (0.55 * img[m] + 0.45 * np.array(CYAN)).astype(np.uint8)
    img = cv2.resize(img, ((x2 - x1) * zoom, (y2 - y1) * zoom), interpolation=cv2.INTER_NEAREST)

    def z(v: float, o: int) -> int:
        return int(round((v - o) * zoom))

    for boxes, color in ((truth, GREEN), (spark, ORANGE), (sam, CYAN)):
        for b in boxes:
            cv2.rectangle(img, (z(b[0], x1), z(b[1], y1)), (z(b[2], x1) - 1, z(b[3], y1) - 1), color, 2)
    for p in points:
        cv2.circle(img, (z(p[0], x1), z(p[1], y1)), 4, ORANGE, -1)
    for t in targets:
        cv2.drawMarker(img, (z(t[0], x1), z(t[1], y1)), GREEN, cv2.MARKER_CROSS, 14, 2)
    return img


def save(path: str, rgb: np.ndarray) -> None:
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
