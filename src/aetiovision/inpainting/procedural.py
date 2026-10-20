"""Procedural baseline inpainter (D28): synthetic fracture lines and opacities, no ML. Deterministic per seed."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, shift

from aetiovision.domain.codes import BODY_STRUCTURE, Code, Triple
from aetiovision.domain.errors import InpaintFailed
from aetiovision.domain.image import Mask, MedicalImage
from aetiovision.domain.request import ResolvedRequest
from aetiovision.domain.template import Template

DX = Code("DCM", "DX")
BONE, FRACTURE, NEOPLASM = "272673000", "72704001", "108369006"
TRANSVERSE, OBLIQUE, COMMINUTED = "70682000", "7905002", "13321001"


def _canvas(shape, polylines, width) -> np.ndarray:
    im = Image.new("L", shape[::-1], 0)
    d = ImageDraw.Draw(im)
    for pts, w in polylines:
        d.line([(x, y) for y, x in pts], fill=255, width=max(1, round(w * width)), joint="curve")
    return np.asarray(im, np.float32) / 255


def _jagged(rng, c, d, n, half, step_amp) -> list[tuple[float, float]]:
    """Polyline through centre c along unit direction d, random-walk offset along the normal."""
    nrm = np.array([-d[1], d[0]])
    off = np.cumsum(rng.normal(0, step_amp, n))
    off -= off[n // 2]
    return [tuple(c + d * s + nrm * o) for s, o in zip(np.linspace(-half, half, n), off)]


class ProceduralInpainter:
    name = "procedural"

    def capabilities(self) -> set[Triple]:
        return {
            Triple(DX, Code.sct(BONE), Code.sct(FRACTURE)),
            Triple(DX, Code.sct(BODY_STRUCTURE), Code.sct(NEOPLASM)),
        }

    def inpaint(self, template: Template, mask: Mask, req: ResolvedRequest) -> MedicalImage:
        img, a = template.image.pixels.astype(np.float32), mask.alpha.astype(np.float32)
        rng = np.random.default_rng(req.seed)
        ys, xs = np.nonzero(a > 0.5)
        if len(ys) < 10:
            raise InpaintFailed(f"mask too small ({len(ys)} px)")
        pts = np.stack([ys, xs], 1).astype(np.float64)
        mean = pts.mean(0)
        w, v = np.linalg.eigh(np.cov((pts - mean).T))
        axis = v[:, 1]  # principal axis (y, x)
        proj = (pts - mean) @ axis
        morph = req.requested.morphology.code
        if req.supported.morphology.code == NEOPLASM:
            out, meta = self._neoplasm(img, a, pts, proj, rng)
        else:
            out, meta = self._fracture(img, a, mean, axis, pts, proj, morph, "displaced" in req.instructions.lower(), rng)
        out = np.clip(img * (1 - a) + out * a, 0, 1).astype(np.float32)
        return MedicalImage(out, {"inpainter": self.name, "morphology": morph, **meta})

    def _fracture(self, img, a, mean, axis, pts, proj, morph, displaced, rng):
        h, w = img.shape
        diag = float(np.hypot(h, w))
        normal = np.array([-axis[1], axis[0]])
        lo, hi = proj.min(), proj.max()
        t = rng.uniform(0.3, 0.7)
        centre = mean + axis * (lo + t * (hi - lo))
        width_px = diag * rng.uniform(0.003, 0.005)
        half = (np.abs((pts - mean) @ normal).max() + 4) * 1.3
        angle = {TRANSVERSE: rng.uniform(0, 10), OBLIQUE: rng.uniform(30, 45), COMMINUTED: rng.uniform(0, 30)}.get(morph, rng.uniform(0, 30))
        angle *= rng.choice([-1, 1])

        def direction(deg):  # perpendicular to the axis, rotated by deg
            r = np.deg2rad(deg)
            return normal * np.cos(r) + axis * np.sin(r)

        n = max(12, int(2 * half / 3))
        lines = [(_jagged(rng, centre, direction(angle), n, half, width_px * 0.6), 1.0)]
        if morph == COMMINUTED:
            for _ in range(int(rng.integers(1, 3))):
                c2 = centre + axis * rng.uniform(-1, 1) * (hi - lo) * 0.12
                lines.append((_jagged(rng, c2, direction(angle + rng.uniform(-35, 35)), n, half * 0.6, width_px * 0.6), 0.7))
            fc = centre + axis * rng.uniform(-1, 1) * width_px * 3  # small free fragment between lines
            r = width_px * rng.uniform(2, 3.5)
            ang = rng.uniform(0, 2 * np.pi, 5)
            ring = [tuple(fc + r * rng.uniform(0.6, 1.2) * np.array([np.sin(t_), np.cos(t_)])) for t_ in ang]
            lines.append((ring + ring[:1], 0.6))
        line = _canvas(img.shape, lines, width_px)
        line = gaussian_filter(line, max(0.6, width_px * 0.4)) * a
        line = np.clip(line * 1.5, 0, 1)
        dark = rng.uniform(0.2, 0.35)
        rim = np.clip(gaussian_filter(line, width_px * 1.5) - gaussian_filter(line, width_px * 0.5), 0, None)
        src = img
        meta = {"angle_deg": float(angle), "position_t": float(t), "darkening": float(dark), "displaced": displaced}
        if displaced:
            d_px = rng.uniform(2, 5) * max(1.0, diag / 700) * rng.choice([-1, 1])
            side = ((np.mgrid[:h, :w].transpose(1, 2, 0) - centre) @ axis) > 0
            moved = shift(img, normal * d_px, order=1, mode="nearest")
            src = np.where(side & (a > 0.5), moved, img)
            meta["shift_px"] = float(d_px)
        out = src * (1 - dark * line) + 0.25 * rim * (1 + src)
        return out, meta

    def _neoplasm(self, img, a, pts, proj, rng):
        h, w = img.shape
        ext = float(np.ptp(proj))
        p = pts[rng.integers(len(pts))]
        ry, rx = (ext * rng.uniform(0.03, 0.08)) * rng.uniform(0.7, 1.3, 2)
        th = rng.uniform(0, np.pi)
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        dy, dx = yy - p[0], xx - p[1]
        u, v = dx * np.cos(th) + dy * np.sin(th), -dx * np.sin(th) + dy * np.cos(th)
        r = np.sqrt((u / max(rx, 1)) ** 2 + (v / max(ry, 1)) ** 2)
        prof = np.clip((1.3 - r) / 0.6, 0, 1) ** 1.5  # soft edge
        tex = gaussian_filter(rng.normal(0, 1, (h, w)).astype(np.float32), 1.5)
        tex /= tex.std() + 1e-6
        gain = rng.uniform(0.1, 0.2)
        return img + prof * (gain + 0.03 * tex), {"centre": (float(p[0]), float(p[1])), "radius_px": float(max(rx, ry)), "gain": float(gain)}
