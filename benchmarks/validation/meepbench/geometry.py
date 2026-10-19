"""Layout sources -> polygons per GDS layer + named ports."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .spec import Geometry, LayerKey, Port


def _from_component(component, extension: float):
    import gdsfactory as gf
    if extension > 0 and len(component.ports):
        component = gf.components.extend_ports(component, length=extension)
    polygons: Dict[LayerKey, List[np.ndarray]] = {}
    for layer, items in component.get_polygons_points(by="tuple").items():
        polygons[tuple(layer)] = [np.asarray(p, dtype=float) for p in items]
    # Report the ports of the *unextended* device: sources and monitors sit at
    # the device boundary, the extensions only carry the guides through PML.
    return polygons


def load(geometry: Geometry) -> Tuple[Dict[LayerKey, List[np.ndarray]], Dict[str, Port],
                                       Tuple[float, float, float, float]]:
    """Return (polygons by layer, ports, device bbox (xmin, ymin, xmax, ymax))."""
    if geometry.polygons is not None:
        out = geometry.polygons()
        polygons, ports = out[0], out[1]
        if len(out) > 2:                     # explicit device bbox
            return polygons, ports, tuple(out[2])
        items = [np.asarray(p) for group in polygons.values() for p in group]
        if not items:                        # empty layout: domain_xy sets the cell
            return polygons, ports, (0.0, 0.0, 0.0, 0.0)
        pts = np.concatenate(items)
        bbox = (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())
        return polygons, ports, bbox

    import gdsfactory as gf
    gf.gpdk.PDK.activate()
    if geometry.gds is not None:
        component = gf.import_gds(geometry.gds, cellname=geometry.cell)
    elif callable(geometry.gdsfactory):
        component = geometry.gdsfactory(**geometry.params)
    else:
        component = getattr(gf.components, geometry.gdsfactory)(**geometry.params)
    ports = {p.name: Port(center=(float(p.center[0]), float(p.center[1])),
                          orientation=float(p.orientation), width=float(p.width))
             for p in component.ports}
    bbox = (float(component.xmin), float(component.ymin), float(component.xmax),
            float(component.ymax))
    return _from_component(component, geometry.port_extension), ports, bbox
