from dataclasses import dataclass

import numpy as np

from app.schemas.project import REGION_PRIORITY, Region, RegionType
from app.services.imaging import decode_mask

_RANK = {region_type: rank for rank, region_type in enumerate(REGION_PRIORITY)}


@dataclass(frozen=True)
class RegionMask:
    id: str
    type: RegionType
    plane_id: str
    mask: np.ndarray


def effective_masks(regions: list[Region]) -> list[RegionMask]:
    """Decode masks and cut overlaps by priority, so a wall never includes its windows or railings."""
    occupied: np.ndarray | None = None
    resolved = []
    for region in sorted(regions, key=lambda r: _RANK[r.type]):
        mask = decode_mask(region.mask_png)
        if occupied is None:
            occupied = np.zeros_like(mask)
        resolved.append(RegionMask(region.id, region.type, region.plane_id, mask & ~occupied))
        occupied |= mask
    return resolved
