import json
from functools import lru_cache
from pathlib import Path

from app.schemas.catalog import Material

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEXTURE_DIR = DATA_DIR / "textures"


def _expand(group: dict) -> list[dict]:
    """One catalog group holds the shared fields; each variant adds an id, a name and its own overrides."""
    shared = {key: value for key, value in group.items() if key not in ("name", "variants", "look")}
    materials = []
    for variant in group["variants"]:
        material = {**shared, "id": variant["id"], "name": f'{group["name"]} – {variant["name"]}', "look": group["look"].format(name=variant["name"])}
        for key in ("render", "rates", "quantity"):
            material[key] = {**shared.get(key, {}), **variant.get(key, {})}
        materials.append(material)
    return materials


@lru_cache
def get_catalog() -> dict[str, Material]:
    groups = json.loads((DATA_DIR / "materials.json").read_text(encoding="utf-8"))
    return {m.id: m for m in map(Material.model_validate, (item for group in groups for item in _expand(group)))}
