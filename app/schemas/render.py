from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RenderResult:
    jpeg: bytes
    method: Literal["ai", "preview"]
    note: str | None = None
