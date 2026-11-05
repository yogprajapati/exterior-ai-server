from pydantic import BaseModel, ConfigDict, Field

MAX_ADJUST_PX = 40


class MaskData(BaseModel):
    model_config = ConfigDict(ser_json_bytes="base64", val_json_bytes="base64")

    mask_png: bytes


class GrowRequest(MaskData):
    pixels: int = Field(ge=-MAX_ADJUST_PX, le=MAX_ADJUST_PX)


class EraseRequest(MaskData):
    erase_png: bytes
