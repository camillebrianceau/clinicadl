from logging import getLogger
from time import time
from typing import List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, field_validator
from pydantic.types import NonNegativeInt

from clinicadl.prepare_data.prepare_data_utils import compute_discarded_slices
from clinicadl.utils.enum import (
    ExtractionMethod,
    SliceDirection,
    SliceMode,
)
from clinicadl.utils.iotools.clinica_utils import FileType

logger = getLogger("clinicadl.preprocessing_config")


class ExtractionConfig(BaseModel):
    """
    Abstract config class for the Extraction procedure.
    """

    extract_method: ExtractionMethod
    save_features: bool = False
    extract_json: Optional[str] = None
    use_uncropped_image: bool = True

    # pydantic config
    model_config = ConfigDict(validate_assignment=True)

    @field_validator("extract_json", mode="before")
    def compute_extract_json(cls, v: str):
        if v is None:
            return f"extract_{int(time())}.json"
        elif not v.endswith(".json"):
            return f"{v}.json"
        else:
            return v


class ExtractionImageConfig(ExtractionConfig):
    extract_method: ExtractionMethod = ExtractionMethod.IMAGE


class ExtractionPatchConfig(ExtractionConfig):
    patch_size: int = 50
    stride_size: int = 50
    extract_method: ExtractionMethod = ExtractionMethod.PATCH


class ExtractionSliceConfig(ExtractionConfig):
    slice_direction: SliceDirection = SliceDirection.SAGITTAL
    slice_mode: SliceMode = SliceMode.RGB
    num_slices: Optional[NonNegativeInt] = None
    discarded_slices: Union[int, tuple] = (0,)
    extract_method: ExtractionMethod = ExtractionMethod.SLICE

    @field_validator("slice_direction", mode="before")
    def check_slice_direction(cls, v: str):
        if isinstance(v, int):
            return SliceDirection(str(v))

    @field_validator("discarded_slices", mode="before")
    def compute_discarded_slice(cls, v: Union[int, tuple]) -> tuple[int, int]:
        return compute_discarded_slices(v)


class ExtractionROIConfig(ExtractionConfig):
    roi_list: List[str] = []
    roi_uncrop_output: bool = False
    roi_custom_template: str = ""
    roi_custom_pattern: str = ""
    roi_custom_suffix: str = ""
    roi_custom_mask_pattern: str = ""
    roi_background_value: int = 0
    extract_method: ExtractionMethod = ExtractionMethod.ROI
