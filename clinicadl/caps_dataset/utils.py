# coding: utf8
# TODO: create a folder for generate/ prepare_data/ data to deal with capsDataset objects ?
import abc
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict
from torch.utils.data import Dataset

from clinicadl.caps_dataset.caps_dataset_config import get_extraction, get_preprocessing
from clinicadl.utils.enum import (
    ExtractionMethod,
    Pattern,
    Preprocessing,
    SliceDirection,
    SliceMode,
    Template,
)
from clinicadl.utils.exceptions import (
    ClinicaDLCAPSError,
    ClinicaDLTSVError,
)

logger = getLogger("clinicadl")


def get_preprocessing_and_mode_from_json(json_path: Path):
    """
    Extracts the preprocessing and mode from a json file.

    Parameters
    ----------
    json_path : Path
        Path to the json file containing the preprocessing and mode.

    Returns
    -------
    Tuple[Preprocessing, SliceMode]
        The preprocessing and mode extracted from the json file.
    """
    from clinicadl.utils.iotools.utils import read_json

    dict_ = read_json(json_path)
    return get_preprocessing_and_mode_from_parameters(**dict_)


def get_preprocessing_and_mode_from_parameters(**kwargs):
    """
    Extracts the preprocessing and mode from a json file.

    Returns
    -------
    Tuple[Preprocessing, SliceMode]
        The preprocessing and mode extracted from the json file.
    """

    if "preprocessing_dict" in kwargs:
        kwargs = kwargs["preprocessing_dict"]

    print(kwargs)
    preprocessing = Preprocessing(kwargs["preprocessing"])
    mode = ExtractionMethod(kwargs["extract_method"])
    return get_preprocessing(preprocessing)(**kwargs), get_extraction(mode)(**kwargs)


class CapsDatasetOutput(BaseModel):
    image: torch.Tensor
    participant_id: Union[int, str]
    session_id: Union[int, str]
    label: Optional[Union[float, int]] = None
    image_id: Optional[Union[float, str]] = None
    image_path: Optional[Path] = None
    # domain: Optional[int]=None
    mode: ExtractionMethod

    # pydantic config
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)
