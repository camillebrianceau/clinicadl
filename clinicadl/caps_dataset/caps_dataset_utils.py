import json
from pathlib import Path
from typing import Any, Dict


def read_json(json_path: Path) -> Dict[str, Any]:
    """
    Ensures retro-compatibility between the different versions of ClinicaDL.

    Parameters
    ----------
    json_path: Path
        path to the JSON file summing the parameters of a MAPS.

    Returns
    -------
    A dictionary of training parameters.
    """
    from clinicadl.utils.iotools.utils import path_decoder

    with json_path.open(mode="r") as f:
        parameters = json.load(f, object_hook=path_decoder)
    # Types of retro-compatibility
    # Change arg name: ex network --> model
    # Change arg value: ex for preprocessing: mni --> t1-extensive
    # New arg with default hard-coded value --> discarded_slice --> 20
    retro_change_name = {
        "model": "architecture",
        "multi": "multi_network",
        "minmaxnormalization": "normalize",
        "num_workers": "n_proc",
        "mode": "extract_method",
    }

    retro_add = {
        "optimizer": "Adam",
        "loss": None,
    }

    for old_name, new_name in retro_change_name.items():
        if old_name in parameters:
            parameters[new_name] = parameters[old_name]
            del parameters[old_name]

    for name, value in retro_add.items():
        if name not in parameters:
            parameters[name] = value

    if "extract_method" in parameters:
        parameters["mode"] = parameters["extract_method"]
    # Value changes
    if "use_cpu" in parameters:
        parameters["gpu"] = not parameters["use_cpu"]
        del parameters["use_cpu"]
    if "nondeterministic" in parameters:
        parameters["deterministic"] = not parameters["nondeterministic"]
        del parameters["nondeterministic"]

    from clinicadl.caps_dataset.caps_dataset_config import CapsDatasetConfig

    config = CapsDatasetConfig.from_preprocessing_and_extraction_method(
        extraction=parameters["mode"],
        preprocessing_type=parameters["preprocessing"],
        **parameters,
    )

    file_type = config.preprocessing.get_filetype()

    return parameters
