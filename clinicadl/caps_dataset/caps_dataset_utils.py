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

    # Build preprocessing_dict
    if "preprocessing_dict" not in parameters:
        parameters["preprocessing_dict"] = {"mode": parameters["mode"]}
        preprocessing_options = [
            "preprocessing",
            "use_uncropped_image",
            "prepare_dl",
            "custom_suffix",
            "tracer",
            "suvr_reference_region",
            "patch_size",
            "stride_size",
            "slice_direction",
            "slice_mode",
            "discarded_slices",
            "roi_list",
            "uncropped_roi",
            "roi_custom_suffix",
            "roi_custom_template",
            "roi_custom_mask_pattern",
        ]
        for preprocessing_var in preprocessing_options:
            if preprocessing_var in parameters:
                parameters["preprocessing_dict"][preprocessing_var] = parameters[
                    preprocessing_var
                ]
                del parameters[preprocessing_var]

    # Add missing parameters in previous version of extract
    if "use_uncropped_image" not in parameters["preprocessing_dict"]:
        parameters["preprocessing_dict"]["use_uncropped_image"] = False

    if (
        "prepare_dl" not in parameters["preprocessing_dict"]
        and parameters["mode"] != "image"
    ):
        parameters["preprocessing_dict"]["prepare_dl"] = False

    if (
        parameters["mode"] == "slice"
        and "slice_mode" not in parameters["preprocessing_dict"]
    ):
        parameters["preprocessing_dict"]["slice_mode"] = "rgb"

    if "preprocessing" not in parameters:
        parameters["preprocessing"] = parameters["preprocessing_dict"]["preprocessing"]

    from clinicadl.caps_dataset.caps_dataset_config import CapsDatasetConfig

    config = CapsDatasetConfig.from_preprocessing_and_extraction_method(
        extraction=parameters["mode"],
        preprocessing_type=parameters["preprocessing"],
        **parameters,
    )
    if "file_type" not in parameters["preprocessing_dict"]:
        file_type = config.preprocessing.get_filetype()
        parameters["preprocessing_dict"]["file_type"] = file_type.model_dump()

    return parameters
