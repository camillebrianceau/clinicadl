# coding: utf8
# TODO: create a folder for generate/ prepare_data/ data to deal with capsDataset objects ?
import abc
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from clinicadl.caps_dataset.caps_dataset_config import CapsDatasetConfig
from clinicadl.caps_dataset.data_config import DataConfig
from clinicadl.caps_dataset.dataloader_config import DataLoaderConfig
from clinicadl.caps_dataset.extraction.config import (
    ExtractionConfig,
    ExtractionImageConfig,
    ExtractionPatchConfig,
    ExtractionROIConfig,
    ExtractionSliceConfig,
)
from clinicadl.caps_dataset.preprocessing.config import PreprocessingConfig
from clinicadl.caps_dataset.utils import (
    CapsDatasetOutput,
    get_preprocessing_and_mode_from_json,
)
from clinicadl.prepare_data.prepare_data_utils import (
    compute_discarded_slices,
    extract_patch_path,
    extract_patch_tensor,
    extract_roi_path,
    extract_roi_tensor,
    extract_slice_path,
    extract_slice_tensor,
    find_mask_path,
)
from clinicadl.transforms.config import TransformsConfig
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
    ClinicaDLConfigurationError,
    ClinicaDLTSVError,
)

logger = getLogger("clinicadl")


#################################
# Datasets loaders
#################################
class CapsDataset(Dataset):
    """Abstract class for all derived CapsDatasets."""

    def __init__(
        self,
        data: DataConfig,
        preprocessing: PreprocessingConfig,
        extraction: ExtractionConfig,
        transforms: TransformsConfig,
        label_presence: bool,
    ):
        self.label_presence = label_presence
        self.eval_mode = False
        self.elem_per_image = self.num_elem_per_image()
        self.size = self[0]["image"].size()

        self.preprocessing = preprocessing
        self.extraction = extraction
        self.transforms = transforms
        self.data = data
        self.caps_dict = data.caps_dict

        if not hasattr(self, "elem_index"):
            raise AttributeError(
                "Child class of CapsDataset must set elem_index attribute."
            )
        if not hasattr(self, "mode"):
            raise AttributeError("Child class of CapsDataset, must set mode attribute.")

        self.df = pd.read_csv(data.tsv_path, sep="\t")

        mandatory_col = {
            "participant_id",
            "session_id",
            "cohort",
        }

        # if label_presence and self.config.data.label is not None:
        #     mandatory_col.add(self.config.data.label)

        if not mandatory_col.issubset(set(self.df.columns.values)):
            raise ClinicaDLTSVError(
                f"the data file is not in the correct format."
                f"Columns should include {mandatory_col}"
            )

    @classmethod
    def from_extract_json(
        cls,
        data: DataConfig,
        extract_json: str,
        transforms: TransformsConfig,
        label_presence: bool,
    ):
        extract_json_path = data.caps_directory / "tensor_extraction" / extract_json
        if not extract_json_path.is_file():
            raise ClinicaDLConfigurationError(f"Could not find {extract_json_path}")

        preprocessing, extraction = get_preprocessing_and_mode_from_json(
            extract_json_path
        )

        return cls(
            data=data,
            preprocessing=preprocessing,
            extraction=extraction,
            transforms=transforms,
            label_presence=label_presence,
        )

    @property
    @abc.abstractmethod
    def elem_index(self):
        pass

    def label_fn(self, target: Union[str, float, int]) -> Union[float, int, None]:
        """
        Returns the label value usable in criterion.

        Args:
            target: value of the target.
        Returns:
            label: value of the label usable in criterion.
        """
        # Reconstruction case (no label)
        if self.data.label is None:
            return None
        # Regression case (no label code)
        elif self.data.label_code is None:
            return np.float32([target])
        # Classification case (label + label_code dict)
        else:
            return self.data.label_code[str(target)]

    def domain_fn(self, target: Union[str, float, int]) -> Union[float, int]:
        """
        Returns the label value usable in criterion.

        """
        domain_code = {"t1": 0, "flair": 1}
        return domain_code[str(target)]

    def __len__(self) -> int:
        return len(self.df) * self.elem_per_image

    def _get_image_path(self, participant: str, session: str, cohort: str) -> Path:
        """
        Gets the path to the tensor image (*.pt)

        Args:
            participant: ID of the participant.
            session: ID of the session.
            cohort: Name of the cohort.
        Returns:
            image_path: path to the tensor containing the whole image.
        """
        from clinicadl.utils.iotools.clinica_utils import clinicadl_file_reader

        # Try to find .nii.gz file
        try:
            folder, file_type = self.preprocessing.compute_folder_and_file_type()

            results = clinicadl_file_reader(
                [participant],
                [session],
                self.caps_dict[cohort],
                file_type.model_dump(),
            )
            logger.debug(f"clinicadl_file_reader output: {results}")
            filepath = Path(results[0][0])
            image_filename = filepath.name.replace(".nii.gz", ".pt")

            image_dir = (
                self.caps_dict[cohort]
                / "subjects"
                / participant
                / session
                / "deeplearning_prepare_data"
                / "image_based"
                / folder
            )
            image_path = image_dir / image_filename
        # Try to find .pt file
        except ClinicaDLCAPSError:
            folder, file_type = self.preprocessing.compute_folder_and_file_type()
            file_type.pattern = file_type.pattern.replace(".nii.gz", ".pt")
            results = clinicadl_file_reader(
                [participant],
                [session],
                self.caps_dict[cohort],
                file_type.model_dump(),
            )
            filepath = results[0]
            image_path = Path(filepath[0])

        return image_path

    def _get_meta_data(
        self, idx: int
    ) -> Tuple[str, str, str, int, Union[float, int, None]]:
        """
        Gets all meta data necessary to compute the path with _get_image_path

        Args:
            idx (int): row number of the meta-data contained in self.df
        Returns:
            participant (str): ID of the participant.
            session (str): ID of the session.
            cohort (str): Name of the cohort.
            elem_index (int): Index of the part of the image.
            label (str or float or int): value of the label to be used in criterion.
        """
        image_idx = idx // self.elem_per_image
        participant = self.df.at[image_idx, "participant_id"]
        session = self.df.at[image_idx, "session_id"]
        cohort = self.df.at[image_idx, "cohort"]

        if self.elem_index is None:
            elem_idx = idx % self.elem_per_image
        else:
            elem_idx = self.elem_index
        if self.label_presence and self.data.label is not None:
            target = self.df.at[image_idx, self.data.label]
            label = self.label_fn(target)
        else:
            label = -1

        # if "domain" in self.df.columns:
        #     domain = self.df.at[image_idx, "domain"]
        #     domain = self.domain_fn(domain)
        # else:
        #     domain = ""  # TO MODIFY
        return participant, session, cohort, elem_idx, label  # , domain

    def _get_full_image(self) -> torch.Tensor:
        """
        Allows to get the an example of the image mode corresponding to the dataset.
        Useful to compute the number of elements if mode != image.

        Returns:
            image tensor of the full image first image.
        """
        import nibabel as nib

        from clinicadl.utils.iotools.clinica_utils import clinicadl_file_reader

        participant_id = self.df.at[0, "participant_id"]
        session_id = self.df.at[0, "session_id"]
        cohort = self.df.at[0, "cohort"]

        try:
            image_path = self._get_image_path(participant_id, session_id, cohort)
            image = torch.load(image_path, weights_only=True)
        except IndexError:
            file_type = self.preprocessing.file_type
            results = clinicadl_file_reader(
                [participant_id],
                [session_id],
                self.caps_dict[cohort],
                file_type.model_dump(),
            )
            image_nii = nib.loadsave.load(results[0])
            image_np = image_nii.get_fdata()
            image = ToTensor()(image_np)

        return image

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Gets the sample containing all the information needed for training and testing tasks.

        Args:
            idx: row number of the meta-data contained in self.df
        Returns:
            dictionary with following items:
                - "image" (torch.Tensor): the input given to the model,
                - "label" (int or float): the label used in criterion,
                - "participant_id" (str): ID of the participant,
                - "session_id" (str): ID of the session,
                - f"{self.mode}_id" (int): number of the element,
                - "image_path": path to the image loaded in CAPS.

        """
        pass

    @abc.abstractmethod
    def num_elem_per_image(self) -> int:
        """Computes the number of elements per image based on the full image."""
        pass

    def eval(self):
        """Put the dataset on evaluation mode (data augmentation is not performed)."""
        self.eval_mode = True
        return self

    def train(self):
        """Put the dataset on training mode (data augmentation is performed)."""
        self.eval_mode = False
        return self


class CapsDatasetImage(CapsDataset):
    """Dataset of MRI organized in a CAPS folder."""

    def __init__(
        self,
        data: DataConfig,
        preprocessing: PreprocessingConfig,
        extraction: ExtractionImageConfig,
        transforms: TransformsConfig,
        label_presence: bool = True,
    ):
        """
        Args:
            caps_directory: Directory of all the images.
            data_file: Path to the tsv file or DataFrame containing the subject/session list.
            train_transformations: Optional transform to be applied only on training mode.
            label_presence: If True the diagnosis will be extracted from the given DataFrame.
            label: Name of the column in data_df containing the label.
            label_code: label code that links the output node number to label value.
            all_transformations: Optional transform to be applied during training and evaluation.
            multi_cohort: If True caps_directory is the path to a TSV file linking cohort names and paths.

        """

        self.mode = "image"
        self.label_presence = label_presence
        super().__init__(
            data=data,
            preprocessing=preprocessing,
            extraction=extraction,
            transforms=transforms,
            label_presence=label_presence,
        )

    @property
    def elem_index(self):
        return None

    def __getitem__(self, idx):
        participant, session, cohort, _, label = self._get_meta_data(idx)

        image_path = self._get_image_path(participant, session, cohort)
        image = torch.load(image_path, weights_only=True)

        train_trf, trf = self.transforms.get_transforms()

        image = trf(image)
        if self.transforms.train_transformations and not self.eval_mode:
            image = train_trf(image)

        sample = CapsDatasetOutput(
            image=image,
            label=label,
            participant_id=participant,
            session_id=session,
            image_id=0,
            image_path=image_path,
            # domain= domain,
            mode=ExtractionMethod.IMAGE,
        )

        return sample

    def num_elem_per_image(self):
        return 1


class CapsDatasetPatch(CapsDataset):
    def __init__(
        self,
        data: DataConfig,
        preprocessing: PreprocessingConfig,
        extraction: ExtractionPatchConfig,
        transforms: TransformsConfig,
        patch_index: Optional[int] = None,
        label_presence: bool = True,
    ):
        """
        caps_directory: Directory of all the images.
        data_file: Path to the tsv file or DataFrame containing the subject/session list.
        train_transformations: Optional transform to be applied only on training mode.
        """
        self.patch_index = patch_index
        self.label_presence = label_presence
        self.extraction = extraction
        self.preprocessing = preprocessing
        self.transforms = transforms
        super().__init__(
            data=data,
            preprocessing=preprocessing,
            extraction=extraction,
            transforms=transforms,
            label_presence=label_presence,
        )

    @property
    def elem_index(self):
        return self.patch_index

    def __getitem__(self, idx):
        participant, session, cohort, patch_idx, label = self._get_meta_data(idx)
        image_path = self._get_image_path(participant, session, cohort)

        if self.extraction.save_features:
            patch_dir = image_path.parent.as_posix().replace(
                "image_based", f"{self.extraction.extract_method}_based"
            )
            patch_filename = extract_patch_path(
                image_path,
                self.extraction.patch_size,
                self.extraction.stride_size,
                patch_idx,
            )
            patch_tensor = torch.load(
                Path(patch_dir).resolve() / patch_filename, weights_only=True
            )

        else:
            image = torch.load(image_path, weights_only=True)
            patch_tensor = extract_patch_tensor(
                image,
                self.extraction.patch_size,
                self.extraction.stride_size,
                patch_idx,
            )

        train_trf, trf = self.transforms.get_transforms()
        patch_tensor = trf(patch_tensor)

        if self.transforms.train_transformations and not self.eval_mode:
            patch_tensor = train_trf(patch_tensor)

        sample = CapsDatasetOutput(
            image=patch_tensor,
            label=label,
            participant_id=participant,
            session_id=session,
            image_id=patch_idx,
            mode=ExtractionMethod.PATCH,
        )

        return sample

    def num_elem_per_image(self):
        if self.elem_index is not None:
            return 1

        image = self._get_full_image()

        patches_tensor = (
            image.unfold(
                1,
                self.extraction.patch_size,
                self.extraction.stride_size,
            )
            .unfold(
                2,
                self.extraction.patch_size,
                self.extraction.stride_size,
            )
            .unfold(
                3,
                self.extraction.patch_size,
                self.extraction.stride_size,
            )
            .contiguous()
        )
        patches_tensor = patches_tensor.view(
            -1,
            self.extraction.patch_size,
            self.extraction.patch_size,
            self.extraction.patch_size,
        )
        num_patches = patches_tensor.shape[0]
        return num_patches


class CapsDatasetRoi(CapsDataset):
    def __init__(
        self,
        data: DataConfig,
        preprocessing: PreprocessingConfig,
        extraction: ExtractionROIConfig,
        transforms: TransformsConfig,
        roi_index: Optional[int] = None,
        label_presence: bool = True,
    ):
        """
        Args:
            caps_directory: Directory of all the images.
            data_file: Path to the tsv file or DataFrame containing the subject/session list.
            roi_index: If a value is given the same region will be extracted for each image.
                else the dataset will load all the regions possible for one image.
            train_transformations: Optional transform to be applied only on training mode.
            label_presence: If True the diagnosis will be extracted from the given DataFrame.
            label: Name of the column in data_df containing the label.
            label_code: label code that links the output node number to label value.
            all_transformations: Optional transform to be applied during training and evaluation.
            multi_cohort: If True caps_directory is the path to a TSV file linking cohort names and paths.

        """
        self.roi_index = roi_index
        self.label_presence = label_presence

        self.extraction = extraction
        self.preprocessing = preprocessing
        self.transforms = transforms

        self.mask_paths, self.mask_arrays = self._get_mask_paths_and_tensors()
        super().__init__(
            data=data,
            preprocessing=preprocessing,
            extraction=extraction,
            transforms=transforms,
            label_presence=label_presence,
        )

    @property
    def elem_index(self):
        return self.roi_index

    def __getitem__(self, idx):
        participant, session, cohort, roi_idx, label = self._get_meta_data(idx)
        image_path = self._get_image_path(participant, session, cohort)

        if self.extraction.roi_list is None:
            raise NotImplementedError(
                "Default regions are not available anymore in ClinicaDL. "
                "Please define appropriate masks and give a roi_list."
            )

        if self.extraction.save_features:
            mask_path = self.mask_paths[roi_idx]
            roi_dir = image_path.parent.as_posix().replace(
                "image_based", f"{self.extraction.extract_method}_based"
            )
            roi_filename = extract_roi_path(
                image_path, mask_path, self.extraction.roi_uncrop_output
            )
            roi_tensor = torch.load(Path(roi_dir) / roi_filename, weights_only=True)

        else:
            image = torch.load(image_path, weights_only=True)
            mask_array = self.mask_arrays[roi_idx]
            roi_tensor = extract_roi_tensor(
                image, mask_array, self.extraction.roi_uncrop_output
            )

        train_trf, trf = self.transforms.get_transforms()

        roi_tensor = trf(roi_tensor)

        if self.transforms.train_transformations and not self.eval_mode:
            roi_tensor = train_trf(roi_tensor)

        sample = CapsDatasetOutput(
            image=roi_tensor,
            label=label,
            participant_id=participant,
            session_id=session,
            image_id=roi_idx,
            mode=ExtractionMethod.ROI,
        )

        return sample

    def num_elem_per_image(self):
        if self.elem_index is not None:
            return 1
        if self.extraction.roi_list is None:
            return 2
        else:
            return len(self.extraction.roi_list)

    def _get_mask_paths_and_tensors(
        self,
    ) -> Tuple[List[Path], List]:
        """Loads the masks necessary to regions extraction"""
        import nibabel as nib

        caps_dict = self.caps_dict
        if len(caps_dict) > 1:
            caps_directory = Path(caps_dict[next(iter(caps_dict))])
            logger.warning(
                f"The equality of masks is not assessed for multi-cohort training. "
                f"The masks stored in {caps_directory} will be used."
            )

        try:
            preprocessing_ = self.preprocessing.preprocessing
        except NotImplementedError:
            print(
                f"Template of preprocessing {self.preprocessing.preprocessing.value} "
                f"is not defined."
            )
        # Find template name and pattern
        if preprocessing_ == Preprocessing.CUSTOM:
            template_name = self.extraction.roi_custom_template
            if template_name is None:
                raise ValueError(
                    "Please provide a name for the template when preprocessing is `custom`."
                )

            pattern = self.extraction.roi_custom_mask_pattern
            if pattern is None:
                raise ValueError(
                    "Please provide a pattern for the masks when preprocessing is `custom`."
                )

        else:
            for template_ in Template:
                if preprocessing_.name == template_.name:
                    template_name = template_

            for pattern_ in Pattern:
                if preprocessing_.name == pattern_.name:
                    pattern = pattern_

        mask_location = self.data.caps_directory / "masks" / f"tpl-{template_name}"

        mask_paths, mask_arrays = list(), list()
        for roi in self.extraction.roi_list:
            logger.info(f"Find mask for roi {roi}.")
            mask_path, desc = find_mask_path(mask_location, roi, pattern, True)
            if mask_path is None:
                raise FileNotFoundError(desc)
            mask_nii = nib.loadsave.load(mask_path)
            mask_paths.append(Path(mask_path))
            mask_arrays.append(mask_nii.get_fdata())

        return mask_paths, mask_arrays


class CapsDatasetSlice(CapsDataset):
    def __init__(
        self,
        data: DataConfig,
        preprocessing: PreprocessingConfig,
        extraction: ExtractionSliceConfig,
        transforms: TransformsConfig,
        slice_index: Optional[int] = None,
        label_presence: bool = True,
    ):
        """
        Args:
            caps_directory: Directory of all the images.
            data_file: Path to the tsv file or DataFrame containing the subject/session list.
            slice_index: If a value is given the same slice will be extracted for each image.
                else the dataset will load all the slices possible for one image.
            train_transformations: Optional transform to be applied only on training mode.
            label_presence: If True the diagnosis will be extracted from the given DataFrame.
            label: Name of the column in data_df containing the label.
            label_code: label code that links the output node number to label value.
            all_transformations: Optional transform to be applied during training and evaluation.
            multi_cohort: If True caps_directory is the path to a TSV file linking cohort names and paths.
        """
        self.slice_index = slice_index
        self.label_presence = label_presence

        self.extraction = extraction
        self.preprocessing = preprocessing
        self.transforms = transforms

        super().__init__(
            data=data,
            preprocessing=preprocessing,
            extraction=extraction,
            transforms=transforms,
            label_presence=label_presence,
        )

    @property
    def elem_index(self):
        return self.slice_index

    def __getitem__(self, idx):
        participant, session, cohort, slice_idx, label = self._get_meta_data(idx)
        slice_idx = slice_idx + self.extraction.discarded_slices[0]
        image_path = self._get_image_path(participant, session, cohort)

        if self.extraction.save_features:
            slice_dir = image_path.parent.as_posix().replace(
                "image_based", f"{self.extraction.extract_method}_based"
            )
            slice_filename = extract_slice_path(
                image_path,
                self.extraction.slice_direction,
                self.extraction.slice_mode,
                slice_idx,
            )
            slice_tensor = torch.load(
                Path(slice_dir) / slice_filename, weights_only=True
            )

        else:
            image_path = self._get_image_path(participant, session, cohort)
            image = torch.load(image_path, weights_only=True)
            slice_tensor = extract_slice_tensor(
                image,
                self.extraction.slice_direction,
                self.extraction.slice_mode,
                slice_idx,
            )

        train_trf, trf = self.transforms.get_transforms()

        slice_tensor = trf(slice_tensor)

        if self.transforms.train_transformations and not self.eval_mode:
            slice_tensor = train_trf(slice_tensor)

        sample = CapsDatasetOutput(
            image=slice_tensor,
            label=label,
            participant_id=participant,
            session_id=session,
            image_id=slice_idx,
            mode=ExtractionMethod.SLICE,
        )

        return sample

    def num_elem_per_image(self):
        if self.elem_index is not None:
            return 1

        if self.extraction.num_slices is not None:
            return self.extraction.num_slices

        image = self._get_full_image()
        return (
            image.size(int(self.extraction.slice_direction) + 1)
            - self.extraction.discarded_slices[0]
            - self.extraction.discarded_slices[1]
        )


def return_dataset(
    data: DataConfig,
    preprocessing: PreprocessingConfig,
    extraction: ExtractionConfig,
    transforms_config: TransformsConfig,
    cnn_index: Optional[int] = None,
    label_presence: bool = True,
) -> CapsDataset:
    """
    Return appropriate Dataset according to given options.
    Args:
        input_dir: path to a directory containing a CAPS structure.
        data_df: List subjects, sessions and diagnoses.
        train_transformations: Optional transform to be applied during training only.
        all_transformations: Optional transform to be applied during training and evaluation.
        label: Name of the column in data_df containing the label.
        label_code: label code that links the output node number to label value.
        cnn_index: Index of the CNN in a multi-CNN paradigm (optional).
        label_presence: If True the diagnosis will be extracted from the given DataFrame.
        multi_cohort: If True caps_directory is the path to a TSV file linking cohort names and paths.

    Returns:
         the corresponding dataset.
    """
    if cnn_index is not None and extraction.extract_method == ExtractionMethod.IMAGE:
        raise NotImplementedError(
            f"Multi-CNN is not implemented for {extraction.extract_method.value} mode."
        )

    if isinstance(extraction, ExtractionImageConfig):
        return CapsDatasetImage(
            data=data,
            extraction=extraction,
            preprocessing=preprocessing,
            transforms=transforms_config,
            label_presence=label_presence,
        )

    elif isinstance(extraction, ExtractionPatchConfig):
        return CapsDatasetPatch(
            data=data,
            extraction=extraction,
            preprocessing=preprocessing,
            transforms=transforms_config,
            label_presence=label_presence,
        )

    elif isinstance(extraction, ExtractionROIConfig):
        return CapsDatasetRoi(
            data=data,
            extraction=extraction,
            preprocessing=preprocessing,
            transforms=transforms_config,
            label_presence=label_presence,
        )

    elif isinstance(extraction, ExtractionSliceConfig):
        return CapsDatasetSlice(
            data=data,
            extraction=extraction,
            preprocessing=preprocessing,
            transforms=transforms_config,
            label_presence=label_presence,
        )
    else:
        raise NotImplementedError(
            f"Mode {extraction.extract_method.value} is not implemented."
        )
