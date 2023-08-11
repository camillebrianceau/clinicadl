from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.cuda.amp import autocast
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader, Sampler

from clinicadl.utils.caps_dataset.data import CapsDataset
from clinicadl.utils.metric_module import MetricModule
from clinicadl.utils.network.network import Network


# TODO: add function to check that the output size of the network corresponds to what is expected to
#  perform the task
class TaskManager:
    def __init__(self, mode: str, n_classes: int = None):
        self.mode = mode
        self.metrics_module = MetricModule(self.evaluation_metrics, n_classes=n_classes)

    @property
    @abstractmethod
    def columns(self, **kwargs):
        """
        List of the columns' names in the TSV file containing the predictions.
        """
        pass

    @property
    @abstractmethod
    def evaluation_metrics(self):
        """
        Evaluation metrics which can be used to evaluate the task.
        """
        pass

    @property
    @abstractmethod
    def save_outputs(self):
        """
        Boolean value indicating if the output values should be saved as tensor for this task.
        """
        pass

    @abstractmethod
    def generate_test_row(
        self, idx: int, data: Dict[str, Any], outputs: Tensor
    ) -> List[List[Any]]:
        """
        Computes an individual row of the prediction TSV file.

        Args:
            idx: index of the individual input and output in the batch.
            data: input batch generated by a DataLoader on a CapsDataset.
            outputs: output batch generated by a forward pass in the model.
        Returns:
            list of items to be contained in a row of the prediction TSV file.
        """
        pass

    @abstractmethod
    def compute_metrics(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute the metrics based on the result of generate_test_row

        Args:
            results_df: results generated based on _results_test_row
        Returns:
            dictionary of metrics
        """
        pass

    @abstractmethod
    def ensemble_prediction(
        self,
        performance_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        selection_threshold: float = None,
        use_labels: bool = True,
        method: str = "soft",
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Compute the results at the image-level by assembling the results on parts of the image.

        Args:
            performance_df: results that need to be assembled.
            validation_df: results on the validation set used to compute the performance
                of each separate part of the image.
            selection_threshold: with soft-voting method, allows to exclude some parts of the image
                if their associated performance is too low.
            use_labels: If True, metrics are computed and the label column values must be different
                from None.
            method: method to assemble the results. Current implementation proposes soft or hard-voting.

        Returns:
            the results and metrics on the image level
        """
        pass

    @staticmethod
    @abstractmethod
    def generate_label_code(df: pd.DataFrame, label: str) -> Optional[Dict[str, int]]:
        """
        Generates a label code that links the output node number to label value.

        Args:
            df: meta-data of the training set.
            label: name of the column containing the labels.
        Returns:
            label_code
        """
        pass

    @staticmethod
    @abstractmethod
    def output_size(
        input_size: Sequence[int], df: pd.DataFrame, label: str
    ) -> Sequence[int]:
        """
        Computes the output_size needed to perform the task.

        Args:
            input_size: size of the input.
            df: meta-data of the training set.
            label: name of the column containing the labels.
        Returns:
            output_size
        """
        pass

    @staticmethod
    @abstractmethod
    def generate_sampler(
        dataset: CapsDataset, sampler_option: str = "random", n_bins: int = 5
    ) -> Sampler:
        """
        Returns sampler according to the wanted options.

        Args:
            dataset: the dataset to sample from.
            sampler_option: choice of sampler.
            n_bins: number of bins to used for a continuous variable (regression task).
        Returns:
             callable given to the training data loader.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_criterion(criterion: str = None) -> _Loss:
        """
        Gives the optimization criterion.
        Must check that it is compatible with the task.

        Args:
            criterion: name of the loss as written in Pytorch.
        Raises:
            ClinicaDLArgumentError: if the criterion is not compatible with the task.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_default_network() -> Network:
        """Returns the default network to use when no architecture is specified."""
        pass

    def test(
        self,
        model: Network,
        dataloader: DataLoader,
        criterion: _Loss,
        monte_carlo: int = 0,
        seed=None,
        use_labels: bool = True,
        amp: bool = False,
        save_reconstruction_tensor=False,
        save_reconstruction_nifti=False,
        save_latent_tensor=False,
        tensor_path=None,
        nifti_path=None,
        latent_tensor_path=None,
    ) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
        """
        Computes the predictions and evaluation metrics.

        Args:
            model: the model trained.
            dataloader: wrapper of a CapsDataset.
            criterion: function to calculate the loss.
            use_labels: If True the true_label will be written in output DataFrame
                and metrics dict will be created.
            monte_carlo: number of monte carlo iterations to perform.
            seed: seed to use for the monte carlo sampling.
            save_reconstruction_tensor: if True, the reconstruction tensor is saved.
            save_reconstruction_nifti: if True, the reconstruction nifti is saved.
            save_latent_tensor: if True, the latent tensor is saved.
            tensor_path: path to the folder where the tensor will be saved.
            nifti_path: path to the folder where the nifti will be saved.
            latent_tensor_path: path to the folder where the latent tensor will be saved.
        Returns:
            the results and metrics on the image level.
        """
        model.eval()
        dataloader.dataset.eval()

        results_df = pd.DataFrame(columns=self.columns)
        mc_results_df = pd.DataFrame()

        total_loss = {}
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                # output = model.predict(data)
                with autocast(enabled=amp):
                    outputs, loss_dict = model.compute_outputs_and_loss(
                        data, criterion, use_labels=use_labels
                    )

                    if i == 0:
                        for loss_component in loss_dict.keys():
                            total_loss[loss_component] = 0
                    for loss_component in total_loss.keys():
                        total_loss[loss_component] += (
                            loss_dict[loss_component].float().item()
                        )
                    try:
                        image = data["image"]
                    except:
                        image = data["data"]
                    # data["data"] = data["data"].unsqueeze(0)
                    participant_id = data["participant_id"][0]
                    session_id = data["session_id"][0]
                    mode_id = data[f"{self.mode}_id"][0]

                # Save tensor
                if save_reconstruction_tensor:
                    reconstruction = outputs["recon_x"].squeeze(0).cpu()
                    input_filename = (
                        f"{participant_id}_{session_id}_{self.mode}-{mode_id}_input.pt"
                    )
                    output_filename = (
                        f"{participant_id}_{session_id}_{self.mode}-{mode_id}_output.pt"
                    )
                    torch.save(image, tensor_path / input_filename)
                    torch.save(reconstruction, tensor_path / output_filename)
                    # logger.debug(f"File saved at {[input_filename, output_filename]}")

                # Save nifti image
                if save_reconstruction_nifti:
                    reconstruction = output["recon_x"].squeeze(0).cpu()
                    # Convert tensor to nifti image with appropriate affine
                    input_nii = nib.Nifti1Image(image[0].detach().numpy(), np.eye(4))
                    output_nii = nib.Nifti1Image(
                        reconstruction[0].detach().numpy(),
                        np.eye(4),
                    )
                    # Create file name according to participant and session id
                    input_filename = f"{participant_id}_{session_id}_image_input.nii.gz"
                    output_filename = (
                        f"{participant_id}_{session_id}_image_output.nii.gz"
                    )
                    nib.save(input_nii, nifti_path / input_filename)
                    nib.save(output_nii, nifti_path / output_filename)

                # Save latent tensor
                if save_latent_tensor:
                    latent = outputs["embedding"].squeeze(0).cpu()
                    output_filename = (
                        f"{participant_id}_{session_id}_{self.mode}-{mode_id}_latent.pt"
                    )
                    torch.save(latent, latent_tensor_path / output_filename)

                # Generate detailed DataFrame
                for idx in range(len(data["participant_id"])):
                    row = self.generate_test_row(idx, data, outputs.float())
                    row_df = pd.DataFrame(row, columns=self.columns)
                    results_df = pd.concat([results_df, row_df])

                del outputs, loss_dict

                if monte_carlo:
                    outputs = model.predict(data, monte_carlo=monte_carlo, seed=seed)

                    for i in range(monte_carlo):
                        output = outputs[i]

                        if save_reconstruction_tensor:
                            reconstruction = output["recon_x"].squeeze(0).cpu()
                            input_filename = f"{participant_id}_{session_id}_{self.mode}-{mode_id}_input.pt"
                            output_filename = f"{participant_id}_{session_id}_{self.mode}-{mode_id}_output-mc{i}.pt"
                            torch.save(image, tensor_path / input_filename)
                            torch.save(reconstruction, tensor_path / output_filename)
                            # logger.debug(
                            #     f"File saved at {[input_filename, output_filename]}"
                            # )

                        if save_reconstruction_nifti:
                            input_nii = nib.Nifti1Image(
                                image[0].detach().numpy(), np.eye(4)
                            )
                            output_nii = nib.Nifti1Image(
                                reconstruction[0].detach().numpy(),
                                np.eye(4),
                            )
                            # Create file name according to participant and session id
                            input_filename = (
                                f"{participant_id}_{session_id}_image_input.nii.gz"
                            )
                            output_filename = f"{participant_id}_{session_id}_image_output-mc{i}.nii.gz"
                            nib.save(input_nii, nifti_path / input_filename)
                            nib.save(output_nii, nifti_path / output_filename)

                        if save_latent_tensor:
                            latent = output["embedding"].squeeze(0).cpu()
                            output_filename = f"{participant_id}_{session_id}_{self.mode}-{mode_id}_latent-mc{i}.pt"
                            torch.save(latent, latent_tensor_path / output_filename)

                        row = self.generate_test_row_monte_carlo(
                            idx, i, data, outputs[i]["recon_x"]
                        )
                        row_df = pd.DataFrame(
                            row, columns=self.columns  # (monte_carlo=monte_carlo)
                        )
                        results_df = pd.concat([results_df, row_df])

                    del outputs

            results_df.reset_index(inplace=True, drop=True)
            results_df[self.evaluation_metrics] = results_df[
                self.evaluation_metrics
            ].apply(pd.to_numeric, axis=1)

            if monte_carlo:
                mc_results_df.reset_index(inplace=True, drop=True)
                mc_results_df[self.evaluation_metrics] = mc_results_df[
                    self.evaluation_metrics
                ].apply(pd.to_numeric, axis=1)

        if not use_labels:
            metrics_df = None
        else:
            metrics_dict = self.compute_metrics(results_df)
            for loss_component in total_loss.keys():
                metrics_dict[loss_component] = total_loss[loss_component]
        torch.cuda.empty_cache()

        return results_df, metrics_df, mc_results_df
