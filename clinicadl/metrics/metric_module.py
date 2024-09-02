from logging import getLogger
from typing import Callable, Dict, List, Union

import numpy as np
from scipy.stats import bootstrap

metric_optimum = {
    "MAE": "min",
    "RMSE": "min",
    "accuracy": "max",
    "sensitivity": "max",
    "specificity": "max",
    "PPV": "max",
    "NPV": "max",
    "F1_score": "max",
    "BA": "max",
    "PSNR": "max",
    "SSIM": "max",
    "LNCC": "max",
    "loss": "min",
}

logger = getLogger("clinicadl.metric")


class MetricModule:
    def __init__(self, metrics: List[str], n_classes: int = 2):
        self.n_classes = n_classes
        self.metrics = self._initialize_metrics(metrics)

    def _initialize_metrics(self, metrics: List[str]) -> Dict[str, Callable]:
        """Initialize the metrics dictionary with the appropriate functions."""
        metric_methods = {
            method_name[8:]: getattr(self, method_name)
            for method_name in dir(self)
            if method_name.startswith("compute_")
        }

        initialized_metrics = {}
        for metric in metrics:
            if metric.lower() in metric_methods:
                initialized_metrics[metric] = metric_methods[metric.lower()]
            else:
                raise NotImplementedError(
                    f"The metric {metric} is not implemented in the module."
                )
        return initialized_metrics

    def apply(
        self, y_: List[float], y_pred_: List[float], report_ci: bool
    ) -> Dict[str, Union[float, List[float]]]:
        """Calculate metrics based on the list of true and predicted labels."""
        if not y_ or not y_pred_:
            return {}

        y = np.array(y_)
        y_pred = np.array(y_pred_)
        results = {}

        for metric_name, metric_fn in self.metrics.items():
            class_numbers = (
                range(self.n_classes)
                if "class_number" in metric_fn.__code__.co_varnames
                else [0]
            )
            self._calculate_metric(
                y, y_pred, metric_name, metric_fn, class_numbers, results, report_ci
            )

        return results

    def _calculate_metric(
        self,
        y: np.ndarray,
        y_pred: np.ndarray,
        metric_name: str,
        metric_fn: Callable,
        class_numbers: range,
        results: Dict[str, Union[float, List[float]]],
        report_ci: bool,
    ):
        """Helper function to calculate metrics and optionally confidence intervals."""
        for class_number in class_numbers:
            metric_result = metric_fn(y, y_pred, class_number)
            metric_key = (
                f"{metric_name}-{class_number}"
                if len(class_numbers) > 1
                else metric_name
            )

            if report_ci and len(y) >= 2:
                lower_ci, upper_ci, se = self._calculate_confidence_interval(
                    y, y_pred, metric_fn, class_number
                )
                results.setdefault("Metric_names", []).append(metric_key)
                results.setdefault("Metric_values", []).append(metric_result)
                results.setdefault("Lower_CI", []).append(lower_ci)
                results.setdefault("Upper_CI", []).append(upper_ci)
                results.setdefault("SE", []).append(se)
            else:
                results[metric_key] = metric_result

    @staticmethod
    def _calculate_confidence_interval(
        y: np.ndarray, y_pred: np.ndarray, metric_fn: Callable, class_number: int
    ) -> tuple:
        """Calculate confidence intervals and standard error for a metric."""
        res = bootstrap(
            (y, y_pred),
            lambda y, y_pred: metric_fn(y, y_pred, class_number),
            n_resamples=3000,
            confidence_level=0.95,
            method="percentile",
            paired=True,
        )
        return (
            res.confidence_interval.low,
            res.confidence_interval.high,
            res.standard_error,
        )

    @staticmethod
    def compute_mae(y, y_pred, *args):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
        Returns:
            (float) mean absolute error
        """

        return np.mean(np.abs(y - y_pred))

    @staticmethod
    def compute_rmse(y, y_pred, *args):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
        Returns:
            (float) root mean squared error
        """

        return np.sqrt(np.mean(np.square(y - y_pred)))

    @staticmethod
    def compute_r2_score(y, y_pred, *args):
        """
        Calculate the R-squared (coefficient of determination) score.

        Args:
            y (List): List of actual labels
            y_pred (List): List of predicted labels

        Returns:
            (float) R-squared score
        """
        mean_y = np.mean(y)
        total_sum_squares = np.sum((y - mean_y) ** 2)
        residual_sum_squares = np.sum((y - y_pred) ** 2)
        r2_score = (
            1 - (residual_sum_squares / total_sum_squares)
            if total_sum_squares != 0
            else 0
        )

        return r2_score

    @staticmethod
    def compute_accuracy(y, y_pred, *args):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
        Returns:
            (float) accuracy
        """
        true = np.sum(y_pred == y)

        return true / len(y)

    @staticmethod
    def compute_sensitivity(y, y_pred, class_number):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
            class_number (int): number of the class studied
        Returns:
            (float) sensitivity
        """
        true_positive = np.sum((y_pred == class_number) & (y == class_number))
        false_negative = np.sum((y_pred != class_number) & (y == class_number))

        if (true_positive + false_negative) != 0:
            return true_positive / (true_positive + false_negative)
        else:
            return 0.0

    @staticmethod
    def compute_specificity(y, y_pred, class_number):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
            class_number (int): number of the class studied
        Returns:
            (float) specificity
        """
        true_negative = np.sum((y_pred != class_number) & (y != class_number))
        false_positive = np.sum((y_pred == class_number) & (y != class_number))

        if (false_positive + true_negative) != 0:
            return true_negative / (false_positive + true_negative)
        else:
            return 0.0

    @staticmethod
    def compute_ppv(y, y_pred, class_number):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
            class_number (int): number of the class studied
        Returns:
            (float) positive predictive value
        """
        true_positive = np.sum((y_pred == class_number) & (y == class_number))
        false_positive = np.sum((y_pred == class_number) & (y != class_number))

        if (true_positive + false_positive) != 0:
            return true_positive / (true_positive + false_positive)
        else:
            return 0.0

    @staticmethod
    def compute_npv(y, y_pred, class_number):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
            class_number (int): number of the class studied
        Returns:
            (float) negative predictive value
        """
        true_negative = np.sum((y_pred != class_number) & (y != class_number))
        false_negative = np.sum((y_pred != class_number) & (y == class_number))

        if (true_negative + false_negative) != 0:
            return true_negative / (true_negative + false_negative)
        else:
            return 0.0

    @staticmethod
    def compute_f1_score(y, y_pred, class_number):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
            class_number (int): number of the class studied
        Returns:
            (float) F1 score
        """

        precision = MetricModule.compute_ppv(y, y_pred, class_number)
        recall = MetricModule.compute_sensitivity(y, y_pred, class_number)

        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) != 0
            else 0
        )

        return f1_score

    @staticmethod
    def compute_ba(y, y_pred, class_number):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
            class_number (int): number of the class studied
        Returns:
            (float) balanced accuracy
        """

        return (
            MetricModule.compute_sensitivity(y, y_pred, class_number)
            + MetricModule.compute_specificity(y, y_pred, class_number)
        ) / 2

    @staticmethod
    def compute_mcc(y, y_pred, class_number):
        """
        Calculate the Matthews correlation coefficient (MCC) for a specific class.

        Args:
            y (List): List of actual labels
            y_pred (List): List of predicted labels
            class_number (int): Number of the class studied

        Returns:
            (float) Matthews correlation coefficient for the specified class
        """
        true_positive = np.sum((y_pred == class_number) & (y == class_number))
        true_negative = np.sum((y_pred != class_number) & (y != class_number))
        false_positive = np.sum((y_pred == class_number) & (y != class_number))
        false_negative = np.sum((y_pred != class_number) & (y == class_number))
        denominator = np.sqrt(
            (true_positive + false_positive)
            * (true_positive + false_negative)
            * (true_negative + false_positive)
            * (true_negative + false_negative)
        )
        mcc = (
            (true_positive * true_negative - false_positive * false_negative)
            / denominator
            if denominator != 0
            else 0
        )
        return mcc

    @staticmethod
    def compute_mk(y, y_pred, class_number):
        """
        Calculate Markedness (MK) for a specific class.

        Args:
            y (List): List of actual labels
            y_pred (List): List of predicted labels
            class_number (int): Number of the class studied

        Returns:
            (float) Markedness for the specified class
        """
        precision = MetricModule.compute_ppv(y, y_pred, class_number)
        npv = MetricModule.compute_npv(y, y_pred, class_number)
        mk = precision + npv - 1
        return mk

    @staticmethod
    def compute_lr_plus(y, y_pred, class_number):
        """
        Calculate Positive Likelihood Ratio (LR+).

        Args:
            y (List): List of actual labels
            y_pred (List): List of predicted labels
            class_number (int): Number of the class studied

        Returns:
            (float) Positive Likelihood Ratio
        """
        sensitivity = MetricModule.compute_sensitivity(y, y_pred, class_number)
        specificity = MetricModule.compute_specificity(y, y_pred, class_number)
        lr_plus = sensitivity / (1 - specificity) if (1 - specificity) != 0 else 0
        return lr_plus

    @staticmethod
    def compute_lr_minus(y, y_pred, class_number):
        """
        Calculate Negative Likelihood Ratio (LR-).

        Args:
            y (List): List of actual labels
            y_pred (List): List of predicted labels
            class_number (int): Number of the class studied

        Returns:
            (float) Negative Likelihood Ratio
        """
        sensitivity = MetricModule.compute_sensitivity(y, y_pred, class_number)
        specificity = MetricModule.compute_specificity(y, y_pred, class_number)
        lr_minus = (1 - sensitivity) / specificity if specificity != 0 else 0
        return lr_minus

    @staticmethod
    def compute_confusion_matrix(y, y_pred, *args):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
        Returns:
            (Dict[str:float]) confusion matrix
        """
        true_positive = np.sum((y_pred == 1) & (y == 1))
        true_negative = np.sum((y_pred == 0) & (y == 0))
        false_positive = np.sum((y_pred == 1) & (y == 0))
        false_negative = np.sum((y_pred == 0) & (y == 1))

        return {
            "tp": true_positive,
            "tn": true_negative,
            "fp": false_positive,
            "fn": false_negative,
        }

    @staticmethod
    def compute_ssim(y, y_pred, *args):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
        Returns:
            (float) SSIM
        """
        from clinicadl.utils.pytorch_ssim import ssim, ssim3D

        if len(y) == 3:
            return ssim(y, y_pred).item()
        else:
            return ssim3D(y, y_pred).item()

    @staticmethod
    def compute_psnr(y, y_pred, *args):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
        Returns:
            (float) PSNR
        """
        from skimage.metrics import peak_signal_noise_ratio

        return peak_signal_noise_ratio(y, y_pred)

    @staticmethod
    def compute_lncc(y, y_pred, *args):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
        Returns:
            (float) LNCC
        """
        from scipy.ndimage import gaussian_filter

        sigma = 2

        mean1 = gaussian_filter(y, sigma)
        mean2 = gaussian_filter(y_pred, sigma)

        mean12 = gaussian_filter(
            y * y_pred, sigma
        )  # the * operator is term by term product
        mean11 = gaussian_filter(y * y, sigma)
        mean22 = gaussian_filter(y_pred * y_pred, sigma)

        covar12 = mean12 - (mean1 * mean2)
        var1 = np.sqrt(mean11 - (mean1 * mean1))
        var2 = np.sqrt(mean22 - (mean2 * mean2))

        lcc_matrix = np.maximum(covar12 / (var1 * var2), 0)
        return np.mean(lcc_matrix)


class RetainBest:
    """
    A class to retain the best and overfitting values for a set of wanted metrics.
    """

    def __init__(self, selection_metrics: List[str], n_classes: int = 0):
        self.selection_metrics = selection_metrics

        if "loss" in selection_metrics:
            selection_metrics.remove("loss")
            metric_module = MetricModule(selection_metrics)
            selection_metrics.append("loss")
        else:
            metric_module = MetricModule(selection_metrics)

        implemented_metrics = set(metric_optimum.keys())
        if not set(self.selection_metrics).issubset(implemented_metrics):
            raise NotImplementedError(
                f"The selection metrics {self.selection_metrics} are not all implemented. "
                f"Available metrics are {implemented_metrics}."
            )
        self.best_metrics = dict()
        for selection in self.selection_metrics:
            if n_classes > 2:
                metric_fn = metric_module.metrics[selection]
                metric_args = list(metric_fn.__code__.co_varnames)
                if "class_number" in metric_args:
                    for class_number in range(n_classes):
                        self.set_optimum(f"{selection}-{class_number}")
                else:
                    self.set_optimum(selection)
            else:
                self.set_optimum(selection)

    def set_optimum(self, selection: str):
        if metric_optimum[selection] == "min":
            self.best_metrics[selection] = np.inf
        elif metric_optimum[selection] == "max":
            self.best_metrics[selection] = -np.inf
        else:
            raise ValueError(
                f"Objective {metric_optimum[selection]} unknown for metric {selection}."
                f"Please choose between 'min' and 'max'."
            )

    def step(self, metrics_valid: Dict[str, float]) -> Dict[str, bool]:
        """
        Computes for each metric if this is the best value ever seen.

        Args:
            metrics_valid: metrics computed on the validation set
        Returns:
            metric is associated to True if it is the best value ever seen.
        """

        metrics_dict = dict()
        for selection in self.selection_metrics:
            if metric_optimum[selection] == "min":
                metrics_dict[selection] = (
                    metrics_valid[selection] < self.best_metrics[selection]
                )
                self.best_metrics[selection] = min(
                    metrics_valid[selection], self.best_metrics[selection]
                )

            else:
                metrics_dict[selection] = (
                    metrics_valid[selection] > self.best_metrics[selection]
                )
                self.best_metrics[selection] = max(
                    metrics_valid[selection], self.best_metrics[selection]
                )

        return metrics_dict
