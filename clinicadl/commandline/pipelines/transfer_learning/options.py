import click

from clinicadl.config.config.transfer_learning import TransferLearningConfig
from clinicadl.config.config_utils import get_default_from_config_class as get_default
from clinicadl.config.config_utils import get_type_from_config_class as get_type

nb_unfrozen_layer = click.option(
    "-nul",
    "--nb_unfrozen_layer",
    type=get_type("nb_unfrozen_layer", TransferLearningConfig),
    default=get_default("nb_unfrozen_layer", TransferLearningConfig),
    help="Number of layer that will be retrain during training. For example, if it is 2, the last two layers of the model will not be freezed.",
    show_default=True,
)
transfer_path = click.option(
    "-tp",
    "--transfer_path",
    type=get_type("transfer_path", TransferLearningConfig),
    default=get_default("transfer_path", TransferLearningConfig),
    help="Path of to a MAPS used for transfer learning.",
    show_default=True,
)
transfer_selection_metric = click.option(
    "-tsm",
    "--transfer_selection_metric",
    type=get_type("transfer_selection_metric", TransferLearningConfig),
    default=get_default("transfer_selection_metric", TransferLearningConfig),
    help="Metric used to select the model for transfer learning in the MAPS defined by transfer_path.",
    show_default=True,
)