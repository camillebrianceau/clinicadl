from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch.nn as nn
from monai.networks.blocks import Convolution
from monai.networks.layers.utils import get_act_layer, get_pool_layer
from monai.utils.misc import ensure_tuple

from .layers.utils import (
    ActFunction,
    ActivationParameters,
    ConvNormalizationParameters,
    ConvNormLayer,
    ConvParameters,
    NormLayer,
    PoolingLayer,
    PoolingParameters,
    SingleLayerPoolingParameters,
)
from .utils import (
    calculate_conv_out_shape,
    calculate_pool_out_shape,
    check_adn_ordering,
    check_norm_layer,
    check_pool_indices,
    ensure_list_of_tuples,
)


class ConvEncoder(nn.Sequential):
    """
    Fully convolutional encoder network with convolutional, pooling, normalization, activation
    and dropout layers.

    Parameters
    ----------
    spatial_dims : int
        number of spatial dimensions of the input image.
    in_channels : int
        number of channels in the input image.
    channels : Sequence[int]
        sequence of integers stating the output channels of each convolutional layer. Thus, this
        parameter also controls the number of convolutional layers.
    kernel_size : ConvParameters (optional, default=3)
        the kernel size of the convolutional layers. Can be an integer, a tuple or a list.\n
        If integer, the value will be used for all layers and all dimensions.\n
        If tuple (of integers), it will be interpreted as the values for each dimension. These values
        will be used for all the layers.\n
        If list (of tuples or integers), it will be interpreted as the kernel sizes for each layer.
        The length of the list must be equal to the number of convolutional layers (i.e. `len(channels)`).
    stride : ConvParameters (optional, default=1)
        the stride of the convolutional layers. Can be an integer, a tuple or a list.\n
        If integer, the value will be used for all layers and all dimensions.\n
        If tuple (of integers), it will be interpreted as the values for each dimension. These values
        will be used for all the layers.\n
        If list (of tuples or integers), it will be interpreted as the strides for each layer.
        The length of the list must be equal to the number of convolutional layers (i.e. `len(channels)`).
    padding : ConvParameters (optional, default=0)
        the padding of the convolutional layers. Can be an integer, a tuple or a list.\n
        If integer, the value will be used for all layers and all dimensions.\n
        If tuple (of integers), it will be interpreted as the values for each dimension. These values
        will be used for all the layers.\n
        If list (of tuples or integers), it will be interpreted as the paddings for each layer.
        The length of the list must be equal to the number of convolutional layers (i.e. `len(channels)`).
    dilation : ConvParameters (optional, default=1)
        the dilation factor of the convolutional layers. Can be an integer, a tuple or a list.\n
        If integer, the value will be used for all layers and all dimensions.\n
        If tuple (of integers), it will be interpreted as the values for each dimension. These values
        will be used for all the layers.\n
        If list (of tuples or integers), it will be interpreted as the dilations for each layer.
        The length of the list must be equal to the number of convolutional layers (i.e. `len(channels)`).
    pooling : Optional[PoolingParameters] (optional, default=(PoolingLayer.MAX, {"kernel_size": 2}))
        the pooling mode and the arguments of the pooling layer, passed as `(pooling_mode, arguments)`.
        If None, no pooling will be performed in the network.\n
        `pooling_mode` can be either `max`, `avg`, `adaptivemax` or `adaptiveavg`. Please refer to PyTorch's [documentation]
        (https://pytorch.org/docs/stable/nn.html#pooling-layers) to know the mandatory and optional arguments.\n
        If a list is passed, it will be understood as `(pooling_mode, arguments)` for each pooling layer.
    pooling_indices : Optional[Sequence[int]] (optional, default=None)
        indices of the convolutional layers after which pooling should be performed.
        If None, no pooling will be performed. An index equal to -1 will be understood as an unpooling layer before
        the first convolution.
    act : Optional[ActivationParameters] (optional, default=ActFunction.PRELU)
        the activation function used after a convolutional layer, and optionally its arguments.
        Should be passed as `activation_name` or `(activation_name, arguments)`. If None, no activation will be used.\n
        `activation_name` can be any value in {`celu`, `elu`, `gelu`, `leakyrelu`, `logsoftmax`, `mish`, `prelu`,
        `relu`, `relu6`, `selu`, `sigmoid`, `softmax`, `tanh`}. Please refer to PyTorch's [activationfunctions]
        (https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity) to know the optional
        arguments for each of them.
    output_act : Optional[ActivationParameters] (optional, default=None)
        a potential activation layer applied to the output of the network. Should be pass in the same way as `act`.
        If None, no last activation will be applied.
    norm : Optional[ConvNormalizationParameters] (optional, default=NormLayer.INSTANCE)
        the normalization type used after a convolutional layer, and optionally the arguments of the normalization
        layer. Should be passed as `norm_type` or `(norm_type, parameters)`. If None, no normalization will be
        performed.\n
        `norm_type` can be any value in {`batch`, `group`, `instance`, `syncbatch`}. Please refer to PyTorch's
        [normalization layers](https://pytorch.org/docs/stable/nn.html#normalization-layers) to know the mandatory and
        optional arguments for each of them.\n
        Please note that arguments `num_channels`, `num_features` of the normalization layer
        should not be passed, as they are automatically inferred from the output of the previous layer in the network.
    dropout : Optional[float] (optional, default=None)
        dropout ratio. If None, no dropout.
    bias : bool (optional, default=True)
        whether to have a bias term in convolutions.
    adn_ordering : str (optional, default="NDA")
        order of operations `Activation`, `Dropout` and `Normalization` after a convolutional layer (except the last
        one).
        For example if "ND" is passed, `Normalization` and then `Dropout` will be performed (without `Activation`).\n
        Note: ADN will not be applied after the last convolution.

    Examples
    --------
    >>> ConvEncoder(
            spatial_dims=2,
            in_channels=1,
            channels=[2, 4, 8],
            kernel_size=(3, 5),
            stride=1,
            padding=[1, (0, 1), 0],
            dilation=1,
            pooling=[("max", {"kernel_size": 2}), ("avg", {"kernel_size": 2})],
            pooling_indices=[0, 1],
            act="elu",
            output_act="relu",
            norm=("batch", {"eps": 1e-05}),
            dropout=0.1,
            bias=True,
            adn_ordering="NDA",
        )
    ConvEncoder(
        (layer0): Convolution(
            (conv): Conv2d(1, 2, kernel_size=(3, 5), stride=(1, 1), padding=(1, 1))
            (adn): ADN(
                (N): BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (D): Dropout(p=0.1, inplace=False)
                (A): ELU(alpha=1.0)
            )
        )
        (pool0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (layer1): Convolution(
            (conv): Conv2d(2, 4, kernel_size=(3, 5), stride=(1, 1), padding=(0, 1))
            (adn): ADN(
                (N): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (D): Dropout(p=0.1, inplace=False)
                (A): ELU(alpha=1.0)
            )
        )
        (pool1): AvgPool2d(kernel_size=2, stride=2, padding=0)
        (layer2): Convolution(
            (conv): Conv2d(4, 8, kernel_size=(3, 5), stride=(1, 1))
        )
        (output_act): ReLU()
    )

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        channels: Sequence[int],
        kernel_size: ConvParameters = 3,
        stride: ConvParameters = 1,
        padding: ConvParameters = 0,
        dilation: ConvParameters = 1,
        pooling: Optional[PoolingParameters] = (
            PoolingLayer.MAX,
            {"kernel_size": 2},
        ),
        pooling_indices: Optional[Sequence[int]] = None,
        act: Optional[ActivationParameters] = ActFunction.PRELU,
        output_act: Optional[ActivationParameters] = None,
        norm: Optional[ConvNormalizationParameters] = ConvNormLayer.INSTANCE,
        dropout: Optional[float] = None,
        bias: bool = True,
        adn_ordering: str = "NDA",
        _input_size: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()

        self._current_size = _input_size if _input_size else None
        self._size_details = [self._current_size] if _input_size else None

        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.channels = ensure_tuple(channels)
        self.n_layers = len(self.channels)

        self.kernel_size = ensure_list_of_tuples(
            kernel_size, self.spatial_dims, self.n_layers, "kernel_size"
        )
        self.stride = ensure_list_of_tuples(
            stride, self.spatial_dims, self.n_layers, "stride"
        )
        self.padding = ensure_list_of_tuples(
            padding, self.spatial_dims, self.n_layers, "padding"
        )
        self.dilation = ensure_list_of_tuples(
            dilation, self.spatial_dims, self.n_layers, "dilation"
        )

        self.pooling_indices = check_pool_indices(pooling_indices, self.n_layers)
        self.pooling = self._check_pool_layers(pooling)
        self.act = act
        self.norm = check_norm_layer(norm)
        if self.norm == NormLayer.LAYER:
            raise ValueError("Layer normalization not implemented in ConvEncoder.")
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = check_adn_ordering(adn_ordering)

        n_poolings = 0
        if self.pooling and -1 in self.pooling_indices:
            pooling_layer = self._get_pool_layer(self.pooling[n_poolings])
            self.add_module("init_pool", pooling_layer)
            n_poolings += 1

        echannel = self.in_channels
        for i, (c, k, s, p, d) in enumerate(
            zip(
                self.channels,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
            )
        ):
            conv_layer = self._get_conv_layer(
                in_channels=echannel,
                out_channels=c,
                kernel_size=k,
                stride=s,
                padding=p,
                dilation=d,
                is_last=(i == len(channels) - 1),
            )
            self.add_module(f"layer{i}", conv_layer)
            echannel = c  # use the output channel number as the input for the next loop
            if self.pooling and i in self.pooling_indices:
                pooling_layer = self._get_pool_layer(self.pooling[n_poolings])
                self.add_module(f"pool{i}", pooling_layer)
                n_poolings += 1

        self.output_act = get_act_layer(output_act) if output_act else None

    @property
    def final_size(self):
        """
        To know the size of an image at the end of the network.
        """
        return self._current_size

    @property
    def size_details(self):
        """
        To know the sizes of intermediate images.
        """
        return self._size_details

    @final_size.setter
    def final_size(self, fct: Callable[[Tuple[int, ...]], Tuple[int, ...]]):
        """
        Takes as input the function used to update the current image size.
        """
        if self._current_size is not None:
            self._current_size = fct(self._current_size)
            self._size_details.append(self._current_size)
        self._check_size()

    def _get_conv_layer(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        padding: Tuple[int, ...],
        dilation: Tuple[int, ...],
        is_last: bool,
    ) -> Convolution:
        """
        Gets the parametrized Convolution-ADN block and updates the current output size.
        """
        self.final_size = lambda size: calculate_conv_out_shape(
            size, kernel_size, stride, padding, dilation
        )

        return Convolution(
            conv_only=is_last,
            spatial_dims=self.spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=stride,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )

    def _get_pool_layer(self, pooling: SingleLayerPoolingParameters) -> nn.Module:
        """
        Gets the parametrized pooling layer and updates the current output size.
        """
        pool_layer = get_pool_layer(pooling, spatial_dims=self.spatial_dims)
        old_size = self.final_size
        self.final_size = lambda size: calculate_pool_out_shape(
            pool_mode=pooling[0], in_shape=size, **pool_layer.__dict__
        )

        if (
            self.final_size is not None
            and (np.array(old_size) < np.array(self.final_size)).any()
        ):
            raise ValueError(
                f"You passed {pooling} as a pooling layer. But before this layer, the size of the image "
                f"was {old_size}. So, pooling can't be performed."
            )

        return pool_layer

    def _check_size(self) -> None:
        """
        Checks that image size never reaches 0.
        """
        if self._current_size is not None and (np.array(self._current_size) <= 0).any():
            raise ValueError(
                f"Failed to build the network. An image of size 0 or less has been reached. Stopped at:\n {self}"
            )

    @classmethod
    def _check_single_pool_layer(
        cls, pooling: SingleLayerPoolingParameters
    ) -> SingleLayerPoolingParameters:
        """
        Checks pooling arguments for a single pooling layer.
        """
        if not isinstance(pooling, tuple) or len(pooling) != 2:
            raise ValueError(
                "pooling must be a double (or a list of doubles) with first the type of pooling and then the parameters "
                f"of the pooling layer in a dict. Got {pooling}"
            )
        pooling_type = PoolingLayer(pooling[0])
        args = pooling[1]
        if not isinstance(args, dict):
            raise ValueError(
                f"The arguments of the pooling layer must be passed in a dict. Got {args}"
            )
        if (
            pooling_type == PoolingLayer.MAX or pooling_type == PoolingLayer.AVG
        ) and "kernel_size" not in args:
            raise ValueError(
                f"For {pooling_type} pooling mode, `kernel_size` argument must be passed. "
                f"Got {args}"
            )
        elif (
            pooling_type == PoolingLayer.ADAPT_AVG
            or pooling_type == PoolingLayer.ADAPT_MAX
        ) and "output_size" not in args:
            raise ValueError(
                f"For {pooling_type} pooling mode, `output_size` argument must be passed. "
                f"Got {args}"
            )

    def _check_pool_layers(
        self, pooling: PoolingParameters
    ) -> List[SingleLayerPoolingParameters]:
        """
        Check argument pooling.
        """
        if pooling is None:
            return pooling
        if isinstance(pooling, list):
            for pool_layer in pooling:
                self._check_single_pool_layer(pool_layer)
            if len(pooling) != len(self.pooling_indices):
                raise ValueError(
                    "If you pass a list for pooling, the size of that list must match "
                    f"the size of pooling_indices. Got: pooling={pooling} and "
                    f"pooling_indices={self.pooling_indices}"
                )
        elif isinstance(pooling, tuple):
            self._check_single_pool_layer(pooling)
            pooling = [pooling] * len(self.pooling_indices)
        else:
            raise ValueError(
                f"pooling can be either None, a double (string, dictionary) or a list of such doubles. Got {pooling}"
            )

        return pooling