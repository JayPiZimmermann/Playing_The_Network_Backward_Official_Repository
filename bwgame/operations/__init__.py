"""
DC Decomposition Operations Module

Forward: [4*batch] = [pos; neg; 0; 0]
Backward: [4*batch] = [delta_pp; delta_np; delta_pn; delta_nn]

This module contains layer-level DC operations (linear, conv, relu, etc.).
For model-level utilities (patch_model, prepare_model_for_dc, make_dc_compatible),
import from bwgame directly.
"""

from .base import (
    cat4, split4,
    init_catted, init_pos_neg, InputMode,
    recenter_dc, reconstruct_output,
    DC_ENABLED, DC_ORIGINAL_FORWARD, DC_IS_OUTPUT_LAYER, DC_RELU_LAMBDA,
    split_input_4, make_output_4, split_grad_4, make_grad_4, make_input_4,
    # Sensitivity utilities
    Sensitivities, extract_sensitivities,
    # Context manager API
    DCForward, dc_forward,
)

from .add import Add, patch_add, unpatch_add

from .linear import patch_linear, unpatch_linear, DCLinearFunction
from .conv import (
    patch_conv2d, unpatch_conv2d, DCConv2dFunction,
    patch_conv1d, unpatch_conv1d, DCConv1dFunction,
)
from .conv_transpose import (
    patch_conv_transpose1d, patch_conv_transpose2d,
    unpatch_conv_transpose1d, unpatch_conv_transpose2d,
    DCConvTranspose1dFunction, DCConvTranspose2dFunction,
)
from .relu import patch_relu, unpatch_relu, DCReLUFunction
from .gelu import patch_gelu, unpatch_gelu, DCGELUFunction, forward_gelu
from .swish import patch_swish, unpatch_swish, DCSwishFunction, forward_swish
from .batchnorm import patch_batchnorm, unpatch_batchnorm, DCBatchNormFunction
from .maxpool import (
    patch_maxpool1d, patch_maxpool2d,
    unpatch_maxpool1d, unpatch_maxpool2d,
    DCMaxPool1dFunction, DCMaxPool2dFunction,
)
from .avgpool import (
    patch_avgpool1d, patch_avgpool2d,
    unpatch_avgpool1d, unpatch_avgpool2d,
    patch_adaptive_avgpool1d, patch_adaptive_avgpool2d,
    unpatch_adaptive_avgpool1d, unpatch_adaptive_avgpool2d,
)
from .shape_ops import (
    patch_flatten, unpatch_flatten, DCFlattenFunction,
    patch_unflatten, unpatch_unflatten, DCUnflattenFunction,
    Reshape, patch_reshape, unpatch_reshape, DCReshapeFunction,
    View, patch_view, unpatch_view,
    Squeeze, patch_squeeze, unpatch_squeeze, DCSqueezeFunction,
    Unsqueeze, patch_unsqueeze, unpatch_unsqueeze, DCUnsqueezeFunction,
    Transpose, patch_transpose, unpatch_transpose, DCTransposeFunction,
    Permute, patch_permute, unpatch_permute, DCPermuteFunction,
    patch_dropout, unpatch_dropout, DCDropoutFunction,
)
from .layernorm import patch_layernorm, unpatch_layernorm
from .softmax import patch_softmax, unpatch_softmax
from .tensor_ops import (
    DCSplit, DCChunk, DCCat, DCSlice
)
from .embedding import patch_embedding, unpatch_embedding
from .gather import Gather, patch_gather, unpatch_gather
from .mean import Mean, patch_mean, unpatch_mean
from .sum import Sum, patch_sum, unpatch_sum
from .contiguous import Contiguous, patch_contiguous, unpatch_contiguous

__all__ = [
    # Base
    'cat4', 'split4',
    'init_catted', 'init_pos_neg', 'InputMode',
    'recenter_dc', 'reconstruct_output',
    'DC_ENABLED', 'DC_ORIGINAL_FORWARD', 'DC_IS_OUTPUT_LAYER', 'DC_RELU_LAMBDA',
    'split_input_4', 'make_output_4', 'split_grad_4', 'make_grad_4', 'make_input_4',
    # Sensitivity utilities
    'Sensitivities', 'extract_sensitivities',
    # Context manager API
    'DCForward', 'dc_forward',
    # Add
    'Add', 'patch_add', 'unpatch_add',
    # Linear
    'patch_linear', 'unpatch_linear', 'DCLinearFunction',
    # Conv
    'patch_conv2d', 'unpatch_conv2d', 'DCConv2dFunction',
    'patch_conv1d', 'unpatch_conv1d', 'DCConv1dFunction',
    # ConvTranspose
    'patch_conv_transpose1d', 'patch_conv_transpose2d',
    'unpatch_conv_transpose1d', 'unpatch_conv_transpose2d',
    'DCConvTranspose1dFunction', 'DCConvTranspose2dFunction',
    # ReLU
    'patch_relu', 'unpatch_relu', 'DCReLUFunction',
    # GELU
    'patch_gelu', 'unpatch_gelu', 'DCGELUFunction', 'forward_gelu',
    # SiLU/Swish
    'patch_swish', 'unpatch_swish', 'DCSwishFunction', 'forward_swish',
    # BatchNorm
    'patch_batchnorm', 'unpatch_batchnorm', 'DCBatchNormFunction',
    # MaxPool
    'patch_maxpool1d', 'patch_maxpool2d',
    'unpatch_maxpool1d', 'unpatch_maxpool2d',
    'DCMaxPool1dFunction', 'DCMaxPool2dFunction',
    # AvgPool
    'patch_avgpool1d', 'patch_avgpool2d',
    'unpatch_avgpool1d', 'unpatch_avgpool2d',
    'patch_adaptive_avgpool1d', 'patch_adaptive_avgpool2d',
    'unpatch_adaptive_avgpool1d', 'unpatch_adaptive_avgpool2d',
    # Shape ops
    'patch_flatten', 'unpatch_flatten', 'DCFlattenFunction',
    'patch_unflatten', 'unpatch_unflatten', 'DCUnflattenFunction',
    'Reshape', 'patch_reshape', 'unpatch_reshape', 'DCReshapeFunction',
    'View', 'patch_view', 'unpatch_view',
    'Squeeze', 'patch_squeeze', 'unpatch_squeeze', 'DCSqueezeFunction',
    'Unsqueeze', 'patch_unsqueeze', 'unpatch_unsqueeze', 'DCUnsqueezeFunction',
    'Transpose', 'patch_transpose', 'unpatch_transpose', 'DCTransposeFunction',
    'Permute', 'patch_permute', 'unpatch_permute', 'DCPermuteFunction',
    'patch_dropout', 'unpatch_dropout', 'DCDropoutFunction',
    # LayerNorm
    'patch_layernorm', 'unpatch_layernorm',
    # Softmax
    'patch_softmax', 'unpatch_softmax',
    # Tensor ops
    'DCSplit', 'DCChunk', 'DCCat', 'DCSlice',
    # Individual patches
    'patch_embedding', 'unpatch_embedding',
    'Gather', 'patch_gather', 'unpatch_gather', 
    'Mean', 'patch_mean', 'unpatch_mean',
    'Sum', 'patch_sum', 'unpatch_sum',
    'Contiguous', 'patch_contiguous', 'unpatch_contiguous',
]
