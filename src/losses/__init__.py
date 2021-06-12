from utils.utils import ObjectBuilder

from .losses import *

builder = ObjectBuilder()
builder.register_constructor('masked_mse', MaskedMSELoss)
builder.register_constructor('acmnet_loss', ACMNetLoss)
builder.register_constructor('masked_smooth_L1', MaskedSmoothL1Loss)
builder.register_constructor('self_sup_sparse_to_dense_loss', SelfSupSparseToDenseLoss)
builder.register_constructor('edge_aware_smoothness', EdgeAwareSmoothnessLoss)
