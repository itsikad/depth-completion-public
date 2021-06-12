from utils.utils import ObjectBuilder

from .acmnet import ACMNet
from .guidenet import GuideNet
from .sparse_to_dense import SparseToDense

builder = ObjectBuilder()
builder.register_constructor('acmnet', ACMNet)
builder.register_constructor('guidenet', GuideNet)
builder.register_constructor('sparse_to_dense', SparseToDense)
