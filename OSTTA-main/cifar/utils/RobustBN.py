import torch
import torch.nn as nn
from copy import deepcopy
"""此文件是用来配置MedBN中值归一化的，但是在 OWTTT + 可学习protos 上起到了副作用"""
def configure_model(model):
    # model.requires_grad_(False)
    model.requires_grad_(True)
    normlayer_names = []

    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.track_running_stats = True
            # module.running_mean = None
            # module.running_var = None

    for name, sub_module in model.named_modules():
        if isinstance(sub_module, nn.BatchNorm1d) or isinstance(sub_module, nn.BatchNorm2d):
            normlayer_names.append(name)
        elif isinstance(sub_module, (nn.LayerNorm, nn.GroupNorm)):
            sub_module.requires_grad_(True)

    for name in normlayer_names:
        bn_layer = get_named_submodule(model, name)
        if isinstance(bn_layer, nn.BatchNorm1d):
            NewBN = RobustBN1d
        elif isinstance(bn_layer, nn.BatchNorm2d):
            NewBN = RobustBN2d
        else:
            raise RuntimeError()

        momentum_bn = NewBN(bn_layer, 0.1)
        momentum_bn.requires_grad_(True)
        set_named_submodule(model, name, momentum_bn)

    return model

def get_named_submodule(model, sub_name: str):
    names = sub_name.split(".")
    module = model
    for name in names:
        module = getattr(module, name)

    return module


def set_named_submodule(model, sub_name, value):
    names = sub_name.split(".")
    module = model
    for i in range(len(names)):
        if i != len(names) - 1:
            module = getattr(module, names[i])

        else:
            setattr(module, names[i], value)


class MomentumBN(nn.Module):
    def __init__(self, bn_layer: nn.BatchNorm2d, momentum):
        super().__init__()
        self.num_features = bn_layer.num_features
        self.momentum = momentum

        if bn_layer.track_running_stats and bn_layer.running_var is not None and bn_layer.running_mean is not None:
            self.register_buffer("running_mean", deepcopy(bn_layer.running_mean))
            self.register_buffer("running_var", deepcopy(bn_layer.running_var))
            self.source_num = bn_layer.num_batches_tracked
        self.weight = deepcopy(bn_layer.weight)
        self.bias = deepcopy(bn_layer.bias)
        self.eps = bn_layer.eps
        self.track_running_stats = bn_layer.track_running_stats
        self.current_mu = None
        self.current_sigma = None

    def forward(self, x):
        raise NotImplementedError


class RobustBN1d(MomentumBN):
    def forward(self, x):
        pass


class RobustBN2d(MomentumBN):

    def forward(self, x):
        y = x.transpose(0, 1)
        return_shape = y.shape
        y = y.contiguous().view(x.size(1), -1)
        mu = y.median(dim=1)[0]
        sigma2 = ((y - mu.repeat(y.shape[1], 1).T) ** 2).sum(dim=1) / y.shape[1]

        if self.training:
            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * sigma2
            if self.track_running_stats and self.running_mean is not None:
                y = y - self.running_mean.view(-1, 1)
            else:
                y = y - mu.view(-1, 1)

            if self.track_running_stats and self.running_var is not None:
                y = y / torch.sqrt(self.running_var.view(-1, 1) + self.eps)
            else:
                y = y / torch.sqrt(sigma2.view(-1, 1) + self.eps)

        elif self.training is not True:
            if self.track_running_stats and self.running_mean is not None:
                y = y - self.running_mean.view(-1, 1)
            else:
                y = y - mu.view(-1, 1)
            if self.track_running_stats and self.running_var is not None:
                y = y / torch.sqrt(self.running_var.view(-1, 1) + self.eps)
            else:
                y = y / torch.sqrt(sigma2.view(-1, 1) + self.eps)

        y = self.weight.view(-1, 1) * y + self.bias.view(-1, 1)
        return y.view(return_shape).transpose(0, 1)