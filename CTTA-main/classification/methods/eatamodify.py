"""
Builds upon: https://github.com/mr-eggplant/EATA
Corresponding paper: https://arxiv.org/abs/2204.02610
"""

import math
import logging
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F

from methods.base import TTAMethod
from datasets.data_loading import get_source_loader

from classification.augmentations.transforms_cotta import get_tta_transforms
from classification.utils import compute_os_variance

logger = logging.getLogger(__name__)


class EATAMODIFY(TTAMethod):
    """EATA adapts a model by entropy minimization during testing.
    Once EATAed, a model adapts itself by updating on every forward.
    """
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        self.num_samples_update_1 = 0  # number of samples after First filtering, exclude unreliable samples
        self.num_samples_update_2 = 0  # number of samples after Second filtering, exclude both unreliable and redundant samples
        self.e_margin = math.log(self.num_classes) * 0.40   # hyper-parameter E_0 (Eqn. 3)
        self.d_margin = cfg.EATA.D_MARGIN   # hyperparameter \epsilon for cosine similarity thresholding (Eqn. 5)

        self.current_model_probs = None # the moving average of probability vector (Eqn. 4)
        self.fisher_alpha = cfg.EATA.FISHER_ALPHA # trade-off \beta for two losses (Eqn. 8)
        self.transform = get_tta_transforms(self.dataset_name)

        self.count = 0
        self.mean_dict = {}
        self.var_dict = {}
        self.mean_perLayer = []
        self.var_perLayer = []
        # if self.fisher_alpha > 0.0:
        #     # compute fisher informatrix
        #     batch_size_src = cfg.TEST.BATCH_SIZE if cfg.TEST.BATCH_SIZE > 1 else cfg.TEST.WINDOW_LENGTH
        #     _, fisher_loader = get_source_loader(dataset_name=cfg.CORRUPTION.DATASET,
        #                                          root_dir=cfg.DATA_DIR, adaptation=cfg.MODEL.ADAPTATION,
        #                                          batch_size=batch_size_src, ckpt_path=cfg.CKPT_PATH,
        #                                          num_samples=cfg.EATA.NUM_SAMPLES)
        #
        #     ewc_optimizer = torch.optim.SGD(self.params, 0.001)
        #     self.fishers = {} # fisher regularizer items for anti-forgetting, need to be calculated pre model adaptation (Eqn. 9)
        #     train_loss_fn = nn.CrossEntropyLoss().cuda()
        #     self.model.eval()
        #     for iter_, batch in enumerate(fisher_loader, start=1):
        #         images = batch[0].cuda(non_blocking=True)
        #         ema_out = self.model(images)
        #         _, targets = ema_out.max(1)
        #         loss = train_loss_fn(ema_out, targets)
        #         loss.backward()
        #         for name, param in model.named_parameters():
        #             if param.grad is not None:
        #                 if iter_ > 1:
        #                     fisher = param.grad.data.clone().detach() ** 2 + self.fishers[name][0]
        #                 else:
        #                     fisher = param.grad.data.clone().detach() ** 2
        #                 if iter_ == len(fisher_loader):
        #                     fisher = fisher / iter_
        #                 self.fishers.update({name: [fisher, param.data.clone().detach()]})
        #         ewc_optimizer.zero_grad()
        #     logger.info("compute fisher matrices finished")
        #     del ewc_optimizer
        # else:
        #     self.fishers = None

 # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        self.model.train()
        imgs_test = x[0]
        outputs = self.model(imgs_test)


        # 遍历模型的每个模块
        for name, module in self.model.named_modules():
            if isinstance(module, RobustBN2d):
                if name not in self.mean_dict:
                    self.mean_dict[name] = []
                    self.var_dict[name] = []

                # 计算当前批次的均值和方差
                mean = module.source_mean
                var = module.source_var

                # 将当前批次的均值和方差添加到列表中
                self.mean_dict[name].append(mean)
                self.var_dict[name].append(var)



        # entropys = softmax_entropy(outputs)
        #
        # # filter unreliable samples
        # filter_ids_1 = torch.where(entropys < self.e_margin)
        # ids1 = filter_ids_1
        # ids2 = torch.where(ids1[0] > -0.1)
        # entropys = entropys[filter_ids_1]
        #
        # # filter redundant samples
        # if self.current_model_probs is not None:
        #     cosine_similarities = F.cosine_similarity(self.current_model_probs.unsqueeze(dim=0),
        #                                               outputs[filter_ids_1].softmax(1), dim=1)
        #     filter_ids_2 = torch.where(torch.abs(cosine_similarities) < self.d_margin)
        #     entropys = entropys[filter_ids_2]
        #     ids2 = filter_ids_2
        #     updated_probs = update_model_probs(self.current_model_probs, outputs[filter_ids_1][filter_ids_2].softmax(1))
        # else:
        #     updated_probs = update_model_probs(self.current_model_probs, outputs[filter_ids_1].softmax(1))
        # coeff = 1 / (torch.exp(entropys.clone().detach() - self.e_margin))
        #
        # # implementation version 1, compute loss, all samples backward (some unselected are masked)
        # entropys = entropys.mul(coeff)  # reweight entropy losses for diff. samples
        # loss = entropys.mean(0)
        #
        # if self.fishers is not None:
        #     ewc_loss = 0
        #     for name, param in self.model.named_parameters():
        #         if name in self.fishers:
        #             ewc_loss += self.fisher_alpha * (self.fishers[name][0] * (param - self.fishers[name][1]) ** 2).sum()
        #     loss += ewc_loss
        # if imgs_test[ids1][ids2].size(0) != 0:
        #     loss.backward()
        #     self.optimizer.step()
        # else:
        #     outputs = outputs.detach()
        # self.optimizer.zero_grad()
        #
        # self.num_samples_update_1 += filter_ids_1[0].size(0)
        # self.num_samples_update_2 += entropys.size(0)
        # self.reset_model_probs(updated_probs)


        # if self.count == 100:
        #     for name in self.mean_dict:
        #         mean_list = torch.stack(self.mean_dict[name])
        #         var_list = torch.stack(self.var_dict[name])
        #         mean = torch.mean(mean_list,dim=0)
        #         var = torch.mean(var_list,dim=0)
        #         self.mean_dict[name] = mean
        #         self.var_dict[name] = var
        #         self.mean_dict[name] = [self.mean]



        self.count += 1
        if self.count == 50:
            for name in self.mean_dict:
                mean_list = torch.stack(self.mean_dict[name])
                var_list = torch.stack(self.var_dict[name])
                mean = torch.mean(mean_list,dim=0)
                var = torch.mean(var_list,dim=0)
                self.mean_perLayer.append(mean)
                self.var_perLayer.append(var)
            print("完成！")
        return outputs



    def reset_steps(self, new_steps):
        self.steps = new_steps

    def reset_model_probs(self, probs):
        self.current_model_probs = probs



    def configure_model(self):
        self.model.requires_grad_(False)
        normlayer_names = []

        for name, sub_module in self.model.named_modules():
            if isinstance(sub_module, nn.BatchNorm1d) or isinstance(sub_module, nn.BatchNorm2d):
                normlayer_names.append(name)
            elif isinstance(sub_module, (nn.LayerNorm, nn.GroupNorm)):
                sub_module.requires_grad_(True)

        for name in normlayer_names:
            bn_layer = get_named_submodule(self.model, name)
            if isinstance(bn_layer, nn.BatchNorm1d):
                NewBN = RobustBN1d
            elif isinstance(bn_layer, nn.BatchNorm2d):
                NewBN = RobustBN2d
            else:
                raise RuntimeError()

            momentum_bn = NewBN(bn_layer, self.cfg.ROTTA.ALPHA)
            momentum_bn.requires_grad_(True)
            set_named_submodule(self.model, name, momentum_bn)


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

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x


def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)


class MomentumBN(nn.Module):
    def __init__(self, bn_layer: nn.BatchNorm2d, momentum):
        super().__init__()
        self.num_features = bn_layer.num_features
        self.momentum = momentum
        self.beta_pre = 0.1
        if bn_layer.track_running_stats and bn_layer.running_var is not None and bn_layer.running_mean is not None:
            self.register_buffer("source_mean", deepcopy(bn_layer.running_mean))
            self.register_buffer("source_var", deepcopy(bn_layer.running_var))
            self.source_num = bn_layer.num_batches_tracked
        self.weight = deepcopy(bn_layer.weight)
        self.bias = deepcopy(bn_layer.bias)

        self.register_buffer("target_mean", torch.zeros_like(self.source_mean))
        self.register_buffer("target_var", torch.ones_like(self.source_var))
        self.eps = bn_layer.eps

        self.current_mu = None
        self.current_sigma = None

    def forward(self, x):
        raise NotImplementedError


class RobustBN1d(MomentumBN):
    def forward(self, x):
        if self.training:
            b_var, b_mean = torch.var_mean(x, dim=0, unbiased=False, keepdim=False)  # (C,)
            mean = (1 - self.momentum) * self.source_mean + self.momentum * b_mean
            var = (1 - self.momentum) * self.source_var + self.momentum * b_var
            self.source_mean, self.source_var = deepcopy(mean.detach()), deepcopy(var.detach())
            mean, var = mean.view(1, -1), var.view(1, -1)
        else:
            mean, var = self.source_mean.view(1, -1), self.source_var.view(1, -1)

        x = (x - mean) / torch.sqrt(var + self.eps)
        weight = self.weight.view(1, -1)
        bias = self.bias.view(1, -1)

        return x * weight + bias


class RobustBN2d(MomentumBN):

    def forward(self, x):

        if self.training:
            b_var, b_mean = torch.var_mean(x, dim=[0, 2, 3], unbiased=False, keepdim=False)  # (C,)
            self.source_mean = b_mean
            self.source_var = b_var
            # mean = (1 - self.momentum) * self.source_mean + self.momentum * b_mean
            # var = (1 - self.momentum) * self.source_var + self.momentum * b_var
            # self.source_mean, self.source_var = deepcopy(mean.detach()), deepcopy(var.detach())
            mean, var = b_mean.view(1, -1, 1, 1), b_var.view(1, -1, 1, 1)

            # b_var, b_mean = torch.var_mean(x, dim=[0, 2, 3], unbiased=False, keepdim=False)  # (C,)
            # mean, var = b_mean.view(1, -1, 1, 1), b_var.view(1, -1, 1, 1)
        else:
            mean, var = self.source_mean.view(1, -1, 1, 1), self.source_var.view(1, -1, 1, 1)

        """ 通过计算得到的新的统计量对当前数据归一化 """
        x = (x - mean) / torch.sqrt(var + self.eps)
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)

        return x * weight + bias