"""
Builds upon: https://github.com/fiveai/LAME
Corresponding paper: https://arxiv.org/abs/2201.05718
"""

import torch
import torch.jit
import logging
import torch.nn.functional as F
from copy import deepcopy
from methods.base import TTAMethod
from models.model import split_up_model

logger = logging.getLogger(__name__)


class LAME(TTAMethod):
    """ Parameter-free Online Test-time Adaptation
    """
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        self.affinity = eval(f'{cfg.LAME.AFFINITY}_affinity')(sigma=cfg.LAME.SIGMA, knn=cfg.LAME.KNN)
        self.force_symmetry = cfg.LAME.FORCE_SYMMETRY

        # split up the model
        self.feature_extractor, self.classifier = split_up_model(self.model, cfg.MODEL.ARCH, self.dataset_name)
        self.model_state, _ = self.copy_model_and_optimizer()

    @torch.no_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        imgs_test = x[0]

        features = self.feature_extractor(imgs_test)
        outputs = self.classifier(features)

        # --- Get unary and terms and kernel ---
        unary = - torch.log(outputs.softmax(dim=1) + 1e-10)  # [N, K]

        features = F.normalize(features, p=2, dim=-1)  # [N, d]
        kernel = self.affinity(features)  # [N, N]
        if self.force_symmetry:
            kernel = 1 / 2 * (kernel + kernel.t())

        # --- Perform optim ---
        outputs = laplacian_optimization(unary, kernel)

        return outputs

    def configure_model(self):
        """Configure model"""
        self.model.eval()
        # disable grad, to (re-)enable only what we update
        self.model.requires_grad_(False)

    def copy_model_and_optimizer(self):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_state = deepcopy(self.model.state_dict())
        return model_state, None

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)


class AffinityMatrix:

    def __init__(self, **kwargs):
        pass

    def __call__(X, **kwargs):
        raise NotImplementedError

    def is_psd(self, mat):
        eigenvalues = torch.eig(mat)[0][:, 0].sort(descending=True)[0]
        return eigenvalues, float((mat == mat.t()).all() and (eigenvalues >= 0).all())

    def symmetrize(self, mat):
        return 1 / 2 * (mat + mat.t())

class Rknn_affinity(AffinityMatrix):
    def __init__(self, knn: int, **kwargs):
        self.knn = knn

    def __call__(self, X):
        N = X.size(0)
        dist = X @ X.T
        n_neighbors = min(self.knn, N)
        idx_near = dist.topk(n_neighbors, dim=-1, largest=True).indices[:, 1:]  # b * K
        fea_near = X[idx_near]  # b * K * d
        fea_bank_re = X.unsqueeze(0).expand(fea_near.shape[0], -1, -1)
        distance_ = torch.bmm(fea_near, fea_bank_re.permute(0, 2, 1))  # batch x K x batch
        idx_near_near = distance_.topk(n_neighbors-1, dim=-1, largest=True).indices[:, :, 1:]  # batch x K x M
        tar_idx = torch.tensor(range(X.size(0))).cuda()
        tar_idx = tar_idx.unsqueeze(-1).unsqueeze(-1)
        match = (idx_near_near == tar_idx).sum(-1).float()
        weight = torch.zeros(N, N, device=X.device)
        weight.scatter_(dim=-1, index=(idx_near * match).type(torch.LongTensor).cuda(), value=1.0)


        # weight = torch.where(match > 0., match,
        #                      torch.ones_like(match).fill_(0.1))
        # weight_kk = weight.unsqueeze(-1).expand(-1, -1, n_neighbors-2)
        #
        # weight_kk = weight_kk.contiguous().view(weight_kk.shape[0], -1)  # batch x KM
        return weight

class kNN_affinity(AffinityMatrix):
    def __init__(self, knn: int, **kwargs):
        self.knn = knn

    def __call__(self, X):
        N = X.size(0)
        dist = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1, p=2)  # 得到任意两个向量之间的距离
        n_neighbors = min(self.knn + 1, N)

        knn_index = dist.topk(n_neighbors, -1, largest=False).indices[:, 1:]  # [N, knn]

        W = torch.zeros(N, N, device=X.device)
        W.scatter_(dim=-1, index=knn_index, value=1.0)

        return W

# 高斯核函数即为 RBF 核函数，φ(x)=exp(−γ||x−c||^2),x是输入的样本，c是RBF中心，γ是控制函数衰减的超参数(可取1/2σ^2)，|| ⋅ || 表示欧几里得距离
class rbf_affinity(AffinityMatrix):
    def __init__(self, sigma: float, **kwargs):
        self.sigma = sigma
        self.k = kwargs['knn']

    def __call__(self, X):
        N = X.size(0)
        dist = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1, p=2)  # [N, N]
        n_neighbors = min(self.k, N)
        kth_dist = dist.topk(k=n_neighbors, dim=-1, largest=False).values[:, -1]  # compute k^th distance for each point, [N, knn + 1]
        sigma = kth_dist.mean()
        rbf = torch.exp(- dist ** 2 / (2 * sigma ** 2))
        # mask = torch.eye(X.size(0)).to(X.device)
        # rbf = rbf * (1 - mask)
        return rbf


class linear_affinity(AffinityMatrix):

    def __call__(self, X: torch.Tensor):
        """
        X: [N, d]
        """
        return torch.matmul(X, X.t())


def laplacian_optimization(unary, kernel, bound_lambda=1, max_steps=100):
    # E_list = []
    oldE = float('inf')
    Y = (-unary).softmax(-1)  # [N, K]
    for i in range(max_steps):
        pairwise = bound_lambda * kernel.matmul(Y)  # [N, K]
        exponent = -unary + pairwise
        Y = exponent.softmax(-1)
        E = entropy_energy(Y, unary, pairwise, bound_lambda).item()
        if (i > 1 and (abs(E - oldE) <= 1e-8 * abs(oldE))):
            break
        else:
            oldE = E
    return Y


def entropy_energy(Y, unary, pairwise, bound_lambda):
    E = (unary * Y - bound_lambda * pairwise * Y + Y * torch.log(Y.clip(1e-20))).sum()
    return E