import torch
import torch.nn.functional as F
@torch.jit.script
def symmetric_cross_entropy(x, x_ema):  # -> torch.Tensor:
    return -0.5*(x_ema.softmax(1) * x.log_softmax(1)).sum(1)-0.5*(x.softmax(1) * x_ema.log_softmax(1)).sum(1)


def update_ema_variables(ema_model, model, alpha_teacher=0.999):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

@torch.jit.script
def self_training(x, x_aug, x_ema):  # -> torch.Tensor:
    return - 0.25 * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - 0.25 * (x.softmax(1) * x_ema.log_softmax(1)).sum(1) \
           - 0.25 * (x_ema.softmax(1) * x_aug.log_softmax(1)).sum(1) - 0.25 * (x_aug.softmax(1) * x_ema.log_softmax(1)).sum(1)

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def warmup(args, net, model_ema, trloader, optimizer):
    print(f"Starting warm up...")
    # trloader.dataset.data = trloader.dataset.data[:100]
    for batch_idx, (inputs, labels) in enumerate(trloader):
        print(f"warmup_{batch_idx}/{len(trloader)}")
        inputs = inputs.cuda()
        #linearly increase the learning rate
        for par in optimizer.param_groups:
            par["lr"] = 0.0001 * (batch_idx+1) / len(trloader)


        """cosine分类"""
        net.head.fc.weight.data = F.normalize(net.head.fc.weight.data)
        feature_ext = net.ext(inputs)
        logit = torch.mm(F.normalize(feature_ext), net.head.fc.weight.t()) / args.delta

        feature_ext_ema = model_ema.ext(inputs)
        logit_ema = torch.mm(F.normalize(feature_ext_ema), net.head.fc.weight.t()) / args.delta

        loss = symmetric_cross_entropy(logit, logit_ema).mean(0)

        try:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        except:
            print('can not backward')
        model_ema = update_ema_variables(model_ema, net)
    print(f"Finished warm up...")
    for par in optimizer.param_groups:
        par["lr"] = args.lr
    return net, model_ema, optimizer


# Integrated from: https://github.com/HobbitLong/SupContrast/blob/master/losses.py
def contrastive_loss(features, labels=None, mask=None, projector=None, temperature=0.1, base_temperature=0.1):
    contrast_mode = "all"

    batch_size = features.shape[0]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).cuda()  # 创建一个对角线上元素均为 1，其余元素均为 0 的正方形矩阵
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().cuda()
    else:
        mask = mask.float().cuda()

    contrast_count = features.shape[1]  # contrast_count = 3
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # 将张量沿着指定维度拆分为多个张量(此处为：prototypes，features_test，features_aug_test)
    contrast_feature = projector(contrast_feature)
    contrast_feature = F.normalize(contrast_feature, p=2, dim=1)
    if contrast_mode == 'one':
        anchor_feature = features[:, 0]
        anchor_count = 1
    elif contrast_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = contrast_count
    else:
        raise ValueError('Unknown mode: {}'.format(contrast_mode))

    # compute logits
    anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), temperature)  # 先内积，再执行元素级别的除法运算
    # 将自己与自己的内积置为0
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)  # 沿第0、1维度对张量重复3次 200 * 200 ==> 600 * 600
    # mask-out self-contrast cases
    logits_mask = torch.scatter(  # 创建了一个对角线为 0 ，其余全为 1 的正方形矩阵
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).cuda(),
        0
    )  # torch.scatter(input, dim, index, src):根据给定的索引，在指定的维度上将源张量 src 的值散射（分散）到输入张量 input 上
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask  # 将logits指数化，同时将对角元素置为0
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()
    return loss

