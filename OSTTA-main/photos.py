import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn import datasets
from sklearn.manifold import TSNE
import torch

features_dir = "cifar10_joint_resnet50"
fix_portos_features = "offline_cifar10"

# # source domain data
source_feat = torch.load(f"/home/tjut_hanlei/OWTTT_3000/cifar/results/{features_dir}/source_data_features_AfterTrain.pth")
# source_feat = source_feat / np.linalg.norm(source_feat, ord=2, axis=1, keepdims=True)


# source domain protos
ground_truth_protos = torch.load(f"/home/tjut_hanlei/OWTTT_3000/cifar/results/{features_dir}/Ground_truth_SourceProtos_AfterTrain.pth")
ground_truth_protos_new = torch.stack((ground_truth_protos)).cpu().detach().numpy()
ground_truth_protos_new = ground_truth_protos_new / np.linalg.norm(ground_truth_protos_new, ord=2, axis=1, keepdims=True)
# 固定 protos
fix_portos1 = torch.load(f"/home/tjut_hanlei/OWTTT_3000/cifar/results/{features_dir}/{fix_portos_features}.pth")[0]
fix_portos_new1 = torch.stack((fix_portos1)).cpu().detach().numpy()
fix_portos2 = torch.load(f"/home/tjut_hanlei/OWTTT_3000/cifar/results/{features_dir}/protos_AfterTrain_OWTTT.pth")
fix_portos_new2 = fix_portos2.cpu().detach().numpy()
# 可更新的protos
trainable_protos = torch.load(f"/home/tjut_hanlei/OWTTT_3000/cifar/results/{features_dir}/protos_AfterTrain_embedding.pth")
trainable_protos_new = trainable_protos.cpu().detach().numpy()



# # 欧式距离
dis1 = np.sqrt(np.square(fix_portos_new1-ground_truth_protos_new).sum(1))
dis2 = np.sqrt(np.square(fix_portos_new2-ground_truth_protos_new).sum(1))
dis3 = np.sqrt(np.square(trainable_protos_new-ground_truth_protos_new).sum(1))
#
#
# # cosine距离
# cosine_similarity1 = np.dot(fix_portos_new1, ground_truth_protos_new.T) / (np.linalg.norm(fix_portos_new1,axis=1).reshape(-1,1) * np.linalg.norm(ground_truth_protos_new.T,axis=0).reshape(1,-1))
# cos1 = cosine_similarity1.max(-1) # 固定protos与真值protos的相似度
# cosine_similarity2 = np.dot(fix_portos_new2, ground_truth_protos_new.T) / (np.linalg.norm(fix_portos_new2,axis=1).reshape(-1,1) * np.linalg.norm(ground_truth_protos_new.T,axis=0).reshape(1,-1))
# cos2 = cosine_similarity2.max(-1) # 固定protos与真值protos的相似度
# cosine_similarity3 = np.dot(trainable_protos_new, ground_truth_protos_new.T) / (np.linalg.norm(trainable_protos_new,axis=1).reshape(-1,1) * np.linalg.norm(ground_truth_protos_new.T,axis=0).reshape(1,-1))
# cos3 = cosine_similarity3.max(-1) # 可训练protos与真值protos的相似度
# print()












def plot_tsne(protos_features, protos_labels, labels_mask):
    select_source_feat = [torch.stack(source_feat[i][:100]) for i in labels_mask]
    select_source_feat = torch.concatenate(select_source_feat).cpu().detach().numpy()
    select_source_feat = select_source_feat / np.linalg.norm(select_source_feat, ord=2, axis=1, keepdims=True)
    print(f'select_source_feat的形状为：{len(select_source_feat)}')
    labels_instance = []
    [labels_instance.append(idx // int(len(select_source_feat)/len(labels_mask))) for idx in range(len(select_source_feat))]
    features = np.concatenate((protos_features, select_source_feat), axis=0)
    labels = protos_labels + labels_instance

    # tsne = TSNE(n_components=2, init='pca', random_state=0, early_exaggeration=5, metric="cosine")
    # tsne = TSNE(n_components=2, init='pca', random_state=0, metric="cosine")
    # tsne = TSNE(n_components=2, init='pca', random_state=0, metric="cosine", method="exact")
    # tsne = TSNE(n_components=2, init='pca', random_state=0, early_exaggeration=20, method="exact")
    # tsne = TSNE(n_components=2, init='pca', random_state=0, early_exaggeration=20)
    tsne = TSNE(n_components=2, perplexity=300, random_state=42, n_iter=500)
    tsne_features = tsne.fit_transform(features)  # 将特征使用PCA降维至2维

    # 归一化tsne_features
    # x_min, x_max = tsne_features.min(0), tsne_features.max(0)
    # tsne_features = (tsne_features - x_min) / (x_max - x_min)

    # 定义颜色映射
    color_map = {str(protos_labels[0]): 'red'}
    color_map.update({str(protos_labels[1]): 'blue'})
    color_map.update({str(protos_labels[2]): 'green'})
    color_map.update({str(protos_labels[3]): 'brown'})
    color_map.update({str(protos_labels[4]): 'purple'})

    color_map.update({str(protos_labels[5]): 'black'})
    color_map.update({str(protos_labels[6]): 'pink'})
    color_map.update({str(protos_labels[7]): 'gray'})
    color_map.update({str(protos_labels[8]): 'cyan'})
    color_map.update({str(protos_labels[9]): 'orange'})

    # 绘制散点图，同时在每个点上添加文本标签，并使用不同的颜色
    fig = plt.figure()
    ax = plt.subplot(111)
    for idx, (x, y, label) in enumerate(zip(tsne_features[:, 0], tsne_features[:, 1], labels)):
        if idx % 500 == 0:
            print(idx/500)
        # protos
        if idx < len(protos_features):
            if idx < (len(protos_labels) // 3):
                plt.scatter(x, y, c=color_map[str(label)], alpha=1, s=80, marker='o', zorder=100, edgecolors='black', linewidth=1)
            elif (idx >= len(protos_labels) // 3) and (idx < 2 * (len(protos_labels) // 3)):
                plt.scatter(x, y, c=color_map[str(label)], alpha=1, s=80, marker='^', zorder=100,edgecolors='black', linewidth=1)
            elif idx >= (2 * (len(protos_labels) // 3)):
                plt.scatter(x, y, c=color_map[str(label)], alpha=1, s=80, marker=(5, 0), zorder=100,edgecolors='black', linewidth=1)

        # source domain data
        else:
            plt.scatter(x, y, c=color_map[str(label)], alpha=0.1, s=10, marker='o')

    # handles = [plt.Line2D([0], [0], marker='o', color=color, label=label, markersize=10) for label, color in color_map.items()]
    # plt.legend(handles=handles)
    plt.title('t-SNE Feature Visualization')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()



labels_mask = [0,1,2,3,4,5,6,7,8,9]
labels = [i for sublist in [[i+len(ground_truth_protos_new)*_ for i in labels_mask] for _ in range(1)] for i in sublist]
mapping = {value: index for index, value in enumerate(labels)}
# 现在我们可以得到每个元素映射到的新值
labels = [mapping[x] for x in labels]
labels = [element for _ in range(3) for element in labels]
feas = np.concatenate((ground_truth_protos_new[labels_mask],fix_portos_new1[labels_mask],trainable_protos_new[labels_mask]),axis=0)
plot_tsne(feas, labels, labels_mask)
