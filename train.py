import torch
from torch import nn
import dam2s_old
import cooper
from VFeatureExtractor import VFeatureExtractor
from DFeatureExtractor import DFeatureExtractor
import cv2
from argparse import ArgumentParser
from tqdm import tqdm
import os
import numpy as np

parser = ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--primal_lr', type=float, default=1e-3)
parser.add_argument('--dual_lr', type=float, default=1e-3)
parser.add_argument('--subspace_dim', type=int, default=500)
parser.add_argument('--c', type=float, default=1.0)
parser.add_argument('--mu', type=float, default=1e5)
parser.add_argument('--l', type=float, default=1e-2)
parser.add_argument('--save_dir', type=str, default='checkpoints')
args = parser.parse_args()

# v_features = torch.ones(3250, 2048)
# t_features = torch.ones(571, 2048)
# d_features = torch.ones(3250, 1594)
# labels = torch.randint(0, 5, (3250,))
# num_epochs = args.num_epochs
# num_samples = 3250
# v_dim = 2048
# d_dim = 1594
# subspace_dim = 500
# num_classes = 5

classes = ['coffee_mug', 'lightbulb', 'mushroom', 'soda_can', 'tomato']
num_classes = 5
source_dir = './data/source/'
target_dir = './data/target/'
v_feature_extractor = VFeatureExtractor()
d_feature_extractor = DFeatureExtractor()
# v_feature_extractor.eval()
# d_feature_extractor.eval()
v_features = []
d_features = []
t_features = []
labels = []
t_labels = []
with torch.no_grad():
    print("start data preparing")
        
    for i in range(num_classes):
        print('{}'.format(classes[i]))
        filenames = []
        t_images = []
        for file in os.listdir(os.path.join(target_dir, classes[i])):
            t_image = cv2.imread(os.path.join(target_dir, classes[i], file))
            t_image = cv2.cvtColor(t_image, cv2.COLOR_BGR2RGB)
            t_image = cv2.resize(t_image, (224, 224))
            t_images.append(t_image)
            t_labels.append(i)
        t_images = np.stack(t_images)
        t_features.append(v_feature_extractor(t_images))
            
        filenames = []
        for file in os.listdir(os.path.join(source_dir, classes[i])):
            filename = file.split('.')[0].rstrip('_depth').rstrip('_mask').rstrip('_loc')
            if filename in filenames:
                continue
            filenames.append(filename)

        v_images = []
        d_images = []
        for filename in filenames:
            v_image = cv2.imread(os.path.join(source_dir, classes[i], filename + '.png'))
            d_image = cv2.imread(os.path.join(source_dir, classes[i], filename + '_depth.png'))
            v_image = cv2.cvtColor(v_image, cv2.COLOR_BGR2RGB)
            v_image = cv2.resize(v_image, (224, 224))
            d_image = cv2.resize(d_image, (224, 224))
            v_images.append(v_image)
            d_images.append(d_image)
            labels.append(i)
        v_images = np.stack(v_images)
        d_images = np.stack(d_images)
        v_features.append(v_feature_extractor(v_images))
        d_features.append(d_feature_extractor(d_images))
        print(len(d_features))
    v_features = torch.concatenate(v_features, 0)
    d_features = torch.concatenate(d_features, 0)
    t_features = torch.concatenate(t_features, 0)
    labels = torch.tensor(labels, dtype=torch.int64)
    t_labels = torch.tensor(t_labels, dtype=torch.int64)

    print("d_features shape: {}".format(d_features.shape))
    print("v_features shape: {}".format(v_features.shape))
    print("t_features shape: {}".format(t_features.shape))
    print("v_label shape: {}".format(labels.shape))
    print("t_label shape: {}".format(t_labels.shape))
    print("start training")
    num_samples = v_features.shape[0]
    v_dim = v_features.shape[1]
    d_dim = d_features.shape[1]
    subspace_dim = args.subspace_dim
    num_epochs = args.num_epochs
# with torch.no_grad():
#     model = dam2s.dam2s_a(num_classes=5, num_samples=num_samples, v_dim=v_dim, d_dim=d_dim, subspace_dim=subspace_dim)
#     problem = dam2s.cmp(subspace_dim, 5)
#     formulation = cooper.LagrangianFormulation(problem)
#     primal_optimizer = torch.optim.SGD(list(model.parameters()), lr=args.primal_lr)
#     dual_optimizer = cooper.optim.partial_optimizer(torch.optim.SGD, lr=args.dual_lr)
#     combined = cooper.ConstrainedOptimizer(formulation, primal_optimizer, dual_optimizer)

#     num_epochs = args.num_epochs
#     progress_bar = tqdm(range(num_epochs), total=num_epochs)
#     for i in progress_bar:
#         combined.zero_grad()
#         lagrangian = formulation.composite_objective(
#             problem.closure,
#             v_features,
#             d_features,
#             labels,
#             t_features,
#             *model.parameters()
#         )
#         formulation.custom_backward(lagrangian)
#         combined.step()
#         s = 'loss: {}'.format(problem.get_loss())
#         progress_bar.set_description(s)





# have not got the data yet. using all-one tensor just for debugging
# v_features = torch.ones(3250, 2048)
# t_features = torch.ones(571, 2048)
# d_features = torch.ones(3250, 1594)
# labels = torch.randint(0, 5, (3250,))
model = dam2s_old.dam2s_a(num_classes=num_classes, num_samples=v_features.shape[0], v_dim=v_features.shape[1], d_dim=d_features.shape[1], subspace_dim=subspace_dim)
problem = dam2s_old.cmp(500, 5, c=args.c, mu=args.mu, l=args.l)
formulation = cooper.LagrangianFormulation(problem)
primal_optimizer = torch.optim.SGD(list(model.parameters()), lr=1e-3)
dual_optimizer = cooper.optim.partial_optimizer(torch.optim.SGD, lr=1e-3)
combined = cooper.ConstrainedOptimizer(formulation, primal_optimizer, dual_optimizer)
progress_bar = tqdm(range(num_epochs), total=num_epochs)
for i in progress_bar:
    combined.zero_grad()
    lagrangian = formulation.composite_objective(
        problem.closure,
        v_features,
        d_features,
        labels,
        t_features,
        *model.parameters()
    )
    formulation.custom_backward(lagrangian)
    combined.step()
    s = 'loss: {}'.format(problem.get_loss())
    progress_bar.set_description(s)