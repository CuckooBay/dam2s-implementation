import torch
from torch import nn
import cooper
from cooper import CMPState
from torch.nn import functional as F

class dam2s_a(nn.Module):
    def __init__(self, num_classes, num_samples, v_dim, d_dim, subspace_dim, c=1.0, mu=1e5, l=1e-2, **kwargs):
        super(dam2s_a, self).__init__(**kwargs)
        self.v_dim = v_dim
        self.d_dim = d_dim
        self.subspace_dim = subspace_dim
        self.num_classes = num_classes
        self.num_sample = num_samples

        # define trainable parameters
        self.v_proj = nn.Linear(v_dim, subspace_dim, bias=False)
        self.d_proj = nn.Linear(d_dim, subspace_dim, bias=False)
        self.v_svm = nn.Linear(subspace_dim, num_classes, bias=True)
        self.d_svm = nn.Linear(subspace_dim, num_classes, bias=True)
        self.v_error_slack = nn.Parameter(torch.rand(num_samples, num_classes))
        self.d_error_slack = nn.Parameter(torch.rand(num_samples, num_classes))

class cmp(cooper.ConstrainedMinimizationProblem):
    def __init__(self, subspace_dim, num_classes, c=1, mu=1e5, l=1e-2):
        super(cmp, self).__init__(is_constrained=True)
        self.c = c
        self.mu = mu
        self.l = l
        self.subspace_dim = subspace_dim
        self.num_classes = num_classes

    def closure(self, v_features, d_features, labels, t_features, *trainable_params) -> CMPState:
        hot_label = F.one_hot(labels, self.num_classes) * 2 - 1
        v_error_slack, d_error_slack, v_proj_mat, d_proj_mat, v_svm_mat, v_svm_bias, d_svm_mat, d_svm_bias = trainable_params
        v_sub = torch.matmul(v_features, v_proj_mat.transpose(0, 1))
        d_sub = torch.matmul(d_features, d_proj_mat.transpose(0, 1))
        t_sub = torch.matmul(t_features, v_proj_mat.transpose(0, 1))

        svm_regularitor_loss = 0.5 * (torch.sum(torch.square(v_svm_mat)) + torch.sum(torch.square(d_svm_mat)))
        slack_loss = torch.sum(v_error_slack + d_error_slack)
        mmd_loss = 0.5 * torch.norm(torch.mean(v_sub, 0) - torch.mean(t_sub, 0), 2)
        mfc_loss = - torch.trace(torch.matmul(d_sub.transpose(0, 1), v_sub))
        total_loss = svm_regularitor_loss + self.c * slack_loss + self.mu * mmd_loss + self.l * mfc_loss
        eq_constraint = torch.matmul(v_sub.transpose(0, 1), v_sub) + torch.matmul(d_sub.transpose(0, 1), d_sub) - torch.eye(self.subspace_dim)
        ineq_constraint = torch.concat([-v_error_slack, -d_error_slack, 1 - v_error_slack - hot_label * (torch.matmul(v_sub, v_svm_mat.transpose(0, 1))) - v_svm_bias, 1 - d_error_slack - hot_label * (torch.matmul(d_sub, d_svm_mat.transpose(0, 1)) + d_svm_bias)], dim=0)
        return  CMPState(loss=total_loss, ineq_defect=ineq_constraint, eq_defect=eq_constraint)