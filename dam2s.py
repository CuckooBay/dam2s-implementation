import torch
from torch import nn
#import cooper
#from cooper import CMPState
import liblinear
from torch.nn import functional as F

class dam2s_a(nn.Module):
    def __init__(self, num_classes, num_samples_s,num_samples_t,dim_sub,feat_v_s,feat_d,feat_v_t,c=1.0, mu=1e5, l=1e-2, **kwargs):
        super(dam2s_a, self).__init__(**kwargs)
        #hyper parameters
        self.mu=mu
        self.l=l
        
        #extracted features with shape (ns, dim_v) and (ns, dim_d)
        self.feat_d=feat_d
        self.feat_v_s=feat_v_s
        self.feat_v_t=feat_v_t
        
        #shape parameters
        self.dim_v = torch.Size(feat_v_s,dim=1) #visual feature dimension mv
        self.dim_d = torch.Size(feat_d,dim=1) #depth feature dimension md
        self.dim_sub = dim_sub #subspace dimension m
        self.num_classes = num_classes
        self.num_samples_s = num_samples_s
        self.num_samples_t= num_samples_t

        
        # define trainable parameters
        self.proj_v = nn.Linear(self.dim_v, dim_sub, bias=False) #project visual to subspace   
        self.proj_d = nn.Linear(self.dim_d, dim_sub, bias=False) #project depth to subspace

        # define kernel matrix
        self.kermat_d=torch.mm(torch.T(feat_d),feat_d)
        self.kermat_v_s=torch.mm(torch.T(feat_v_s),feat_v_s)
        self.kermat_v_st=torch.mm(torch.T(feat_v_s),feat_v_t)

        # define matrix for dual problem
        #calculate matrix B_kcca
        mat_zeros_ss=torch.zeros((self.num_samples_s,self.num_samples_s)) #zeros matrix with ns*ns
        mat_B_kcca_U=torch.cat((mat_zeros_ss,torch.mm(self.kermat_d,self.kermat_v_s)),1)  #B_kcca_U matrix
        mat_B_kcca_L=torch.cat((torch.mm(self.kermat_v_s,self.kermat_d),mat_zeros_ss),1) #B_kcca_L matrix
        self.mat_B_kcca=torch.cat((mat_B_kcca_U,mat_B_kcca_L),0) #B_kcca matrix

        #calculate matrix B_mmd
        mat_zeros_st=torch.zeros((self.num_samples_s,self.num_samples_t)) #zeros matrix with ns*nt
        mat_B_mmdtemp_U=torch.cat((self.kermat_d,mat_zeros_st),1) #B_mmdtemp_U matrix
        mat_B_mmdtemp_L=torch.cat(self.kermat_v_s,2*self.kermat_v_st),1 #B_mmdtemp_L matrix
        mat_B_mmdtemp=torch.cat((mat_B_mmdtemp_U,mat_B_mmdtemp_L),0) #B_mmdtemp matrix
        mat_S=torch.cat(1/self.num_samples_s*torch.ones(self.num_samples_s),-1/self.num_samples_t*torch.ones(self.num_samples_t)) #S matrix
    
        mat_B_mmd_left=torch.mm(mat_B_mmdtemp,mat_S) #B_mmd matrix left
        self.mat_B_mmd=torch.mm(mat_B_mmd_left,mat_B_mmd_left.T) #B_mmd matrix

        #calculate overall matrix B
        self.mat_B=self.mat_B_kcca-self.l*self.mat_B_mmd

        #calculate B_tilde
        mat_G_U=torch.cat((self.kermat_d,mat_zeros_ss),1)
        mat_G_L=torch.cat((mat_zeros_ss,self.kermat_v_s),1)


        
    def dual_descent(self,label,kermat_d,kermat_v_s,kermat_v_st):
        t=0
        mat_zeros=torch.zeros((self.num_samples,self.num_samples)) #zeros matrix with ns*ns
        mat_D_U=torch.cat((torch.mm(kermat_d,kermat_d),mat_zeros),1)  #D_U matrix
        mat_D_L=torch.cat((mat_zeros,torch.mm(kermat_v_st,kermat_v_st)),1) #D_L matrix
        mat_D=torch.cat((mat_D_U,mat_D_L),0) #D matrix

        

'''
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
'''
