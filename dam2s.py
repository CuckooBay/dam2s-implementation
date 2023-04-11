import torch
from torch import nn
#import cooper
#from cooper import CMPState
import liblinear
import scipy
from torch.nn import functional as F

class dam2s_a(nn.Module):
    def __init__(self, num_classes, num_samples_s,num_samples_t,label,dim_sub,feat_v_s,feat_d,feat_v_t,c=1.0, mu=1e5, l=1e-2, **kwargs):
        super(dam2s_a, self).__init__(**kwargs)
        #hyper parameters
        self.mu=mu
        self.l=l
        
        #label
        self.label=label
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

        #calculate matrix Gamma
        mat_Gamma=0  #TBD
    
        #calculate G
        mat_H_U=torch.cat((self.kermat_d,mat_zeros_ss),1)  
        mat_H_L=torch.cat((mat_zeros_ss,self.kermat_v_s),1)
        self.mat_H=torch.cat((mat_H_U,mat_H_L),0)     #H=[[Kd,0],[0,Kvs]]
        mat_G=torch.mm(self.mat_H,mat_Gamma)
        
        #calculate B_tilde
        self.mat_B_tilde=mu*self.mat_B+torch.mm(mat_G,mat_G.T)

        #calculate E
        self.mat_E=torch.ones((2*self.num_samples_s,self.num_classes))
        for i in range(2*self.num_samples_s):
            self.mat_E[i,label[i]]=0


        
    def dual_descent(self,label,kermat_d,kermat_v_s,kermat_v_st):
        t=0
        mat_zeros_ss=torch.zeros((self.num_samples,self.num_samples)) #zeros matrix with ns*ns
        mat_D_U=torch.cat((torch.mm(kermat_d,kermat_d),mat_zeros_ss),1)  #D_U matrix
        mat_D_L=torch.cat((mat_zeros_ss,torch.mm(kermat_v_st,kermat_v_st)),1) #D_L matrix
        mat_D=torch.cat((mat_D_U,mat_D_L),0) #D matrix

        #Choelesky decomposition to get matrix C
        mat_C=torch.cholesky(mat_D)
        mat_C_inv=torch.linalg.inv(mat_C)
        mat_CBC=torch.mm(mat_C_inv.T,self.mat_B_tilde)
        mat_CBC=torch.mm(mat_CBC,mat_C_inv)

        #U_tilde containing the m leading eigenvectors in U corresponding to the largest eigenvalues.
        mat_U_tilde=scipy.linalg.eigh(mat_CBC,subset_by_index=[2*self.num_samples_s-self.dim_sub,2*self.num_samples_s-1])

        #calculate A
        mat_A=torch.mm(mat_C_inv,mat_U_tilde)
        
        #calculate kermat A
        kermat_A=torch.mm(self.mat_H,mat_A)

        

        
        #calculate H


        #calculate mat_Gamma
         


        #calculating 
        pass

        


