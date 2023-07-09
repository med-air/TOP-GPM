import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.distributions import Normal, Independent, kl

import torch.nn.init as init
from torchvision import models

from resnet3d import *
from torch.autograd import Variable
import math
import numpy as np
import copy
import random

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out

class EpsilonLayer(nn.Module):
    def __init__(self):
        super(EpsilonLayer, self).__init__()

        # building epsilon trainable weight
        self.weights = nn.Parameter(torch.Tensor(1, 1))

        # initializing weight parameter with RandomNormal
        nn.init.normal_(self.weights, mean=0, std=0.05)

    def forward(self, inputs):
        return torch.mm(torch.ones_like(inputs)[:, 0:1], self.weights.T)

class VAE_ours2(nn.Module):
    def __init__(self,input_size,hidden_size,latent_size,out_features=[200, 64, 1]):
        super(VAE_ours2,self).__init__()
        dropout = False
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        
        # self.encoder = nn.Sequential(
        # nn.Linear(input_size, hidden_size),
        # nn.ReLU(),
        # nn.Linear(hidden_size,latent_size*4)        
        # )
        
        self.decoder0 = nn.Sequential(
        #nn.Linear(latent_size+1,4),
        #nn.ReLU(),
        nn.Linear(latent_size+1,1)
        )
        
        
        self.dropout = nn.Dropout(p=0.2)
        self.experts = ProductOfExperts()
        self.epsilon = EpsilonLayer()
        c=[64,64,128,256,512]
        layers = [3, 4, 6, 3]
        self.inplanes = c[0]
        self.share = torch.nn.Sequential()
        self.share.add_module('conv1', nn.Conv3d(1, c[0],kernel_size=7, stride=2, padding=0, bias=False))
        self.share.add_module('bn1', nn.BatchNorm3d(c[0]))
        self.share.add_module('relu', nn.ReLU(inplace=True))
        self.share.add_module('maxpool',nn.MaxPool3d(kernel_size=3, stride=2, padding=1))
        self.share.add_module('layer1', self._make_layer(BasicBlock, c[1], layers[0]))
        self.share.add_module('layer2', self._make_layer(BasicBlock, c[2], layers[1], stride=2))
        self.share.add_module('layer3', self._make_layer(BasicBlock, c[3], layers[2], stride=2))
        self.share.add_module('layer4', self._make_layer(BasicBlock, c[4], layers[3], stride=2))
        self.share.add_module('avgpool', nn.AvgPool3d([1,7,7])) 
        
        
        if dropout is True:
            self.share.add_module('dropout', nn.Dropout(0.5))
        self.resenet_head = nn.Sequential(nn.Linear(512, out_features[1]), nn.BatchNorm1d(out_features[1]), nn.ReLU())
        self.resenet_head_t = nn.Sequential(nn.Linear(out_features[1]+1, out_features[1]), nn.BatchNorm1d(out_features[1]), nn.ReLU())
        self.pnet_net_im_0 = nn.Sequential(nn.Linear(out_features[1], latent_size*2), nn.BatchNorm1d(latent_size*2))
        self.pnet_net_im_1 = nn.Sequential(nn.Linear(out_features[1], latent_size*2), nn.BatchNorm1d(latent_size*2))
        self.qnet_net_im_0 = nn.Sequential(nn.Linear(out_features[1]+1, latent_size*2), nn.BatchNorm1d(latent_size*2))
        self.qnet_net_im_1 = nn.Sequential(nn.Linear(out_features[1]+1, latent_size*2), nn.BatchNorm1d(latent_size*2))
        
        self.pnet_net_tab_0 = nn.Sequential(nn.Linear(out_features[1], latent_size*2), nn.BatchNorm1d(latent_size*2))
        self.pnet_net_tab_1 = nn.Sequential(nn.Linear(out_features[1], latent_size*2), nn.BatchNorm1d(latent_size*2))
        self.qnet_net_tab_0 = nn.Sequential(nn.Linear(out_features[1]+1, latent_size*2), nn.BatchNorm1d(latent_size*2))
        self.qnet_net_tab_1 = nn.Sequential(nn.Linear(out_features[1]+1, latent_size*2), nn.BatchNorm1d(latent_size*2))
        
        self.representation_block = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=out_features[0]),
            nn.BatchNorm1d(out_features[0]),
            nn.ReLU(),
            nn.Linear(in_features=out_features[0], out_features=out_features[1]),
            nn.BatchNorm1d(out_features[1]),
            nn.ReLU()     
        )
        
        self.representation_block_0 = nn.Sequential(
            nn.Linear(in_features=out_features[1]+1, out_features=out_features[1]),
            nn.BatchNorm1d(out_features[1]),
            nn.ReLU()     
        )
        
        self.representation_block_1 = nn.Sequential(
            nn.Linear(in_features=out_features[1], out_features=out_features[1]),
            nn.BatchNorm1d(out_features[1]),
            nn.ReLU()     
        )
        
        for m in self.modules():
            if isinstance(m,nn.Conv3d): 
                nn.init.kaiming_uniform_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm1d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.BatchNorm3d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            if isinstance(m,nn.Linear): 
                nn.init.normal_(m.weight, std=0.01)
                
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes*block.expansion,kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm3d(planes*block.expansion))
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            
        return nn.Sequential(*layers)    
    
    def encode_tab(self,x,treatment):
        x = self.representation_block(x)
        treatment_0 = torch.zeros(treatment.shape).cuda()
        treatment_1 = torch.ones(treatment.shape).cuda()
        x_0 = torch.cat((x,treatment_0),dim=1)
        x_1 = torch.cat((x,treatment_1),dim=1)
        h_tab_0 = self.representation_block_0(x_0)
        h_tab_1 = self.representation_block_0(x_1)
        return h_tab_0, h_tab_1
        
    def encode_img(self,x,treatment):
        x = self.resenet_head(x)
        treatment_0 = torch.zeros(treatment.shape).cuda()
        treatment_1 = torch.ones(treatment.shape).cuda()
        x_0 = torch.cat((x,treatment_0),dim=1)
        x_1 = torch.cat((x,treatment_1),dim=1)
        h_im_0 = self.resenet_head_t(x_0)
        h_im_1 = self.resenet_head_t(x_1)
        return h_im_0, h_im_1
    
    def p_net_im(self,phi0_im,phi1_im):
        h_im0 = self.pnet_net_im_0(phi0_im)
        mu0_im, logvar0_im = torch.chunk(h_im0,2,dim=-1)
        h_im1 = self.pnet_net_im_0(phi1_im)
        mu1_im, logvar1_im = torch.chunk(h_im1,2,dim=-1)
        return mu0_im, logvar0_im, mu1_im, logvar1_im
    
    def p_net_tab(self,phi0_tab,phi1_tab):
        h_tab0 = self.pnet_net_tab_0(phi0_tab)
        mu0_tab, logvar0_tab = torch.chunk(h_tab0,2,dim=-1)
        h_tab1 = self.pnet_net_tab_0(phi1_tab)
        mu1_tab, logvar1_tab = torch.chunk(h_tab1,2,dim=-1)
        return mu0_tab, logvar0_tab, mu1_tab, logvar1_tab
        
    def q_net_im(self,phi0_im,phi1_im):
        h_im0 = self.qnet_net_im_0(phi0_im)
        mu0_im, logvar0_im = torch.chunk(h_im0,2,dim=-1)
        h_im1 = self.qnet_net_im_0(phi1_im)
        mu1_im, logvar1_im = torch.chunk(h_im1,2,dim=-1)
        return mu0_im, logvar0_im, mu1_im, logvar1_im
    
    def q_net_tab(self,phi0_tab,phi1_tab):
        h_tab0 = self.qnet_net_tab_0(phi0_tab)
        mu0_tab, logvar0_tab = torch.chunk(h_tab0,2,dim=-1)
        h_tab1 = self.qnet_net_tab_0(phi1_tab)
        mu1_tab, logvar1_tab = torch.chunk(h_tab1,2,dim=-1)
        return mu0_tab, logvar0_tab, mu1_tab, logvar1_tab
    
    def PoE(self, mu0_im, logvar0_im, mu1_im, logvar1_im,
        mu0_tab, logvar0_tab, mu1_tab, logvar1_tab):
        mu0, logvar0, mu1, logvar1 = prior_expert((1,mu0_im.shape[0],self.latent_size))
        mu0 = torch.cat((mu0,mu0_tab.unsqueeze(0)),dim=0)
        logvar0 = torch.cat((logvar0,logvar0_tab.unsqueeze(0)),dim=0)
        mu1 = torch.cat((mu1,mu1_tab.unsqueeze(0)),dim=0)
        logvar1 = torch.cat((logvar1,logvar1_tab.unsqueeze(0)),dim=0)
        
        mu0 = torch.cat((mu0,mu0_im.unsqueeze(0)),dim=0)
        logvar0 = torch.cat((logvar0,logvar0_im.unsqueeze(0)),dim=0)
        mu1 = torch.cat((mu1,mu1_im.unsqueeze(0)),dim=0)
        logvar1 = torch.cat((logvar1,logvar1_im.unsqueeze(0)),dim=0)
        mu0, logvar0 = self.experts(mu0, logvar0)
        mu1, logvar1 = self.experts(mu1, logvar1)
        
        return  mu0, logvar0, mu1, logvar1
    
    def VDC_p(self,phi0_im, phi1_im, phi0_tab, phi1_tab):
        mu0_im_p, logvar0_im_p, mu1_im_p, logvar1_im_p = self.p_net_im(phi0_im,phi1_im)
        mu0_tab_p, logvar0_tab_p, mu1_tab_p, logvar1_tab_p = self.p_net_tab(phi0_tab, phi1_tab)
        
        return self.PoE(mu0_im_p, logvar0_im_p, mu1_im_p, logvar1_im_p,
        mu0_tab_p, logvar0_tab_p, mu1_tab_p, logvar1_tab_p)
         
    def VDC_q(self, phi0_im, phi1_im, phi0_tab, phi1_tab,labels):
        phi0_im = torch.cat((phi0_im,labels[:,0].unsqueeze(1)),dim=1)
        phi1_im = torch.cat((phi1_im,labels[:,0].unsqueeze(1)),dim=1)
        phi0_tab = torch.cat((phi0_tab,labels[:,0].unsqueeze(1)),dim=1)
        phi1_tab = torch.cat((phi1_tab,labels[:,0].unsqueeze(1)),dim=1)
        mu0_im_p, logvar0_im_p, mu1_im_p, logvar1_im_p = self.q_net_im(phi0_im,phi1_im)
        mu0_tab_p, logvar0_tab_p, mu1_tab_p, logvar1_tab_p = self.q_net_tab(phi0_tab, phi1_tab)
        
        return self.PoE(mu0_im_p, logvar0_im_p, mu1_im_p, logvar1_im_p,
        mu0_tab_p, logvar0_tab_p, mu1_tab_p, logvar1_tab_p)
    
    def reparameterize(self,mu,logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z
    
   
    def decode(self,z0,z1, treatment):
        treatment_0 = torch.zeros(treatment.shape).cuda()
        treatment_1 = torch.ones(treatment.shape).cuda()
        z0 = torch.cat((z0,treatment_0),dim=1)
        z1 = torch.cat((z1,treatment_1),dim=1)
        y0 = self.decoder0(z0)
        y1 = self.decoder0(z1)
        
        return F.sigmoid(y0)[:,0], F.sigmoid(y1)[:,0]
    
    
    def forward(self,x,labels,image,is_train=False):
        
        treatment = labels[:,1]
        treatment = torch.unsqueeze(treatment,dim=1)
        
        image = self.share.forward(image)
        image = self.dropout(image)
        image = image[:,:,0,0,0]
        phi0_im, phi1_im  = self.encode_img(image,treatment)
        phi0_tab, phi1_tab = self.encode_tab(x,treatment)
        
        mu0_p, logvar0_p, mu1_p, logvar1_p = self.VDC_p(phi0_im, phi1_im,phi0_tab, phi1_tab)
        if is_train:
            mu0_q, logvar0_q, mu1_q, logvar1_q = self.VDC_q(phi0_im, phi1_im,phi0_tab, phi1_tab,labels)
            z0 = self.reparameterize(mu0_q, logvar0_q)
            z1 = self.reparameterize(mu1_q, logvar1_q)
            y0,y1 = self.decode(z0,z1,treatment)
            dist_p_0 = Independent(Normal(loc=mu0_p,scale=torch.exp(logvar0_p)),1)
            dist_q_0 = Independent(Normal(loc=mu0_q,scale=torch.exp(logvar0_q)),1)
            dist_p_1 = Independent(Normal(loc=mu1_p,scale=torch.exp(logvar1_p)),1)
            dist_q_1 = Independent(Normal(loc=mu1_q,scale=torch.exp(logvar1_q)),1)
            return y0, y1, dist_p_0, dist_q_0, dist_p_1, dist_q_1
        else:
            z0 = self.reparameterize(mu0_p, logvar0_p)
            z1 = self.reparameterize(mu1_p, logvar1_p)
            y0,y1 = self.decode(z0,z1,treatment)
            return y0, y1

def prior_expert(size):
    mu0 = Variable(torch.zeros(size))
    logvar0 = Variable(torch.log(torch.ones(size)))
    mu1 = Variable(torch.zeros(size))
    logvar1 = Variable(torch.log(torch.ones(size)))
    mu0, logvar0, mu1, logvar1 = mu0.cuda(), logvar0.cuda(), mu1.cuda(), logvar1.cuda()
    return mu0, logvar0, mu1, logvar1

class ProductOfExperts(nn.Module):
    def forward(self,mu,logvar,eps=1e-8):
        var = torch.exp(logvar)+eps
        T = 1./var
        pd_mu = torch.sum(mu*T,dim=0) / torch.sum(T, dim=0)
        pd_var = 1./torch.sum(T,dim=0)
        pd_logvar = torch.log(pd_var)
        
        return pd_mu, pd_logvar
   
def VAE_loss_function(y0, y1, dist_p_0, dist_q_0, dist_p_1, dist_q_1, labels, class_ratio,ratio_as_t1):
    treatment = labels[:,1]
    map_0 = torch.zeros(treatment.shape)
    map_0[treatment==0] = 1
    map_1 = torch.zeros(treatment.shape)
    map_1[treatment==1] = 1
    BCE = torch.sum(map_0.cuda()*F.binary_cross_entropy(y0, labels[:,0], reduction='none'))/(torch.sum(map_0)+1e-8) + torch.sum(map_1.cuda()*F.binary_cross_entropy(y1, labels[:,0], reduction='none'))/(torch.sum(map_1)+1e-8)
    
    KLD = torch.mean(kl.kl_divergence(dist_q_0,dist_p_0)+kl.kl_divergence(dist_q_1,dist_p_1))
    return BCE, 1.0*KLD

def normal_loss_function(y0, y1, labels, class_ratio,ratio_as_t1):
    treatment = labels[:,1]
    map_0 = torch.zeros(treatment.shape)
    map_0[treatment==0] = 1
    map_1 = torch.zeros(treatment.shape)
    map_1[treatment==1] = 1
    BCE = torch.sum(map_0.cuda()*F.binary_cross_entropy(y0, labels[:,0], reduction='none'))/(torch.sum(map_0)+1e-8) + torch.sum(map_1.cuda()*F.binary_cross_entropy(y1, labels[:,0], reduction='none'))/(torch.sum(map_1)+1e-8)

    return BCE