import torch
import torch.nn as nn
import numpy as np
from configure.cfgs import cfg
import copy

import torch.nn as nn


class SpiralConv(nn.Module):
    def __init__(self, in_c, spiral_size,out_c,activation='elu',bias=True,device=None):
        super(SpiralConv,self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.device = device

        self.conv = nn.Linear(in_c*spiral_size,out_c,bias=bias)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.02)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'identity':
            self.activation = lambda x: x
        else:
            raise NotImplementedError()

    def forward(self,x,spiral_adj):
        # print(x.size())
        # print(spiral_adj.size())
        bsize, num_pts, feats = x.size()
        _, _, spiral_size = spiral_adj.size()
  
        spirals_index = spiral_adj.view(bsize*num_pts*spiral_size) # [1d array of batch,vertx,vertx-adj]
        batch_index = torch.arange(bsize, device=self.device).view(-1,1).repeat([1,num_pts*spiral_size]).view(-1).long() # [0*numpt,1*numpt,etc.]
        spirals = x[batch_index,spirals_index,:].view(bsize*num_pts,spiral_size*feats) # [bsize*numpt, spiral*feats]


        out_feat = self.conv(spirals)
        out_feat = self.activation(out_feat)

        out_feat = out_feat.view(bsize,num_pts,self.out_c)
        zero_padding = torch.ones((1,x.size(1),1), device=self.device)
        zero_padding[0,-1,0] = 0.0
        out_feat = out_feat * zero_padding

        return out_feat

class SpiralAutoencoder(nn.Module):
    def __init__(self, filters_enc, filters_dec, latent_size, sizes, spiral_sizes, spirals, D, U, device, VAE_flag = False, activation = 'elu'):
        super(SpiralAutoencoder,self).__init__()
        self.latent_size = latent_size
        self.sizes = sizes
        self.spirals = spirals
        self.filters_enc = filters_enc
        self.filters_dec = filters_dec
        self.spiral_sizes = spiral_sizes
        self.D = D
        self.U = U
        self.device = device
        self.activation = activation
        self.VAE_flag = VAE_flag
        self.conv = []
        input_size = filters_enc[0][0]
        for i in range(len(spiral_sizes)-1):
            if filters_enc[1][i]:
                self.conv.append(SpiralConv(input_size, spiral_sizes[i], filters_enc[1][i],
                                            activation=self.activation, device=device).to(device))
                input_size = filters_enc[1][i]

            self.conv.append(SpiralConv(input_size, spiral_sizes[i], filters_enc[0][i+1],
                                        activation=self.activation, device=device).to(device))
            input_size = filters_enc[0][i+1]

        self.conv = nn.ModuleList(self.conv)   
        if self.VAE_flag:
            self.fc_latent_enc = nn.Linear((sizes[-1]+1)*input_size, 2 * latent_size)
        else:
            self.fc_latent_enc = nn.Linear((sizes[-1]+1)*input_size, latent_size)
        self.fc_latent_dec = nn.Linear(latent_size, (sizes[-1]+1)*filters_dec[0][0])
        
        self.dconv = []
        input_size = filters_dec[0][0]
        for i in range(len(spiral_sizes)-1):
            if i != len(spiral_sizes)-2:
                self.dconv.append(SpiralConv(input_size, spiral_sizes[-2-i], filters_dec[0][i+1],
                                             activation=self.activation, device=device).to(device))
                input_size = filters_dec[0][i+1]  
                
                if filters_dec[1][i+1]:
                    self.dconv.append(SpiralConv(input_size,spiral_sizes[-2-i], filters_dec[1][i+1],
                                                 activation=self.activation, device=device).to(device))
                    input_size = filters_dec[1][i+1]
            else:
                if filters_dec[1][i+1]:
                    self.dconv.append(SpiralConv(input_size, spiral_sizes[-2-i], filters_dec[0][i+1],
                                                 activation=self.activation, device=device).to(device))
                    input_size = filters_dec[0][i+1]                      
                    self.dconv.append(SpiralConv(input_size,spiral_sizes[-2-i], filters_dec[1][i+1],
                                                 activation='identity', device=device).to(device)) 
                    input_size = filters_dec[1][i+1] 
                else:
                    self.dconv.append(SpiralConv(input_size, spiral_sizes[-2-i], filters_dec[0][i+1],
                                                 activation='identity', device=device).to(device))
                    input_size = filters_dec[0][i+1]                      
                    
        self.dconv = nn.ModuleList(self.dconv)

    def encode(self, x, VAE_flag):
        bsize = x.size(0)
        S = self.spirals
        D = self.D
        # print(x.shape)
        j = 0
        for i in range(len(self.spiral_sizes)-1):
            x = self.conv[j](x,S[i].repeat(bsize,1,1))
            j+=1
            if self.filters_enc[1][i]:
                x = self.conv[j](x,S[i].repeat(bsize,1,1))
                j+=1
            x = torch.matmul(D[i],x)
            #  print(x.shape)
        x = x.view(bsize,-1)
        z = self.fc_latent_enc(x)
        if VAE_flag:
            self.z_mu = z[...,:self.latent_size]
            self.z_var  = z[...,self.latent_size:]
            std = torch.exp(self.z_var / 2)
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(self.z_mu) 
        return z
    
    def decode(self,z):
        bsize = z.size(0)
        S = self.spirals
        U = self.U
        
        x = self.fc_latent_dec(z)
        x = x.view(bsize,self.sizes[-1]+1,-1)
        j=0
        for i in range(len(self.spiral_sizes)-1):
            x = torch.matmul(U[-1-i],x)
            x = self.dconv[j](x,S[-2-i].repeat(bsize,1,1))
            j+=1
            if self.filters_dec[1][i+1]: 
                x = self.dconv[j](x,S[-2-i].repeat(bsize,1,1))
                j+=1
        return x

    
    def forward(self,x):
        bsize = x.size(0)
        z = self.encode(x, self.VAE_flag)
        x = self.decode(z)
        
        return x, z   



class SpiralAutoencoder_multiz_partkps(nn.Module):
    def __init__(self, kps_index_list, vert_part_index_dict, filters_enc, filters_dec, latent_size, part_kps_latent_size, sizes, spiral_sizes, spirals, D, U, device, VAE_flag = False, activation = 'elu'):
        super(SpiralAutoencoder_multiz_partkps,self).__init__()
        self.kps_keep = list(range(len(cfg.CONSTANTS.newskl_list) + 4))
        for i in [3,13,14]:
            self.kps_keep.remove(i)
        self.kps_index_list = kps_index_list
        self.vert_part_index_dict = vert_part_index_dict
        self.part_kps_latent_size = part_kps_latent_size
        self.latent_size = latent_size 
        self.sizes = sizes
        self.spirals = spirals
        self.filters_enc = filters_enc
        self.filters_dec = filters_dec
        self.spiral_sizes = spiral_sizes
        self.D = D
        self.U = U
        self.device = device
        self.activation = activation
        self.VAE_flag = VAE_flag
        self.conv = []
        input_size = filters_enc[0][0]
        for i in range(len(spiral_sizes)-1):
            if filters_enc[1][i]:
                self.conv.append(SpiralConv(input_size, spiral_sizes[i], filters_enc[1][i],
                                            activation=self.activation, device=device).to(device))
                input_size = filters_enc[1][i]

            self.conv.append(SpiralConv(input_size, spiral_sizes[i], filters_enc[0][i+1],
                                        activation=self.activation, device=device).to(device))
            input_size = filters_enc[0][i+1]

        self.conv = nn.ModuleList(self.conv)   
        if self.VAE_flag:
            self.fc_latent_enc_list = nn.ModuleList([nn.Linear((v.shape[0])*input_size, 2 * self.latent_size).to(device) for v in self.vert_part_index_dict.values()])
        else:
            self.fc_latent_enc_list = nn.ModuleList([nn.Linear((v.shape[0])*input_size, self.latent_size).to(device) for v in self.vert_part_index_dict.values()])
        self.fc_latent_dec_list = nn.ModuleList([nn.Linear(self.latent_size + self.part_kps_latent_size, (v.shape[0])*filters_dec[0][0]).to(device) for v in self.vert_part_index_dict.values()])
        self.kps_enc_list = nn.ModuleList([nn.Linear(len(kps_index)*3, self.part_kps_latent_size).to(device) for kps_index in self.kps_index_list])
        self.dconv = []
        input_size = filters_dec[0][0]
        for i in range(len(spiral_sizes)-1):
            if i != len(spiral_sizes)-2:
                self.dconv.append(SpiralConv(input_size, spiral_sizes[-2-i], filters_dec[0][i+1],
                                             activation=self.activation, device=device).to(device))
                input_size = filters_dec[0][i+1]  
                
                if filters_dec[1][i+1]:
                    self.dconv.append(SpiralConv(input_size,spiral_sizes[-2-i], filters_dec[1][i+1],
                                                 activation=self.activation, device=device).to(device))
                    input_size = filters_dec[1][i+1]
            else:
                if filters_dec[1][i+1]:
                    self.dconv.append(SpiralConv(input_size, spiral_sizes[-2-i], filters_dec[0][i+1],
                                                 activation=self.activation, device=device).to(device))
                    input_size = filters_dec[0][i+1]                      
                    self.dconv.append(SpiralConv(input_size,spiral_sizes[-2-i], filters_dec[1][i+1],
                                                 activation='identity', device=device).to(device)) 
                    input_size = filters_dec[1][i+1] 
                else:
                    self.dconv.append(SpiralConv(input_size, spiral_sizes[-2-i], filters_dec[0][i+1],
                                                 activation='identity', device=device).to(device))
                    input_size = filters_dec[0][i+1]                      
                    
        self.dconv = nn.ModuleList(self.dconv)


    def kps_encode(self, kps):
        z_part_kps = torch.cat([self.kps_enc_list[k](kps[:, kps_index, :].reshape(kps.shape[0], -1))[:, None] for k, kps_index in enumerate(self.kps_index_list)], dim = 1)
        # z_part_kps = torch.cat([self.kps_enc_list[k](self.emb(kps[:, kps_index, :]).reshape(kps.shape[0], -1))[:, None] for k, kps_index in enumerate(self.kps_index_list)], dim = 1)
        return z_part_kps

    def encode(self, x, kps, VAE_flag = None):
        bsize = x.size(0)
        S = self.spirals
        D = self.D
        # print(x.shape)
        j = 0
        for i in range(len(self.spiral_sizes)-1):
            x = self.conv[j](x,S[i].repeat(bsize,1,1))
            j+=1
            if self.filters_enc[1][i]:
                x = self.conv[j](x,S[i].repeat(bsize,1,1))
                j+=1
            x = torch.matmul(D[i],x)
            #  print(x.shape)
        z = torch.cat([self.fc_latent_enc_list[k](x[:, torch.from_numpy(part_index).to(self.device), :].view(bsize,-1))[:, None] for k, part_index in enumerate(self.vert_part_index_dict.values())], dim = 1)
        z_part_kps = self.kps_encode(kps)
        
        # x = x.view(bsize,-1)
        # z = self.fc_latent_enc(x)
        # if VAE_flag:
        #     self.z_mu = z[...,:self.latent_size]
        #     self.z_var  = z[...,self.latent_size:]
        #     std = torch.exp(self.z_var / 2)
        #     eps = torch.randn_like(std)
        #     z = eps.mul(std).add_(self.z_mu) 
        return z, z_part_kps, x[:, -1:, :]
    
    def decode(self,z,z_part_kps,dummy):
        bsize = z.size(0)
        S = self.spirals
        U = self.U
        x = torch.cat([self.fc_latent_dec_list[index](torch.cat([z[:, index, :], z_part_kps[:, index, :]], dim = 1)) for index in range(z.shape[1])], dim = 1).view(bsize,self.sizes[-1],-1)
        arange_index = torch.arange(self.sizes[-1], device=self.device)
        re_index = torch.from_numpy(np.concatenate([v for v in self.vert_part_index_dict.values()])).to(self.device)
        x[:, re_index, :] = x[:, arange_index, :]
        x = torch.cat([x, dummy], dim = 1)
        j=0
        for i in range(len(self.spiral_sizes)-1):
            x = torch.matmul(U[-1-i],x)
            x = self.dconv[j](x,S[-2-i].repeat(bsize,1,1))
            j+=1
            if self.filters_dec[1][i+1]: 
                x = self.dconv[j](x,S[-2-i].repeat(bsize,1,1))
                j+=1
        return x

    def kps2skl(self, kps_tmp):
        skl_list = cfg.CONSTANTS.newskl_list # measure_skl_list newskl_list
        # print(skl_list)
        if kps_tmp.shape[1] == len(skl_list) + 4:
            kps = copy.deepcopy(kps_tmp)
        else:
            kps_keep = list(range(len(skl_list) + 4))
            for i in [3,13,14]:
                kps_keep.remove(i)
            kps = torch.zeros((kps_tmp.shape[0], len(skl_list) + 4, 3), device = kps_tmp.device)
            kps[:, kps_keep, :] = kps_tmp

        skl = torch.zeros((kps.shape[0], len(skl_list), 4), device = kps.device)
        for index in range(len(skl_list)):
            if len(skl_list[index]) == 2:
                skl[:, index, :3] = (kps[:, skl_list[index][0], :] - kps[:, skl_list[index][1], :]) / (torch.sqrt(torch.sum((kps[:, skl_list[index][0], :] -  kps[:, skl_list[index][1], :]) ** 2, dim = 1)))[:, None]
                skl[:, index, -1] = torch.sqrt(torch.sum((kps[:, skl_list[index][0], :] -  kps[:, skl_list[index][1], :]) ** 2, dim = 1))
            elif len(skl_list[index]) == 3:
                skl[:, index, :3] = (kps[:, skl_list[index][0], :] - (kps[:, skl_list[index][1], :] + kps[:, skl_list[index][2], :]) / 2) / torch.sqrt(torch.sum((kps[:, skl_list[index][0], :] - (kps[:, skl_list[index][1], :] + kps[:, skl_list[index][2], :]) / 2) ** 2, dim = 1))[:, None]
                skl[:, index, -1] = torch.sqrt(torch.sum((kps[:, skl_list[index][0], :] - (kps[:, skl_list[index][1], :] + kps[:, skl_list[index][2], :]) / 2) ** 2, dim = 1))
        return skl
    
    def forward(self,x,kps):
        bsize = x.size(0)
        z, z_part_kps, dummy = self.encode(x, kps, self.VAE_flag)
        x = self.decode(z, z_part_kps, dummy)
        return x, z, z_part_kps