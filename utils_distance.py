import torch
import torch.nn.functional as F
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
import time

import scipy.io as sio
from scipy.sparse import csr_matrix, csc_matrix
from torch_scatter import scatter_add


SAVE_MEMORY = False

def VF_adjacency_matrix(V, F):
    """
    Input:
    V: N x 3
    F: F x 3
    Outputs:
    C: V x F adjacency matrix
    """
    #tensor type and device
    device = V.device
    dtype = V.dtype
    
    VF_adj = torch.zeros((V.shape[0], F.shape[0]), dtype=dtype, device=device)
    v_idx = F.view(-1)
    f_idx = torch.arange(F.shape[0]).repeat(3).reshape(3, F.shape[0]).transpose(1, 0).contiguous().view(
        -1)  # [000111...FFF]

    VF_adj[v_idx, f_idx] = 1
    return VF_adj


def _grad_div(V,T):
    
    #tensor type and device
    device = V.device
    dtype = V.dtype
    
    #WARNING not sure about this
    bs = V.shape[0];
    V = V.reshape([-1,3])
    T = T.reshape([-1,3])

    XF = V[T,:].transpose(0,1)

    Na = torch.cross(XF[1]-XF[0],XF[2]-XF[0])
    A = torch.sqrt(torch.sum(Na**2,-1,keepdim=True))+1e-6
    N = Na/A
    dA = 0.5/A

    m = T.shape[0]
    n = V.shape[0]
    def grad(f):
        gf = torch.zeros(m,3,f.shape[-1], device=device, dtype=dtype)
        for i in range(3):
            s = (i+1)%3
            t = (i+2)%3
            v = -torch.cross(XF[t]-XF[s],N)
            if SAVE_MEMORY:
                gf.add_(f[T[:,i],None,:]*(dA[:,0,None,None]*v[:,:,None])) #Slower less-memeory
            else:
                gf.add_(f[T[:,i],None,:]*(dA[:,0,None,None]*v[:,:,None])) 
        return gf
    
    def div(f):
        gf = torch.zeros(f.shape[-1],n, device=device, dtype=dtype)        
        for i in range(3):
            s = (i+1)%3
            t = (i+2)%3
            v = torch.cross(XF[t]-XF[s],N)
            if SAVE_MEMORY:
                gf.add_(scatter_add( torch.bmm(v[:,None,:],f)[:,0,:].t(), T[:,i], dim_size=n))# slower but uses less memory
            else:
                gf.add_(scatter_add( (f*v[:,:,None]).sum(1).t(), T[:,i], dim_size=n))
        return gf.t()
    
#     W = div(grad(torch.eye(n).cuda().double()))
#     A = scatter_add(A[:,0],T[:,0]).scatter_add(0,T[:,1],A[:,0]).scatter_add(0,T[:,2],A[:,0])/6
    return grad, div, A

def _geodesics_in_heat(grad, div, W, A, t=1e-1):
    
    
    nsplits=1
    if SAVE_MEMORY:
        nsplits=5
        
    #tensor type and device
    device = W.device
    dtype = W.dtype
    
    n = W.shape[0]  
    n_chunk = int(n/nsplits)
    D = torch.zeros(n,  n, dtype=dtype, device=device)
    
    B = torch.diag(A) + t * W
    
    for i in range(nsplits):
        i1 = i*n_chunk
        i2 = np.min([n,(i+1)*n_chunk]).item()

        #U = torch.eye(n, dtype=dtype, device=device)
        U = torch.zeros(n, i2 - i1, dtype=dtype, device=device)
        U[i1:i2, :(i2 - i1)] = torch.eye((i2 - i1), dtype=dtype, device=device)
        f = torch.linalg.solve(B, U)#torch.solve(U, B)[0]
        gf = grad(f)
        gf = gf*(gf.pow(2).sum(1,keepdims=True)+1e-12).rsqrt()
        
        Di = torch.linalg.solve(W, div(gf))#torch.solve(div(gf),W)[0]
        D[:,i1:i2] = Di
    return D






def distance_GIH(V, T, t=1e-1):
    
    W,A = LBO_slim(V, T)
    grad,div,N = _grad_div(V,T)
    
    D = _geodesics_in_heat(grad,div,W[0],A,t)
    d = torch.diag(D)[:,None]
    
    #WARNIG: original D is not symmetric, it is symmetrized and shifted to have diagonal equal to zero
    D = (D + D.t()-d-d.t())/2
#     d = torch.min(D,dim=0)[0][:,None]
#     D = D-d.t()
    
    return D, grad, div, W, A, N


#Not stable
def distance_BH(V,T):
    W, S = LBO_slim(V,T)

    n=W.shape[1]
    A = W[0].mm(((1/ (S[:,None]+1e-6) )*W[0]))
    A[0,:] = 0
    A[:,0] = 0
    A[0,0] = 1

    h = torch.eye(n, dtype=W.dtype, device=W.device)- (1/n)*torch.ones(n,n, dtype=W.dtype, device=W.device)
    h[0,:] = 0

    g = torch.solve(h.double(),A.double())[0].float()
    g = g - torch.sum(g,0,keepdims=True)/n

    v = torch.diag(g)
    D = torch.sqrt(torch.relu(v[None,:]+v[:,None] - 2*g))
    return D

def calc_volume(V,T):    
    T1 = V[:,T[:,:,0],:]
    T2 = V[:,T[:,:,1],:]
    T3 = V[:,T[:,:,2],:]
    XP = torch.cross(T2-T1, T3-T1, -1)
    T_C = (T1+T2+T3)/3
    return (XP*T_C/6).sum([1,2,3])

def LBO_slim(V, F):
    """
    Input:
      V: B x N x 3
      F: B x F x 3
    Outputs:
      C: B x F x 3 list of cotangents corresponding
        angles for triangles, columns correspond to edges 23,31,12
    """
    #tensor type and device
    device = V.device
    dtype = V.dtype
    
    indices_repeat = torch.stack([F, F, F], dim=2)

    # v1 is the list of first triangles B*F*3, v2 second and v3 third
    v1 = torch.gather(V, 1, indices_repeat[:, :, :, 0].long())
    v2 = torch.gather(V, 1, indices_repeat[:, :, :, 1].long())
    v3 = torch.gather(V, 1, indices_repeat[:, :, :, 2].long())

    l1 = torch.sqrt(((v2 - v3) ** 2).sum(2))  # distance of edge 2-3 for every face B*F
    l2 = torch.sqrt(((v3 - v1) ** 2).sum(2))
    l3 = torch.sqrt(((v1 - v2) ** 2).sum(2))

    # semiperimieters
    sp = (l1 + l2 + l3) * 0.5

    # Heron's formula for area
    # A = torch.sqrt( sp * (sp-l1)*(sp-l2)*(sp-l3)).cuda()
    A = 0.5 * (torch.sum(torch.cross(v2 - v1, v3 - v2, dim=2) ** 2, dim=2) ** 0.5)  # VALIDATED

    # Theoreme d Al Kashi : c2 = a2 + b2 - 2ab cos(angle(ab))
    cot23 = (l1 ** 2 - l2 ** 2 - l3 ** 2) / (8 * A)
    cot31 = (l2 ** 2 - l3 ** 2 - l1 ** 2) / (8 * A)
    cot12 = (l3 ** 2 - l1 ** 2 - l2 ** 2) / (8 * A)

    batch_cot23 = cot23.view(-1)
    batch_cot31 = cot31.view(-1)
    batch_cot12 = cot12.view(-1)

    # proof page 98 http://www.cs.toronto.edu/~jacobson/images/alec-jacobson-thesis-2013-compressed.pdf
    # C = torch.stack([cot23, cot31, cot12], 2) / torch.unsqueeze(A, 2) / 8 # dim: [B x F x 3] cotangent of angle at vertex 1,2,3 correspondingly

    B = V.shape[0]
    num_vertices_full = V.shape[1]
    num_faces = F.shape[1]

    edges_23 = F[:, :, [1, 2]]
    edges_31 = F[:, :, [2, 0]]
    edges_12 = F[:, :, [0, 1]]

    batch_edges_23 = edges_23.view(-1, 2)
    batch_edges_31 = edges_31.view(-1, 2)
    batch_edges_12 = edges_12.view(-1, 2)

    W = torch.zeros(B, num_vertices_full, num_vertices_full, dtype=dtype, device=device)

    repeated_batch_idx_f = torch.arange(0, B).repeat(num_faces).reshape(num_faces, B).transpose(1, 0).contiguous().view(
        -1)  # [000...111...BBB...], number of repetitions is: num_faces
    repeated_batch_idx_v = torch.arange(0, B).repeat(num_vertices_full).reshape(num_vertices_full, B).transpose(1,
                                                                                                      0).contiguous().view(
        -1)  # [000...111...BBB...], number of repetitions is: num_vertices_full
    repeated_vertex_idx_b = torch.arange(0, num_vertices_full).repeat(B)

    W[repeated_batch_idx_f, batch_edges_23[:, 0], batch_edges_23[:, 1]] = batch_cot23
    W[repeated_batch_idx_f, batch_edges_31[:, 0], batch_edges_31[:, 1]] = batch_cot31
    W[repeated_batch_idx_f, batch_edges_12[:, 0], batch_edges_12[:, 1]] = batch_cot12

    W = W + W.transpose(2, 1)

    batch_rows_sum_W = torch.sum(W, dim=1).view(-1)
    W[repeated_batch_idx_v, repeated_vertex_idx_b, repeated_vertex_idx_b] = -batch_rows_sum_W
    # W is the contangent matrix VALIDATED
    # Warning: residual error of torch.max(torch.sum(W,dim = 1).view(-1)) is ~ 1e-18

    VF_adj = VF_adjacency_matrix(V[0], F[0]).unsqueeze(0).expand(B, num_vertices_full, num_faces)  # VALIDATED
    V_area = (torch.bmm(VF_adj, A.unsqueeze(2)) / 3).squeeze()  # VALIDATED

    return W, V_area
    

def LBO(V, F):
    W, V_area = LBO_slim(V,F)
    area_matrix = torch.diag_embed(V_area)
    area_matrix_inv = torch.diag_embed(1 / V_area)
    L = torch.bmm(area_matrix_inv, W)  # VALIDATED
    return L, area_matrix, area_matrix_inv, W


def normals(V,T):
    
    #tensor type and device
    device = V.device
    dtype = V.dtype
    
    #WARNING not sure about this
    bs = V.shape[0];
    V = V.reshape([-1,3])
    T = T.reshape([-1,3])

    XF = V[T,:].transpose(0,1)

    Na = torch.cross(XF[1]-XF[0],XF[2]-XF[0])
#     A = torch.sqrt(torch.sum(Na**2,-1,keepdim=True))+1e-6
    
    n = V.shape[0]
    m = T.shape[0]
#     Na=torch.index_select(Na,0,T.view(-1)).view(m,3,3).sum(1)
#      
    Nva = scatter_add( Na.t(), T[:,0], dim_size=n) + \
          scatter_add( Na.t(), T[:,1], dim_size=n) + \
          scatter_add( Na.t(), T[:,2], dim_size=n)
    Nva=Nva.t()
    
    A = torch.sqrt(torch.sum(Nva**2,-1,keepdim=True))+1e-6
    Nva = Nva/A
    return Nva



class Eigendecomposition(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input_matrix, K, N):
        input_matrix_np = input_matrix.data.numpy()
        Knp = K.data.numpy()
        eigvals, eigvecs = eigh(input_matrix_np, eigvals=(0, Knp - 1), lower=False)

        eigvals = torch.from_numpy(eigvals)
        eigvecs = torch.from_numpy(eigvecs)
        ctx.save_for_backward(input_matrix, eigvals, eigvecs, K, N)
        return (eigvecs, eigvals)

    @staticmethod
    def backward(ctx, grad_output, grad_output2):
        # grad_output stands for the grad of eigvecs
        #grad_output2 stands for the grad of eigvecs

        input_matrix, eigvals, eigvecs, K, N = ctx.saved_tensors
        Knp = K.data.numpy()
        Nnp = N.data.numpy()
        grad_K = None
        grad_N = None
        grad_input_matrix = csr_matrix((1, Nnp ** 2), dtype=np.double)

        # constructing the indices for the calculation of sparse du/dL
        #Todo: refactor this call and the same call in optimize.py
        #This is the path for the ground truth mesh
        x = sio.loadmat("./../data/eigendecomposition/downsampled_tr_reg_004.mat")
        adj_VV = x['adj_VV']
        print(adj_VV.shape)
        print(Nnp)
        
        L_mask_flatten = csc_matrix.reshape(adj_VV, (1, Nnp ** 2))
        _, col_ind = L_mask_flatten.nonzero()
        Lnnz = col_ind.shape[0]
        row_ind = np.arange(Nnp)
        row_ind = np.repeat(row_ind, Lnnz)
        col_ind = np.tile(col_ind, Nnp)

        # TODO: parallelize
        for k in range(K):
            print(k)
            lambdaI = eigvals[k] * torch.eye(N).double()
            M = lambdaI - input_matrix
            t = time.time()
            P = torch.pinverse(M.cuda())
            elapsed = time.time() - t
            print(elapsed)

            t = time.time()
            P = csr_matrix(P.cpu().data.numpy())
            elapsed = time.time() - t
            print(elapsed)

            uk = eigvecs[:, k].data.numpy()  # dims: n

            # Sparsity emerges from the knowledge that LBO will always be zero for most of the entries (nomatter of the xyz embedding of the vertices)
            # If dL_ij/dx = 0, then we don't have to consider the path  dF/dL_ij * dL_ij/dx in the calculation of dF/dx
            # Therefore we can assume dF/dL_ij = 0, without changing the result, allowing the matrix Rk = du_kn/dL to be stored in the RAM

            t = time.time()
            data = np.squeeze(np.asarray(uk[col_ind // Nnp])) * np.squeeze(np.asarray(P[row_ind, col_ind % Nnp]))
            Rk = csr_matrix((data, (row_ind, col_ind)), shape=(Nnp, Nnp ** 2))
            elapsed = time.time() - t
            print(elapsed)

            t = time.time()
            grad_input_matrix = grad_input_matrix + csr_matrix(grad_output[:, k], shape=(1, Nnp)).dot(Rk)
            elapsed = time.time() - t
            print(elapsed)

        grad_input_matrix = torch.from_numpy(np.transpose(np.reshape(grad_input_matrix.todense(), (Nnp, Nnp))))
        return grad_input_matrix, grad_K, grad_N


# Aliasing
Eigendecomposition = Eigendecomposition.apply


def calc_euclidean_dist_matrix(x):
    #OH: x contains the coordinates of the mesh,
    #x dimensions are [batch_size x num_nodes x 3]

    #x = x.transpose(2,1)
    r = torch.sum(x ** 2, dim=2).unsqueeze(2)  # OH: [batch_size  x num_points x 1]
    r_t = r.transpose(2, 1) # OH: [batch_size x 1 x num_points]
    inner = torch.bmm(x,x.transpose(2, 1))
    D = F.relu(r - 2 * inner + r_t)**0.5 # OH: the residual numerical error can be negative ~1e-16
    # D = (r - 2 * inner + r_t)**0.5
    return D

#def apply_different_rotation_for_each_point(R,x):
    # OH: R is a torch tensor of dimensions [batch x num_points x 3 x 3]
    #     x i a torch tensor of dimenstions [batch x num_points x 3    ]
    #     the result has the same dimensions as x

#initialize the weights of the network for Convolutional layers and batchnorm layers
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class AverageValueMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):  # OH: n is the weight corresponding to the current value val
        self.val = val
        self.sum += val * n  # OH: weighted sum
        self.count += n  # OH: sum of weights
        self.avg = self.sum / self.count  # OH: weighted average

