import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """标准VQ层"""
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # 码本
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, z_e):
        """
        z_e: (B, D, H, W)
        返回: z_q, loss, perplexity, encodings, distances
        """
        # 转换形状: (B, D, H, W) -> (B, H, W, D) -> (B*H*W, D)
        z_e_permuted = z_e.permute(0, 2, 3, 1).contiguous()
        flat_z_e = z_e_permuted.view(-1, self.embedding_dim)
        
        # 计算距离: (B*H*W, K)
        distances = (torch.sum(flat_z_e**2, dim=1, keepdim=True)
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_z_e, self.embedding.weight.t()))
        
        # 硬量化
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # 查表
        z_q = torch.matmul(encodings, self.embedding.weight)
        z_q = z_q.view(z_e_permuted.shape)
        
        # 损失
        e_latent_loss = F.mse_loss(z_q.detach(), z_e_permuted)
        q_latent_loss = F.mse_loss(z_q, z_e_permuted.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # 直通估计器
        z_q = z_e_permuted + (z_q - z_e_permuted).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        # 困惑度（衡量码本使用多样性）
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return z_q, loss, perplexity, encoding_indices, distances
