import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder
from .quantizer import VectorQuantizer


class Discriminator(nn.Module):
    """判别器：区分真实Dirichlet样本和生成的软分配h"""
    def __init__(self, num_embeddings):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embeddings, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, h):
        """h: (B, K) 概率向量"""
        return self.net(h).squeeze(-1)

class DoraVQ(nn.Module):
    """VQ-VAE + Dirichlet对抗正则化 (Ours)"""
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25, 
                 temperature=1.0, dirichlet_alpha=0.1, h_dim=128, n_res_layers=3, res_h_dim=64):
        super().__init__()
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(h_dim, embedding_dim, kernel_size=1, stride=1)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)
        self.discriminator = Discriminator(num_embeddings)
        
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.num_embeddings = num_embeddings
        
    def compute_soft_assignment(self, distances):
        """
        从距离计算软分配概率h (未平均)
        distances: (B*H*W, K)
        返回: (B*H*W, K) - 每个编码向量的软分配概率
        """
        # 转换为概率 (B*H*W, K)
        h = F.softmax(-distances / self.temperature, dim=-1)
        return h
    
    def sample_dirichlet_prior(self, batch_size):
        """从Dirichlet(alpha, ..., alpha)采样"""
        alpha = torch.full((self.num_embeddings,), self.dirichlet_alpha)
        dist = torch.distributions.Dirichlet(alpha)
        samples = dist.sample((batch_size,))
        return samples.to(next(self.parameters()).device)
        
    def forward(self, x, return_soft_assignment=False):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        # 动态获取形状，消除硬编码
        B, D, H, W = z_e.shape
        
        z_q, loss, perplexity, encoding_indices, distances = self.vq(z_e)
        x_recon = self.decoder(z_q)
        
        if return_soft_assignment:
            # 计算软分配h（用于对抗训练）
            h_flat = self.compute_soft_assignment(distances) # (B*H*W, K)
            
            # 动态地reshape和求平均
            h_reshaped = h_flat.view(B, H * W, -1)      # (B, H*W, K)
            h = h_reshaped.mean(dim=1)      # (B, K)
            
            return x_recon, loss, perplexity, encoding_indices, h
        
        return x_recon, loss, perplexity, encoding_indices, distances
    
    def reconstruct(self, x):
        """
        重构输入图像
        """
        with torch.no_grad():
            x_recon, _, _, _, _ = self.forward(x)
            return x_recon
