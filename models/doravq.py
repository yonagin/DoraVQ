import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder
from .quantizer import VectorQuantizer


class Discriminator(nn.Module):
    """判别器：区分真实Dirichlet样本和生成的软分配h"""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
        )
        
    def forward(self, h):
        """h: (B, K) 概率向量"""
        return self.net(h).squeeze(-1)

class DoraVQ(nn.Module):
    """VQ-VAE + Dirichlet对抗正则化 (Ours)"""
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25, 
                 temperature=1.0, dirichlet_alpha=0.1, h_dim=128, n_res_layers=3, res_h_dim=64,
                 top_k=None):
        super().__init__()
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(h_dim, embedding_dim, kernel_size=1, stride=1)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)
        
        # 根据是否使用top-K过滤来决定判别器的输入维度
        discriminator_input_dim = top_k if top_k is not None and top_k < num_embeddings else num_embeddings
        self.discriminator = Discriminator(discriminator_input_dim)
        
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.num_embeddings = num_embeddings
        self.top_k = top_k
        
    def get_h(self, x):
        """
        从距离计算软分配概率h (未平均)
        distances: (B*H*W, K)
        返回: (B*H*W, K) - 每个编码向量的软分配概率
        """
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        B = z_e.shape[0]
        z_q, loss, perplexity, encoding_indices, distances = self.vq(z_e)
        # 转换为概率 (B*H*W, K)
        h_flat = F.softmax(-distances / self.temperature, dim=-1)
         # 动态地reshape和求平均
        h_reshaped = h_flat.view(B, -1, self.num_embeddings)    # (B, H*W, K)
        h = h_reshaped.mean(dim=1)      # (B, K)
        
        # 如果设置了top_k参数，则进行top-K过滤
        if self.top_k is not None and self.top_k < self.num_embeddings:
            # 获取每个样本中概率最大的前K个索引和值
            topk_values, topk_indices = torch.topk(h, self.top_k, dim=-1)
            
            # 直接使用top-K个值，而不是保持完整维度
            # 这样h的维度就从(B, num_embeddings)变为(B, top_k)
            h = F.normalize(topk_values, p=1, dim=-1)

        return z_q, loss, perplexity, encoding_indices, distances, h
    
    def sample_dirichlet_prior(self, batch_size):
        """从Dirichlet(alpha, ..., alpha)采样"""
        # 根据是否使用top-K过滤来决定采样的维度
        if self.top_k is not None and self.top_k < self.num_embeddings:
            # 使用top-K过滤时，只采样top-K个维度
            alpha = torch.full((self.top_k,), self.dirichlet_alpha)
        else:
            # 不使用top-K过滤时，采样完整的维度
            alpha = torch.full((self.num_embeddings,), self.dirichlet_alpha)
            
        dist = torch.distributions.Dirichlet(alpha)
        samples = dist.sample((batch_size,))
        
        # 如果使用top-K过滤，直接返回top-K维度的采样结果
        # 不需要扩展到完整维度，因为判别器期望的输入维度是top_k
        return samples.to(next(self.parameters()).device)
        
    def forward(self, x):
        z_q, loss, perplexity, encoding_indices, distances, h = self.get_h(x)
        x_recon = self.decoder(z_q) 
        return x_recon, loss, perplexity, encoding_indices, distances, h
    
    def reconstruct(self, x):
        """
        重构输入图像
        """
        with torch.no_grad():
            x_recon, _, _, _, _, _ = self.forward(x)
            return x_recon
