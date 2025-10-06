import torch
import torch.nn as nn
from models.vqvae import VQVAE
from models.doravq import DoraVQ

def test_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model parameters
    h_dim = 128
    res_h_dim = 32
    n_res_layers = 2
    n_embeddings = 512
    embedding_dim = 64
    beta = 0.25
    temperature = 1.0
    
    # Create models
    vqvae = VQVAE(h_dim, res_h_dim, n_res_layers, n_embeddings, embedding_dim, beta).to(device)    
    # Test input
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32).to(device)
    
    print("Testing VQVAE...")
    # Test forward pass without loss
    embedding_loss, x_hat, perplexity = vqvae(x)
    print(f"VQVAE - Input shape: {x.shape}")
    print(f"VQVAE - Output shape: {x_hat.shape}")
    print(f"VQVAE - Embedding loss: {embedding_loss.item():.4f}")
    print(f"VQVAE - Perplexity: {perplexity.item():.4f}")
    
    # Test forward pass with loss
    total_loss, recon_loss, embedding_loss, perplexity, x_hat_loss = vqvae(x, return_loss=True)
    print(f"VQVAE - Total loss: {total_loss.item():.4f}")
    print(f"VQVAE - Recon loss: {recon_loss.item():.4f}")
    print(f"VQVAE - Embedding loss: {embedding_loss.item():.4f}")
    print(f"VQVAE - Output shapes match: {x_hat.shape == x_hat_loss.shape}")
    
    print("\n" + "="*50)
    print("Testing DoraVQ...")
    
    # Create DoraVQ model
    doravq = DoraVQ(
        num_embeddings=n_embeddings,
        embedding_dim=embedding_dim,
        commitment_cost=beta,
        temperature=temperature,
        dirichlet_alpha=0.1,
        h_dim=h_dim,
        n_res_layers=n_res_layers,
        res_h_dim=res_h_dim,
        top_k=None  # Test without top-K filtering by default
    ).to(device)
    
    # Test forward pass without soft assignment
    x_recon, loss, perplexity, encoding_indices, distances = doravq(x)
    print(f"DoraVQ - Input shape: {x.shape}")
    print(f"DoraVQ - Output shape: {x_recon.shape}")
    print(f"DoraVQ - Loss: {loss.item():.4f}")
    print(f"DoraVQ - Perplexity: {perplexity.item():.4f}")
    print(f"DoraVQ - Encoding indices shape: {encoding_indices.shape}")
    print(f"DoraVQ - Distances shape: {distances.shape}")
    
    # Test forward pass with soft assignment
    x_recon_soft, loss_soft, perplexity_soft, encoding_indices_soft, h_per_sample = doravq(x, return_soft_assignment=True)
    print(f"DoraVQ (soft) - Output shape: {x_recon_soft.shape}")
    print(f"DoraVQ (soft) - Loss: {loss_soft.item():.4f}")
    print(f"DoraVQ (soft) - Perplexity: {perplexity_soft.item():.4f}")
    print(f"DoraVQ (soft) - Encoding indices shape: {encoding_indices_soft.shape}")
    print(f"DoraVQ (soft) - Soft assignment shape: {h_per_sample.shape}")
    
    # Test compute_soft_assignment method
    distances_flat = distances.view(-1, n_embeddings)
    h_flat = doravq.compute_soft_assignment(distances_flat)
    print(f"DoraVQ - Soft assignment flat shape: {h_flat.shape}")
    
    # Test sample_dirichlet_prior method
    dirichlet_samples = doravq.sample_dirichlet_prior(batch_size)
    print(f"DoraVQ - Dirichlet samples shape: {dirichlet_samples.shape}")
    
    # Test discriminator
    discriminator_output = doravq.discriminator(h_per_sample)
    print(f"DoraVQ - Discriminator output shape: {discriminator_output.shape}")
    
    # Verify that outputs are consistent
    print(f"DoraVQ - Output shapes consistent: {x_recon.shape == x_recon_soft.shape}")
    print(f"DoraVQ - Loss values consistent: {abs(loss.item() - loss_soft.item()) < 1e-6}")
    
    print("\n" + "="*50)
    print("All tests passed successfully!")

if __name__ == "__main__":
    test_models()