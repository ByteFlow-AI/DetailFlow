import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class SoftVectorQuantizer(nn.Module):
    def __init__(
        self,
        n_e,
        e_dim,
        entropy_loss_ratio=0.01,
        tau=0.07,
        num_codebooks=1,
        l2_norm=False,
        show_usage=False,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.n_e = n_e
        self.e_dim = e_dim
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage
        self.tau = tau
        
        # Single embedding layer for all codebooks
        self.embedding = nn.Parameter(torch.randn(num_codebooks, n_e, e_dim))
        self.embedding.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        
        if self.l2_norm:
            self.embedding.data = F.normalize(self.embedding.data, p=2, dim=-1)
        
        if self.show_usage:
            self.register_buffer("codebook_used", torch.zeros(num_codebooks, 65536))

    def forward(self, z):
        # Handle different input shapes
        if z.dim() == 4:
            z = torch.einsum('b c h w -> b h w c', z).contiguous()
            z = z.view(z.size(0), -1, z.size(-1))
        
        batch_size, seq_length, _ = z.shape
        
        # Ensure sequence length is divisible by number of codebooks
        assert seq_length % self.num_codebooks == 0, \
            f"Sequence length ({seq_length}) must be divisible by number of codebooks ({self.num_codebooks})"
        
        segment_length = seq_length // self.num_codebooks
        z_segments = z.view(batch_size, self.num_codebooks, segment_length, self.e_dim)
        
        # Apply L2 norm if needed
        embedding = F.normalize(self.embedding, p=2, dim=-1) if self.l2_norm else self.embedding
        if self.l2_norm:
            z_segments = F.normalize(z_segments, p=2, dim=-1)
            
        z_flat = z_segments.permute(1, 0, 2, 3).contiguous().view(self.num_codebooks, -1, self.e_dim)
        
        logits = torch.einsum('nbe, nke -> nbk', z_flat, embedding.detach())
        
        # Calculate probabilities
        probs = F.softmax(logits / self.tau, dim=-1)  
        
        
        # Quantize
        z_q = torch.einsum('nbk, nke -> nbe', probs, embedding)
        
        # Reshape back
        z_q = z_q.view(self.num_codebooks, batch_size, segment_length, self.e_dim).permute(1, 0, 2, 3).contiguous()
        
        
        # Calculate cosine similarity
        with torch.no_grad():
            zq_z_cos = F.cosine_similarity(
                z_segments.view(-1, self.e_dim),
                z_q.view(-1, self.e_dim),
                dim=-1
            ).mean()
        
        # Get indices for usage tracking
        indices = torch.argmax(probs, dim=-1)  # (batch*segment_length, num_codebooks)
        
        # Track codebook usage
        if self.show_usage and self.training:
            for k in range(self.num_codebooks):
                cur_len = indices.size(0)
                self.codebook_used[k, :-cur_len].copy_(self.codebook_used[k, cur_len:].clone())
                self.codebook_used[k, -cur_len:].copy_(indices[:, k])
        
        # Calculate losses if training
        if self.training:
            vq_loss = commit_loss = 0.0            
            entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(logits.view(-1, self.n_e))
        else:
            vq_loss = commit_loss = entropy_loss = None
        
        # Calculate codebook usage
        codebook_usage = torch.tensor([
            len(torch.unique(self.codebook_used[k])) / self.n_e 
            for k in range(self.num_codebooks)
        ]).mean() if self.show_usage else 0

        z_q = z_q.view(batch_size, -1, self.e_dim)
        
        # Reshape back to match original input shape
        if len(z.shape) == 4:
            z_q = torch.einsum('b h w c -> b c h w', z_q)
        
        # Calculate average probabilities
        avg_probs = torch.mean(torch.mean(probs, dim=-1))
        max_probs = torch.mean(torch.max(probs, dim=-1)[0])
        
        return z_q, (vq_loss, commit_loss, entropy_loss, codebook_usage), (
            None,  # perplexity
            None,  # min_encodings
            indices.view(batch_size, self.num_codebooks, segment_length),
            avg_probs,
            max_probs,
            z_q.detach(),
            z.detach(),
            zq_z_cos
        )


def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    avg_probs = torch.mean(target_probs, dim=0)
    avg_entropy = - torch.sum(avg_probs * torch.log(avg_probs + 1e-6))
    sample_entropy = - torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss
