import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, entropy_loss_ratio, l2_norm, show_usage, vq_loss_ratio):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.vq_loss_ratio = vq_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        if self.l2_norm:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)
        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(torch.zeros(65536), requires_grad=False))

    
    def forward(self, z, top_k=None, noise_end=0):
        
        # reshape z -> (batch, height, width, channel) and flatten
        if z.dim() == 4:
            z = torch.einsum('b c h w -> b h w c', z).contiguous()
        else:
            z = z.contiguous()
        
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        if self.l2_norm:
            z = F.normalize(z, p=2, dim=-1)
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, torch.einsum('n d -> d n', embedding))
        
        if top_k is None:
            min_encoding_indices = torch.argmin(d, dim=1)
            z_q_noise = None
            selected_indices = None
        else:
            top_k_indices = torch.topk(d, k=top_k, dim=1, largest=False).indices

            random_indices = torch.randint(0, top_k, (d.shape[0],), device=d.device)
            selected_indices = top_k_indices[torch.arange(d.shape[0], device=d.device), random_indices]
            min_encoding_indices = top_k_indices[:, 0]
            z_q_noise = embedding[selected_indices].view(z.shape)
            z_q_noise = z + (z_q_noise - z).detach()

        z_q = embedding[min_encoding_indices].view(z.shape)
        vq_loss = None
        commit_loss = None
        entropy_loss = None
        codebook_usage = 0

        if self.show_usage and self.training:
            cur_len = min_encoding_indices.shape[0]
            self.codebook_used[:-cur_len].copy_(self.codebook_used[cur_len:].clone())
            self.codebook_used[-cur_len:].copy_(min_encoding_indices)
            codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e

        # compute loss for embedding
        if self.training:
            vq_loss = torch.mean((z_q[:, noise_end:] - z.detach()[:, noise_end:]) ** 2) * self.vq_loss_ratio
            commit_loss = self.beta * torch.mean((z_q.detach()[:, noise_end:] - z[:, noise_end:]) ** 2) 
            entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-d)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        if z.dim() == 4:
            z_q = torch.einsum('b h w c -> b c h w', z_q)

        return z_q, (vq_loss, commit_loss, entropy_loss, codebook_usage), (z_q_noise, selected_indices, min_encoding_indices)

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        # shape = (batch, channel, height, width) if channel_first else (batch, height, width, channel)
        if self.l2_norm:
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight
        z_q = embedding[indices]  # (b*h*w, c)

        if shape is not None:
            if channel_first and len(shape) == 4:
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[1])
                # reshape back to match original input shape
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
            else:
                z_q = z_q.view(shape)
        return z_q
    
    def get_embedding(self):
        if self.l2_norm:
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight
        return embedding


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
    avg_entropy = - torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    sample_entropy = - torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss
