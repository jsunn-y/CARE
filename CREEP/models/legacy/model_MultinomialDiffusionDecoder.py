import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists
from transformers import BertConfig
from protein.models.model_SDE import VESDE, VPSDE
from protein.models.model_Sampler import ReverseDiffusionPredictor, LangevinCorrector
from protein.models.score_networks import ToyScoreNetwork, RNNScoreNetwork, BertScoreNetwork

EPS = 1e-6


class MultinomialDiffusion(nn.Module):
    def __init__(
        self, hidden_dim, condition_dim, beta_min, beta_max, num_diffusion_timesteps, mask_id, num_classes, score_network_type
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.num_classes = num_classes
        self.mask_id = mask_id

        # self.SDE_func = VPSDE(beta_min=self.beta_min, beta_max=self.beta_max, N=self.num_diffusion_timesteps)
        

        output_dim = hidden_dim
        if score_network_type == "Toy":
            word_embedding_dim = self.hidden_dim
            self.score_network = ToyScoreNetwork(hidden_dim=hidden_dim, output_dim=output_dim)

        elif score_network_type == "RNN":
            word_embedding_dim = self.hidden_dim
            self.score_network = RNNScoreNetwork(hidden_dim=hidden_dim, output_dim=output_dim)

        elif score_network_type == "BertBase":
            config = BertConfig.from_pretrained(
                "bert-base-uncased",
                cache_dir="../data/temp_Bert_base",
                vocab_size=self.num_classes,
                hidden_size=hidden_dim,
                num_attention_heads=8
            )
            word_embedding_dim = self.hidden_dim
            self.score_network = BertScoreNetwork(config=config, output_dim=output_dim)

        self.word_embedding_dim = word_embedding_dim
        self.embedding_layer = nn.Linear(self.num_classes, self.word_embedding_dim, bias=False)
        self.decoder_layer = nn.Linear(word_embedding_dim, self.num_classes)
        self.condition_proj_layer = nn.Linear(self.condition_dim, self.word_embedding_dim)

        self.CE_criterion = nn.CrossEntropyLoss(reduction='none')
        return
    
    def forward(self, protein_seq_input_ids, protein_seq_attention_mask, condition):
        B = protein_seq_input_ids.size()[0]
        device = protein_seq_input_ids.device

        # TODO: need double-check range of timesteps
        timesteps = torch.rand(B, device=device) * (1 - EPS) + EPS  # (B)
        timesteps = torch.randint(1, 1+self.num_diffusion_timesteps, (B,), device=device)
        # pt = torch.ones_like(timesteps).float() / self.num_diffusion_timesteps

        condition = condition.float()
        condition = self.condition_proj_layer(condition)  # (B, max_seq_len, condition_dim) ---> (B, max_seq_len, hidden_dim)
        
        protein_seq_onehot = F.one_hot(protein_seq_input_ids, num_classes=self.num_classes)  # (B, max_seq_len, num_class)

        x_t, x_0_ignore = protein_seq_input_ids.clone(), protein_seq_input_ids.clone()
        
        mask = torch.rand_like(x_t.float()) < timesteps.float().unsqueeze(-1) / self.num_diffusion_timesteps  # (B, max_seq_len)
        x_t[mask] = self.mask_id  # (B, max_seq_len)
        x_0_ignore[torch.bitwise_not(mask)] = -1  # (B, max_seq_len)

        x_t_one_hot = F.one_hot(x_t, num_classes=self.num_classes)  # (B, max_seq_len, num_class)
        x_t_one_hot = x_t_one_hot.float()
        x_t_repr = self.embedding_layer(x_t_one_hot)  # (B, max_seq_len, hidden_dim)
        
        x_0_repr = self.score_network(protein_seq_repr=x_t_repr, protein_seq_attention_mask=protein_seq_attention_mask, condition=condition)  # (B, max_seq_len, hidden_dim)
        x_0_logits = self.decoder_layer(x_0_repr)   # (B*max_sequence_len, num_class)

        flattened_logits = x_0_logits.view(-1, x_0_logits.size(-1))  # (B*max_sequence_len, num_class)
        flattened_ids = protein_seq_input_ids.view(-1)  # (B*max_sequence_len)
        flattened_mask = protein_seq_attention_mask.view(-1)  # (B*max_sequence_len)
        total_SDE_loss = self.CE_criterion(flattened_logits, flattened_ids)  # (B*max_sequence_len)
        masked_SDE_loss = total_SDE_loss * flattened_mask  # (B*max_sequence_len)
        total_SDE_loss = torch.mean(total_SDE_loss)
        masked_SDE_loss = masked_SDE_loss.sum() / flattened_mask.sum()
        # previously:
        # masked_SDE_loss = total_SDE_loss.sum() / flattened_mask.sum()

        SDE_loss = total_SDE_loss + masked_SDE_loss
        decoding_loss = 0

        return SDE_loss, decoding_loss

    @torch.no_grad()
    def inference(self, condition, max_seq_len, protein_seq_attention_mask):
        B = condition.size()[0]
        device = condition.device
        
        shape = (B, max_seq_len, self.num_classes)

        condition = condition.float()
        condition = self.condition_proj_layer(condition)  # (B, max_seq_len, condition_dim) ---> (B, max_seq_len, hidden_dim)

        x_t = torch.ones((B, max_seq_len), device=device).long() * self.mask_id
        unmasked = torch.zeros_like(x_t, device=device).bool()
        temperature = 1.

        for t in reversed(range(1, 1+self.num_diffusion_timesteps)):
            t = torch.full((B,), t, device=device).long()  # (B)
            
            # where to unmask
            changes = torch.rand(x_t.shape, device=device) < 1/t.float().unsqueeze(-1)  # (B, max_seq_len)
            # don't unmask somwhere already masked
            changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))  # (B, max_seq_len)
            # update mask with changes
            unmasked = torch.bitwise_or(unmasked, changes)  # (B, max_seq_len)

            x_t_one_hot = F.one_hot(x_t, num_classes=self.num_classes)  # (B, max_seq_len, num_class)
            x_t_one_hot = x_t_one_hot.float()
            x_t_repr = self.embedding_layer(x_t_one_hot)  # (B, max_seq_len, hidden_dim)

            x_0_repr = self.score_network(protein_seq_repr=x_t_repr, protein_seq_attention_mask=protein_seq_attention_mask, condition=condition)  # (B, max_seq_len, hidden_dim)
            x_0_logits = self.decoder_layer(x_0_repr)   # (B, max_sequence_len, num_class)

            x_0_logits /= temperature
            x_0_dist = dists.Categorical(logits=x_0_logits)
            x_0_hat = x_0_dist.sample().long()

            x_t[changes] = x_0_hat[changes]

        x = x_t
        x = F.one_hot(x, num_classes=self.num_classes)

        return x