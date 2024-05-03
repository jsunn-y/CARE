import torch.nn as nn

#use this model for the facilitator model for generative
class AEFacilitatorModel(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.MLP = nn.Sequential(
            nn.Linear( self.latent_dim, self.latent_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.latent_dim, self.latent_dim)
        )
        return
    
    def forward(self, input_repr):
        output_repr_pred = self.MLP(input_repr)
        return output_repr_pred
    
class GaussianFacilitatorModel(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.MLP = nn.Sequential(
            nn.Linear( self.latent_dim, self.latent_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

        self.criterion = nn.MSELoss()
        return
    
    def forward(self, output_repr, input_repr):
        protein_repr_pred = self.MLP(input_repr)
        loss = self.criterion(output_repr, protein_repr_pred)
        return loss
    
    def inference(self, input_repr):
        output_repr_pred = self.MLP(input_repr)
        return output_repr_pred