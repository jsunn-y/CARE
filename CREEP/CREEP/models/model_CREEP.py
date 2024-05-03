import torch.nn as nn

class CREEPModel(nn.Module):
    def __init__(self, protein_model, text_model, reaction_model, protein2latent_model, text2latent_model, reaction2latent_model, reaction2protein_facilitator_model, protein2reaction_facilitator_model,protein_model_name, text_model_name, reaction_model_name):
        super().__init__()
        self.protein_model = protein_model
        self.text_model = text_model
        self.reaction_model = reaction_model
        self.protein2latent_model = protein2latent_model
        self.text2latent_model = text2latent_model
        self.reaction2latent_model = reaction2latent_model
        self.reaction2protein_facilitator_model = reaction2protein_facilitator_model
        self.protein2reaction_facilitator_model = protein2reaction_facilitator_model

        self.protein_model_name = protein_model_name
        self.text_model_name = text_model_name
        self.reaction_model_name = reaction_model_name
        return
    
    def forward(self, protein_sequence_input_ids, protein_sequence_attention_mask, text_sequence_input_ids, text_sequence_attention_mask, reaction_sequence_input_ids, reaction_sequence_attention_mask):

        protein_output = self.protein_model(protein_sequence_input_ids, protein_sequence_attention_mask)
        if self.protein_model_name == "ProtT5":
            protein_repr = protein_output["last_hidden_state"]
            protein_repr = protein_repr.mean(dim=1)

            #try using the first CLS token repsentation instead (doesn't seem to work as well)
            #protein_repr = protein_output["last_hidden_state"][:,0,:]

        else: #for BERT-based protein models
            protein_repr = protein_output["pooler_output"]
        protein_repr = self.protein2latent_model(protein_repr)

        description_output = self.text_model(text_sequence_input_ids, text_sequence_attention_mask)
        description_repr = description_output["pooler_output"]
        description_repr = self.text2latent_model(description_repr)

        reaction_output = self.reaction_model(reaction_sequence_input_ids, reaction_sequence_attention_mask)
        #reaction_repr = reaction_output["pooler_output"]
        #CLS token representation seems to work better here
        reaction_repr = reaction_output["last_hidden_state"][:,0,:]
        reaction_repr = self.reaction2latent_model(reaction_repr)
        
        reaction2protein_repr = self.reaction2protein_facilitator_model(reaction_repr)
        protein2reaction_repr = self.protein2reaction_facilitator_model(protein_repr)

        return protein_repr, description_repr, reaction_repr, reaction2protein_repr, protein2reaction_repr

class SingleModalityModel(nn.Module):
    def __init__(self, modality_model, modality2latent_model, modality_model_name, modality='protein'):
        super().__init__()
        self.modality_model = modality_model
        self.modality2latent_model = modality2latent_model
        self.modality = modality
        self.modality_model_name = modality_model_name
        return
    
    def forward(self, sequence_input_ids, sequence_attention_mask):

        output = self.modality_model(sequence_input_ids, sequence_attention_mask)

        if self.modality == 'protein':
            if self.modality_model_name == "ProtT5":
                repr = output["last_hidden_state"]
                repr = repr.mean(dim=1)
                #try using the first CLS token repsentation instead (doesn't seem to work as well)
                #protein_repr = protein_output["last_hidden_state"][:,0,:]
            else: #for BERT-based protein models
                repr = output["pooler_output"]
        elif self.modality == 'reaction':
            repr = output["last_hidden_state"][:,0,:]
        elif self.modality == 'text':
            repr = output["pooler_output"]
        
        repr = self.modality2latent_model(repr)

        return repr

# class ProteinTextReactionModel_with_mine_EC(ProteinTextReactionModel):
#     def __init__(self, protein_model, text_model, reaction_model, protein2latent_model, text2latent_model, reaction2latent_model, reaction2protein_facilitator_model, protein2reaction_facilitator_model, protein_model_name, text_model_name, reaction_model_name):
#         super().__init__(protein_model, text_model, reaction_model, protein2latent_model, text2latent_model, reaction2latent_model, reaction2protein_facilitator_model, protein2reaction_facilitator_model, protein_model_name, text_model_name, reaction_model_name)
    
#     def forward(self, protein_sequence_input_ids, protein_sequence_attention_mask, text_sequence_input_ids, text_sequence_attention_mask, reaction_sequence_input_ids, reaction_sequence_attention_mask):

#         protein_repr, description_repr, reaction_repr, reaction2protein_repr, protein2reaction_repr = super().forward(protein_sequence_input_ids, protein_sequence_attention_mask, text_sequence_input_ids, text_sequence_attention_mask, reaction_sequence_input_ids, reaction_sequence_attention_mask)
