import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

def gumbel_softmax(logits, tau, hard=False, dim=-1):
    # Sample noise from Gumbel(0, 1)
    gumbels = -torch.empty_like(logits).exponential_().log()  # Sample from Gumbel(0, 1)
    gumbels = (logits + gumbels) / tau  # Add noise and apply temperature
    y_soft = gumbels.softmax(dim)

    if hard:
        # Create one-hot encoded vectors from y_soft
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft

    return ret

class YieldPredictor(nn.Module):
    def __init__(self, 
                 feature_dim,
                 label_dim,
                 hidden_size,
                 threshold,
                 layers,
                 dropout,
                 mlp_output_size):
        super(YieldPredictor, self).__init__()
        # Soft label embeddings will remain the same for each instance in the batch
        self.threshold = threshold
        self.label_embeddings = nn.Parameter(torch.Tensor(3, label_dim))
        nn.init.xavier_uniform_(self.label_embeddings, gain=nn.init.calculate_gain('relu'))
        self.label_dim = label_dim
        self.feat_mlp = nn.Sequential(nn.Linear(feature_dim, hidden_size),
                                        nn.ReLU(),
                                        nn.Dropout(dropout))
        self.label_mlp = nn.Linear(label_dim, hidden_size)
        # Define MLPs for high, medium, and low yield predictions
        self.high_mlp = self._build_mlp(hidden_size*2, hidden_size,layers,dropout,mlp_output_size)
        self.medium_mlp = self._build_mlp(hidden_size*2, hidden_size,layers,dropout,mlp_output_size)
        self.low_mlp = self._build_mlp(hidden_size*2, hidden_size,layers,dropout,mlp_output_size)

        
        self.high_pre_mlp = nn.Sequential(nn.Linear(hidden_size*2, hidden_size*4),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(hidden_size*4, hidden_size*2),
                                        nn.ReLU(),
                                        nn.Dropout(0.2)
                                        )
        
        self.medium_pre_mlp = nn.Sequential(nn.Linear(hidden_size*2, hidden_size*4),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(hidden_size*4, hidden_size*2),
                                        nn.ReLU(),
                                        nn.Dropout(0.2)
                                        )
        
        self.low_pre_mlp = nn.Sequential(nn.Linear(hidden_size*2, hidden_size*4),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(hidden_size*4, hidden_size*2),
                                        nn.ReLU(),
                                        nn.Dropout(0.2)
                                        )
        
        self.cls_mlp = nn.Sequential(nn.Linear(hidden_size*6, hidden_size*3),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(hidden_size*3, hidden_size),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(hidden_size, 3)
                
                                        )

    def _build_mlp(self, input_dim, hidden_size, layers, dropout, output_size):
        # Initialize list to hold the layers
        mlp_layers = []

        # Input layer
        mlp_layers.append(nn.Linear(input_dim, hidden_size))
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(layers - 1):  # Subtracting one because we've already added the first layer
            mlp_layers.append(nn.Linear(hidden_size, hidden_size//2))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            hidden_size = hidden_size // 2  # Reduce the size for the next layer

        # Output layer
        mlp_layers.append(nn.Linear(hidden_size, output_size))

        # Combine layers into a single sequential module
        return nn.Sequential(*mlp_layers)


    def forward(self, reaction_encodings,hard,tau):
        reaction_encodings = self.feat_mlp(reaction_encodings) # Shape: [batch_size, hidden_size]
        batch_size, hidden_size = reaction_encodings.size()  

        # Expand label embeddings to match the batch size
        label_embeddings_batch = self.label_embeddings.expand(batch_size, -1, -1)

        # Prepare query and keys for attention
        query = reaction_encodings.unsqueeze(1)  # Shape: [batch_size, 1, hidden_size
        label_embeddings_batch = self.label_mlp(label_embeddings_batch) # Shape: [batch_size, 3, hidden_size]
        keys = label_embeddings_batch.transpose(1, 2) # Shape: [batch_size, hidden_size, 3]

        # # Compute attention scores using batch matrix multiplication
        scores = torch.bmm(query, keys).squeeze(1)  # Shape: [batch_size, 3]
        attention_weights = F.softmax(scores, dim=1)  # Shape: [batch_size, 3]
        updated_label_embeddings = torch.bmm(attention_weights.unsqueeze(1), label_embeddings_batch)
        updated_label_embeddings = updated_label_embeddings.squeeze(1)  # Shape: [batch_size, hidden_size]
        reaction_encodings = reaction_encodings + updated_label_embeddings  # Simple update rule
        gumbel_attention_probs = gumbel_softmax(scores, tau=tau, hard=hard, dim=1)

        concatenated_inputs = torch.cat([reaction_encodings.unsqueeze(1).expand(-1, 3, -1),
                                         label_embeddings_batch], dim=2) # Shape: [batch_size, 3, hidden_size*2]
        # pdb.set_trace()
        high_features = self.high_pre_mlp(concatenated_inputs[:,0,:])
        medium_features = self.medium_pre_mlp(concatenated_inputs[:,1,:])
        low_features = self.low_pre_mlp(concatenated_inputs[:,2,:])
        concatenated_cls_inputs = torch.cat([high_features, medium_features, low_features], dim=1)
        cls_logits = self.cls_mlp(concatenated_cls_inputs)
    # Shape: [batch_size, 3]

    # Use softmax to ensure the outputs sum up to 1 for each instance
        cls_prob = F.softmax(cls_logits, dim=1)

        high_yield = self.high_mlp(high_features)
        medium_yield = self.medium_mlp(medium_features)
        low_yield= self.low_mlp(low_features)
        
        # Combine the MLP outputs
        yields = torch.stack([high_yield, medium_yield, low_yield], dim=1)  # Shape: [batch_size, 3]
        # cls = torch.stack([torch.sigmoid(high_yield[:,1]),
        #                     torch.sigmoid(medium_yield[:,1]), 
        #                     torch.sigmoid(low_yield[:,1])], dim=1)  # Shape: [batch_size, 3]
        # apply an auxiliary classifier
        # pdb.set_trace()

        return yields,cls_prob,label_embeddings_batch,gumbel_attention_probs,concatenated_cls_inputs

