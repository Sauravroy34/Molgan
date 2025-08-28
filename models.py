import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Generator", "MolGANDiscriminator"]


class Generator(nn.Module):
    def __init__(self, latent_dim=32, N=9, T=5, Y=5, dropout_rate=0.01):
        super().__init__()
        
        self.latent_dim = latent_dim 
        self.N = N  
        self.T = T  
        self.Y = Y  
        
        self.atom_mlp = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate), 
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate), 
            nn.Linear(512, N * T),
        )
        
        self.adj_mlp = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate), 
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate), 
            nn.Linear(512, N * N * Y),
        )
        
    def forward(self, z):
        batch_size = z.size(0)
        
        x = self.atom_mlp(z) 
        x = x.view(batch_size, self.N, self.T)
        
        a = self.adj_mlp(z)
        a = a.view(batch_size, self.N, self.N, self.Y)
        a = (a + a.permute(0, 2, 1, 3)) / 2

        
        return a, x

    
class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features, num_bond_types, activation=torch.sigmoid):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_bond_types = num_bond_types
        self.activation = activation
        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features, out_features) for _ in range(num_bond_types)
        ])
        self.self_connection_layer = nn.Linear(in_features, out_features)

    def forward(self, node_features, adjacency_tensor):
        transformed_features_by_type = [linear_layer(node_features) for linear_layer in self.linear_layers]
        aggregated_features = torch.zeros(
            node_features.size(0), 
            node_features.size(1), 
            self.out_features,
            device=node_features.device # Ensure tensor is on the correct device
        )
        for i in range(self.num_bond_types):
            aggregated_features += torch.matmul(adjacency_tensor[:, :, :, i], transformed_features_by_type[i])
        self_connection_features = self.self_connection_layer(node_features)
        output = aggregated_features + self_connection_features
        if self.activation is not None:
            output = self.activation(output)
        return output


class GraphAggregationLayer(nn.Module):
    # This class remains unchanged
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.i = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Sigmoid()
        )
        self.j = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Tanh()
        )

    def forward(self, node_features):
        i_output = self.i(node_features)
        j_output = self.j(node_features)
        aggregated_vector = torch.sum(i_output * j_output, dim=1)
        return torch.tanh(aggregated_vector)


class MolGANDiscriminator(nn.Module):
    def __init__(self, node_feature_dim=5, num_bond_types=4, dropout_rate=0.01):
        super().__init__()
        self.gcn_layer_1 = GraphConvolutionLayer(node_feature_dim, 64, num_bond_types)
        self.gcn_layer_2 = GraphConvolutionLayer(64, 32, num_bond_types)
        
        # Added dropout layer for use between GCN layers
        self.dropout = nn.Dropout(p=dropout_rate)

        self.aggregation_layer = GraphAggregationLayer(32, 128)

        self.mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Dropout(p=dropout_rate), # Added Dropout
            nn.Linear(128, 1),
        )

    def forward(self, node_features, adjacency_tensor, reward=False):
        gcn_output_1 = self.gcn_layer_1(node_features, adjacency_tensor)
        gcn_output_1 = self.dropout(gcn_output_1) # Applied Dropout

        gcn_output_2 = self.gcn_layer_2(gcn_output_1, adjacency_tensor)
        gcn_output_2 = self.dropout(gcn_output_2) # Applied Dropout

        graph_representation = self.aggregation_layer(gcn_output_2)
        raw_output = self.mlp(graph_representation)

        if reward:
            return torch.sigmoid(raw_output)
        else:
            return raw_output