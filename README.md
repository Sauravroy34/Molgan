# MolGan
Molecular genration using Genrative adversal networks and Reinforcement learing objective 


Output using pure RL (lamda = 0)

<img width="1250" height="250" alt="output_2" src="https://github.com/user-attachments/assets/082af58b-2d99-4f9b-83c7-0cf8ec52f89b" />

## Architecture

<img width="1140" height="363" alt="download" src="https://github.com/user-attachments/assets/95fedfec-9ee7-44ac-90be-cd5b965ddb1a" />

The architecture consits of 3 main sections: a generator, a discriminator, and a reward network.

The generator takes a sample (z) from a standard normal distribution to generate an a graph using a MLP (this limits the network to a fixed maximum size) to generate the graph at once. Sepcifically a dense adjacency tensor A (bond types) and an annotation matrix X (atom types) are produced. Since these are probabilities, a discrete, sparse x and a are generated through categorical sampling.

The discriminator and reward network have the same architectures and recieve graphs as inputs. A Relational-GCN and MLPs are used to produce the singular output


## Refrences:
1) [Molgan paper](https://arxiv.org/pdf/1805.11973)
2) [Implementation-MolGan-Pytorch](https://github.com/kfzyqin/Implementation-MolGAN-PyTorch)

