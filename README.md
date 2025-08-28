# MolGan
Molecular genration using Genrative adversal networks and Reinforcement learing objective 


Here is the output using pure RL

## Architecture

The architecture consits of 3 main sections: a generator, a discriminator, and a reward network.

The generator takes a sample (z) from a standard normal distribution to generate an a graph using a MLP (this limits the network to a fixed maximum size) to generate the graph at once. Sepcifically a dense adjacency tensor A (bond types) and an annotation matrix X (atom types) are produced. Since these are probabilities, a discrete, sparse x and a are generated through categorical sampling.

The discriminator and reward network have the same architectures and recieve graphs as inputs. A Relational-GCN and MLPs are used to produce the singular output


## Refrences:
Molgan paper
Implementation-MolGan-Pytorch

