# MolGan
MolGAN (Molecular Generative Adversarial Network), a deep learning model designed to create new, chemically valid molecular structures.

## Output 
1) Using pure RL (lamda = 0)

<img width="1250" height="250" alt="output_pure_rl" src="https://github.com/user-attachments/assets/524d059e-aabe-4081-b11d-427593cfa8d3" />

2) Half wgan and RL (lambda = 0.5)
<img width="1250" height="250" alt="output_0 5_lambda" src="https://github.com/user-attachments/assets/af1bbee3-7c51-4d6b-9e5f-7901f68118a5" />
3) Pure wgan (lambda = 1)
<img width="1250" height="250" alt="pure_wgan" src="https://github.com/user-attachments/assets/bdd69193-3ff0-4005-adb9-7465a9eaa542" />

## Architecture

<img width="1140" height="363" alt="download" src="https://github.com/user-attachments/assets/95fedfec-9ee7-44ac-90be-cd5b965ddb1a" />

The architecture consists of 3 main sections: a generator, a discriminator, and a reward network.

The generator takes a sample (z) from a standard normal distribution to generate a graph using an MLP (this limits the network to a fixed maximum size) to generate the graph at once. Specifically a dense adjacency tensor A (bond types) and an annotation matrix X (atom types) are produced. Since these are probabilities, a discrete, sparse x and a are generated through categorical sampling.

The discriminator and reward network have the same architectures and receive graphs as inputs. A Relational-GCN and MLPs are used to produce the singular output
### Training Steps
1. **Clone the Repository**  
   Clone the MolGAN implementation repository to your local machine:
   ```bash
   git clone https://github.com/kfzyqin/Implementation-MolGAN-PyTorch.git
   cd Implementation-MolGAN-PyTorch
   ```

2. **Download Molecular Metrics Models**  
   Run the provided script to download pre-trained models for evaluating molecular metrics (e.g., validity, uniqueness, novelty):
   ```bash
   sh download_dataset.sh
   ```
   This script downloads necessary utilities defined in `utils.py`.

3. **Generate the Dataset**  
   Process the QM9 dataset to create a sparse molecular dataset compatible with MolGAN:
   ```bash
   python sparse_molecular_dataset.py
   ```
   **Note**: Uncomment the last few lines in `sparse_molecular_dataset.py` to enable dataset generation.

4. **Train the Model**  
   Execute the training script to start training MolGAN:
   ```bash
   python train.py
   ```
   - Hyperparameters (e.g., learning rate, batch size, λ) can be adjusted in the `train.py` or `Solve` module for experimentation.
   - Monitor training progress through loss metrics and generated molecule quality.

## References:
1) [Molgan paper](https://arxiv.org/pdf/1805.11973)
2) [Implementation-MolGan-Pytorch](https://github.com/kfzyqin/Implementation-MolGAN-PyTorch)

