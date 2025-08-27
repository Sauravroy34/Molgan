import torch
from models import Generator
from tools import postprocess,get_reward,sample_z,label2onehot,reward,get_gen_mols,save_mol_img

state_dict = torch.load('Generator.pth',weights_only=True)

model = Generator()
model.load_state_dict(state_dict)
model.eval()


from rdkit import Chem
valid_mols = []
total_generated = 0
target_count = 10
with torch.no_grad(): # Disable gradient calculations for inference
    while len(valid_mols) < target_count:
        # Sample a batch from the latent space
        z = sample_z(32)
        z = torch.from_numpy(z).float()
        
        # Generate raw node and edge predictions from the model
        raw_edges, raw_nodes = model(z)
        (adj_hat,nodes_hat) = postprocess((raw_edges , raw_nodes))
        # Convert raw predictions to RDKit molecule objects
        generated_mols_batch = get_gen_mols(raw_edges, raw_nodes)
        total_generated += len(generated_mols_batch)
        
        # 3. Filtering for valid molecules
        for mol in generated_mols_batch:
            # A common way to check for validity is to see if a SMILES string can be generated.
            # RDKit returns None for invalid molecular graphs.
            if mol is not None and Chem.MolToSmiles(mol) is not None:
                valid_mols.append(mol)
        
        # Progress update
        print(f"\rCollected {len(valid_mols)} / {target_count} valid molecules...", end="")
        save_mol_img(valid_mols,f"mol {len(valid_mols)}.png")