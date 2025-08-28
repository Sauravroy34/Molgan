import torch
from rdkit import Chem
from rdkit.Chem import Draw
import os

from models import Generator
from tools import postprocess, sample_z, get_gen_mols

print("Loading the generator model...")
device = torch.device("cpu") 
state_dict = torch.load('Generator.pth', map_location=device, weights_only=True)

model = Generator()
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print("Model loaded successfully.")

valid_mols = []
seen_smiles = set()  # --- NEW: Set to store SMILES of found molecules
total_generated = 0
target_count = 5
batch_size = 32

print(f"Starting molecule generation to find {target_count} unique valid molecules...")
with torch.no_grad():
    while len(valid_mols) < target_count:
        z = sample_z(batch_size)
        z = torch.from_numpy(z).float()
        
        raw_edges, raw_nodes = model(z)
        generated_mols_batch = get_gen_mols(raw_nodes, raw_edges)
        total_generated += len(generated_mols_batch)
        
        for mol in generated_mols_batch:
            if mol is not None and Chem.MolToSmiles(mol) is not None:
                if len(Chem.GetMolFrags(mol)) == 1:
                    # --- START: UNIQUENESS CHECK ---
                    smi = Chem.MolToSmiles(mol)  # Get canonical SMILES string
                    if smi not in seen_smiles:
                        if len(valid_mols) < target_count:
                            valid_mols.append(mol)
                            seen_smiles.add(smi)  # Add the new SMILES to our set
                        else:
                            break
                    # --- END: UNIQUENESS CHECK ---
        
        if len(valid_mols) >= target_count:
            break

        print(f"\rCollected {len(valid_mols)} / {target_count} unique molecules... (Total processed: {total_generated})", end="")

print(f"\n\nGeneration complete. Found {len(valid_mols)} unique molecules.")

if valid_mols:
    print("Generating grid image of all molecules...")
    
    grid_img = Draw.MolsToGridImage(
        valid_mols, 
        molsPerRow=5,        
        subImgSize=(250, 250), 
        legends=[f'Molecule {i+1}' for i in range(len(valid_mols))] 
    )
    
    output_filename = 'generated_molecules_grid.png'
    grid_img.save(output_filename)
    
    print(f"Successfully saved all molecules to '{output_filename}'")

else:
    print("No valid molecules were generated to save.")