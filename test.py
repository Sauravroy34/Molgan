import torch
from rdkit import Chem
from rdkit.Chem import Draw
import os

# Assuming these are your custom utility files
from models import Generator
from tools import postprocess, sample_z, get_gen_mols

# --- 1. Model Setup and Loading ---
print("Loading the generator model...")
# Use torch.device("cuda" if torch.cuda.is_available() else "cpu") for GPU support
device = torch.device("cpu") 
state_dict = torch.load('Generator.pth', map_location=device, weights_only=True)

model = Generator()
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print("Model loaded successfully.")

# --- 2. Molecule Generation ---
valid_mols = []
total_generated = 0
target_count = 10
batch_size = 32 # You can adjust this based on your system's memory

print(f"Starting molecule generation to find {target_count} valid molecules...")
with torch.no_grad(): # Disable gradient calculations for inference
    while len(valid_mols) < target_count:
        # Sample a batch from the latent space
        z = sample_z(8)
        z = torch.from_numpy(z).float()
        
        # Generate molecules from the model
        raw_edges, raw_nodes = model(z)
        
        # Process the raw output to get RDKit molecule objects
        generated_mols_batch = get_gen_mols(raw_nodes, raw_edges)
        total_generated += len(generated_mols_batch)
        
        # Filter for valid molecules
        for mol in generated_mols_batch:
            if mol is not None and Chem.MolToSmiles(mol) is not None:
                if len(valid_mols) < target_count:
                    valid_mols.append(mol)
                else:
                    break # Stop adding once the target is reached
        
        if len(valid_mols) >= target_count:
            break

        # Progress update
        print(f"\rCollected {len(valid_mols)} / {target_count} valid molecules... (Total processed: {total_generated})", end="")

print(f"\nGeneration complete. Found {len(valid_mols)} valid molecules.")

# --- 3. Save All Molecules to a Single Grid Image ---
if valid_mols:
    print("Generating grid image of all molecules...")
    
    # Create a grid image from the list of molecule objects 🖼️
    grid_img = Draw.MolsToGridImage(
        valid_mols, 
        molsPerRow=5,         # Arrange molecules in rows of 5
        subImgSize=(250, 250), # Set the size for each molecule's image
        legends=[f'Molecule {i+1}' for i in range(len(valid_mols))] # Add a label to each
    )
    
    # Save the generated grid image to a PNG file
    output_filename = 'generated_molecules_grid.png'
    grid_img.save(output_filename)
    
    print(f"Successfully saved all molecules to '{output_filename}'")
    

else:
    print("No valid molecules were generated to save.")