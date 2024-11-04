from Bio import PDB
import os

clip_len = 512
RAW_DIR = "raw_dnmt"
OUT_DIR = f"clipped_{clip_len}"

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

all_file_paths = [os.path.join(RAW_DIR, x) for x in os.listdir(RAW_DIR) if '.pdb' in x]
total_num_paths = len(all_file_paths)

print(total_num_paths)


def clip_protein(pdb_path, output_path, length=128):
    parser = PDB.PDBParser()
    structure = parser.get_structure('protein', pdb_path)

    for model in structure:
        for chain in model:
            # Create new structure for the clipped protein
            clipped_structure = PDB.Structure.Structure('clipped_protein')
            clipped_model = PDB.Model.Model(0)
            clipped_chain = PDB.Chain.Chain(chain.id)
            # Manually add residues up to the specified length
            for i, residue in enumerate(chain):
                if i >= length:
                    break  # Stop adding residues once we reach the desired length
                clipped_chain.add(residue)
            # Add the clipped chain to the model and the model to the structure
            clipped_model.add(clipped_chain)
            clipped_structure.add(clipped_model)
            # Save the clipped structure to a new PDB file
            io = PDB.PDBIO()
            io.set_structure(clipped_structure)
            io.save(output_path)


for i, file_path in enumerate(all_file_paths):
    pdb_file_path = file_path
    pdb_name = os.path.basename(file_path).replace('.pdb', '')
    clipped_pdb_file_path = os.path.join(OUT_DIR, pdb_name + ".pdb")
    clip_protein(pdb_file_path, clipped_pdb_file_path, length=clip_len)
