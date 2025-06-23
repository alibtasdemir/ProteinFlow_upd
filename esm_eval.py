"""
import os
from helpers.sequence import extract_seqs_from_dir

mpnn_dir = os.path.join("mpnn_results", "seqs")
mpnn_sequences = extract_seqs_from_dir(mpnn_dir, extension="fa")
print(mpnn_sequences)
"""
import torch

from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from tqdm import tqdm

import pandas as pd

import os


tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1", cache_dir="esm_cache")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True, cache_dir="esm_cache")
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
model.to(device)
print("Model is loaded!")

torch.backends.cuda.matmul.allow_tf32 = True
model.trunk.set_chunk_size(64)

def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs


def write_pdb(pdb_path, seq_num, pdbcontent):
    outfile_name = os.path.splitext(os.path.basename(pdb_path))[0]
    outfile_name = f"{outfile_name}_{seq_num}.pdb"
    out_path = os.path.join("esmfold_outputs", outfile_name)
    with open(out_path, "w") as f:
        f.write("".join(pdbcontent))

    
# Multiple files
"""
mpnn_dir = os.path.join("mpnn_results", "seqs")
mpnn_sequences = extract_seqs_from_dir(mpnn_dir, extension="fa")
"""
df = pd.read_csv("mpnn_sequences.csv")
sequences_tokenized = tokenizer(df.seq.tolist(), padding=False, add_special_tokens=False)["input_ids"]

# sequences_tokenized = tokenizer(mpnn_sequences, padding=False, add_special_tokens=False)["input_ids"]
# outputs = []
if not os.path.exists("esmfold_outputs"):
    os.makedirs("esmfold_outputs")

tk = tqdm(zip(sequences_tokenized, df.pdb_path.tolist(), df.seq_num.tolist()), total=df.shape[0])
with torch.no_grad():
    #for input_ids, pdb_path, seq_num in tqdm(sequences_tokenized):
    for input_ids, pdb_path, seq_num in tk:
        input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)
        output = model(input_ids)
        # outputs.append({key: val.cpu() for key, val in output.items()})
        # Write to a file
        pdbfile = convert_outputs_to_pdb({key: val.cpu() for key, val in output.items()})
        write_pdb(pdb_path, seq_num, pdbfile)
        

"""
pdb_list = [convert_outputs_to_pdb(output) for output in outputs]

for pdb_path, seq_num, pdb_content in zip(df.pdb_path.tolist(), df.seq_num.tolist(), pdb_list):
    outfile_name = os.path.splitext(os.path.basename(pdb_path))[0]
    outfile_name = f"{outfile_name}_{seq_num}.pdb"
    out_path = os.path.join("esmfold_outputs", outfile_name)
    with open(out_path, "w") as f:
        f.write("".join(pdb_content))
"""

