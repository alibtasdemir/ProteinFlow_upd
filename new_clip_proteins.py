import os

def truncate_pdb_sequence(file_path, max_length=512):
    """
    Truncate the protein sequence in a PDB file to a maximum length.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    truncated_lines = []
    current_residue_num = None
    residue_count = 0

    for line in lines:
        if line.startswith('ATOM'):
            residue_num = int(line[22:26].strip())  # Extract residue sequence number
            if current_residue_num != residue_num:
                current_residue_num = residue_num
                residue_count += 1
            if residue_count <= max_length:
                truncated_lines.append(line)
        else:
            truncated_lines.append(line)

    return truncated_lines

def process_pdb_files(folder_path, output_dir, max_length=512):
    """
    Process all PDB files in the specified folder.
    """
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pdb'):
            file_path = os.path.join(folder_path, file_name)
            truncated_content = truncate_pdb_sequence(file_path, max_length=max_length)
            # Write the truncated content to a new file
            truncated_file_path = os.path.join(output_dir, file_name)
            with open(truncated_file_path, 'w') as truncated_file:
                truncated_file.writelines(truncated_content)
            print(f"Processed: {file_name}")

# Replace 'your_folder_path_here' with the path to your folder of PDB files
clip_length=256
RAW_DIR = "raw_dnmt"
OUT_DIR = f"clipped_{clip_length}"

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

process_pdb_files(RAW_DIR, OUT_DIR, max_length=clip_length)
