from Bio.SeqIO.FastaIO import FastaIterator
import os

def list_files_in_dir(dir, extension=".pdb"):
    paths = []
    for filename in os.listdir(dir):
        full_path = os.path.abspath(os.path.join(dir, filename))
        if filename.endswith(extension):
            paths.append(full_path)
    
    paths.sort()
    return paths

def extract_seqs_from_dir(dir, extension=".fa"):
    file_list = list_files_in_dir(dir, extension)
    sequences = []
    for file in file_list:
        with open(file, "r") as f:
            sequences.extend([str(record.seq) for record in FastaIterator(f)])
    return sequences