import os
import sys
import time

import torch
import esm
from tqdm import tqdm

from config import max_gpu_length, protein_fasta_file, esmfold_structure_dir
from common_methods import create_dir


def read_fasta(sequence_file):
    """Read FASTA records, supporting sequences spanning multiple lines."""
    records = []
    name = None
    sequence_parts = []

    with open(sequence_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(sequence_parts)))
                name = line[1:].strip()
                sequence_parts = []
            elif name is not None:
                sequence_parts.append(line)

    if name is not None:
        records.append((name, "".join(sequence_parts)))
    return records


def predict_structures(workdir):
    sequence_file = os.path.join(workdir, protein_fasta_file)
    pdb_dir = os.path.join(workdir, esmfold_structure_dir)

    if not os.path.isfile(sequence_file):
        raise FileNotFoundError(f"FASTA file not found: {sequence_file}")

    create_dir(pdb_dir)

    records = read_fasta(sequence_file)
    if not records:
        raise ValueError(f"No protein sequences found in: {sequence_file}")

    print("Loading the ESMFold model...")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. ESMFold prediction requires a CUDA-enabled GPU.")

    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()
    model.set_chunk_size(16)

    print(f"Loaded {len(records)} protein sequences.")
    success_count = 0
    skipped_count = 0
    failed_count = 0
    start_time = time.time()

    for name, sequence in tqdm(
        records,
        total=len(records),
        desc="ESMFold structure prediction",
        unit="protein",
    ):
        pdb_file = os.path.join(pdb_dir, name + ".pdb")

        if os.path.isfile(pdb_file) and os.path.getsize(pdb_file) > 0:
            skipped_count += 1
            continue

        if not sequence:
            tqdm.write(f"Warning: Empty sequence for {name}; skipped.")
            failed_count += 1
            continue

        if len(sequence) < max_gpu_length:
            model.set_chunk_size(32)
        else:
            model.set_chunk_size(16)

        try:
            with torch.no_grad():
                output = model.infer_pdb(sequence)
            with open(pdb_file, "w", encoding="utf-8") as f:
                f.write(output)
            success_count += 1
        except Exception as e:
            tqdm.write(f"Warning: ESMFold prediction failed for {name}: {e}")
            failed_count += 1
            torch.cuda.empty_cache()

    elapsed = (time.time() - start_time) / 60
    print("\n========== ESMFold Prediction Summary ==========")
    print(f"Total proteins:       {len(records)}")
    print(f"Successfully saved:   {success_count}")
    print(f"Skipped (existing):   {skipped_count}")
    print(f"Failed or empty:      {failed_count}")
    print(f"Output directory:     {pdb_dir}")
    print(f"Prediction time:      {elapsed:.1f} minutes")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python predict_structures_esmfold.py <workdir>")

    workdir = os.path.abspath(sys.argv[1])
    if not os.path.isdir(workdir):
        raise NotADirectoryError(f"Working directory not found: {workdir}")
    predict_structures(workdir)
