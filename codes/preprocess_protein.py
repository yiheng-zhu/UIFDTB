import os
import sys
import shutil
import pandas as pd

from config import sequence_cut_off


def remove_forbidden_letters(sequence):
    """
    Remove unsupported amino acid symbols.
    """
    forbidden_letters = {'B', 'J', 'O', 'U', 'X', 'Z'}
    filtered_sequence = [
        char for char in sequence if char not in forbidden_letters
    ]
    return ''.join(filtered_sequence)


def main(workdir):

    protein_file = os.path.join(workdir, "protein.csv")
    backup_file = os.path.join(workdir, "protein_original.csv")

    # Check input file
    if not os.path.isfile(protein_file):
        raise FileNotFoundError(
            f"Protein file not found: {protein_file}"
        )

    # Backup the original file
    if not os.path.exists(backup_file):
        shutil.copy2(protein_file, backup_file)
        print(f"Original file backed up to: {backup_file}")
    else:
        print(f"Existing backup found: {backup_file}")
        print("Using the original backup as the preprocessing source.")

    # Always read the original file to ensure reproducibility
    df = pd.read_csv(backup_file)

    if "Protein" not in df.columns:
        raise ValueError("Column 'Protein' not found in protein.csv")

    original_sequences = df["Protein"].fillna("").astype(str)

    # Preprocess protein sequences
    processed_sequences = []

    for sequence in original_sequences:

        # Convert to uppercase
        protein = sequence.upper()

        # Remove forbidden amino acid symbols
        protein = remove_forbidden_letters(protein)

        # Truncate sequences longer than 1000 residues
        if len(protein) > sequence_cut_off:
            protein = protein[:sequence_cut_off]

        processed_sequences.append(protein)

    # Update Protein column
    df["Protein"] = processed_sequences

    # Save processed file
    df.to_csv(protein_file, index=False)

    # Statistics
    print("\n========== Protein Preprocessing Summary ==========")
    print(f"Total proteins: {len(df)}")
    print(f"Original sequences: {len(original_sequences)}")
    print(f"Processed sequences: {len(processed_sequences)}")
    print(f"Maximum sequence length: {sequence_cut_off}")
    print(f"Saved processed file to: {protein_file}")

    print("\nProtein preprocessing completed successfully!")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python preprocess_protein.py <workdir>"
        )

    workdir = os.path.abspath(sys.argv[1])

    if not os.path.isdir(workdir):
        raise NotADirectoryError(
            f"Working directory not found: {workdir}"
        )

    main(workdir)