import os
import sys
import time
import shutil
import subprocess
import numpy as np
from tqdm import tqdm

from config import (
    interproscan_script_file,
    interproscan_workspace,
    protein_csv_file,
    protein_fasta_file,
    interproscan_result_filename,
    interproscan_temp_filedir,
    interproscan_entry_list_file,
    interproscan_resultdir,
    interproscan_featuredir,
)
from common_methods import create_dir, csv_to_fasta


def run_interpro(workdir):
    """Run InterProScan and display elapsed time while it is running."""
    seq_file = os.path.join(workdir, protein_fasta_file)
    workspace = os.path.join(workdir, interproscan_workspace)
    result_file = os.path.join(workspace, interproscan_result_filename)
    temp_dir = os.path.join(workspace, interproscan_temp_filedir)
    log_file = os.path.join(workspace, "interproscan.log")

    csv_to_fasta(os.path.join(workdir, protein_csv_file), seq_file)
    os.makedirs(workspace, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    cmd = [
        interproscan_script_file,
        "-i", seq_file,
        "-f", "tsv",
        "-o", result_file,
        "-T", temp_dir,
    ]

    print("\n[1/3] Running InterProScan...")
    print("Progress is reported as elapsed time; InterProScan does not provide")
    print("a reliable overall completion percentage for this execution.")

    start_time = time.time()
    process = None

    try:
        with open(log_file, "w", encoding="utf-8") as log:
            process = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )

            with tqdm(
                desc="InterProScan",
                bar_format="{desc} | elapsed: {elapsed}",
                dynamic_ncols=True,
            ) as progress:
                while process.poll() is None:
                    progress.refresh()
                    time.sleep(5)
                progress.refresh()

        return_code = process.wait()
        elapsed = time.time() - start_time

        if return_code != 0:
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                log_text = f.read()
            raise RuntimeError(
                f"InterProScan exited with code {return_code}. "
                f"See log file: {log_file}\n{log_text[-4000:]}"
            )

        print(f"InterProScan completed in {elapsed / 60:.2f} minutes.")
        print(f"InterProScan log: {log_file}")

    except Exception as e:
        print("\nInterProScan failed!")
        print(e)
        raise

    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def read_result_file(result_file):
    """Read InterProScan TSV output and collect unique InterPro entries."""
    result_dict = {}

    if not os.path.isfile(result_file):
        raise FileNotFoundError(f"InterProScan result file not found: {result_file}")

    with open(result_file, "r", encoding="utf-8") as f:
        for line in f:
            values = line.rstrip("\n").split("\t")
            if len(values) <= 11:
                continue

            name = values[0]
            entry = values[11]
            result_dict.setdefault(name, [])

            if entry != "-":
                result_dict[name].append(entry)

    for name in result_dict:
        result_dict[name] = sorted(set(result_dict[name]))

    return result_dict


def read_all_entry_list(entry_list_file):
    """Read the ordered list of all InterPro entries."""
    with open(entry_list_file, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    return [line.split("\t")[0] for line in lines[1:] if line.strip()]


def create_entry_array(current_entry_list, entry_to_index, entry_number):
    """Create a binary vector using a precomputed entry-to-index mapping."""
    entry_array = np.zeros(entry_number, dtype=np.int8)

    for entry in current_entry_list:
        index = entry_to_index.get(entry)
        if index is not None:
            entry_array[index] = 1

    return entry_array


def read_fasta_names(seq_file):
    """Read protein identifiers from a FASTA file."""
    names = []
    with open(seq_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                names.append(line[1:])
    return names


def create_interpro_features(workdir):
    """Convert InterProScan annotations into binary feature vectors."""
    result_file = os.path.join(
        workdir, interproscan_workspace, interproscan_result_filename
    )
    if not os.path.isfile(result_file):
        raise FileNotFoundError(f"InterProScan result file not found: {result_file}")

    result_dir = os.path.join(
        workdir, interproscan_workspace, interproscan_resultdir
    )
    feature_dir = os.path.join(
        workdir, interproscan_workspace, interproscan_featuredir
    )
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)

    print("\n[2/3] Converting InterProScan results into binary vectors...")

    result_dict = read_result_file(result_file)
    all_entry_list = read_all_entry_list(interproscan_entry_list_file)
    entry_to_index = {entry: i for i, entry in enumerate(all_entry_list)}
    entry_number = len(all_entry_list)

    for name, entry_name_list in tqdm(
        result_dict.items(),
        total=len(result_dict),
        desc="Generating InterPro features",
        unit="protein",
        dynamic_ncols=True,
    ):
        entry_name_file = os.path.join(result_dir, name)
        entry_array_file = os.path.join(feature_dir, name + ".npy")

        with open(entry_name_file, "w", encoding="utf-8") as f:
            f.write("\n".join(entry_name_list))
            if entry_name_list:
                f.write("\n")

        entry_array = create_entry_array(
            entry_name_list, entry_to_index, entry_number
        )
        np.save(entry_array_file, entry_array)

    print(f"Generated feature vectors for {len(result_dict)} annotated proteins.")


def check_features(workdir):
    """Create empty annotation files and zero vectors for proteins without hits."""
    seq_file = os.path.join(workdir, protein_fasta_file)
    result_dir = os.path.join(
        workdir, interproscan_workspace, interproscan_resultdir
    )
    feature_dir = os.path.join(
        workdir, interproscan_workspace, interproscan_featuredir
    )

    names = read_fasta_names(seq_file)
    all_entry_list = read_all_entry_list(interproscan_entry_list_file)
    entry_number = len(all_entry_list)

    print("\n[3/3] Checking proteins without InterPro annotations...")

    zero_vector_count = 0
    for name in tqdm(
        names,
        total=len(names),
        desc="Checking protein features",
        unit="protein",
        dynamic_ncols=True,
    ):
        entry_name_file = os.path.join(result_dir, name)
        entry_array_file = os.path.join(feature_dir, name + ".npy")

        if not os.path.exists(entry_name_file):
            with open(entry_name_file, "w", encoding="utf-8"):
                pass

        if not os.path.exists(entry_array_file):
            np.save(
                entry_array_file,
                np.zeros(entry_number, dtype=np.int8),
            )
            zero_vector_count += 1

    print(f"Created zero vectors for {zero_vector_count} proteins without annotations.")


def main_process(workdir):
    print("=" * 60)
    print("InterPro Feature Generation")
    print(f"Working directory: {os.path.abspath(workdir)}")
    print("=" * 60)

    create_dir(os.path.join(workdir, interproscan_workspace))

    start_time = time.time()
    run_interpro(workdir)
    create_interpro_features(workdir)
    check_features(workdir)

    elapsed = time.time() - start_time
    print("\nInterPro feature generation completed successfully.")
    print(f"Total elapsed time: {elapsed / 60:.2f} minutes.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python generate_interpro_feature.py <workdir>")

    workdir = os.path.abspath(sys.argv[1])
    if not os.path.isdir(workdir):
        raise NotADirectoryError(f"Working directory not found: {workdir}")

    main_process(workdir)
