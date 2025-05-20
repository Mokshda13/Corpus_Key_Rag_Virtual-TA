#CreateFinalData.py
import os
import pandas as pd
import numpy as np
from pathlib import Path

from transformers import AutoTokenizer, AutoModel


# change all local mapping

# project_folders now = ["input", "chunks", "embeddings", "bundle", "output"]

def create_final_data(directory, chunks_folder, embeddings_folder):
    embeddings_path = Path(embeddings_folder)
    path_suffix = embeddings_path.name.split('_')[-1]

    output_csv = os.path.join(directory, "bundle", "chunks-originaltext.csv")
    output_npy = os.path.join(directory, "bundle", f"chunks_{path_suffix}.npy")
    
    # Get the sorted list of CSV and .npy files
    csv_files = sorted([f for f in os.listdir(chunks_folder) if f.endswith('.csv')])
    npy_files = sorted([f for f in os.listdir(embeddings_folder) if f.endswith('.npy')])

    # Initialize empty DataFrame and NumPy array for concatenation
    concatenated_csv = pd.DataFrame()
    concatenated_npy = None

    for csv_file, npy_file in zip(csv_files, npy_files):
        print(npy_file)
        # Read the CSV file and concatenate

        if not os.path.exists(output_csv):
            csv_path = os.path.join(chunks_folder, csv_file)
            csv_data = pd.read_csv(csv_path, encoding='utf-8', escapechar='\\')
            concatenated_csv = pd.concat([concatenated_csv, csv_data], ignore_index=True)

        npy_path = os.path.join(embeddings_folder, npy_file)
        npy_data = np.load(npy_path)

        if concatenated_npy is None:
            concatenated_npy = npy_data
        else:
            concatenated_npy = np.concatenate([concatenated_npy, npy_data], axis=0)

    # Save the concatenated data to the base folder
    if not os.path.exists(output_csv):
        concatenated_csv.to_csv(output_csv, encoding='utf-8', escapechar='\\', index=False)

    np.save(output_npy, concatenated_npy)
    print(f"Files saved: chunks-originaltext.csv and chunks_{path_suffix}.npy")
    # Print the dimensions of the concatenated files
    print(f"chunks-originaltext.csv dimensions: {concatenated_csv.shape}")
    print(f"chunks_{path_suffix}.npy dimensions: {concatenated_npy.shape}")

if __name__ == "__main__":
    print("This is module.py being run directly.")
    os.chdir(r"C:\Users\Hadi\Desktop\Code\Projects\All Day-TA - Updated\embeddings_testing")
    current_dir = os.getcwd()

    directory = r"C:\Users\Hadi\Desktop\Code\Projects\All Day-TA - Updated\embeddings_testing"

    chunks_folder = os.path.join(directory, "chunks")
    embeddings_folder_ll = os.path.join(directory, "embeddings_ll")
    embeddings_folder_sll = os.path.join(directory, "embeddings_sll")

    create_final_data(directory, chunks_folder, embeddings_folder_ll)
    create_final_data(directory, chunks_folder, embeddings_folder_sll)