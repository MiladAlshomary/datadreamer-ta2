import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Initialize Sentence Transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def process_filtered_files(input_base_path, output_file, batch_size=100):
    """
    Reads multiple files matching the pattern, adds genre based on parent folder name, 
    generates embeddings, appends data with embeddings, and writes all processed data 
    to a single output file.

    Args:
        input_base_path (str): Base path containing subfolders with filtered files.
        output_file (str): Path to the single output file to save processed data.
        batch_size (int): Number of lines to process in each batch.
    """
    input_path = Path(input_base_path)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find all files matching the pattern "train*filtered.jsonl" in subdirectories
    files = list(input_path.glob("*/dev*filtered.jsonl"))

    with output_path.open('w') as outfile:
        for file in files:
            genre = file.parent.name  # Extract the genre from the parent folder name
            print(f"Processing file: {file} with name: {file.name} with genre: {genre}")
            
            # Read file line by line to avoid memory overload
            with file.open('r') as infile:
                batch = []
                for line in tqdm(infile, desc=f"Reading lines from {file.name}"):
                    batch.append((line, genre))  # Pass the line and genre
                    if len(batch) >= batch_size:
                        process_batch(batch, outfile)
                        batch.clear()
                
                # Process remaining lines
                if batch:
                    process_batch(batch, outfile)

def process_batch(batch, outfile):
    """
    Processes a batch of lines, adds genre, generates embeddings, and appends them as a new column.

    Args:
        batch (list): List of tuples with lines (JSON strings) and genres.
        outfile (file object): File object to write the processed data.
    """
    # Separate lines and genres
    lines, genres = zip(*batch)

    # Load batch into DataFrame
    df = pd.DataFrame([json.loads(line) for line in lines])  # Use json.loads instead of eval
    df['genre'] = genres  # Add genre to the DataFrame

    # Ensure column 'fullText' exists
    if 'fullText' not in df.columns:
        raise ValueError("Column 'fullText' is required in the input data.")

    # Generate embeddings for the batch
    embeddings = model.encode(df['fullText'].tolist(), batch_size=len(df))

    # Convert embeddings to list and assign to column
    df['embeddings'] = [embedding.tolist() for embedding in embeddings]

    # Write processed DataFrame to the output file in JSON Lines format
    for _, row in df.iterrows():
        outfile.write(f"{row.to_json()}\n")


if __name__ == "__main__":
    # Base path to input files
    input_base_path = "../data/sadiri/"

    # Output file to write combined processed data
    output_file = "../data/dev_sadiri_processed_with_embeddings_wo_ao3_filtered.jsonl"

    # Batch size for processing lines
    batch_size = 2048

    # Process filtered files in the input path and write to one output file
    process_filtered_files(input_base_path, output_file, batch_size=batch_size)
