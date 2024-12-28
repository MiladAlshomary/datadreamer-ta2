import json
import torch
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

def calculate_cosine_similarity_pairwise_gpu(input_file, output_file, chunk_size=100, device='cuda'):
    """
    Optimized pairwise cosine similarity computation for large files using GPU.
    
    Args:
        input_file (str): Path to the JSONL file containing document embeddings.
        output_file (str): Path to the file where pairwise similarities will be stored.
        chunk_size (int): Number of lines to process in each chunk.
        device (str): Device to use ('cuda' for GPU or 'cpu').
    """
    print("Called calculate_cosine_similarity_pairwise_gpu")

    def load_lines(file, start_offset=None, num_lines=None):
        """Generator to read lines from a file, starting from a specific offset."""
        if start_offset is not None:
            file.seek(start_offset)
        lines = []
        for _ in range(num_lines or chunk_size):
            line = file.readline()
            if not line:  # End of file
                break
            lines.append(json.loads(line))
        return lines

    # Open input and output files
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        total_lines = sum(1 for _ in open(input_file, 'r'))  # Count total lines
        infile.seek(0)  # Reset pointer

        total_comparisons = (total_lines * (total_lines - 1)) // 2  # Total pairwise comparisons
        print(f"Total pairwise comparisons: {total_comparisons}")

        start_offsets = []
        current_offset = infile.tell()
        print(f"Opened input: {input_file} and output file: {output_file}")

        # Precompute starting offsets for each line
        for _ in range(total_lines):
            start_offsets.append(current_offset)
            infile.readline()
            current_offset = infile.tell()

        print("Precomputed starting offset for each line")
        
        infile.seek(0)  # Reset file for actual processing
        
        # Process each document
        with tqdm(total=total_comparisons, desc="Pairwise Comparisons") as pbar:
            for i, start_offset in enumerate(start_offsets):
                infile.seek(start_offset)
                current_doc = json.loads(infile.readline())
                current_id = current_doc['documentID']
                current_embedding = torch.tensor(current_doc['embeddings'], device=device).unsqueeze(0)

                # Process remaining lines in chunks
                for chunk_start in range(i + 1, total_lines, chunk_size):
                    chunk = load_lines(infile, start_offsets[chunk_start], chunk_size)
                    if not chunk:  # No more lines
                        break

                    # Extract embeddings and IDs
                    chunk_embeddings = torch.tensor([doc['embeddings'] for doc in chunk], device=device)
                    chunk_ids = [doc['documentID'] for doc in chunk]

                    # Compute cosine similarities using PyTorch
                    similarities = cosine_similarity(current_embedding, chunk_embeddings).cpu().numpy()

                    # Write results to the output file
                    for doc2, similarity_score in zip(chunk_ids, similarities):
                        result = {
                            "doc1": current_id,
                            "doc2": doc2,
                            "similarity": similarity_score
                        }
                        outfile.write(json.dumps(result) + '\n')
                        pbar.update(1)  # Update progress for each pair

if __name__ == "__main__":
    # Input file containing document embeddings and IDs
    input_file = "../data/sadiri_processed_with_embeddings_wo_ao3.jsonl"

    # Output file to save pairwise cosine similarity data
    output_file = "../data/dpcs_sadiri_processed_with_embeddings_wo_ao3.jsonl"

    # Chunk size for processing
    chunk_size = 8192  # Increase chunk size for better GPU utilization

    # Calculate and save pairwise cosine similarity
    calculate_cosine_similarity_pairwise_gpu(input_file, output_file, chunk_size=chunk_size, device='cuda')
