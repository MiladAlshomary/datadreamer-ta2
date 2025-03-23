import random
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import semantic_search_faiss
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from tqdm import tqdm
import jsonlines


class HardBatchCreator:
    def __init__(
            self,
            data_by_author: dict,
            batch_size: int = 32,
            ceiling_threshold: float = 0.2,
            embed_model_name: str = "gabrielloiseau/LUAR-MUD-sentence-transformers",
            genre_model_name: str = "classla/xlm-roberta-base-multilingual-text-genre-classifier",
    ):
        """
        Args:
            data_by_author: Dictionary mapping author IDs to lists of document texts.
            batch_size: Desired batch size (number of (anchor, positive) pairs per batch).
            ceiling_threshold: Cosine similarity threshold below which a pair is considered a hard positive.
            embed_model_name: Name of the SentenceTransformer model.
            genre_model_name: Name of the genre classifier model.
        """
        self.data_by_author = data_by_author
        self.batch_size = batch_size
        self.ceiling_threshold = ceiling_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embed_model = SentenceTransformer(embed_model_name, device=self.device)
        # We'll use the same embed_model for vectorized operations.
        classifier_device = 0 if self.device == "cuda" else -1
        self.genre_classifier = pipeline("text-classification",
                                         model=genre_model_name,
                                         device=classifier_device)

    def mine_hard_positives(self):
        """
        Mines hard positive pairs per author.
        For each author with at least two documents, computes embeddings and cosine similarities.
        For each anchor, any candidate document with similarity below the threshold is considered.
        Then, using a genre classifier, only candidates with a different genre from the anchor are accepted.
        One candidate is randomly chosen as the hard positive.

        Returns:
            A list of dictionaries with keys:
              - "author": author ID,
              - "anchor": anchor document text,
              - "positive": corresponding hard positive text.
        """
        hard_pairs = []
        # Progress bar for mining hard positives over authors.
        for author, docs in tqdm(self.data_by_author.items(), desc="Mining hard positives",
                                 total=len(self.data_by_author)):
            if len(docs) <= 1:
                continue

            embeddings = self.embed_model.encode(docs, convert_to_tensor=False)
            sim_matrix = cosine_similarity(embeddings)
            n = len(docs)

            # For each anchor, gather candidate indices with similarity below threshold.
            hard_candidates = {}
            for i in range(n):
                for j in range(i + 1, n):
                    if sim_matrix[i, j] < self.ceiling_threshold:
                        hard_candidates.setdefault(i, []).append(j)

            # Compute the genre for each document.
            doc_genres = []
            for doc in docs:
                classification = self.genre_classifier(doc, truncation=True)
                genre = classification[0]["label"]
                doc_genres.append(genre)

            # For each anchor with candidate positives, choose one candidate with a different genre.
            for anchor_idx, candidates in hard_candidates.items():
                anchor_genre = doc_genres[anchor_idx]
                valid_candidates = [cand for cand in candidates if doc_genres[cand] != anchor_genre]
                if valid_candidates:
                    chosen_positive_idx = random.choice(valid_candidates)
                    hard_pairs.append({
                        "author": author,
                        "anchor": docs[anchor_idx],
                        "positive": docs[chosen_positive_idx]
                    })
        return hard_pairs

    def create_batches_faiss(self):
        """
        Constructs batches using a vectorized FAISS search.
        Steps:
         1. Mine all hard positive pairs.
         2. While available pairs remain:
             a. Compute embeddings for all available anchors (converted to a NumPy array).
             b. Select the first available pair as the seed.
             c. Define the corpus as all remaining pairs from different authors.
             d. Convert both seed and corpus embeddings to NumPy.
             e. Use semantic_search_faiss to search the corpus for hard negatives.
             f. Map FAISS results back to pairs, filter to one negative per unique author, and (if needed) sample down.
             g. Form the batch as [seed] + negatives and remove these pairs from the available pool.
         3. Return the list of batches.

        Returns:
            A list of batches, where each batch is a list of dictionaries (with keys "author", "anchor", "positive").
        """
        hard_pairs = self.mine_hard_positives()
        if not hard_pairs:
            raise ValueError("No hard positive pairs were mined. Check your data and threshold.")

        batches = []
        available_pairs = hard_pairs[:]  # copy of the list
        print(f"Number of available pairs: {len(available_pairs)}")

        # Initialize a progress bar based on the number of pairs to process.
        pbar = tqdm(total=len(available_pairs), desc="Batching progress")

        while available_pairs:
            # Compute embeddings for all available pairs.
            anchors = [pair["anchor"] for pair in available_pairs]
            corpus_embeddings = self.embed_model.encode(anchors, convert_to_tensor=False)
            corpus_embeddings_np = np.array(corpus_embeddings)  # shape: (num_pairs, dim)

            # Select the first available pair as the seed.
            seed = available_pairs[0]
            seed_author = seed["author"]
            seed_embedding = self.embed_model.encode([seed["anchor"]], convert_to_tensor=False)
            seed_embedding_np = np.array(seed_embedding)  # shape: (1, dim)

            # Build corpus indices for pairs with a different author.
            corpus_indices = [i for i, pair in enumerate(available_pairs) if pair["author"] != seed_author]
            if not corpus_indices:
                # No negatives available; form a batch with only the seed.
                batch_candidates = [seed]
                available_pairs.pop(0)
                batches.append(batch_candidates)
                pbar.update(1)
                continue

            # Extract the corpus subset embeddings.
            corpus_subset = corpus_embeddings_np[corpus_indices]
            top_k = len(corpus_indices)
            # Use semantic_search_faiss with rescore disabled to avoid quantization warnings.
            results, search_time = semantic_search_faiss(
                query_embeddings=seed_embedding_np,
                corpus_embeddings=corpus_subset,
                top_k=top_k,
                exact=True,
                rescore=False,
                output_index=False
            )
            # 'results' is a list (one per query) of lists of dictionaries.
            search_results = results[0]
            negatives = []
            for res in search_results:
                corpus_id = res["corpus_id"]  # index in corpus_subset
                original_index = corpus_indices[corpus_id]  # map back to available_pairs index
                candidate_pair = available_pairs[original_index]
                negatives.append(candidate_pair)

            # Filter negatives to keep one candidate per unique author.
            unique_negatives = {}
            for pair in negatives:
                if pair["author"] not in unique_negatives:
                    unique_negatives[pair["author"]] = pair
            negatives_filtered = list(unique_negatives.values())

            # Limit negatives to (batch_size - 1) if necessary.
            required_negatives = self.batch_size - 1
            if len(negatives_filtered) > required_negatives:
                negatives_filtered = random.sample(negatives_filtered, required_negatives)

            batch_candidates = [seed] + negatives_filtered

            # If the candidate batch is ridiculously small (e.g. less than half of batch_size),
            # stop batching.
            if len(batch_candidates) < (self.batch_size // 2):
                break

            batches.append(batch_candidates)

            # Remove used pairs from available_pairs.
            used_anchors = {pair["anchor"] for pair in batch_candidates}
            available_pairs = [pair for pair in available_pairs if pair["anchor"] not in used_anchors]
            pbar.update(len(batch_candidates))

        pbar.close()
        return batches


data_by_author = {}
with jsonlines.open('/content/author/2_reduced_compound_subreddit_hrs_redacted.jsonl') as reader:
    for obj in reader:
        # "authorIDs" as key
        for author in obj['authorIDs']:
            # documents list as value
            if author not in data_by_author:
                data_by_author[author] = []
            data_by_author[author].append(obj['fullText'])

# I will only use first 100 authors here
data_by_author = dict(list(data_by_author.items())[:100])
creator = HardBatchCreator(data_by_author, batch_size=32, ceiling_threshold=0.2)
batches = creator.create_batches_faiss()
