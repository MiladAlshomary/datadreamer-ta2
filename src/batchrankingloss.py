import pandas as pd
from batching_for_rankingloss import HardBatchCreator
input_file = "/data/araghavan/HIATUS/datadreamer-ta2/data/ta2_jan_2025_trian_data/train_sadiri_processed_with_embeddings_wo_ao3_filtered.jsonl"
df = pd.read_json(input_file, lines=True)
print(f"Sadiri Dataset Shape: {df.shape}")
auth_df = df.groupby('authorID', as_index=False).agg({'documentID':list, 'fullText':list})
print(f"Grouped by Author Shape: {auth_df.shape}")
data_by_author = auth_df.set_index('authorID')['fullText'].to_dict()
print(f"Number of authors: {len(data_by_author)}")
creator = HardBatchCreator(data_by_author, batch_size=32, ceiling_threshold=0.4)
batches = creator.create_batches_faiss()
print(len(batches))