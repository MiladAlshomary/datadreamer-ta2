import pandas as pd
import fasttext
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from argparse import ArgumentParser

def v1_add_language_column_to_file(
    input_file_path,
    output_file_path,
    model,
    text_column='fullText',
    lang_column='language_x_biber',  # 'x' replaced with detection method
    method='fasttext',
    chunksize=20000
):
    """
    Reads a JSONL file in chunks, detects the language of the text column,
    and writes a new JSONL file with the detected language column added.

    Uses batch predictions per chunk for efficiency.
    Keeps the output file open for all chunks to reduce overhead,
    and uses tqdm progress bar instead of print statements.

    Parameters
    ----------
    input_file_path : str
        Path to the input JSONL file (one document per line).
    output_file_path : str
        Path to the output JSONL file.
    model : fasttext.FastText._FastText
        A loaded fastText language detection model.
    text_column : str
        Column name in the dataframe that contains the text to analyze.
    lang_column : str
        Column name to store the detected language. 'x' in the string
        will be replaced by the `method` name. Default is 'language_x_biber'.
    method : str
        Language detection method name. Defaults to 'fasttext'.
    chunksize : int
        Number of lines/rows to process at a time.
    """

    # Replace 'x' in lang_column with the detection method name
    lang_column = lang_column.replace("x", method)

    print(f"Reading from: {input_file_path}")
    print(f"Writing to:   {output_file_path}")
    print(f"Text column:  {text_column}")
    print(f"New language column: {lang_column}")
    print(f"Detection method: {method}")
    print(f"Chunk size: {chunksize}\n")

    if method.lower() != 'fasttext':
        raise ValueError(f"Only 'fasttext' is supported in this example, got '{method}'.")

    def detect_language_batch(text_list):
        # Create a cleaned copy for prediction: remove newline characters,
        # but do not modify the original texts.
        cleaned_texts = [
            t.replace("\n", " ") if isinstance(t, str) and t.strip() else ""
            for t in text_list
        ]
        # Use the cleaned texts for prediction
        labels, _ = model.predict(cleaned_texts)
        # Convert labels from "__label__en" to "en"
        return [lbl[0].replace('__label__', '') if lbl else 'unknown' for lbl in labels]

    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        with pd.read_json(input_file_path, lines=True, chunksize=chunksize, convert_dates=False) as reader:
            for chunk_df in tqdm(reader, desc="Processing chunks", unit="chunk"):
                # Get the original text column (with newlines preserved)
                chunk_texts = chunk_df[text_column].tolist()
                # Get predictions using the cleaned text copy
                predicted_langs = detect_language_batch(chunk_texts)
                # Add new language column to the chunk; original text remains unchanged
                chunk_df[lang_column] = predicted_langs
                # Write the processed chunk to the output file as JSON lines
                chunk_df.to_json(outfile, orient='records', lines=True, force_ascii=False)

    print(f"\nDone. The combined file with language annotations is ready here: {output_file_path}")

if __name__ == '__main__':
    parser = ArgumentParser(description="Language detect fullText column and add new field to every document in the file")
    parser.add_argument("--input_file_path", type=str, help="Input file path")
    parser.add_argument("--output_file_path", type=str, help="Output file path")
    parser.add_argument("--chunksize", type=int, default=10000, help="Batches of data to infer texts")
    args = parser.parse_args()

    # Download the fastText model (unquantized in this case) from Hugging Face Hub
    model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
    model = fasttext.load_model(model_path)

    v1_add_language_column_to_file(
        input_file_path=args.input_file_path,
        output_file_path=args.output_file_path,
        model=model,
        chunksize=args.chunksize
    )
