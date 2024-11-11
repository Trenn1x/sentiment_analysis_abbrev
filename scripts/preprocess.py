# scripts/preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(csv_path, output_path, chunk_size=5000, batch_size=5):
    train_chunks = []
    test_chunks = []
    batch_counter = 0

    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, header=None):
        # Select columns and add sentiment labels
        chunk = chunk[[3, 7]]  # Adjust for actual columns: 4th (rating), 8th (review text)
        chunk.columns = ['stars', 'text']
        chunk['sentiment'] = chunk['stars'].apply(lambda x: 1 if x >= 4 else 0)

        # Split the data and add to the list
        train_chunk, test_chunk = train_test_split(chunk, test_size=0.2, random_state=42)
        train_chunks.append(train_chunk)
        test_chunks.append(test_chunk)
        batch_counter += 1

        # Write to CSV every `batch_size` chunks
        if batch_counter >= batch_size:
            pd.concat(train_chunks).to_csv(f"{output_path}/train.csv", mode='a', header=False, index=False)
            pd.concat(test_chunks).to_csv(f"{output_path}/test.csv", mode='a', header=False, index=False)
            train_chunks, test_chunks = [], []  # Clear lists after writing
            batch_counter = 0

    # Write any remaining data after the loop
    if train_chunks:
        pd.concat(train_chunks).to_csv(f"{output_path}/train.csv", mode='a', header=False, index=False)
    if test_chunks:
        pd.concat(test_chunks).to_csv(f"{output_path}/test.csv", mode='a', header=False, index=False)

    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    csv_path = "../data/yelp_reviews.csv"
    output_path = "../data"
    preprocess_data(csv_path, output_path)

