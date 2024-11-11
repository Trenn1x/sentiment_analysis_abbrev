# scripts/data_conversion.py
import pandas as pd
import json

def convert_json_to_csv(json_path, csv_path, chunk_size=10000):
    with open(json_path, 'r') as file:
        # Process the file in chunks to manage memory better
        batch_data = []
        for i, line in enumerate(file):
            batch_data.append(json.loads(line))
            if (i + 1) % chunk_size == 0:
                df = pd.DataFrame(batch_data)
                df.to_csv(csv_path, mode='a', header=not i, index=False)
                batch_data = []  # Reset batch

        # Final batch write
        if batch_data:
            df = pd.DataFrame(batch_data)
            df.to_csv(csv_path, mode='a', header=False, index=False)
        print(f"Converted {json_path} to {csv_path}")

if __name__ == "__main__":
    json_path = "../data/yelp_academic_dataset_review.json"
    csv_path = "../data/yelp_reviews.csv"
    convert_json_to_csv(json_path, csv_path)

