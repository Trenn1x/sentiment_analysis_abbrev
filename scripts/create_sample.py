import pandas as pd

# Load the full training data
df = pd.read_csv("../data/train.csv", header=None, names=['stars', 'text', 'sentiment'])

# Take a sample of 50 rows
sample_df = df.sample(n=50, random_state=42)

# Save the sample to a new file
sample_df.to_csv("../data/train_sample.csv", index=False, header=False)

