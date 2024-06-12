import pandas as pd
from sklearn.utils import resample

# Load the train data
train_data = pd.read_csv('val_asset_embedding_label.csv')

# Separate the data into the two classes
bad_data = train_data[train_data['label'] == 'Bad']
good_data = train_data[train_data['label'] == 'Good']

# Oversample the minority class (Good)
good_data_oversampled = resample(good_data,
                                 replace=True,  # Sample with replacement
                                 n_samples=len(bad_data),  # Match number of majority class
                                 random_state=42)  # Reproducible results

# Combine the majority class with the oversampled minority class
balanced_train_data = pd.concat([bad_data, good_data_oversampled])

# Shuffle the data
balanced_train_data = balanced_train_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the balanced data to a new CSV file
balanced_train_data.to_csv('balanced_val_data.csv', index=False)

print("Balanced training data saved to 'balanced_train_data.csv'.")
