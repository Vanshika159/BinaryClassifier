import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('labeled_data.csv')

# Separate the 'Good' and 'Bad' examples
good_data = data[data['label'] == 'Good']
bad_data = data[data['label'] == 'Bad']

# Split the 'Good' examples into training, testing, and validation sets
good_train, good_temp = train_test_split(good_data, test_size=0.4, random_state=42)
good_val, good_test = train_test_split(good_temp, test_size=0.5, random_state=42)

# Split the 'Bad' examples into training, testing, and validation sets
bad_train, bad_temp = train_test_split(bad_data, test_size=0.4, random_state=42)
bad_val, bad_test = train_test_split(bad_temp, test_size=0.5, random_state=42)

# Combine 'Good' and 'Bad' examples to form training, testing, and validation sets
train_data = pd.concat([good_train, bad_train])
val_data = pd.concat([good_val, bad_val])
test_data = pd.concat([good_test, bad_test])

# Shuffle the combined sets to ensure a good mix of examples
train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
val_data = val_data.sample(frac=1, random_state=42).reset_index(drop=True)
test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Write the splits to separate files
train_data.to_csv('train_data.csv', index=False)
val_data.to_csv('val_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

print(f"Training data: {train_data.shape[0]} samples")
print(f"Validation data: {val_data.shape[0]} samples")
print(f"Testing data: {test_data.shape[0]} samples")
