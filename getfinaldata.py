import pandas as pd
import json

# Load embeddings
with open('embeddings1.txt', 'r') as f:
    embeddings_data = json.load(f)

# Function to get embedding for an asset ID
def get_embedding(asset_id):
    return embeddings_data.get(asset_id, {}).get('embedding1', {}).get('field_double_nonsearch_nonstore_dense_vector_dims_1024_1', None)

# Process and save data for each split
def process_and_save_data(file_name, output_name):
    # Load labeled data
    labeled_data = pd.read_csv(file_name)
    
    # Create a new DataFrame to store the result
    result_data = []
    
    # Iterate through asset IDs in labeled data
    for index, row in labeled_data.iterrows():
        asset_id = row['Asset Id']
        embedding = get_embedding(asset_id)
        
        # If embedding is found for the asset ID, append it to the result
        if embedding:
            label = row['label']
            result_data.append({'asset_id': asset_id, 'embedding': embedding, 'label': label})
    
    # Create DataFrame from result data
    result_df = pd.DataFrame(result_data)
    
    # Write the result to a new file
    result_df.to_csv(output_name, index=False)
    print(f"Data written to {output_name} successfully.")

# Process train, test, and validation data
process_and_save_data('train_data.csv', 'train_asset_embedding_label.csv')
process_and_save_data('test_data.csv', 'test_asset_embedding_label.csv')
process_and_save_data('val_data.csv', 'val_asset_embedding_label.csv')
