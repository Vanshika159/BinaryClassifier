import pandas as pd

# Load the dataset
data = pd.read_csv('output1_filtered.csv')

# Define thresholds (these are arbitrary and should be set based on your data analysis)
clicks_threshold = 50
downloads_threshold = 10
dtr_threshold = 0.001
impression_threshold = 1000

# Derive labels based on thresholds and rules
def classify_template(row):
    if ((row['all.true_impressions.asset'] > impression_threshold and row['all.clicks.asset'] > clicks_threshold and
         row['all.downloads.asset'] > downloads_threshold) or (
            row['all.true_impressions.asset'] <= impression_threshold and row['all.clicks.asset'] >= 10 and
            row['all.downloads.asset'] >= 5)):
        return 1  # Good template
    else:
        return 0  # Bad template

# Apply the classification function to derive labels
data['template_label'] = data.apply(classify_template, axis=1)

# Write the labeled data to a text file
with open('labeled_data1.txt', 'w') as file:
    file.write("Asset Id, Impressions, Clicks, Downloads, DTR, Label\n")
    for index, row in data.iterrows():
        info = (f"{row['asset_id']},"
                f"{row['all.true_impressions.asset']},"
                f"{row['all.clicks.asset']},"
                f"{row['all.downloads.asset']},"
                f" {row['dtr']},"
                f"{'Good' if row['template_label'] == 1 else 'Bad'}\n")
        file.write(info)

print("Data written to labeled_data.txt successfully.")
