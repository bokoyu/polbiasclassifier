import pandas as pd
import os

# Define absolute paths for the folders
counts_folder = r"C:\Users\Borko\politicalbiasclassifier\data\phrasebias_data\phrase_counts"
phrases_folder = r"C:\Users\Borko\politicalbiasclassifier\data\phrasebias_data\phrase_selection"
output_folder = r"C:\Users\Borko\politicalbiasclassifier\data\phrasebias_data\combined_data"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Debugging: Check if folders exist
print("Counts folder exists:", os.path.exists(counts_folder))
print("Phrases folder exists:", os.path.exists(phrases_folder))
print("Output folder exists:", os.path.exists(output_folder))

# List all files in each folder
if os.path.exists(counts_folder) and os.path.exists(phrases_folder):
    counts_files = [f for f in os.listdir(counts_folder) if f.endswith('_counts.csv')]
    phrases_files = [f for f in os.listdir(phrases_folder) if f.endswith('_phrases.csv')]
else:
    print("Error: One or more folders do not exist.")
    exit(1)

bias_map = {
    'ap': 'Neutral', 'atlantic': 'Left', 'bbc': 'Neutral', 'breitbart': 'Right',
    'buzzfeed': 'Left', 'cbs': 'Neutral', 'cnn': 'Left', 'commondreams': 'Left',
    'counterpunch': 'Left', 'dailycaller': 'Right', 'dailymail': 'Right',
    'dailywire': 'Right', 'economist': 'Neutral', 'federalist': 'Right', 'fox': 'Right',
    'guardian': 'Left', 'huffingtonpost': 'Left', 'infowars': 'Right', 'intercept': 'Left',
    'jacobinmag': 'Left', 'motherjones': 'Left', 'nationalreview': 'Right', 'nbc': 'Neutral',
    'npr': 'Neutral', 'nypost': 'Right', 'nytimes': 'Left', 'pbs': 'Neutral', 'pjmedia': 'Right',
    'rawstory': 'Left', 'redstate': 'Right', 'rt': 'Right', 'slate': 'Left', 'spectator': 'Right',
    'townhall': 'Right', 'truthdig': 'Left', 'usatoday': 'left', 'vice': 'Left',
    'vox': 'Left', 'wapo': 'Left', 'wsj': 'Right'
}

# Numeric bias scores
bias_scores = {'Left': -1, 'Neutral': 0, 'Right': 1}

# Function to combine corresponding files
def combine_files(counts_file, phrases_file, output_folder):
    try:
        # Load the datasets
        counts_path = os.path.join(counts_folder, counts_file)
        phrases_path = os.path.join(phrases_folder, phrases_file)

        print(f"Processing: {counts_file} and {phrases_file}")
        counts_df = pd.read_csv(counts_path)
        phrases_df = pd.read_csv(phrases_path)

        # Check if 'PHRASE' column exists in both
        if 'PHRASE' not in counts_df.columns or 'PHRASE' not in phrases_df.columns:
            print(f"Error: 'PHRASE' column missing in {counts_file} or {phrases_file}")
            return

        # Merge the datasets on the 'PHRASE' column
        combined_df = pd.merge(counts_df, phrases_df, on="PHRASE", how="outer")

        # Fill missing values (optional)
        combined_df = combined_df.fillna({'TOTAL': 0, 'COUNT': 0, 'ON_TOPIC': 0, 'UNIQUE': 0, 'SPECIFIC': 0})

        # Identify source columns dynamically
        irrelevant_columns = ['PHRASE', 'TOTAL', 'COUNT', 'ON_TOPIC', 'UNIQUE', 'SPECIFIC', 'bias', 'Unnamed: 0']
        source_columns = [col for col in combined_df.columns if col not in irrelevant_columns]

        # Function to calculate weighted bias
        def calculate_bias(row):
            bias_score = 0
            for source in source_columns:
                if source in bias_map:
                    bias = bias_map[source]
                    score = bias_scores[bias]  # Get numeric score for the bias
                    bias_score += row[source] * score  # Weighted contribution
            return bias_score

        # Apply the bias calculation
        combined_df['bias_score'] = combined_df.apply(calculate_bias, axis=1)

        # Categorize phrases based on the bias score
        def categorize_bias(score):
            if score > 0:
                return 'Right'
            elif score < 0:
                return 'Left'
            else:
                return 'Neutral'

        combined_df['calculated_bias'] = combined_df['bias_score'].apply(categorize_bias)

        # Save the combined dataset
        output_file = os.path.join(output_folder, counts_file.replace('_counts.csv', '_combined.csv'))
        combined_df.to_csv(output_file, index=False)
        print(f"Combined dataset saved to {output_file}")

    except Exception as e:
        print(f"Error processing {counts_file} and {phrases_file}: {e}")

# Iterate through corresponding files and combine them
for counts_file in counts_files:
    # Find the matching phrases file
    phrases_file = counts_file.replace('_counts.csv', '_phrases.csv')
    if phrases_file in phrases_files:
        combine_files(counts_file, phrases_file, output_folder)
    else:
        print(f"Warning: No matching phrases file found for {counts_file}")
