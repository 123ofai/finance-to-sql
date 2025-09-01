import pandas as pd

# List of CSV files to merge
csv_files = [
    '/home/gaurav/Downloads/Natural_Language_Queries_300.csv',
    '/home/gaurav/Downloads/Natural_Language_Queries_300_Simple.csv',
    '/home/gaurav/Downloads/Natural_Language_Queries_300_Noisy.csv'
]

# Initialize an empty list to store dataframes
dataframes = []

# Read each CSV file and append to the list
for file in csv_files:
    try:
        df = pd.read_csv(file)
        dataframes.append(df)
    except FileNotFoundError:
        print(f"Error: The file '{file}' was not found. Please check the file name and path.")
    except Exception as e:
        print(f"Error reading '{file}': {e}")

# Merge all dataframes into one
if dataframes:
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    # Save the merged dataframe to a new CSV file
    output_file = 'Merged_Natural_Language_Queries.csv'
    try:
        merged_df.to_csv(output_file, index=False)
        print(f"Successfully merged {len(csv_files)} files into '{output_file}' with {len(merged_df)} rows.")
    except Exception as e:
        print(f"Error saving merged file: {e}")
else:
    print("No files were successfully read. Merging aborted.")