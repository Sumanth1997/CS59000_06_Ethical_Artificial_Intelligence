import pandas as pd
import os

def analyze_merged_data(csv_file_path):
    """
    Reads a CSV file, prints its data description, and handles potential errors.

    Args:
        csv_file_path (str): The path to the CSV file.
    """
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_file_path)

        # Print basic information about the DataFrame
        print("\n--- DataFrame Information ---\n")
        df.info()

        # Print descriptive statistics for numerical columns
        print("\n--- Descriptive Statistics ---\n")
        print(df.describe())

        # Print the first few rows of the DataFrame
        print("\n--- First 5 Rows of Data ---\n")
        print(df.head())

        # Print the shape of the dataframe
        print("\n--- Shape of Data ---\n")
        print(df.shape)
        
        #Print the number of values by label
        print("\n--- Number of values by label ---\n")
        print(df['label'].value_counts())

        # Print the number of values by source
        print("\n--- Number of values by source ---\n")
        print(df['source'].value_counts())

    except FileNotFoundError:
        print(f"Error: File not found: {csv_file_path}")
    except pd.errors.EmptyDataError:
        print(f"Error: The file is empty: {csv_file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # construct absolute path
    merged_csv_path = os.path.join(script_dir, "fakenewnet_dataset/FakeNewsNet/dataset/FakeNewsNet.csv")

    # Analyze the data
    analyze_merged_data(merged_csv_path)
