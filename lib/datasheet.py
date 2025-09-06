import pandas as pd
import numpy as np

class DataSheet:
    def __init__(self, file_path, scaler):
        self.df = None
        self.scaler = scaler
        self.file_path   = file_path
        self.data_scaled = None

    def get_df(self):
        return self.df

    def get_scaler(self):
        return self.scaler

    def get_data_scaled(self):
        return self.data_scaled

    # --- Load datasheet ---
    def load(self):
        try:
            self.df = pd.read_excel(self.file_path)
            print("Dataset loaded successfully!")
        except FileNotFoundError:
            print("Error: 'Attribute DataSet.xlsx' not found. Please make sure the file is in the correct directory.")
            exit()

        # --- Initial Inspection ---
        print("\nFirst 5 rows of the dataset:")
        print(self.df[['Dress_ID', 'Style', 'Price', 'Rating', 'Size', 'Season', 'NeckLine', 'SleeveLength']].head())

        print("\nDataset Information (dtypes, non-null counts):")
        self.df.info()
        return self.df

    # --- Training data ---
    def training(self):
        # Replace comma with dot and convert to numeric
        self.df['Price'] = self.df['Price'].map(self.encoding_price())

        # Replace the string 'null' with NumPy's NaN (Not a Number)
        self.df.replace('null', np.nan, inplace=True)

        # Fill missing values with the mode (most frequent value) of each column
        for column in self.df.columns:
            if self.df[column].isnull().any():
                mode_value = self.df[column].mode()[0]
                self.df[column].fillna(mode_value, inplace=True)
                print(f"Filled missing values in '{column}' with mode: '{mode_value}'")

        self.scaling()
        # return self.df

    # --- Scaling ---
    def scaling(self):
        X_2d = self.df[['Price', 'Rating']].copy()
        self.data_scaled = self.scaler.fit_transform(X_2d)

        print("\nFeature scaling complete.")
        return self.data_scaled

    # --- Encoding Price to define the order of the categories ---
    def encoding_price(self):
        pricing_map = {'Low': 1, 'low': 1, 'Average': 2, 'average': 2, 'Medium': 3, 'medium': 3, 'High': 4, 'high': 4,
                       'very-high': 5}
        return pricing_map

    # --- Write new datasheet with cluster label ---
    def save(self, cluster, output_dataset_file_path):
        self.df['Cluster'] = cluster.labels_

        # Save the final dataframe to a new CSV file
        self.df.to_csv(output_dataset_file_path, index=False, sep=';')

        print(f"\nSuccessfully saved the new dataset with cluster labels to '{output_dataset_file_path}'")
        print("\nFirst 5 rows of the new dataset:")
        print(self.df.head())