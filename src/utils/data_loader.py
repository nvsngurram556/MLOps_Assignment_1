"""
Data loading and preprocessing utilities for California Housing dataset.
"""

import logging
import os
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handle data loading and preprocessing for California Housing dataset."""

    def __init__(self, data_dir: str = "data", random_state: int = 42):
        """
        Initialize DataLoader.

        Args:
            data_dir: Directory to save/load data
            random_state: Random state for reproducibility
        """
        self.data_dir = data_dir
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")

        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

    def fetch_and_save_data(self, force_download: bool = False) -> pd.DataFrame:
        """
        Fetch California Housing data and save to CSV.

        Args:
            force_download: Whether to force re-download even if data exists

        Returns:
            DataFrame with features and target
        """
        data_path = os.path.join(self.data_dir, "california_housing_raw.csv")

        if os.path.exists(data_path) and not force_download:
            logger.info(f"Loading existing data from {data_path}")
            df = pd.read_csv(data_path)
        else:
            logger.info("Fetching California Housing dataset...")
            housing = fetch_california_housing(as_frame=True)

            # Combine features and target
            df = housing.data.copy()
            df["target"] = housing.target

            # Add feature names for clarity
            df.columns = [
                "MedInc",
                "HouseAge",
                "AveRooms",
                "AveBedrms",
                "Population",
                "AveOccup",
                "Latitude",
                "Longitude",
                "Price",
            ]

            # Save raw data
            df.to_csv(data_path, index=False)
            logger.info(f"Data saved to {data_path}")

        logger.info(f"Dataset shape: {df.shape}")
        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data with cleaning and feature engineering.

        Args:
            df: Raw dataframe

        Returns:
            Preprocessed dataframe
        """
        logger.info("Starting data preprocessing...")

        # Create a copy to avoid modifying original
        processed_df = df.copy()

        # Handle missing values (though California Housing typically has none)
        if processed_df.isnull().sum().sum() > 0:
            logger.warning("Found missing values, imputing...")
            numeric_columns = processed_df.select_dtypes(include=[np.number]).columns
            processed_df[numeric_columns] = self.imputer.fit_transform(
                processed_df[numeric_columns]
            )

        # Feature engineering
        processed_df["RoomsPerHousehold"] = (
            processed_df["AveRooms"] / processed_df["AveOccup"]
        )
        processed_df["BedroomsPerRoom"] = (
            processed_df["AveBedrms"] / processed_df["AveRooms"]
        )
        processed_df["PopulationPerHousehold"] = (
            processed_df["Population"] / processed_df["AveOccup"]
        )

        # Handle infinite values
        processed_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        processed_df.fillna(0, inplace=True)

        # Remove outliers using IQR method
        Q1 = processed_df.quantile(0.25)
        Q3 = processed_df.quantile(0.75)
        IQR = Q3 - Q1

        # Filter out outliers
        outlier_condition = ~(
            (processed_df < (Q1 - 1.5 * IQR)) | (processed_df > (Q3 + 1.5 * IQR))
        ).any(axis=1)
        processed_df = processed_df[outlier_condition]

        logger.info(f"Data after preprocessing: {processed_df.shape}")
        logger.info(f"Removed {len(df) - len(processed_df)} outliers")

        return processed_df

    def split_and_scale_data(
        self, df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train/validation/test sets and scale features.

        Args:
            df: Preprocessed dataframe
            test_size: Proportion of test set
            val_size: Proportion of validation set

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Splitting and scaling data...")

        # Separate features and target
        X = df.drop("Price", axis=1)
        y = df["Price"]

        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=self.random_state
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # Save scaler for later use
        scaler_path = os.path.join(self.data_dir, "scaler.joblib")
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")

        logger.info(f"Train set shape: {X_train_scaled.shape}")
        logger.info(f"Validation set shape: {X_val_scaled.shape}")
        logger.info(f"Test set shape: {X_test_scaled.shape}")

        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test

    def save_processed_data(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """
        Save processed data to disk.

        Args:
            X_train, X_val, X_test: Feature arrays
            y_train, y_val, y_test: Target arrays
        """
        logger.info("Saving processed data...")

        data_files = {
            "X_train.npy": X_train,
            "X_val.npy": X_val,
            "X_test.npy": X_test,
            "y_train.npy": y_train,
            "y_val.npy": y_val,
            "y_test.npy": y_test,
        }

        for filename, data in data_files.items():
            filepath = os.path.join(self.data_dir, filename)
            np.save(filepath, data)
            logger.info(f"Saved {filename}")

    def load_processed_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load processed data from disk.

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Loading processed data...")

        data_files = [
            "X_train.npy",
            "X_val.npy",
            "X_test.npy",
            "y_train.npy",
            "y_val.npy",
            "y_test.npy",
        ]
        data = []

        for filename in data_files:
            filepath = os.path.join(self.data_dir, filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Processed data file not found: {filepath}")
            data.append(np.load(filepath))

        return tuple(data)

    def get_feature_names(self) -> list:
        """Get feature names including engineered features."""
        return [
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
            "RoomsPerHousehold",
            "BedroomsPerRoom",
            "PopulationPerHousehold",
        ]


def main():
    """Example usage of DataLoader."""
    loader = DataLoader()

    # Fetch and preprocess data
    raw_data = loader.fetch_and_save_data()
    processed_data = loader.preprocess_data(raw_data)

    # Split and scale
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_and_scale_data(
        processed_data
    )

    # Save processed data
    loader.save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)

    print("Data preparation completed successfully!")


if __name__ == "__main__":
    main()
