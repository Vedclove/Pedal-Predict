import os
import sys
import sys
from pathlib import Path

# Add the src folder to the Python module search path
sys.path.append(str(Path(__file__).resolve().parent))

import calendar

# Add the parent directory to the Python path
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Union

import zipfile
import numpy as np
import pandas as pd
import pytz
import requests


from config import RAW_DATA_DIR

def fetch_raw_trip_data(year: int, month: int) -> Path:
    URL = f"https://s3.amazonaws.com/tripdata/JC-{year}{month:02}-citibike-tripdata.csv.zip"
    response = requests.get(URL)

    if response.status_code == 200:
        # Ensure the RAW_DATA_DIR exists
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Save the zip file to the RAW_DATA_DIR
        zip_path = RAW_DATA_DIR / f"rides_{year}_{month:02}.zip"
        with open(zip_path, "wb") as zip_file:
            zip_file.write(response.content)

        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)

        # Find the CSV file in the extracted directory
        csv_files = list(RAW_DATA_DIR.glob("*.csv"))
        if not csv_files:
            raise Exception(f"No CSV file found in the extracted zip: {zip_path}")

        return csv_files[0]
    else:
        raise Exception(f"{URL} is not available")

def fill_missing_rides_full_range(df, hour_col, location_col, rides_col):
    """
    Fills in missing rides for all hours in the range and all unique locations.

    Parameters:
    - df: DataFrame with columns [hour_col, location_col, rides_col]
    - hour_col: Name of the column containing hourly timestamps
    - location_col: Name of the column containing location IDs
    - rides_col: Name of the column containing ride counts

    Returns:
    - DataFrame with missing hours and locations filled in with 0 rides
    """
    # Ensure the hour column is in datetime format
    df[hour_col] = pd.to_datetime(df[hour_col])

    # Get the full range of hours (from min to max) with hourly frequency
    full_hours = pd.date_range(
        start=df[hour_col].min(),
        end=df[hour_col].max(),
        freq="h"
    )

    # Get all unique location IDs
    all_locations = df[location_col].unique()

    # Create a DataFrame with all combinations of hours and locations
    full_combinations = pd.DataFrame(
        [(hour, location) for hour in full_hours for location in all_locations],
        columns=[hour_col, location_col]
    )

    # Merge the original DataFrame with the full combinations DataFrame
    merged_df = pd.merge(full_combinations, df, on=[hour_col, location_col], how='left')

    # Fill missing rides with 0
    merged_df[rides_col] = merged_df[rides_col].fillna(0).astype(int)

    return merged_df

def fetch_batch_raw_data(from_date: Union[datetime, str], to_date: Union[datetime, str]) -> pd.DataFrame:
    """
    Simulate production data by sampling historical data from 52 weeks ago (i.e., 1 year).

    Args:
        from_date (datetime or str): The start date for the data batch.
        to_date (datetime or str): The end date for the data batch.

    Returns:
        pd.DataFrame: A DataFrame containing the simulated production data.
    """
    # Convert string inputs to datetime if necessary
    if isinstance(from_date, str):
        from_date = datetime.fromisoformat(from_date)
    if isinstance(to_date, str):
        to_date = datetime.fromisoformat(to_date)

    # Validate input dates
    if not isinstance(from_date, datetime) or not isinstance(to_date, datetime):
        raise ValueError("Both 'from_date' and 'to_date' must be datetime objects or valid ISO format strings.")
    if from_date >= to_date:
        raise ValueError("'from_date' must be earlier than 'to_date'.")

    # Shift dates back by 52 weeks (1 year)
    historical_from_date = from_date - timedelta(weeks=52)
    historical_to_date = to_date - timedelta(weeks=52)

    # Load and filter data for the historical period
    year = [historical_from_date.year]
    a , rides_from = load_and_process_citi_data(year, months=[historical_from_date.month])
    rides_from['pickup_hour'] = pd.to_datetime(rides_from['pickup_hour'])
    historical_from_date = pd.to_datetime(historical_from_date)
    rides_from = rides_from[rides_from.pickup_hour >= historical_from_date]

    if historical_to_date.month != historical_from_date.month:
        a , rides_to = load_and_process_citi_data(year, months=[historical_to_date.month])
        rides_to['pickup_hour'] = pd.to_datetime(rides_to['pickup_hour'])
        historical_to_date = pd.to_datetime(historical_to_date)
        rides_to = rides_to[rides_to.pickup_hour < historical_to_date]
        # Combine the filtered data
        rides = pd.concat([rides_from, rides_to], ignore_index=True)
    else:
        rides = rides_from
    # Shift the data forward by 52 weeks to simulate recent data
    rides['pickup_hour'] += timedelta(weeks=52)

    # Sort the data for consistency
    rides.sort_values(by=['station_id', 'pickup_hour'], inplace=True)

    return rides

def load_and_process_citi_data(years: list,  months: Optional[List[int]] = None) -> pd.DataFrame:
    if months is None:
        months = list(range(1, 13))
    # List to store DataFrames for each month
    monthly_rides = []

    for year in years:
        for month in months:
            # Construct the file path
            file_path = RAW_DATA_DIR / f"JC-{year}{month:02}-citibike-tripdata.csv"

            # Load the data
            print(f"Loading data for {year}-{month:02}.")
            try:
                rides = pd.read_csv(file_path)
            # Append the processed DataFrame to the list
                monthly_rides.append(rides)
            except FileNotFoundError:
                continue

        # Combine all monthly data
        if not monthly_rides:
            raise Exception(
                f"No data could be loaded for the year {year} and specified months: {months}"
            )

        print("Combining all monthly data...")
        combined_rides = pd.concat(monthly_rides, ignore_index=True)
        print("Data loading and processing complete!")

        columns_to_drop = ['ride_id', 'end_station_name', 'rideable_type', 'ended_at', 'end_station_id','start_lat', 'start_lng', 'end_lat', 'end_lng', 'member_casual']  # Specify the columns to drop
        processed_rides = combined_rides.drop(columns=columns_to_drop)

        processed_rides.rename(columns={"started_at": "pickup_hour", "start_station_name": "station_name", "start_station_id": "station_id"}, inplace=True)        

    return combined_rides, processed_rides


def transform_data_into_ts_data(df):
    # Filter data for the desired data locations
    station_ids = ["HB101", "HB202", "JC103","HB404", "JC009", "HB201", "JC005", 'JC006', 'JC106', 'JC115']
    df = df[df["station_id"].isin(station_ids)]

    # convert the datatype for pickup
    df["pickup_hour"] = pd.to_datetime(df["pickup_hour"])
    df["pickup_hour"] = df["pickup_hour"].dt.floor('h') # floor is to the nearest hour

    # Group data and make it more better looking
    df = df.groupby(["pickup_hour", "station_id"]).size().reset_index()
    df.rename(columns={0: "rides"}, inplace=True)

    hour_col = "pickup_hour"
    location_col = "station_id"
    rides_col = "rides"
    interval = "6H" # update
    df = fill_missing_rides_full_range(df, hour_col, location_col, rides_col).sort_values(["station_id", "pickup_hour"]).reset_index(drop=True)

    particular_date_6h = datetime(2021, 7, 17) # Update
    df = df[df["pickup_hour"] >= particular_date_6h]     

    # Set the hour column as the index
    df = df.set_index(hour_col)
#
    ## Resample and aggregate rides
    df = df.groupby("station_id").resample(interval)[rides_col].sum().reset_index()

    return df

def transform_ts_data_info_features_and_target(
    df, feature_col="rides", window_size=12, step_size=1
):
    # Get all unique location IDs
    location_ids = df["station_id"].unique()
    # List to store transformed data for each location
    transformed_data = []

    # Loop through each location ID and transform the data
    for location_id in location_ids:
        try:
            # Filter the data for the given location ID
            location_data = df[df["station_id"] == location_id].reset_index(
                drop=True
            )

            # Extract the feature column and pickup_hour as NumPy arrays
            values = location_data[feature_col].values
            times = location_data["pickup_hour"].values

            # Ensure there are enough rows to create at least one window
            if len(values) <= window_size:
                raise ValueError("Not enough data to create even one window.")

            # Create the tabular data using a sliding window approach
            rows = []
            for i in range(0, len(values) - window_size, step_size):
                # The first `window_size` values are features, and the next value is the target
                features = values[i : i + window_size]
                target = values[i + window_size]
                # Get the corresponding target timestamp
                target_time = times[i + window_size]
                # Combine features, target, location_id, and timestamp
                row = np.append(features, [target, location_id, target_time])
                rows.append(row)

            # Convert the list of rows into a DataFrame
            feature_columns = [
                f"{feature_col}_t-{window_size - i}" for i in range(window_size)
            ]
            all_columns = feature_columns + [
                "target",
                "station_id",
                "pickup_hour",
            ]
            transformed_df = pd.DataFrame(rows, columns=all_columns)

            # Append the transformed data to the list
            transformed_data.append(transformed_df)

        except ValueError as e:
            print(f"Skipping location_id {location_id}: {str(e)}")

    # Combine all transformed data into a single DataFrame
    if not transformed_data:
        raise ValueError(
            "No data could be transformed. Check if input DataFrame is empty or window size is too large."
        )

    final_df = pd.concat(transformed_data, ignore_index=True)

    # Extract features (including pickup_hour), targets, and keep the complete DataFrame
    features = final_df[feature_columns + ["pickup_hour", "station_id"]]
    targets = final_df["target"]

    return features, targets