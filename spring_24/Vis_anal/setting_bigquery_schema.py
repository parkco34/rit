#!/usr/bin/env python
from google.cloud import bigquery
import pandas as pd

# Read textfile data
text_file_path = 'opioid.txt'
csv_file_path = 'opioid_data.csv'

# convert text file to csv file
def text_to_csv(text_file_path, csv_file_path):
    # Open the text file in read mode and the CSV file in write mode
    with open(text_file_path, 'r') as text_file, open(csv_file_path, 'w', newline='') as csv_file:
        # Assuming fields in the text file are separated by spaces
        # You can change ' ' to any other delimiter like ',', '\t', etc.
        delimiter = ' '

        # Create a CSV writer object for writing to CSV file
        csv_writer = csv.writer(csv_file)

        # Read each line from the text file
        for line in text_file:
            # Split the line into fields using the defined delimiter
            fields = line.strip().split(delimiter)

            # Write the fields as a row in the CSV file
            csv_writer.writerow(fields)

# Data
df = pd.read_csv("opioid_data.csv", low_memory=False)

# Cleaning data
def remove_columns_with_many_nans(df, threshold=0.5):
    """
    Remove columns from the DataFrame where the proportion of NaN values is greater than the specified threshold.

    :param df: pandas DataFrame from which to remove columns.
    :param threshold: float, the proportion of NaN values (between 0 and 1) to use as a threshold for removing columns.
    :return: DataFrame with columns removed.
    """
    # Calculate the proportion of NaNs for each column
    proportion_of_nans = df.isnull().mean()
    # Find columns where the proportion of NaNs is above the threshold
    columns_to_drop = proportion_of_nans[proportion_of_nans > threshold].index
    # Drop these columns
    df_dropped = df.drop(columns=columns_to_drop)

    return df_dropped

# Generate a BigQuery schema from the DataFrame dtypes
def generate_schema(df):
    schema = []
    for col_name, dtype in df.dtypes.items():
        if dtype in ['int16', 'int32', 'int64']:
            field = bigquery.SchemaField(name=col_name, field_type='INT64')
        elif dtype in ['float16', 'float32', 'float64']:
            field = bigquery.SchemaField(name=col_name, field_type='FLOAT64')
        elif dtype == 'object':  # object type, which is often string
            field = bigquery.SchemaField(name=col_name, field_type='STRING')
        else:
            field = bigquery.SchemaField(name=col_name, field_type='STRING')  # Default to STRING for safety
        schema.append(field)
    return schema


breakpoint()
# Set the threshold as you see fit, here it's set to 50% NaN values.
df_cleaned = remove_columns_with_many_nans(df, threshold=0.5)

# Write dataframe to csv file
df_cleaned.to_csv("opioid_data.csv")

# Assuming your DataFrame is named df
bq_schema = generate_schema(df_cleaned)

# Big Query client stuff
client = bigquery.Client()
breakpoint()

## Specify your dataset and table names
#dataset_name = 'your_dataset_name'  # Update this with your dataset name
#table_name = 'your_table_name'    # Update this with your table name
#
## Construct the dataset ID and table ID
#dataset_id = f"{client.project}.{dataset_name}"
#table_id = f"{dataset_id}.{table_name}"
#
## Create Dataset if it doesn't exist
#dataset = bigquery.Dataset(dataset_id)
#dataset.location = "US"  # you can change this to your preferred location
#try:
#    dataset = client.create_dataset(dataset, exists_ok=True)  # Make an API request.
#    print(f"Dataset {dataset_id} created.")
#except Exception as e:
#    print(e)
#    print(f"Failed to create dataset {dataset_id}.")
#
## Define the table with the schema
#table = bigquery.Table(table_id, schema=bq_schema)
#
## Create the table
#try:
#    table = client.create_table(table, exists_ok=True)  # Make an API request.
#    print(f"Table {table_id} created.")
#except Exception as e:
#    print(e)
#    print(f"Failed to create table {table_id}.")
#
## Now you can insert data into the table from your DataFrame
#df_cleaned.to_gbq(destination_table=table_id, project_id=client.project, if_exists='replace', progress_bar=True)
#
#
#
