#!/usr/bin/env python
# Re-import necessary libraries and re-define functions since the execution state was reset.
import pandas as pd
import numpy as np
import ast


df = pd.read_csv('chicago_data.csv')
df = df.fillna(0)
chosen_wards = [1, 2, 26, 32, 43]

def haversine_miles(lat1, lon1, lat2, lon2):
    """
    Uses the Haversine function to calculate the distance between the centers
    of the wards to the center of Chicago.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 3958.8 # Radius of Earth in miles
    return c * r

def ward_centroids(dataframe, chosen_wards):
    """
    Calculates the centroids of each fo the chosen wards.
    """
    centroids = []

    for ward in chosen_wards:
        # Remove all zeros
        ward_df = df[df["ward"] == ward].reset_index(drop=True)
        ward_df['location'] = ward_df['location'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        ward_df = ward_df[ward_df['location'] != 0]
        x_coord = ward_df["location"].apply(lambda x: x[0]).mean()
        y_coord = ward_df["location"].apply(lambda x: x[1]).mean()
        centroids.append((ward, x_coord, y_coord))

    centroid_df = pd.DataFrame(centroids, columns=["ward", "centroid_lat",
                                                   "centroid_long"])
    return centroid_df

# Center of Chicago coordinates
chicago_center = (41.8268944184666, -87.67148314232931)

ward_centers = ward_centroids(df, chosen_wards)
breakpoint()
# Calculate the distance in miles of each ward center to the center of Chicago
ward_centers['distance_to_chicago_center_mi'] = ward_centers.apply(
    lambda row: haversine_miles(chicago_center[0], chicago_center[1],
                                row['centroid_lat'], row['centroid_long']),
    axis=1
)

print(ward_centers)

# KILOMETERS
## Define the Haversine formula again
#def haversine(lat1, lon1, lat2, lon2):
#    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
#    dlat = lat2 - lat1
#    dlon = lon2 - lon1
#    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
#    c = 2 * np.arcsin(np.sqrt(a))
#    r = 6371 # Radius of Earth in kilometers. Use 3956 for miles
#    return c * r
#
## Load the dataset
#
#np.random.seed(0)  # Ensure reproducible results
#df['latitude'] = 41.8 + np.random.rand(len(df)) * 0.1
#df['longitude'] = -87.7 + np.random.rand(len(df)) * -0.1
#df['ward'] = np.random.choice([1, 2, 26, 32, 43], len(df))
#
## Selected wards based on your criteria
#selected_wards = [43, 32, 26, 2, 1]
#
## Filter for selected wards
#df_selected_wards = df[df['ward'].isin(selected_wards)]
#
## Calculate the approximate center of each ward
#ward_centers = df_selected_wards.groupby('ward').agg({'latitude': 'mean', 'longitude': 'mean'}).reset_index()
#
## Center of Chicago coordinates
#chicago_center = (41.8268944184666, -87.67148314232931)
#
## Calculate the distance of each ward center to the center of Chicago
#ward_centers['distance_to_chicago_center_km'] = ward_centers.apply(
#    lambda row: haversine(chicago_center[0], chicago_center[1], row['latitude'], row['longitude']),
#    axis=1
#)
#
#ward_centers[['ward', 'distance_to_chicago_center_km']]
#


