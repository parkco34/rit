#!/usr/bin/env python
# Re-import necessary libraries and re-define functions since the execution state was reset.
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import numpy as np
from math import sqrt
import ast

"""
! ---> Convert ward column to integer!
"""

# Disances between center of wards and the given highway: {highway, ward,
# distance (miles)}
hiways = {
    "Kennedy Expressway": {
        "1": 1.9,
        "2": 4.4,
        "26": 2.9,
        "32": 0.9,
        "43": 2.7
    },
    "Jane Addams Memorial Tollway":
    {
        "1": 45.8,
        "2": 48.2,
        "26":44.9,
        "32": 44.7,
        "43": 46.7
    },
    "Edens Expressway":
    {
        "1": 14.5,
        "2": 16.9,
        "26": 13.6,
        "32": 13.4,
        "43": 3.8

    },
    "Dan Ryan Expressway": {
        "1": 9.7,
        "2": 9.3,
        "26":12.2,
        "32": 10.1,
        "43": 3.8
    },
    "Eisenhower Expressway": {
        "1": 6.7,
        "2": 8.5,
        "26":5.7,
        "32":10.6,
        "43": 5.4
    },
    "Stevenson Expressway": {
        "1": 16.7,
        "2": 16.3,
        "26": 13.0,
        "32": 17.0,
        "43": 19.1

    },
    "Veterans Memorial Tollway": {
        "1": 2.28,
        "2": 25.7,
        "26": 30.4,
        "32": 32.2,
        "43": 34.3
    },
} # Highways and their coordinates

dframe = pd.read_csv('chicago_data.csv')
dframe["arrest"] = dframe["arrest"].astype(bool)
dframe["domestic"] = dframe["domestic"].astype(bool)
df = dframe.dropna(axis=0, how='any', inplace=False)
df = df.fillna(0)
chosen_wards = [1, 2, 26, 32, 43]
# Convert string tuple of cooridinates for "location"
df['location'] = df['location'].apply(lambda x: ast.literal_eval(x) if
                                      isinstance(x, str) else x)

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
# Calculate the distance in miles of each ward center to the center of Chicago
ward_centers['distance_to_chicago_center_mi'] = ward_centers.apply(
    lambda row: haversine_miles(chicago_center[0], chicago_center[1],
                                row['centroid_lat'], row['centroid_long']),
    axis=1
)

print(ward_centers)

data = []
for highway, wards in hiways.items():
    for ward, distance in wards.items():
        data.append({"ward": ward, "highway": highway, "distance": distance})

highways_df = pd.DataFrame(data)

# Convert ward to integer for consistent plotting
highways_df["ward"] = highways_df["ward"].astype(int)
# Ensure wards are in the specified order
highways_df = highways_df[highways_df["ward"].isin(chosen_wards)]

# Merge Higway dataframe with chicago crime dataframe
merged_df = df.merge(highways_df, on="ward")

# Get separate dataframes for each ward
ward_dfs = {}

for ward in chosen_wards:
    ward_dfs[ward] = merged_df.loc[merged_df["ward"] == ward].copy()
    ward_dfs[ward]["date"] = pd.to_datetime(ward_dfs[ward]["date"])
    ward_dfs[ward].set_index('date', inplace=True)

# Example

chicago_map = gpd.read_file(
"chicago_shapefiles/geo_export_33ca7ae0-c469-46ed-84da-cc7587ccbfe6.shp"
)

# Define a colormap for different crime weights, this could be any mapping you choose
weight_colors = {
    1: 'blue', 
    2: 'green', 
    3: 'yellow', 
    4: 'orange', 
    5: 'red'
}

# Plotting for all wards
for ward in chosen_wards:
    # Check if the ward is in the ward_dfs dictionary
    if ward in ward_dfs:
        # Create a new GeoDataFrame for each ward's crime data
        ward_crime_df = ward_dfs[ward]
        ward_crime_df["lat"] = ward_crime_df["location"].apply(lambda x: x[0])
        ward_crime_df["lon"] = ward_crime_df["location"].apply(lambda x: x[1])
        geometry = [Point(xy) for xy in zip(ward_crime_df["lon"], ward_crime_df["lat"])]
        geo_ward_crime_df = gpd.GeoDataFrame(ward_crime_df, geometry=geometry)

        # Plotting
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        chicago_map.plot(ax=ax, color='lightgrey')

        # Plot crimes for each weight
        for weight, color in weight_colors.items():
            geo_ward_crime_df[geo_ward_crime_df['weights'] == weight].plot(ax=ax, markersize=50, color=color, label=f'Weight {weight}')

        plt.legend()
        plt.title(f'Ward {ward}: Crimes by Weight')
        plt.show()
#
#ward_example = 1  # Example ward number
#if ward_example in ward_dfs:
#    severe_crimes_df = ward_dfs[ward_example][ward_dfs[ward_example]['weights'
#                                                                ].isin([4,
#                                                                        5])].copy()
#    # Creating GeoDataFrame for severe crimes
#    severe_crimes_df.loc[:, "lat"] = severe_crimes_df["location"].apply(lambda x: x[0])
#    severe_crimes_df.loc[:, "lon"] = severe_crimes_df["location"].apply(lambda x: x[1])
#    geometry_severe = [Point(xy) for xy in zip(severe_crimes_df["lon"],
#                                               severe_crimes_df["lat"])]
#    geo_severe_df = gpd.GeoDataFrame(severe_crimes_df, geometry=geometry_severe)
#
#    # Plotting
#    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
#    chicago_map.plot(ax=ax, color='lightgrey')
#    
#    # Plot severe (4) and most severe (5) crimes with different markers/colors
#    for severity, color in [(4, 'orange'), (5, 'red')]:
#        geo_severe_df[geo_severe_df['weights'] == severity].plot(ax=ax, markersize=50, color=color, label=f'Severity {severity}')
#    
#    plt.legend()
#    plt.title(f'Ward {ward_example}: Severe and Most Severe Crimes')
#    plt.show()
#
#
##chicago_map.plot()
#plt.title('Chicago Map')
#plt.show()


