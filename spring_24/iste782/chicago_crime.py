#!/usr/bin/env python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
"""
data-driven approach to classifying the crime weights,
calculating frequency and arrest rates, then normalizing the data 
for equal weighting in clustering process
"""

def clean_data(df):
    """
    Clean data and ensure it's ready for analysis
    """
    pass

# setting the style for the plots
sns.set(style="whitegrid")
def most_occurred_crime_type(df):
    """
    plot the most occurred crime type, adjusted to avoid deprecation warning.
    """
    crime_counts = df['primary_type'].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    # adjusted call to sns.barplot to avoid futurewarning about palette usage
    sns.barplot(x=crime_counts.values, y=crime_counts.index)  # removed 'palette' to adhere to new guidelines
    plt.title('top 10 most occurred crime types')
    plt.xlabel('number of occurrences')
    plt.ylabel('crime type')
    plt.show()

def top_crimes_by_year(df):
    """
    plot the top crimes by year.
    """
    top_crimes = df[df['year'].isin(df['year'].unique())].groupby(['year', 'primary_type']).size().reset_index(name='counts')
    top_crimes = top_crimes.sort_values(['year', 'counts'], ascending=[true, false])
    plt.figure(figsize=(12, 8))
    for year in df['year'].unique():
        sns.lineplot(x='primary_type', y='counts', data=top_crimes[top_crimes['year'] == year], label=str(year))
    plt.xticks(rotation=45, ha="right")
    plt.title('top crimes by year')
    plt.xlabel('crime type')
    plt.ylabel('number of occurrences')
    plt.legend(title='year')
    plt.show()

def crime_rate_over_the_years(df):
    """
    plot the crime rate over the years.
    """
    yearly_counts = df.groupby('year').size()
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=yearly_counts.index, y=yearly_counts.values, marker="o", color="coral")
    plt.title('crime rate over the years')
    plt.xlabel('year')
    plt.ylabel('number of crimes')
    plt.show()

def arrests_ratio_per_crime(df):
    """
    plot the arrests ratio per crime type.
    """
    arrest_ratio = df.groupby('primary_type')['arrest'].mean().sort_values(ascending=false).head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=arrest_ratio.values, y=arrest_ratio.index, palette="magma")
    plt.title('arrests ratio per crime type (top 10)')
    plt.xlabel('arrest ratio')
    plt.ylabel('crime type')
    plt.show()

def arrests_vs_non_arrests_ratio(df):
    """
    plot arrests vs non-arrests ratio.
    """
    arrest_counts = df['arrest'].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(arrest_counts, labels=arrest_counts.index.map({true: 'arrests', false: 'non-arrests'}), autopct='%1.1f%%', startangle=140, colors=["lightblue", "lightgreen"])
    plt.title('arrests vs non-arrests ratio')
    plt.show()

def number_of_arrests_rate(df):
    """
    plot the number of arrests over the years.
    """
    yearly_arrests = df[df['arrest'] == true].groupby('year').size()
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=yearly_arrests.index, y=yearly_arrests.values, marker="o", color="purple")
    plt.title('number of arrests over the years')
    plt.xlabel('year')
    plt.ylabel('number of arrests')

def domestic_violence_crimes(df):
    """
    plot the number of domestic violence crimes over the years.
    """
    domestic_counts = df[df['domestic'] == true].groupby('year').size()
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=domestic_counts.index, y=domestic_counts.values, marker="o", color="red")
    plt.title('domestic violence crimes over the years')
    plt.xlabel('year')
    plt.ylabel('number of domestic crimes')
    plt.show()

def number_of_crimes_by_location(df):
    """
    plot the number of crimes by location description.
    """
    location_counts = df['location_description'].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=location_counts.values, y=location_counts.index, palette="viridis")
    plt.title('top 10 locations for crimes')
    plt.xlabel('number of crimes')
    plt.ylabel('location description')
    plt.show()

# read data and put into dataframe
df = pd.read_csv("chicago_data.csv")
df['date'] = pd.to_datetime(df['date']) # time series

#most_occurred_crime_type(df)
# top_crimes_by_year(df)
# crime_rate_over_the_years(df)
# arrests_ratio_per_crime(df)
# arrests_vs_non_arrests_ratio(df)
# number_of_arrests_rate(df)
# domestic_violence_crimes(df)
# =============================================================
"""
K-means Clustering
"""
crime_stats = df.groupby("primary_type").agg({
    "iucr": "count",
    "arrest": "mean"
}).rename(columns={"iucr" : "frequency", "arrest": "arrest_rate"})

stats_normal = (crime_stats - crime_stats.mean()) / crime_stats.std()

kmeans = KMeans(n_clusters=5, random_state=0).fit(stats_normal)

#labels
crime_stats["cluster"] = kmeans.labels_
# sort
cluster_stats = crime_stats.sort_values("cluster")
# Assign weights to clusters
cluster_weights = {
    0: 8,
    1: 10,
    2: 7,
    3: 5,
    4: 9
}

descriptions_sample = df['description'].sample(n=500, random_state=42).values

# Use TF-IDF to convert the textual data into a matrix of TF-IDF features.
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(descriptions_sample)

# Use K-Means clustering to identify potential categories based on the descriptions.
# The number of clusters is set arbitrarily for this example; in practice, it should be determined based on the data.
num_clusters = 10
km = KMeans(n_clusters=num_clusters, random_state=42)
km.fit(tfidf_matrix)

# Examine the top terms per cluster to understand the kind of categories formed.
# This can give us an idea of how the descriptions are being grouped.
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = tfidf_vectorizer.get_feature_names_out()

# Print out the top terms per cluster
clusters_top_terms = {}
for i in range(num_clusters):
    top_terms = [terms[ind] for ind in order_centroids[i, :10]]  # Top 10 terms for each cluster
    clusters_top_terms[f'Cluster {i}'] = top_terms

cluster_severity_weights = {
    0: 4,  # Unlawful use and possession of firearms: High Severity
    1: 5,  # Aggravated domestic battery and dangerous weapons: Very High Severity
    2: 4,  # Drug offenses, specifically heroin and cocaine: High Severity
    3: 2,  # Vehicle offenses: Low Severity
    4: 3,  # Cannabis-related offenses: Medium Severity
    5: 3,  # Stolen property, indicating theft or burglary: Medium Severity
    6: 4,  # Aggravated battery without firearms: High Severity
    7: 3,  # Violations of weapons regulations and orders of protection: Medium Severity
    8: 4,  # Battery involving physical injury: High Severity
    9: 5,  # Theft within residences, criminal abuse, sexual offenses: Very High Severity
}

# Assign severity weights to each description in the sample based on its cluster assignment
sample_descriptions_severity = [cluster_severity_weights[cluster] for cluster in km.labels_]

# To illustrate, let's create a DataFrame showing the original description and its assigned severity weight
sample_descriptions_df = pd.DataFrame({
    'Description': descriptions_sample,
    'Assigned Severity': sample_descriptions_severity
})

# Map each primary_type to its cluster
primary_type_to_cluster = crime_stats['cluster'].to_dict()
# Assign a weight to each crime in the dataset based on its cluster
df['cluster_weight'] = df['primary_type'].apply(lambda x: cluster_weights[primary_type_to_cluster[x]])
crime_scores_by_ward_cluster = df.groupby('ward')['cluster_weight'].sum().reset_index()
sorted_wards_by_cluster_crime_score = crime_scores_by_ward_cluster.sort_values(by='cluster_weight', ascending=True)
top_wards_cluster = sorted_wards_by_cluster_crime_score.head(5)

breakpoint()


