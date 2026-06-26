# Customer Segmentation Using K-Means Clustering

## Project Overview

This project applies **unsupervised machine learning** techniques to segment customers based on their demographic characteristics, purchasing behavior, and responses to marketing campaigns. The objective is to identify distinct customer groups that enable businesses to develop targeted marketing strategies, improve customer engagement, and support data-driven decision-making.

## What Makes This Project Different

Unlike a basic K-Means clustering implementation, this project incorporates **Principal Component Analysis (PCA)** to reduce the dataset's dimensionality while preserving **95% of the original variance**. The optimal number of principal components was determined using the **cumulative explained variance (cumsum)**, resulting in a more efficient clustering process with minimal information loss.

To ensure the clustering results were both accurate and meaningful, multiple evaluation techniques were used. The optimal number of clusters was determined using:

* **Elbow Method** – identifies the point where adding more clusters provides diminishing improvements.
* **Silhouette Score** – measures how well-separated and cohesive the clusters are.
* **Silhouette Diagrams** – provide a visual assessment of cluster quality and overlap.

This comprehensive evaluation helps validate the effectiveness of the clustering model rather than relying on a single metric.

## Data Visualization

Several visualizations were created to explore customer behavior and interpret the clustering results, including:

* Correlation Heatmap to analyze relationships between variables.
* Elbow Curve for selecting the optimal number of clusters.
* Silhouette Score Plot to compare clustering performance across different values of *k*.
* Silhouette Diagrams to visualize cluster cohesion and separation.
* Promotion Acceptance Count Plot to examine how customers in different clusters respond to marketing campaigns.
* Box Plot of Deals Purchased by Cluster to compare purchasing behavior.
* Customer Spending vs. Personal Characteristics to explore how demographics such as family size, age, education, and parental status relate to spending patterns across clusters.

These visualizations provide actionable insights into customer segments and support the interpretation of the clustering results.
