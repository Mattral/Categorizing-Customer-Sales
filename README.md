# Categorizing Customer Sales in SuperMarket

This project focuses on analyzing and categorizing customer sales data in a supermarket. It involves exploring the dataset, extracting useful features, visualizing relationships, and building a machine learning model.

## Dataset

The dataset used for this analysis is named "supermarket_sales - Sheet1.csv." It contains information about sales transactions, including features like 'Date,' 'Time,' 'Branch,' 'Product line,' 'Quantity,' 'Tax 5%,' 'Total,' 'Gross income,' 'Rating,' and more.

## Exploratory Data Analysis (EDA)

- The initial steps involve loading the dataset, inspecting data types, and converting 'Date' and 'Time' columns to datetime format.
- Derived attributes such as day, month, year, and hour are extracted from the 'Date' and 'Time' columns.
- Correlation analysis is performed using a heatmap, highlighting strong correlations between 'Tax 5%,' 'Total,' 'Gross income,' and 'cogs' (Cost of Goods Sold).
- Visualizations, such as regression plots, are used to explore relationships between variables like 'Tax 5%' vs. 'Gross Income' and 'Quantity' vs. 'Cost of Goods Sold.'
- The distribution of the 'Rating' variable is visualized using a histogram with a vertical line indicating the mean rating.

## Categorical Analysis

- Unique values in categorical columns are explored to understand the diversity of data.
- Functions for count plots, box plots, line plots, and relational plots are defined to analyze categorical variables.

## Product Analysis

- The distribution of product quantity within each product line is visualized using box plots.
- Count plots provide insights into the distribution of product lines.

## Customer Analysis

- Customer type distribution in each branch is analyzed using count plots.
- The influence of customer type on total sales is examined.
- The impact of customer type on customer ratings is visualized using swarm plots.

## Word Cloud

- A word cloud is generated to visualize the frequency of product lines in the dataset.
  
## Building a Machine Learning Model

In this section, a machine learning model is developed using K-Means clustering to categorize customers based on their spending patterns.

```python
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Fetching the 'cogs' (Cost of Goods) column
X = sales.iloc[:, -8].values.reshape(-1,1)

# Using the Elbow Method to find the optimal number of clusters (k)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Method graph
plt.figure(figsize=(20,7))
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Training the K-Means model on the dataset with k=5
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Visualizing the clusters
plt.figure(figsize=(20,8))
plt.scatter(X[y_kmeans == 0], X[y_kmeans == 0], s=100, c='red', label='Lower Class')
plt.scatter(X[y_kmeans == 1], X[y_kmeans == 1], s=100, c='blue', label='Upper Lower Class')
plt.scatter(X[y_kmeans == 2], X[y_kmeans == 2], s=100, c='green', label='Medium Class')
plt.scatter(X[y_kmeans == 3], X[y_kmeans == 3], s=100, c='cyan', label='Upper Medium Class')
plt.scatter(X[y_kmeans == 4], X[y_kmeans == 4], s=100, c='magenta', label='Upper Class5')
plt.title('Clusters of customers')
plt.xlabel('Total expenditure')
plt.ylabel('COGs earned')
plt.legend()
plt.show()
```
