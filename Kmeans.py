#Kmeans clustering is a method of unsupervisied learning where by the algrothrim learns from the data set's patterns rather than labels.

#importing some functions.
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#load the data set
df = pd.read_csv(r'C:\Users\Harry\Desktop\Kmeans\creditcard.csv')

#remove missing values, machine learning algrothrims cant handle
df.dropna(inplace=True)
print(df.describe())

#data standardisation and transformation, want each feature to be relative to each 
#to do this we just need a standard scaler, equasion is (featre - mean) / standard deviation

scaler = StandardScaler()

# Standardize only the feature columns (excluding 'Class' if it's the target)
scaled_features = scaler.fit_transform(df.iloc[:, :-1])

# Convert to DataFrame with modified column names (e.g., "V1_scaled", "V2_scaled", ...)
scaled_df = pd.DataFrame(scaled_features, columns=[col + "_scaled" for col in df.columns[:-1]])

# Concatenate original and scaled values
df_combined = pd.concat([df, scaled_df], axis=1)

# Print the first few rows to check
print(df_combined.head())

#this project will have 4 steps
#1. define the number (k) of clusters to split the data into.

def optimise_k_means(data, max_k):
    means = []
    inertias = []

    for k in range(1, max_k+1):  # Iterate until max_k (inclusive)
        kmeans = KMeans(n_clusters=k, random_state=42)  # Add random_state for reproducibility
        kmeans.fit(data)  # Fit KMeans model

        means.append(k)
        inertias.append(kmeans.inertia_)  # inertia_ stores the sum of squared distances to centroids

    # Generate the elbow plot
    plt.figure(figsize=(10, 5))  # Correct way to set the figure size
    plt.plot(means, inertias, 'o-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')

    plt.grid(True)
    plt.show()

# Corrected function call with the proper column names
optimise_k_means(df_combined[['V1_scaled', 'V2_scaled']], 10)
#we can see that 4 clusters is the best choice.

#2. Applying Kmeans
Kmeans = KMeans(n_clusters = 4)
Kmeans.fit(df_combined[['V1_scaled', 'V2_scaled']])
KMeans(n_clusters=4)

# Add the cluster labels to the original dataframe
df['kmeans_4'] = Kmeans.labels_ 

#print(df)

#3. Plotting the results
plt.scatter(x=df_combined['V1_scaled'], y=df_combined['V2_scaled'], c=df['kmeans_4'], cmap='viridis')
plt.xlabel('V1_scaled')
plt.ylabel('V2_scaled')
plt.title('KMeans Clustering (k=4)')
plt.colorbar(label='Cluster Label')  # Show color bar with cluster labels
plt.show()
