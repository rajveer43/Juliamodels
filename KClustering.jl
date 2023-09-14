using Clustering
using Random

# Generate synthetic data (replace with your own data)
Random.seed!(123)
n_samples = 200
n_features = 2
k = 3  # Number of clusters

data = [randn(n_samples) for _ in 1:n_features]

# Perform k-means clustering
result = kmeans(transpose(hcat(data...)), k)

# Get cluster assignments and centroids
cluster_assignments = assignments(result)
centroids = result.centers'

# Print cluster assignments and centroids
println("Cluster Assignments: ", cluster_assignments)
println("Cluster Centroids: ")
println(centroids)
