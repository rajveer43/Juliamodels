using LinearAlgebra
using Statistics

# Sample data matrix (each row represents an observation, and each column represents a feature)
data = [
    1.0  2.0  3.0;
    4.0  5.0  6.0;
    7.0  8.0  9.0;
    10.0 11.0 12.0;
]

# Step 1: Standardize the data (subtract the mean and divide by the standard deviation)
mean_data = mean(data, dims=1)
std_data = std(data, dims=1)
standardized_data = (data .- mean_data) ./ std_data

# Step 2: Compute the covariance matrix
cov_matrix = cov(standardized_data, dims=1)

# Step 3: Perform eigenvalue decomposition of the covariance matrix
eigenvalues, eigenvectors = eigen(cov_matrix)

# Step 4: Sort the eigenvalues and eigenvectors in descending order
sorted_indices = sortperm(eigenvalues, rev=true)
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Step 5: Select the top k eigenvectors to reduce dimensionality
k = 2  # Number of principal components to keep (adjust as needed)
selected_eigenvectors = eigenvectors[:, 1:k]

# Step 6: Project the data onto the selected eigenvectors to obtain the reduced-dimensional representation
reduced_data = standardized_data * selected_eigenvectors

# Display the reduced-dimensional data
println("Reduced-Dimensional Data:")
println(reduced_data)
