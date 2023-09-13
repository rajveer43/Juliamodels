using Flux
using Random
using LinearAlgebra: norm

# Define a simple 2D dataset (replace this with your own data)
data = rand(100, 2)

# SOM parameters
grid_size = (10, 10)  # SOM grid size
input_size = size(data, 2)
epochs = 100          # Number of training epochs
learning_rate = 0.1  # Learning rate

# Initialize the SOM with random weights
som_weights = randn(Float32, grid_size[1], grid_size[2], input_size)

# Training loop
for epoch in 1:epochs
    for point in data
        # Find the best-matching unit (BMU) or winning neuron
        bmu_index = argmin(norm.(point .- som_weights))
        
        # Update the weights of the BMU and its neighbors
        for i in 1:grid_size[1]
            for j in 1:grid_size[2]
                weight = som_weights[i, j, :]
                distance = norm([i, j] .- bmu_index)
                influence = exp(-distance / (2 * (0.5 * epoch / epochs)^2))  # Decay the influence with time
                weight_delta = learning_rate * influence * (point - weight)
                weight += weight_delta
                som_weights[i, j, :] = weight
            end
        end
    end
end

# Perform data clustering by finding the BMU for each data point
cluster_assignments = [argmin(norm.(point .- som_weights)) for point in data]

# Print the cluster assignments
println("Cluster Assignments:")
for (idx, cluster) in enumerate(cluster_assignments)
    println("Data point $idx assigned to cluster $cluster")
end
