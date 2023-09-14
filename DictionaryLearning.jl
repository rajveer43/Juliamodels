using DictionaryLearning
using Random

# Generate synthetic data
Random.seed!(123)
n_samples = 100
n_features = 20
n_components = 10
X = randn(n_samples, n_features)

# Create a dictionary learning model
model = DictionaryLearning(n_components)

# Fit the model to the data
fit!(model, X)

# Transform the data using the learned dictionary
transformed_data = transform(model, X)

# Print the learned dictionary and transformed data
println("Learned Dictionary:")
println(model.components)

println("\nTransformed Data:")
println(transformed_data)
