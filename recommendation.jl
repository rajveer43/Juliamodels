using Flux

# Define a synthetic user-item interaction matrix (replace with your data)
num_users = 100
num_items = 50
interaction_matrix = rand(0:1, num_users, num_items)

# Define the recommender model
function RecommenderModel(num_users, num_items, embedding_dim)
    user_embeddings = param(randn(Float32, num_users, embedding_dim))
    item_embeddings = param(randn(Float32, num_items, embedding_dim))
    
    return user_embeddings, item_embeddings
end

# Define the loss function (mean squared error)
function loss(user_embeddings, item_embeddings, interaction_matrix)
    predicted_matrix = user_embeddings * transpose(item_embeddings)
    mse_loss = Flux.mse(predicted_matrix, interaction_matrix)
    return mse_loss
end

# Training parameters
embedding_dim = 10
learning_rate = 0.01
num_epochs = 100

# Create the model and optimizer
user_embeddings, item_embeddings = RecommenderModel(num_users, num_items, embedding_dim)
optimizer = ADAM(learning_rate)

# Training loop
for epoch in 1:num_epochs
    Flux.train!(loss, [(user_embeddings, item_embeddings, interaction_matrix)], optimizer)
    
    # Compute and print the loss for monitoring
    training_loss = loss(user_embeddings, item_embeddings, interaction_matrix)
    println("Epoch $epoch, Loss: $training_loss")
end

# Perform recommendations for a user (replace with a specific user ID)
user_id = 1
user_embedding = user_embeddings[user_id, :]

# Calculate similarity scores between the user and items
similarity_scores = sum(user_embedding .* transpose(item_embeddings), dims=1)

# Sort items by similarity scores to recommend the top items
sorted_items = sortperm(vec(similarity_scores), rev=true)

# Print the top recommended items
println("Top Recommended Items for User $user_id:")
for i in sorted_items[1:10]
    println("Item $i")
end
