using Flux
using Flux: @functor

# Define a hypergraph (a list of hyperedges)
hypergraph = [(1, [1, 2]), (2, [2, 3, 4])]

# Define node embeddings
node_embeddings = Dict(1 => rand(2), 2 => rand(2), 3 => rand(2), 4 => rand(2))

# Define hyperedge embeddings
hyperedge_embeddings = Dict(1 => rand(2), 2 => rand(2))

# Define a HyperGNN model
@functor HyperGNN(embed_dim) = Chain(
    Dense(embed_dim, embed_dim, relu),
    Dense(embed_dim, embed_dim)
)

# Message passing function
function message_passing(hyperedge, node_embeddings, hyperedge_embeddings, model)
    hyperedge_id, nodes = hyperedge
    node_messages = [node_embeddings[node] for node in nodes]
    hyperedge_message = hyperedge_embeddings[hyperedge_id]
    
    combined_message = sum(node_messages, dims=1) + hyperedge_message
    
    return model(combined_message)
end

# Update node embeddings
for hyperedge in hypergraph
    node_id, _ = hyperedge
    new_embedding = message_passing(hyperedge, node_embeddings, hyperedge_embeddings, HyperGNN(2))
    node_embeddings[node_id] = new_embedding
end

# Updated node embeddings after message passing
println("Updated Node Embeddings: ", node_embeddings)
