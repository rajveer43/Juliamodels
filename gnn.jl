using Pkg
Pkg.add("LightGraphs")
Pkg.add("Flux")


using LightGraphs
using Flux
using Flux: Chain, Dense, relu

# Define a simple graph
num_nodes = 5
graph = Graph(num_nodes)

# Add edges to the graph (replace this with your own graph structure)
add_edge!(graph, 1, 2)
add_edge!(graph, 1, 3)
add_edge!(graph, 2, 4)
add_edge!(graph, 3, 5)

# Create random features for each node (replace this with your node features)
node_features = randn(Float32, num_nodes, 64)

# Define a basic GNN layer
mutable struct GNNLayer
    weight::Dense
end

function GNNLayer(input_size, output_size)
    return GNNLayer(Dense(input_size, output_size, relu))
end

# Define the GNN model
mutable struct GNNModel
    layers::Vector{GNNLayer}
end

function GNNModel(input_size, hidden_sizes, output_size)
    layers = GNNLayer.([input_size; hidden_sizes; output_size][1:end-1], [hidden_sizes; output_size])
    return GNNModel(layers)
end

# Forward pass for the GNN
function (model::GNNModel)(graph, node_features)
    for layer in model.layers
        node_features = layer.weight(node_features)
        node_features = graph * node_features  # Graph convolution
        node_features = relu(node_features)    # Activation function (ReLU)
    end
    return node_features
end

# Create the GNN model
input_size = size(node_features, 2)
hidden_sizes = [32, 16]
output_size = 2  # Number of classes for node classification

gnn_model = GNNModel(input_size, hidden_sizes, output_size)

# Example forward pass
output = gnn_model(graph, node_features)

# Print the output shape
println(size(output))  # Should be (num_nodes, output_size)
