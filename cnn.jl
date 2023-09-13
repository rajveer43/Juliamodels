using Flux
using Flux: Chain, Conv, maxpool, relu, flatten, Dense, softmax

# Define the CNN architecture
function CNN()
    return Chain(
        Conv((5, 5), 1 => 16, relu),
        maxpool((2, 2)),
        
        Conv((5, 5), 16 => 32, relu),
        maxpool((2, 2)),
        
        flatten,
        
        Dense(32 * 5 * 5, 256, relu),
        Dense(256, 10),  # 10 is the number of classes in this example (replace as needed)
        softmax
    )
end

# Create the CNN model
model = CNN()

# Example input image dimensions
input_size = (28, 28, 1)  # Assuming grayscale images with a size of 28x28

# Generate a random input image (replace this with your actual data)
x = randn(Float32, input_size)

# Forward pass
y = model(x)

# Print the output shape
println(size(y))  # Should be (10,), representing class probabilities for 10 classes
