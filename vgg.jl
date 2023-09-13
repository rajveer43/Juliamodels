using Flux
using Flux: Chain, Dense, Conv, maxpool, softmax

# Define the VGG architecture
function VGG()
    return Chain(
        Conv((3, 3), 3 => 64, relu, pad=(1, 1)),
        Conv((3, 3), 64 => 64, relu, pad=(1, 1)),
        maxpool((2, 2)),
        
        Conv((3, 3), 64 => 128, relu, pad=(1, 1)),
        Conv((3, 3), 128 => 128, relu, pad=(1, 1)),
        maxpool((2, 2)),
        
        Conv((3, 3), 128 => 256, relu, pad=(1, 1)),
        Conv((3, 3), 256 => 256, relu, pad=(1, 1)),
        Conv((3, 3), 256 => 256, relu, pad=(1, 1)),
        maxpool((2, 2)),
        
        Conv((3, 3), 256 => 512, relu, pad=(1, 1)),
        Conv((3, 3), 512 => 512, relu, pad=(1, 1)),
        Conv((3, 3), 512 => 512, relu, pad=(1, 1)),
        maxpool((2, 2)),
        
        Conv((3, 3), 512 => 512, relu, pad=(1, 1)),
        Conv((3, 3), 512 => 512, relu, pad=(1, 1)),
        Conv((3, 3), 512 => 512, relu, pad=(1, 1)),
        maxpool((2, 2)),
        
        flatten,
        Dense(512, 4096, relu),
        Dense(4096, 4096, relu),
        Dense(4096, 1000),  # 1000 is the number of classes for ImageNet, replace it for your task
        softmax
    )
end

# Create the VGG model
model = VGG()

# Example input image dimensions
input_size = (224, 224, 3)

# Generate a random input image (replace this with your actual data)
x = randn(Float32, input_size)

# Forward pass
y = model(x)

# Print the output shape
println(size(y))  # Should be (1000,), representing the class probabilities
