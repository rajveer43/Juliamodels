using Flux
using Flux: Chain, Dense, Conv, maxpool, leakyrelu, reshape

# Define the YOLO-like object detection model
function YOLO()
    return Chain(
        Conv((3, 3), 3 => 16, leakyrelu, pad=(1, 1)),
        maxpool((2, 2)),
        
        Conv((3, 3), 16 => 32, leakyrelu, pad=(1, 1)),
        maxpool((2, 2)),
        
        Conv((3, 3), 32 => 64, leakyrelu, pad=(1, 1)),
        maxpool((2, 2)),
        
        Conv((3, 3), 64 => 128, leakyrelu, pad=(1, 1)),
        maxpool((2, 2)),
        
        Conv((3, 3), 128 => 256, leakyrelu, pad=(1, 1)),
        maxpool((2, 2)),
        
        flatten,
        Dense(256 * 7 * 7, 4096, leakyrelu),
        Dense(4096, 1470)  # 7x7x30 for a simplified YOLO-like model with two classes and bounding box predictions
    )
end

# Create the YOLO-like model
model = YOLO()

# Example input image dimensions
input_size = (224, 224, 3)

# Generate a random input image (replace this with your actual data)
x = randn(Float32, input_size)

# Forward pass
y = model(x)

# Print the output shape
println(size(y))  # Should be (1470,), representing bounding box predictions and class scores
