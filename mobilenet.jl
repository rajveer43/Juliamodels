using Flux
using Flux: @epochs, onehotbatch, onecold
using Flux.Data: DataLoader
using Base.Iterators: partition

# Depthwise separable convolution block
struct DepthwiseSeparableConv
    depthwise_conv::Conv
    pointwise_conv::Conv
end

function DepthwiseSeparableConv(in_channels, out_channels, stride=1)
    return DepthwiseSeparableConv(
        Conv((3, 3), in_channels=>in_channels, stride=stride, pad=(1, 1), group=in_channels),
        Conv((1, 1), in_channels=>out_channels)
    )
end

# Define the MobileNetV2 model
struct MobileNetV2
    conv1::Conv
    blocks::Vector{DepthwiseSeparableConv}
    fc::Dense
end

function MobileNetV2(input_size, num_classes, block_configs)
    # Initial convolution layer
    conv1 = Conv((3, 3), input_size=>32, stride=(2, 2), pad=(1, 1), relu)
    
    # Create the depthwise separable convolution blocks
    blocks = DepthwiseSeparableConv.(32, block_configs)
    
    # Fully connected layer
    fc = Dense(1280, num_classes)
    
    return MobileNetV2(conv1, blocks, fc)
end

# Define the forward pass
function (model::MobileNetV2)(x)
    x = model.conv1(x)
    for block in model.blocks
        x = block.depthwise_conv(x)
        x = block.pointwise_conv(x)
    end
    x = maxpool(x, (2, 2))
    x = reshape(x, :, size(x, 4))
    x = model.fc(x)
    return x
end

# Define training and evaluation functions
function train(model, dataloader, loss_fn, optimizer)
    for (x, y) in dataloader
        gs = gradient(params(model)) do
            ŷ = model(x)
            loss = loss_fn(ŷ, y)
            return loss
        end
        Flux.update!(optimizer, gs)
    end
end

function evaluate(model, dataloader)
    acc = 0.0
    for (x, y) in dataloader
        ŷ = model(x)
        acc += sum(onecold(ŷ) .== onecold(y))
    end
    return acc / length(dataloader)
end

# Example usage
input_size = (224, 224, 3)
num_classes = 1000
block_configs = [(32, 1), (64, 2), (128, 1), (128, 2), (256, 1), (256, 2), (512, 1)]  # Depthwise separable block configurations

model = MobileNetV2(input_size, num_classes, block_configs)
