using Flux
using Random

# Generator network
function build_generator(latent_dim, output_dim)
    return Chain(
        Dense(latent_dim, 128, leakyrelu),
        Dense(128, 256, leakyrelu),
        Dense(256, output_dim, sigmoid)
    )
end
