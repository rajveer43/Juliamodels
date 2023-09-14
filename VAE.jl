using Flux
using Distributions

# Define the VAE architecture
function VAE(latent_dim::Int, hidden_dim::Int)
    encoder = Chain(
        Dense(28 * 28, hidden_dim, relu),
        Dense(hidden_dim, 2 * latent_dim)
    )
    
    decoder = Chain(
        Dense(latent_dim, hidden_dim, relu),
        Dense(hidden_dim, 28 * 28, sigmoid)
    )
    
    return encoder, decoder
end

# Define the VAE loss function
function vae_loss(encoder, decoder, x)
    μ, logσ = split(encoder(x), 2, dims=2)
    σ = exp.(logσ)
    ε = rand(Normal(), size(μ))
    z = μ .+ ε .* σ
    x̂ = decoder(z)
    
    # Reconstruction loss (binary cross-entropy)
    reconstruction_loss = Flux.mse(x̂, x)
    
    # KL divergence between the latent distribution and the prior (Gaussian)
    kl_divergence = -0.5 * sum(1 + logσ .- μ.^2 .- σ.^2)
    
    return reconstruction_loss + kl_divergence
end

# Create a VAE model
latent_dim = 2
hidden_dim = 256
encoder, decoder = VAE(latent_dim, hidden_dim)
vae_model = Chain(encoder, decoder)

# Define an optimizer
optimizer = ADAM(0.001)

# Load your dataset (replace this with your data loading code)
# For simplicity, we use the MNIST dataset for illustration
using Flux.Data.MNIST
images, _ = MNIST.images()

# Preprocess the data (flatten and normalize)
data = [reshape(float(image), :) for image in images]

# Training loop
epochs = 10
batch_size = 64

for epoch in 1:epochs
    for batch in 1:batch_size:length(data)
        batch_data = data[batch:min(batch+batch_size-1, end)]
        loss_value = vae_loss(encoder, decoder, batch_data)
        Flux.back!(loss_value)
        Flux.update!(optimizer, encoder, batch_data)
        Flux.update!(optimizer, decoder, batch_data)
    end
    println("Epoch $epoch, Loss: $loss_value")
end
