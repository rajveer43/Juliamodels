using Turing
using MCMCChains
using Random

# Generate synthetic data for a linear regression problem
Random.seed!(123)
n_samples = 100
X = randn(n_samples)
Y = 2.0 .* X .+ 1.0 .+ 0.5 .* randn(n_samples)

# Define a Bayesian linear regression model using Turing.jl
@model function linear_regression(X, Y)
    # Priors
    α ~ Normal(0, 10)
    β ~ Normal(0, 10)
    σ ~ Exponential(1)
    
    # Likelihood
    Y ~ MvNormal(X * β + α, σ)
end

# Compile the model
model = linear_regression(X, Y)

# Run a Bayesian inference algorithm (e.g., Hamiltonian Monte Carlo)
chain = sample(model, NUTS(0.65), 1000)

# Summarize the results
describe(chain)
