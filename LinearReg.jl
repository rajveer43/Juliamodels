using GLM
using DataFrames
using Random

# Generate synthetic data for linear regression (replace with your own data)
Random.seed!(123)
n_samples = 100
X = rand(1:10, n_samples)  # Independent variable
β₀ = 2.0                   # Intercept
β₁ = 3.0                   # Slope
ϵ = randn(n_samples) * 2   # Gaussian noise
y = β₀ .+ β₁ .* X + ϵ      # Dependent variable

# Create a DataFrame from the data
data = DataFrame(X=X, y=y)

# Fit a simple linear regression model
model = lm(@formula(y ~ X), data)

# Print the model summary
println(model)

# Extract model coefficients
intercept = coef(model)[1]
slope = coef(model)[2]

# Print the regression equation
println("Regression Equation: y = $intercept + $slope * X")
