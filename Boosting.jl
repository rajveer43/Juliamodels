using XGBoost
using Random

# Generate synthetic regression data
Random.seed!(123)
n_samples = 100
n_features = 5

X = randn(n_samples, n_features)
y = sum(X, dims=2) + 0.5 * randn(n_samples)

# Create a DMatrix for the data
data = DMatrix(X, label=y)

# Define hyperparameters for the XGBoost model
params = Dict(
    "objective" => "reg:squarederror",  # Regression objective
    "eta" => 0.1,                       # Learning rate
    "max_depth" => 3,                   # Maximum tree depth
    "nrounds" => 100                    # Number of boosting rounds (iterations)
)

# Train an XGBoost regression model
model = xgboost(data, params=params, num_round=params["nrounds"])

# Make predictions with the trained model
predictions = predict(model, XGBoost.DMatrix(X))

# Evaluate the model (e.g., calculate Mean Squared Error)
mse = sum((y .- predictions).^2) / n_samples
println("Mean Squared Error (MSE): $mse")

