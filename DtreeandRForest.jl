
#classification DTree
using DecisionTree
using Random

# Generate synthetic data for classification (replace with your own data)
Random.seed!(123)
n_samples = 100
n_features = 2
X = randn(n_samples, n_features)
y = [rand(1:2) for _ in 1:n_samples]  # Two classes

# Train a decision tree classifier
model = DecisionTreeClassifier(max_depth=3)
fit!(model, X, y)

# Make predictions
predictions = predict(model, X)

# Calculate accuracy
accuracy = sum(y .== predictions) / n_samples
println("Decision Tree Classification Accuracy: $accuracy")

#DTree for Regreation!
using DecisionTree
using Random

# Generate synthetic data for regression (replace with your own data)
Random.seed!(123)
n_samples = 100
n_features = 1
X = randn(n_samples, n_features)
y = 2 * X + randn(n_samples)  # Linear relationship

# Train a decision tree regressor
model = DecisionTreeRegressor(max_depth=3)
fit!(model, X, y)

# Make predictions
predictions = predict(model, X)

# Calculate Mean Squared Error
mse = sum((y .- predictions).^2) / n_samples
println("Decision Tree Regression MSE: $mse")

#random forest fo classification
using RandomForest
using Random

# Generate synthetic data for classification (replace with your own data)
Random.seed!(123)
n_samples = 100
n_features = 2
X = randn(n_samples, n_features)
y = [rand(1:2) for _ in 1:n_samples]  # Two classes

# Train a random forest classifier
model = RandomForestClassifier(n_trees=100, max_depth=3)
fit!(model, X, y)

# Make predictions
predictions = predict(model, X)

# Calculate accuracy
accuracy = sum(y .== predictions) / n_samples
println("Random Forest Classification Accuracy: $accuracy")

#randomforst for regression
using RandomForest
using Random

# Generate synthetic data for regression (replace with your own data)
Random.seed!(123)
n_samples = 100
n_features = 1
X = randn(n_samples, n_features)
y = 2 * X + randn(n_samples)  # Linear relationship

# Train a random forest regressor
model = RandomForestRegressor(n_trees=100, max_depth=3)
fit!(model, X, y)

# Make predictions
predictions = predict(model, X)

# Calculate Mean Squared Error
mse = sum((y .- predictions).^2) / n_samples
println("Random Forest Regression MSE: $mse")
