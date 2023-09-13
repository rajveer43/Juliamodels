using DecisionTree
using Random

# Generate synthetic data
Random.seed!(123)
X = rand(100, 4)
y = rand([0, 1], 100)

# Create and train a decision tree classifier
model = DecisionTreeClassifier(max_depth=3)
fit!(model, X, y)

# Make predictions
predictions = predict(model, X)

# Evaluate the model
accuracy = sum(predictions .== y) / length(y)
println("Decision Tree Classifier Accuracy: $accuracy")


using RandomForest

# Create and train a random forest classifier
model = RandomForestClassifier(n_trees=100, max_depth=3)
fit!(model, X, y)

# Make predictions
predictions = predict(model, X)

# Evaluate the model
accuracy = sum(predictions .== y) / length(y)
println("Random Forest Classifier Accuracy: $accuracy")


using LIBSVM

# Create and train an SVM classifier
model = svmtrain(X, y, kernel=LIBSVM.Kernel.Linear)
predictions, decision_values = svmpredict(model, X)

# Evaluate the model
accuracy = sum(predictions .== y) / length(y)
println("SVM Classifier Accuracy: $accuracy")


using NearestNeighbors

# Create and train a KNN classifier
model = KNNClassifier(n_neighbors=3)
fit!(model, X, y)

# Make predictions
predictions = predict(model, X)

# Evaluate the model
accuracy = sum(predictions .== y) / length(y)
println("K-Nearest Neighbors (KNN) Classifier Accuracy: $accuracy")
