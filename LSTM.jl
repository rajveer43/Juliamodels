using Flux
using Flux: Chain, LSTM, Dense

# Define the LSTM-based model
function LSTMModel(input_size::Int, hidden_size::Int, output_size::Int)
    return Chain(
        LSTM(input_size, hidden_size),
        Dense(hidden_size, output_size)
    )
end

# Define the input size, hidden size, and output size
input_size = 1
hidden_size = 64
output_size = 1

# Create the LSTM model
model = LSTMModel(input_size, hidden_size, output_size)

# Define a sample input sequence (a sequence of numbers)
input_sequence = [0.1, 0.2, 0.3, 0.4, 0.5]

# Convert the input sequence to a Flux.jl-friendly format
input_data = [(x,) for x in input_sequence]

# Initialize the model
Flux.reset!(model)

# Forward pass through the LSTM model
output = model.(input_data)

# Print the output
println("Input Sequence: ", input_sequence)
println("Predicted Output: ", Flux.data(output))
