# Define the transition matrix for a simple 2-state Markov Chain
transition_matrix = [0.8 0.2;
                     0.3 0.7]

# Define the initial state probabilities
initial_state = [0.6, 0.4]

# Define the number of time steps
n_steps = 5

# Initialize the current state
current_state = rand(1:2, 1)

# Create a function to perform Markov Chain simulation
function simulate_markov_chain(transition_matrix, initial_state, n_steps)
    state_sequence = Int[current_state]
    
    for step in 1:n_steps
        current_state = rand(1:2, 1, weights(transition_matrix[current_state, :]))
        push!(state_sequence, current_state)
    end
    
    return state_sequence
end

# Simulate the Markov Chain
state_sequence = simulate_markov_chain(transition_matrix, initial_state, n_steps)

# Print the state sequence
println("Markov Chain State Sequence: ", state_sequence)
