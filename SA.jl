using Survival
using DataFrames

# Create a sample dataset (replace with your own data)
data = DataFrame(
    Time = [5, 10, 12, 18, 20, 25, 30, 35, 40, 45],
    Event = [1, 1, 1, 0, 1, 1, 0, 1, 0, 1]
)

# Create a survival object
survival_data = Survival.Surv(data.Time, data.Event)

# Fit a Kaplan-Meier survival curve
km_curve = Survival.kmf(survival_data)

# Print Kaplan-Meier survival probabilities at specific time points
time_points = [10, 20, 30, 40]
for t in time_points
    prob = Survival survival_probability(km_curve, t)
    println("Survival Probability at Time $t: $prob")
end
