from llamea import LLaMEA

# Define your evaluation function
def your_evaluation_function(solution):
    # Implementation of your function
    # return feedback, quality score, error information
    return "feedback for LLM", 0.1, ""

# Initialize LLaMEA with your API key and other parameters
optimizer = LLaMEA(f=your_evaluation_function, api_key="your_api_key_here")

# Run the optimizer
best_solution, best_fitness = optimizer.run()
print(f"Best Solution: {best_solution}, Fitness: {best_fitness}")