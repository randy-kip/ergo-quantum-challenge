import math

def newton_raphson(target_function, gradient_function, initial_guess, threshold=1e-6, max_iterations=100):
    """
    Newton-Raphson method for finding roots of the target function.

    Parameters:
    - target_function (function): The target function f(x) = 0.
    - gradient_function (function): The gradient function df(x)/dx.
    - initial_guess (float): The initial guess x0.
    - threshold (float): The convergence threshold.
    - max_iterations (int): Maximum number of iterations.

    Returns:
    - float: The root of the target function.
    """
    x = initial_guess
    iteration = 0

    # Iterate until convergence or maximum iterations reached
    while iteration < max_iterations:
        # Calculate the function value and its derivative at the current guess
        fx = target_function(x)
        dfx = gradient_function(x)

        # Update the guess using the Newton-Raphson formula
        x = x - fx / dfx
        
        # Check for convergence
        if abs(fx) < threshold:
            break
        
        iteration += 1

    return x

# Example usage:

# Define the target function f(x) = sin(x) - 0.5x
def target_function(x):
    return math.sin(x) - 0.5 * x

# Define the gradient function df(x)/dx = cos(x) - 0.5
def gradient_function(x):
    return math.cos(x) - 0.5

# Initial guess
initial_guess = 5.0

# Convergence threshold
threshold = 1e-6

# Maximum iterations
max_iterations = 100

# Find the root using Newton-Raphson method
root = newton_raphson(target_function, gradient_function, initial_guess, threshold, max_iterations)

print("Root found:", root)
print("Initial guess: {:.6f}, Root: {:.6f}".format(initial_guess, root))
