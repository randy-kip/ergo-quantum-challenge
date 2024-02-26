def compute_square_root(a, initial_guess, threshold=1e-6, max_iterations=100):
    """
    Compute the square root of a number using an iterative method.

    Parameters:
    - a (float): The number for which the square root is to be computed.
    - initial_guess (float): The initial guess for the square root.
    - threshold (float, optional): The threshold for the difference between successive iterations.
    - max_iterations (int, optional): The maximum number of iterations allowed.

    Returns:
    - float: The computed square root.
    """

    x = initial_guess  # Initialize the initial guess
    iteration = 0  # Initialize the iteration counter

    # Iterate until the stopping criteria are met
    while True:
        # Compute the next guess using the provided equation
        xn_plus_1 = x - (x ** 2 - a) / (2 * x)
        
        # Stopping criteria: check the difference between successive iterations
        if abs(xn_plus_1 - x) < threshold or iteration >= max_iterations:
            break
        
        # Update the current guess for the next iteration
        x = xn_plus_1
        iteration += 1
    
    return x


# Example usage
a = 2
initial_guess = 1.0  # You can adjust the initial guess
square_root = compute_square_root(a, initial_guess)
print("Square root of", a, "is:", square_root)
