import math

def f(x):
  """
  Defines the function f(x) = sin(x) + 3/4 * x - 1.
  """
  return math.sin(x) + 3/4 * x - 1

def derivative_f(x):
  """
  Calculates the derivative of f(x) = cos(x) + 3/4.
  """
  return math.cos(x) + 3/4

def newton_raphson(f, derivative_f, x_0, max_iterations=100, domain_min=0.0, domain_max=6.0):
  """
  Implements the Newton-Raphson method to find the root of f(x) within the given domain.

  Args:
      f: The function to find the root of.
      derivative_f: The derivative of the function.
      x_0: The initial guess for the root.
      max_iterations: The maximum number of iterations allowed (default is 100).
      domain_min: The minimum value of the domain (default is 0.0).
      domain_max: The maximum value of the domain (default is 6.0).

  Returns:
      The root (or None if not found within the domain and iterations).
  """

  x = x_0
  for i in range(max_iterations):
    # Check if within domain
    if x < domain_min or x > domain_max:
      return None
    
    # Calculate new guess
    x_new = x - f(x) / derivative_f(x)

    # Check for convergence through a threshold
    if abs(x_new - x) < 1e-6:
      return x_new

    # Update guess
    x = x_new

  # Maximum iterations reached without convergence
  return None

# Function for graphing
def plot_function(f, domain_min, domain_max):
  import matplotlib.pyplot as plt
  import numpy as np

  x = np.linspace(domain_min, domain_max, 1000)
  # Apply f(x) element-wise using vectorized operations
  y = np.vectorize(f)(x)  # or y = f(x)  (if f already handles vectorized input)

  plt.plot(x, y, label="sin(x) + 3/4 * x - 1")
  plt.axhline(y=0, color='r', linestyle='--', label="y = 0")
  plt.xlabel("x")
  plt.ylabel("f(x)")
  plt.title("f(x) = sin(x) + 3/4 * x - 1")
  plt.legend()
  plt.grid(True)
  plt.show()

# Initial values
initial_values = [1.0, 2.0, 4.0, 1.5]

# Find solutions and print results
for x_0 in initial_values:
  root = newton_raphson(f, derivative_f, x_0)
  if root is not None:
    print(f"Initial guess: {x_0:.6f}, Root: {root:.6f}")
  else:
    print(f"No solution found within domain for initial guess: {x_0:.6f}")

# Plot the function
plot_function(f, domain_min=0.0, domain_max=6.0)
