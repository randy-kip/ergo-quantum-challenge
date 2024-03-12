#include <stdio.h>
#include <math.h>

// Define the target function f(x) = sin(x) - 0.5x
double target_function(double x) {
    return sin(x) - 0.5 * x;
}

// Define the gradient function df(x)/dx = cos(x) - 0.5
double gradient_function(double x) {
    return cos(x) - 0.5;
}

// Newton-Raphson method for finding roots of the target function
double newton_raphson(double (*target_function)(double), double (*gradient_function)(double), double initial_guess, double threshold, int max_iterations) {
    double x = initial_guess;
    int iteration = 0;

    // Iterate until convergence or maximum iterations reached
    while (iteration < max_iterations) {
        // Calculate the function value and its derivative at the current guess
        double fx = target_function(x);
        double dfx = gradient_function(x);

        // Update the guess using the Newton-Raphson formula
        x = x - fx / dfx;
        
        // Check for convergence
        if (fabs(fx) < threshold) {
            break;
        }
        
        iteration++;
    }

    return x;
}

int main() {
    // Initial guess
    double initial_guess = 5.0;
    
    // Convergence threshold
    double threshold = 1e-6;
    
    // Maximum iterations
    int max_iterations = 100;
    
    // Find the root using Newton-Raphson method
    double root = newton_raphson(target_function, gradient_function, initial_guess, threshold, max_iterations);
    
    printf("Root found: %.6f\n", root);
    printf("Initial guess: %.6f, Root: %.6f\n", initial_guess, root);

    return 0;
}
