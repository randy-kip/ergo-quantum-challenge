import numpy as np

def mps_encode(x):
    """Encodes a vector x as a matrix product state (MPS)."""

    # Pad with zeros to nearest 2^N length
    N = int(np.ceil(np.log2(len(x))))
    x_padded = np.pad(x, (0, 2**N - len(x)), mode='constant')

    # Normalize
    x_norm = x_padded / np.linalg.norm(x_padded)

    # Reshape to 2^(N-1) * 2 matrix
    matrix = x_norm.reshape((2**(N-1), 2))

    # Initialize empty list for MPS tensors
    psi = []

    for i in range(N):
        # Perform SVD
        U, s, V = np.linalg.svd(matrix)

        # Adjusted reshaping with minimum allowed length
        min_s_len = 2
        V_reshaped = V.reshape((min_s_len, 2, len(V) // min_s_len)).swapaxes(0, 1)
        psi.append(V_reshaped)

        # Construct and reshape A
        A = U @ np.diag(s)

        # Reshape A based on the sizes of U, s, and V
        A_shape_before = A.shape
        A = A.reshape((len(U), len(s)))
        A_shape_after = A.shape

        # Print shapes and sizes before and after reshaping A
        print(f"\nShapes and Sizes before reshaping A for iteration {i + 1}:")
        print(f"U: {U.shape}")
        print(f"s: {s.shape}")
        print(f"V: {V.shape}")
        print(f"A: {A_shape_before}")
        print(f"Reshaped A: {A_shape_after}")

        # Repeat for all steps except the last one
        if i < N - 1:
            matrix = A

    # Reverse psi list
    psi.reverse()

    return psi

# Example usage
x = np.array([1, 2, 3])
psi = mps_encode(x)
print("\nFinal MPS Tensors (psi):", psi)