# Import the NumPy library and give it the alias 'np' for convenience
import numpy as np 

# Create a 3x3 matrix (2D array) using NumPy's array function
# Each inner list represents a row in the matrix
A = np.array([
    [100, 150, 200],  # First row
    [50, 200, 150],   # Second row
    [0, 50, 300]      # Third row
])

# Print a label for the original matrix
print("\nOriginal matrix:")
# Display the entire matrix A
print(A)
print()

# Access and print the element at row index 1, column index 1 (center element)
# Remember Python uses 0-based indexing
print("Elemento A[1,1]")
print(A[1, 1])  # Outputs: 200
print()

# Extract and print a 2x2 submatrix from the upper left corner
# Syntax: array[rows, columns]
# 0:2 means from index 0 up to (but not including) index 2
print("Submatriz A[0:2, 0:2]:")
print(A[0:2, 0:2])  # Outputs [[100, 150], [50, 200]]
print("\n" + "-"*50 + "\n")

# Modify the center element (1,1) to be the sum of:
# - top-left (0,0), center (1,1), and bottom-right (2,2) elements
A[1, 1] = A[0, 0] + A[1 , 1] + A[2, 2]  # 100 + 200 + 300 = 600
# Print the modified matrix (not shown in original code, but implied)
print("Matriz A após modificação:")
print(A)
print("\n" + "="*50 + "\n")

# Create another 3x3 matrix B where all elements are 100
B = np.array([
    [100, 100, 100],
    [100, 100, 100],
    [100, 100, 100]
])

# Perform element-wise addition between matrices A and B
# This adds corresponding elements (A[0,0]+B[0,0], A[0,1]+B[0,1], etc.)
C = A + B

# Print the resulting matrix C
print("Matriz C (A + B):")
print(C)
print("\nFim dos resultados.")
