## **Python Matrix Operations**

**Introduction**

This Python script provides a collection of functions for performing various matrix operations, including:

* Matrix creation and initialization  
* Matrix addition  
* Matrix multiplication  
* Matrix power  
* Matrix trace  
* Matrix transpose  
* Matrix inversion  
* Matrix row operations  
* Matrix determinant  
* Solving systems of linear equations

**Usage**

1. **Import the script:**  
   Python  
   `import matrix_operations as mo`

2. **Create matrices:**  
   * **Manually:**  
     Python  
     `matrix_a = [[1, 2], [3, 4]]`

   * **Randomly:**  
     Python  
     `matrix_b = mo.random_fill_matrix(3, 3)`

   * **Identity matrix:**  
     Python  
     `identity_matrix = mo.matrix_identity(4)`

3. **Perform operations:**  
   Python  
   `sum_matrix = mo.add_matrices(matrix_a, matrix_b)`  
   `product_matrix = mo.multiply_matrices(matrix_a, matrix_b)`  
   `powered_matrix = mo.matrix_power(matrix_a, 2)`  
   `trace = mo.trace_of_matrix(matrix_a)`  
   `transpose_matrix = mo.transpose_matrix(matrix_a)`  
   `inverse_matrix = mo.matrix_inverse(matrix_a)`

4. **Solve systems of linear equations:**  
   Python  
   `M = [[1, 2], [3, 4]]`  
   `Y = [[5], [6]]`  
   `solution = mo.system_solver(M, Y)`

**Functions and Their Descriptions:**

* input\_integer: Prompts for a positive integer input.  
* input\_float: Prompts for a floating-point number input.  
* get\_matrix\_dimensions: Prompts for matrix dimensions.  
* matrix\_identity: Generates an identity matrix.  
* fill\_matrix: Prompts for manual matrix input.  
* random\_fill\_matrix: Generates a random matrix.  
* is\_matrix and is\_square\_matrix: Check matrix validity.  
* zero\_matrix: Creates a zero matrix.  
* add\_matrices: Adds two matrices.  
* multiply\_matrices: Multiplies two matrices.  
* matrix\_power: Raises a matrix to a power.  
* trace\_of\_matrix: Calculates the trace of a matrix.  
* transpose\_matrix: Transposes a matrix.  
* matrix\_echelon: Converts a matrix to echelon form.  
* matrix\_inverse: Computes the inverse of a matrix.  
* remove\_column: Removes a column from a matrix.  
* remove\_row: Removes a row from a matrix.  
* determinant: Calculates the determinant of a matrix.  
* print\_matrix: Prints a matrix in a formatted way.  
* system\_solver: Solves a system of linear equations.

**Note:**

* Ensure that the input matrices are valid and have compatible dimensions for operations.  
* The matrix\_echelon function uses a Gaussian elimination-based approach to find the inverse and solve systems of linear equations.  
* The determinant function uses a recursive approach to calculate the determinant.

Feel free to explore and extend this code to suit your specific matrix operations needs\!