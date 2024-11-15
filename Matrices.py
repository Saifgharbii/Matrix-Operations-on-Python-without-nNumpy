from copy import deepcopy
from random import randint

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

def input_integer(message):
    """Prompts the user for a positive integer input."""
    while True:
        try:
            value = int(input(message))
            if value > 0:
                break
        except ValueError:
            print("Please provide a valid integer.")
    return value


def input_float(message):
    """Prompts the user for a floating-point number input."""
    while True:
        try:
            value = float(input(message))
            break
        except ValueError:
            print("Please provide a valid float.")
    return value


# ------------------------------------------------------------------------------
# Matrix Initialization
# ------------------------------------------------------------------------------

def get_matrix_dimensions():
    """Prompts the user to input dimensions for a matrix."""
    rows = input_integer("Enter the number of rows: ")
    columns = input_integer("Enter the number of columns: ")
    return rows, columns


def matrix_identity(size):
    """Generates an identity matrix of a given size."""
    if isinstance(size, int) and size > 0:
        matrix = [[1 if i == j else 0 for j in range(size)] for i in range(size)]
        return matrix


def fill_matrix():
    """Prompts the user to fill a matrix with custom values."""
    rows, cols = get_matrix_dimensions()
    matrix = []
    for i in range(rows):
        row = []
        for j in range(cols):
            value = input_float(f"Enter the value for row {i+1}, column {j+1}: ")
            row.append(value)
        matrix.append(row)
    print("Your matrix is:\n", matrix)
    return matrix


def random_fill_matrix(rows, cols, min_value=-100, max_value=100):
    """Generates a matrix filled with random integers within a given range."""
    if max_value < min_value:
        print("Error: max_value cannot be less than min_value.")
    else:
        return [[randint(min_value, max_value) for _ in range(cols)] for _ in range(rows)]


# ------------------------------------------------------------------------------
# Matrix Validations
# ------------------------------------------------------------------------------

def is_matrix(matrix):
    """Checks if the input is a valid matrix."""
    if isinstance(matrix, list) and all(isinstance(row, list) for row in matrix):
        row_length = len(matrix[0])
        return all(len(row) == row_length for row in matrix)
    return False


def is_square_matrix(matrix):
    """Checks if the input matrix is square."""
    if is_matrix(matrix):
        return len(matrix) == len(matrix[0])
    return False


# ------------------------------------------------------------------------------
# Matrix Operations
# ------------------------------------------------------------------------------

def zero_matrix(rows, cols):
    """Generates a zero matrix with the specified dimensions."""
    return [[0] * cols for _ in range(rows)]


def add_matrices(matrix_a, matrix_b):
    """Performs element-wise addition of two matrices."""
    if is_matrix(matrix_a) and is_matrix(matrix_b) and \
            len(matrix_a) == len(matrix_b) and len(matrix_a[0]) == len(matrix_b[0]):
        result = zero_matrix(len(matrix_a), len(matrix_a[0]))
        for i in range(len(matrix_a)):
            for j in range(len(matrix_a[0])):
                result[i][j] = matrix_a[i][j] + matrix_b[i][j]
        return result
    else:
        print("Error: Matrices must have the same dimensions for addition.")


def multiply_matrices(matrix_a, matrix_b):
    """Performs matrix multiplication."""
    if is_matrix(matrix_a) and is_matrix(matrix_b):
        rows_a, cols_a = len(matrix_a), len(matrix_a[0])
        rows_b, cols_b = len(matrix_b), len(matrix_b[0])
        if cols_a != rows_b:
            print("Error: Matrix multiplication is not possible with these dimensions.")
            return None
        result = zero_matrix(rows_a, cols_b)
        for i in range(rows_a):
            for j in range(cols_b):
                result[i][j] = sum(matrix_a[i][k] * matrix_b[k][j] for k in range(cols_a))
        return result
    else:
        print("Error: Input must be valid matrices.")


def matrix_power(matrix, power):
    """Calculates the matrix raised to a given power."""
    if is_matrix(matrix) and is_square_matrix(matrix):
        result = matrix_identity(len(matrix))
        for _ in range(power):
            result = multiply_matrices(result, matrix)
        return result
    else:
        print("Error: Matrix must be square to raise to a power.")


# ------------------------------------------------------------------------------
# Other Matrix Functions
# ------------------------------------------------------------------------------

def trace_of_matrix(matrix):
    """Calculates the trace of a square matrix."""
    if is_matrix(matrix) and is_square_matrix(matrix):
        return sum(matrix[i][i] for i in range(len(matrix)))
    print("Error: Matrix must be square to calculate the trace.")


def transpose_matrix(matrix):
    """Calculates the transpose of a matrix."""
    if is_matrix(matrix):
        return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
    else :
        print("Error: Input must be a valid matrix.")
        return None

# -----------------------------------------------------------------------------

def matrix_row_exchange(matrix, row1, row2):
    """
    Exchanges two rows in a matrix.
    """
    if is_matrix(matrix) and (row1 and row2) in range(len(matrix)):
        new_matrix = deepcopy(matrix)
        new_matrix[row2] = new_matrix[row1]
        new_matrix[row1] = new_matrix[row2]
        return new_matrix
    else:
        print('Matrix error.')

# -----------------------------------------------------------------------------

def matrix_row_product_with_scalar(matrix, row, scalar):
    """
    Multiplies a row by a scalar.
    """
    new_matrix = deepcopy(matrix)
    for i in range(len(new_matrix[row])):
        new_matrix[row][i] *= scalar
    return new_matrix

# -----------------------------------------------------------------------------

def matrix_row_sum(matrix, row1, row2, scalar=1):
    """
    Adds a scaled row2 to row1 in the matrix.
    """
    if is_matrix(matrix) and isinstance(row1, int) and isinstance(row2, int) and isinstance(scalar, (int, float)) \
            and row1 in range(len(matrix)) and row2 in range(len(matrix)):
        new_matrix = deepcopy(matrix)
        for i in range(len(new_matrix[0])):
            new_matrix[row1][i] += scalar * new_matrix[row2][i]
        return new_matrix
    else:
        print('Error in matrix row sum.')

# -----------------------------------------------------------------------------

def matrix_echelon(matrix):
    """
    Converts a matrix to echelon form.
    """
    if is_matrix(matrix) and is_square_matrix(matrix):
        A = deepcopy(matrix)
        k = 0
        matrix_identity = matrix_identity(len(matrix))
        pivot_test = False
        for i in range(len(matrix[0])):  # column-wise search for pivots
            if A[k][i] != 0:  # A[k][i] is the i-th pivot
                pivot_test = False
                for j in range(k + 1, len(matrix)):  # row reduction below k-th pivot
                    if abs(A[j][i]) < 10**(-10):
                        A[j][i] = 0
                    matrix_identity = matrix_row_sum(matrix_identity, j, k, -A[j][i] / A[k][i])
                    A = matrix_row_sum(A, j, k, -A[j][i] / A[k][i])
                    if A[j][i] != 0:
                        print(f'Row {j} <= Row {j} + {(-A[j][i] / A[k][i])} * Row {k}')

            elif A[k][i] == 0:  # pivot search loop
                pivot_test = False
                for p in range(k + 1, len(matrix) + 1):
                    if p == len(matrix):
                        pivot_test = True  # no pivot in the i-th column
                    elif A[p][i] != 0:  # pivot exists in this column
                        A = matrix_row_exchange(A, k, p)  # swap pivot row into k-th row
                        matrix_identity = matrix_row_exchange(matrix_identity, k, p)
                        print(f'Row {p} <=> Row {k}')
                        for j in range(k + 1, len(matrix)):  # row reduction below k-th pivot
                            if abs(A[j][i]) < 10**(-10):
                                A[j][i] = 0
                            matrix_identity = matrix_row_sum(matrix_identity, j, k, -A[j][i] / A[k][i])
                            A = matrix_row_sum(A, j, k, -A[j][i] / A[k][i])
                            if A[j][i] != 0:
                                print(f'Row {j + 1} <= Row {j + 1} + {(-A[j][i] / A[k][i])} * Row {k + 1}')
                        break
            if not pivot_test:
                k += 1
        return A, k, matrix_identity
    else:
        print('Error: Not a square matrix.')

# -----------------------------------------------------------------------------

def matrix_inverse(matrix):
    """
    Returns the inverse of a matrix.
    """
    A, k, matrix_identity = matrix_echelon(matrix)
    A = matrix_echelon(A)[0]
    determinant = 1
    if k == len(matrix):
        k = len(matrix) - 1
        for i in range(len(matrix) - 1, -1, -1):  # column-wise search for pivots
            for j in range(k - 1, -1, -1):  # row reduction above k-th pivot
                if abs(A[j][i]) < 10**(-10):
                    A[j][i] = 0
                matrix_identity = matrix_row_sum(matrix_identity, j, k, -A[j][i] / A[k][i])
                A = matrix_row_sum(A, j, k, -A[j][i] / A[k][i])
            k -= 1
        for i in range(len(matrix)):
            scalar = 1 / A[i][i]
            determinant *= A[i][i]
            matrix_identity = matrix_row_product_with_scalar(matrix_identity, i, scalar)
            print(f'Row {i + 1} <= {scalar} * Row {i + 1}')
            A[i][i] = 1
        # eliminate zeros
        for row in matrix_identity:
            for j in range(len(row)):
                if abs(row[j]) < 10**(-10):
                    row[j] = 0
        return matrix_identity, determinant
    else:
        print("Matrix is not invertible.")

# -----------------------------------------------------------------------------

def remove_column(matrix, index):
    """
    Removes the specified column from the matrix.
    """
    new_matrix = []
    if index < len(matrix[0]):
        for row in matrix:
            new_matrix.append(row[:index] + row[index + 1:])
    else:
        new_matrix = [row[:index] for row in matrix]
    return new_matrix

# -----------------------------------------------------------------------------

def remove_row(matrix, index):
    """
    Removes the specified row from the matrix.
    """
    new_matrix = []
    if index < len(matrix):
        new_matrix = matrix[:index] + matrix[index + 1:]
    else:
        new_matrix = matrix[:index]
    return new_matrix

# -----------------------------------------------------------------------------

def determinant(matrix):
    """
    Computes the determinant of a square matrix.
    """
    if is_matrix(matrix) and is_square_matrix(matrix):
        if len(matrix) == 2:
            det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
            return det
        else:
            det = 0
            for i in range(len(matrix)):
                sign = matrix[i][0] if i % 2 == 0 else -matrix[i][0]
                minor_matrix = remove_row(matrix, i)
                minor_matrix = remove_column(minor_matrix, 0)
                det += sign * determinant(minor_matrix)
            return det

# -----------------------------------------------------------------------------

def print_matrix(matrix):
    """
    Prints the matrix in a formatted way.
    """
    max_lengths = []
    for i in range(len(matrix[0])):  # find the maximum length of each column
        max_len = len(str(matrix[0][i]))
        for j in range(len(matrix)):
            if len(str(matrix[j][i])) > max_len:
                max_len = len(str(matrix[j][i]))
        max_lengths.append(max_len)

    for row in matrix:
        row_str = ''
        for i in range(len(row)):
            space = (max_lengths[i] - len(str(row[i]))) // 2 + 1
            row_str += ' ' * space + str(row[i]) + ' ' * space
        print(f'|{row_str}|')

# -----------------------------------------------------------------------------


def system_solver(M,Y):
    if  is_matrix(M) and is_square_matrix(M) and is_matrix(Y) and len(Y)==len(M) and len(Y[0])==1:
        A = deepcopy(M)
        k = 0
        test_pivot = False

        for i in range(len(M[0])):  # Iterate over columns
            if A[k][i] != 0:  # If A[k][i] is a pivot
                test_pivot = False
                for j in range(k+1, len(M)):  # Row reduction for rows below the pivot
                    if abs(A[j][i]) < 10**(-10):
                        A[j][i] = 0
                    A = add_matrices(A, j, k, -A[j][i] / A[k][i])
                    Y = add_matrices(Y, j, k, -A[j][i] / A[k][i])
                    
            elif A[k][i] == 0:  # If no pivot in the current column
                test_pivot = False
                for p in range(k+1, len(M)+1):
                    if p == len(M):  # No pivot in this column
                        test_pivot = True
                    elif A[p][i] != 0:  # Found a pivot in this column
                        A = matrix_row_exchange(A, k, p)
                        Y = matrix_row_exchange(Y, k, p)

                        for j in range(k+1, len(M)):  # Row reduction for rows below the pivot
                            if abs(A[j][i]) < 10**(-10):
                                A[j][i] = 0
                            A = add_matrices(A, j, k, -A[j][i] / A[k][i])
                            Y = add_matrices(Y, j, k, -A[j][i] / A[k][i])
                            if A[j][i] != 0:
                                print(f'line {j+1} <= line {j+1} + {(-A[j][i]/A[k][i])} * line {k+1}')
                        break

            if not test_pivot:
                k += 1

        if k == len(M):
            k = len(M) - 1
            for i in range(len(M) - 1, -1, -1):  # Back substitution on the columns
                for j in range(k-1, -1, -1):  # Row reduction for rows above the pivot
                    if abs(A[j][i]) < 10**(-10):
                        A[j][i] = 0
                    A = add_matrices(A, j, k, -A[j][i] / A[k][i])
                    Y = add_matrices(Y, j, k, -A[j][i] / A[k][i])
                k -= 1
            for i in range(len(M)):
                a = 1 / A[i][i]
                Y = matrix_row_product_with_scalar(Y, i, a)
                A[i][i] = 1
            return Y

        else:
            # Check for incompatible system
            incompatible_system = False
            for i in range(k, len(A)):
                if Y[i][0] != 0:
                    incompatible_system = True

            if incompatible_system:
                print("the system is incompatible!")
                return None  # Incompatible system, no solution
            else:
                # The system has infinitely many solutions
                print("The system has infinitely many solutions!")
                return None
    else:
        print('Error: Invalid input matrices.')
        return None


