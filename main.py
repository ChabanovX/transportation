import numpy as np

def check_balance(supply, demand):
    """Check if the problem is balanced."""
    total_supply = sum(supply)
    total_demand = sum(demand)
    if total_supply != total_demand:
        print("The problem is not balanced!")
        return False
    return True

def print_table(cost_matrix, supply, demand):
    m, n = len(supply), len(demand)

    # Print header row with demands
    print("    ", end="")
    for j in range(n):
        print(f"D{j+1:<4}", end=" ")
    print("Supply")

    # Print cost matrix rows with supply
    for i in range(m):
        print(f"S{i+1:<2}", end=" ")
        for j in range(n):
            print(f"{cost_matrix[i][j]:<5}", end="")
        print(f"\t{supply[i]}")

    # Print demand row
    print("Demand ", end="")
    for d in demand:
        print(f"{d:<5}", end="")
    print()

def north_west_corner(supply, demand):
    """North-West Corner Method for finding initial feasible solution."""
    m, n = len(supply), len(demand)
    x = np.zeros((m, n))
    i, j = 0, 0
    while i < m and j < n:
        allocation = min(supply[i], demand[j])
        x[i, j] = allocation
        supply[i] -= allocation
        demand[j] -= allocation
        if supply[i] == 0:
            i += 1
        if demand[j] == 0:
            j += 1
    return x

def vogel_approximation(supply, demand, cost_matrix):
    """Vogel's Approximation Method for finding initial feasible solution."""
    m, n = len(supply), len(demand)
    x = np.zeros((m, n))
    cost_matrix = cost_matrix.astype(float)  # Convert cost matrix to float to use np.inf

    while any(supply) and any(demand):
        # Calculate penalties for rows
        penalty_rows = []
        for row in cost_matrix:
            finite_values = sorted([val for val in row if val != np.inf])
            if len(finite_values) > 1:
                penalty_rows.append(finite_values[1] - finite_values[0])
            elif len(finite_values) == 1:
                penalty_rows.append(finite_values[0])
            else:
                penalty_rows.append(0)

        # Calculate penalties for columns
        penalty_cols = []
        for col in cost_matrix.T:
            finite_values = sorted([val for val in col if val != np.inf])
            if len(finite_values) > 1:
                penalty_cols.append(finite_values[1] - finite_values[0])
            elif len(finite_values) == 1:
                penalty_cols.append(finite_values[0])
            else:
                penalty_cols.append(0)

        # Determine row or column with the highest penalty
        row_idx = np.argmax(penalty_rows)
        col_idx = np.argmax(penalty_cols)
        if penalty_rows[row_idx] >= penalty_cols[col_idx]:
            i = row_idx
            j = np.argmin(cost_matrix[i])
        else:
            j = col_idx
            i = np.argmin(cost_matrix[:, j])
        
        # Allocate to the chosen cell
        allocation = min(supply[i], demand[j])
        x[i, j] = allocation
        supply[i] -= allocation
        demand[j] -= allocation
        cost_matrix[i, j] = np.inf  # Mark the cell as allocated

    return x

def russell_approximation(supply, demand, cost_matrix):
    """Russell's Approximation Method for finding initial feasible solution."""
    m, n = len(supply), len(demand)
    x = np.zeros((m, n))
    u = np.zeros(m)  # Row potentials
    v = np.zeros(n)  # Column potentials
    cost_matrix = cost_matrix.astype(float)  # Convert cost matrix to float

    for i in range(m):
        u[i] = min(cost_matrix[i])
    for j in range(n):
        v[j] = min(cost_matrix[:, j])

    for i in range(m):
        for j in range(n):
            cost_adjusted = cost_matrix[i, j] - u[i] - v[j]
            if cost_adjusted < 0:
                allocation = min(supply[i], demand[j])
                x[i, j] = allocation
                supply[i] -= allocation
                demand[j] -= allocation
    return x

# Main function to solve the transportation problem
def transportation_problem(supply, demand, cost_matrix):
    # Convert cost_matrix to float for compatibility with np.inf
    cost_matrix = cost_matrix.astype(float)
    
    # Check if the problem is balanced
    if not check_balance(supply, demand):
        return
    
    print("The input parameter table:")
    print_table(cost_matrix, supply, demand)
    
    # North-West Corner Method
    nw_corner_solution = north_west_corner(supply.copy(), demand.copy())
    print("\nNorth-West Corner Method:")
    print(nw_corner_solution)
    
    # Vogel's Approximation Method
    vogel_solution = vogel_approximation(supply.copy(), demand.copy(), cost_matrix.copy())
    print("\nVogel's Approximation Method:")
    print(vogel_solution)
    
    # Russell's Approximation Method
    russell_solution = russell_approximation(supply.copy(), demand.copy(), cost_matrix.copy())
    print("\nRussell's Approximation Method:")
    print(russell_solution)

# TEST CASE 1
# supply = np.array([140, 180, 160])          # S -- a vector of coefficients of supply
# cost_matrix = np.array([
#     [2, 3, 4, 2, 4],
#     [8, 4, 1, 4, 1],
#     [9, 7, 3, 7, 2]])           # C -- a matrix of coefficients of costs
# demand = np.array([60, 70, 120, 130, 100])             # D -- a vector of coefficients of demand
# !! NOT WORKING FOR RUSSELS

# TEST CASE 2
# supply = np.array([20, 30, 25])          # S -- a vector of coefficients of supply
# cost_matrix = np.array([[8, 6, 10],
#                         [9, 12, 13],
#                         [14, 9, 16]])           # C -- a matrix of coefficients of costs          
# demand = np.array([10, 35, 30])             # D -- a vector of coefficients of demand
# WORKING FOR RUSSELS


# TEST CASE 3
# supply = np.array([160, 140, 170])          # S -- a vector of coefficients of supply
# cost_matrix = np.array([
#     [7, 8, 1, 2],
#     [4, 5, 9, 8],
#     [9, 2, 3, 6]])           # C -- a matrix of coefficients of costs
# demand = np.array([120, 50, 190, 110])             # D -- a vector of coefficients of demand
# !! NOT WORKING FOR RUSSELS

# NEW TEST CASE 4
# supply = np.array([7, 9, 18])          # S -- a vector of coefficients of supply
# cost_matrix = np.array([
#     [19, 30, 50, 10],
#     [70, 30, 40, 60],
#     [40, 8, 70, 20]])           # C -- a matrix of coefficients of costs
# demand = np.array([5, 8, 7, 14])             # D -- a vector of coefficients of demand

# EXPECTED OUTPUT TABLE: 
# Russell's Approximation Method:
# [[5. 2. 0. 2.]
#  [0. 2. 7. 0.]
#  [0. 4. 0. 14.]]

# CURRENT OUTPUT TABLE:
# Russell's Approximation Method:
# [[5. 0. 0. 2.]
#  [0. 8. 1. 0.]
#  [0. 0. 0. 0.]]


transportation_problem(supply, demand, cost_matrix)


# TEST CASE 1
# print("\n=== TEST CASE 1 ===")
# # Test case with balanced supply and demand
# test_supply = np.array([15, 25, 10])
# test_demand = np.array([20, 15, 15]) 
# test_cost = np.array([
#     [6, 4, 8],
#     [7, 9, 11],
#     [5, 3, 7]
# ])

# print("\nTest Case Parameters:")
# print("Supply:", test_supply)
# print("Demand:", test_demand)
# print("Cost Matrix:")
# print(test_cost)
# print("\nSolving test case...")
# transportation_problem(test_supply, test_demand, test_cost)

# Expected optimal solution should have minimum total cost
# Can be verified by comparing results from different methods
