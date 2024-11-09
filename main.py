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
    cost_matrix = cost_matrix.astype(float)

    while np.any(supply > 0) and np.any(demand > 0):
        # Calculate penalties for rows
        penalty_rows = []
        for row in cost_matrix:
            finite_values = sorted([val for val in row if val != np.inf])
            if len(finite_values) > 1:
                min1, min2 = finite_values[0], finite_values[1]
                diff = min2 - min1
            elif len(finite_values) == 1:
                diff = finite_values[0]
            else:
                diff = 0
            penalty_rows.append(diff)

        # Calculate penalties for columns
        penalty_cols = []
        for col in cost_matrix.T:
            finite_values = sorted([val for val in col if val != np.inf])
            if len(finite_values) > 1:
                min1, min2 = finite_values[0], finite_values[1]
                diff = min2 - min1
            elif len(finite_values) == 1:
                diff = finite_values[0]
            else:
                diff = 0
            penalty_cols.append(diff)

        # print(f"Row penalties: {penalty_rows}")
        # print(f"Column penalties: {penalty_cols}")

        # Determine row or column with the highest penalty
        row_idx = np.argmax(penalty_rows)
        col_idx = np.argmax(penalty_cols)
        if penalty_rows[row_idx] >= penalty_cols[col_idx]:
            i = row_idx
            j = np.argmin(cost_matrix[i])
        else:
            j = col_idx
            i = np.argmin(cost_matrix[:, j])
        
        allocation = min(supply[i], demand[j])
        x[i, j] = allocation
        supply[i] -= allocation
        demand[j] -= allocation
        if supply[i] == 0:
            cost_matrix[i, :] = np.inf  # Mark entire row as unusable
        if demand[j] == 0:
            cost_matrix[:, j] = np.inf  # Mark entire column as unusable


    return x

def russell_approximation(supply, demand, cost_matrix):
    """Russell's Approximation Method for finding initial feasible solution."""
    m, n = len(supply), len(demand)
    x = np.zeros((m, n))
    u = np.zeros(m)  # Row potentials
    v = np.zeros(n)  # Column potentials
    cost_matrix = cost_matrix.astype(float)  # Convert cost matrix to float
    reduced_cost_matrix = np.zeros((m, n))


    # Calculate U_i is the largest cost in row and V_j is the largest cost in column
    for i in range(m):
        u[i] = max(cost_matrix[i])
    for j in range(n):
        v[j] = max(cost_matrix[:, j])

    # Compute reduced cost of each cell Δ_ij = c_ij - U_i - V_j

    for i in range(m):
        for j in range(n):
            reduced_cost_matrix[i, j] = cost_matrix[i, j] - u[i] - v[j]


    # Select the variable having the most negative Δ value, break ties arbitrarily
    most_negative_value = np.min(reduced_cost_matrix)

    # Allocate as much as possible. Eliminate necessary cells from consideration
    while np.min(reduced_cost_matrix) < 0:
        # Find the cell with the most negative reduced cost
        min_index = np.unravel_index(np.argmin(reduced_cost_matrix), reduced_cost_matrix.shape)
        i, j = min_index

        # Allocate as much as possible to this cell
        allocation = min(supply[i], demand[j])
        x[i, j] = allocation
        supply[i] -= allocation
        demand[j] -= allocation

        # If supply or demand is exhausted, remove the corresponding row or column
        if supply[i] == 0:
            reduced_cost_matrix[i, :] = np.inf
        if demand[j] == 0:
            reduced_cost_matrix[:, j] = np.inf


    return x

def display_total_cost(allocation, cost_matrix):
    terms = []
    total_cost = 0

    print("The minimum total transportation cost is:")
    for i in range(allocation.shape[0]):
        for j in range(allocation.shape[1]):
            if allocation[i, j] != 0:
                cost = allocation[i, j] * cost_matrix[i, j]
                total_cost += cost

                allocation_str = int(allocation[i, j]) if allocation[i, j].is_integer() else allocation[i, j]
                cost_str = int(cost_matrix[i, j]) if cost_matrix[i, j].is_integer() else cost_matrix[i, j]
                terms.append(f"({allocation_str} * {cost_str})")


    expression = " + ".join(terms)
    print(f"{expression} = {int(total_cost) if total_cost.is_integer() else total_cost}")

    return total_cost

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
    display_total_cost(nw_corner_solution, cost_matrix)

    # Vogel's Approximation Method
    vogel_solution = vogel_approximation(supply.copy(), demand.copy(), cost_matrix.copy())
    print("\nVogel's Approximation Method:")
    print(vogel_solution)
    display_total_cost(vogel_solution, cost_matrix)
    
    # Russell's Approximation Method
    russell_solution = russell_approximation(supply.copy(), demand.copy(), cost_matrix.copy())
    print("\nRussell's Approximation Method:")
    print(russell_solution)
    display_total_cost(russell_solution, cost_matrix)

print("FIRST CASE")

# TEST CASE 1
supply = np.array([20, 30, 25])          # S -- a vector of coefficients of supply
cost_matrix = np.array([[8, 6, 10],
                        [9, 12, 13],
                        [14, 9, 16]])           # C -- a matrix of coefficients of costs          
demand = np.array([10, 35, 30])             # D -- a vector of coefficients of demand

transportation_problem(supply, demand, cost_matrix)

print("\n\nSECOND CASE")

# TEST CASE 2
supply = np.array([7, 9, 18])          # S -- a vector of coefficients of supply
cost_matrix = np.array([
    [19, 30, 50, 10],
    [70, 30, 40, 60],
    [40, 8, 70, 20]])           # C -- a matrix of coefficients of costs
demand = np.array([5, 8, 7, 14])             # D -- a vector of coefficients of demand

transportation_problem(supply, demand, cost_matrix)

print("\n\nTHIRD CASE")

# NEW TEST CASE 3
supply = np.array([20, 60, 70])          # S -- a vector of coefficients of supply
cost_matrix = np.array([
    [13, 11, 15, 40],
    [17, 14, 12, 13],
    [18, 18, 15, 12]])           # C -- a matrix of coefficients of costs
demand = np.array([30, 30, 40, 50])

transportation_problem(supply, demand, cost_matrix)

