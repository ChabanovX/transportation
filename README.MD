
# Transportation Problem Solver

This project provides a solution to a balanced transportation problem with three sources and four destinations. It includes implementations of three methods for finding the initial basic feasible solution: the North-West Corner method, Vogel’s Approximation method, and Russell’s Approximation method.

## Table of Contents
- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Input and Output Format](#input-and-output-format)
- [Testing](#testing)
- [Contributors](#contributors)

## Project Overview
This project is a continuation from previous programming tasks, focusing on solving transportation problems. The assignment requires a balanced setup where the total supply equals the total demand, and the solution needs to find initial feasible solutions for transportation cost minimization.

## Getting Started

### Requirements
- Python 3.8+
- Libraries for matrix and vector manipulation (optional, e.g., NumPy)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ChabanovX/transportation.git
   cd transportation
   ```
2. Install required libraries:
   ```bash
   pip install numpy
   ```

## Usage

### Running the Program
Execute the main Python script to solve the transportation problem:
```bash
python main.py
```

### Input and Output Format

#### Input
1. **Supply Vector (S)**: Defines the supply values from each source.
2. **Cost Matrix (C)**: Represents transportation costs between sources and destinations.
3. **Demand Vector (D)**: Defines the demand values for each destination.

#### Output
- **String Responses**:
  - `"The method is not applicable!"` if any of the methods cannot be used.
  - `"The problem is not balanced!"` if the total supply does not equal total demand.
- **Parameter Table**: A printed table of input values, using the cost matrix (C), supply vector (S), and demand vector (D).
- **Initial Feasible Solutions (x₀)**: Three vectors representing solutions from:
  - North-West Corner method
  - Vogel’s Approximation method
  - Russell’s Approximation method

## Testing
- The code is tested on three different scenarios with varying supply, demand, and cost matrices. These cases are included in the report.
- Run tests using the provided test cases in the comments at the end of main.py file

## Contributors
This project is developed by DSAI-03 group, with contributions documented by the team leader in the report. Ratings are assigned based on individual contributions.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
