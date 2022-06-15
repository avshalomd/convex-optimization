import os
import sys
import time
import statistics
from typing import Tuple

import scs
import cvxpy as cp
import numpy as np


def get_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(os.path.join('data', 'parameters_values.npy'), 'rb') as f:
        P = np.load(f)
    with open(os.path.join('data', 'trace_constraints_matrices.npy'), 'rb') as f:
        A = np.load(f)
    with open(os.path.join('data', 'trace_constraints_results.npy'), 'rb') as f:
        b = np.load(f)
    return P, A, b


def print_versions():
    print('Software versions:')
    print(f'python == {sys.version}')
    print(f'cvxpy == {cp.__version__}')
    print(f'numpy == {np.__version__}')
    print(f'scs == {scs.__version__}\n')


if __name__ == '__main__':

    print_versions()

    P, A, b = get_data()
    n = P.shape[1]

    num_iterations = 10
    solver_time = []

    for i in range(num_iterations):

        # Define problem parameters
        U = cp.Parameter((n, n))
        U.value = P

        # Define PSD variable
        X = cp.Variable((n, n), PSD=True)

        # Define PSD constraint - unit diagonal Gram matrix
        constraints = [cp.diag(X) == 1]

        for A_i, b_i in zip(A, b):
            constraints += [cp.trace(A_i @ X) == b_i]

        # Define an SDP program
        problem = cp.Problem(cp.Maximize(cp.trace(U @ X)), constraints)

        start_t = time.time()
        if cp.__version__ == '1.0.21':
            problem.solve(solver=cp.SCS, verbose=True)
        else:
            problem.solve(solver=cp.SCS, ignore_dpp=True, verbose=True)
        final_time = time.time() - start_t
        print(f"({i+1}/{num_iterations}) {round(final_time*1000)}[ms]")
        solver_time.append(final_time)

    print(f"\nSolver average time ({n} parameters, {len(b) + 1} constraints): "
          f"{round(statistics.mean(solver_time)*1000)}[ms]")
