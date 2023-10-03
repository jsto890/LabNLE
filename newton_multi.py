import numpy as np
from core import *
from jacobian import jacobian


def newton_multi(n, f, x0, h, max_iter, tol):
    x = [x0]
    k = 0
    while True:
        xk = x[k]
        fk = f(xk)

        if max(fk) <= tol and max(-fk) <= tol:
            return x, ExitFlag.Converged

        if np.linalg.norm(fk, np.inf) <= tol:
            return x, ExitFlag.Converged

        Jk = jacobian(f, xk, h, n)

        # Check if the Jacobian is singular
        if np.linalg.det(Jk) == 0:
            return x, ExitFlag.SingularJacobian

        # Solve for delta_x
        delta_x = -np.linalg.solve(Jk, fk)

        # Update the solution
        x.append(xk + delta_x)

        k += 1
        if k == max_iter:
            return x, ExitFlag.MaxIterations
