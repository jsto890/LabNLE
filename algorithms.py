from core import *


# Nonlinear equation root finding by the bisection method.
# Inputs
# f        : nonlinear function
# xl, xr   : initial root bracket
# max_iter : maximum number of iterations performed
# tol      : numerical tolerance used to check for root
# Outputs
# x        : one-dimensional array containing estimates of root
# i        : number of iterations (number of bisections)
# e        : ExitFlag (enumeration)

# Hint 1:
# Iterate until either a root has been found or maximum number of iterations has been reached

# Hint 2:
# Check for root each iteration, making use of tol

# Hint 3:
# Update the bracket each iteration

def bisection(f, xl, xr, max_iter, tol):
    x = [xl]  # Initialize the list with the left bracket
    fxl = f(xl)

    # check the left side of interval for convergence
    if abs(fxl) <= tol:
        return x, 0, ExitFlag.Converged

    x.append(xr)  # Append the right bracket to the list
    fxr = f(xr)

    # check the right side of interval for convergence / check that the l/r function values have different signs.
    if abs(fxr) <= tol:
        return x, 0, ExitFlag.Converged
    elif fxl * fxr > 0:
        return x, 0, ExitFlag.NoRoot

    i = 1
    while True:
        xm = (xl + xr) / 2  # Midpoint
        x.append(xm)
        fxm = f(xm)

        if abs(fxm) <= tol:
            return x, i, ExitFlag.Converged
        elif i == max_iter:
            return x, i, ExitFlag.MaxIterations

        if fxl * fxm < 0:
            xr = xm
        else:
            xl = xm
            fxl = fxm

        i += 1


# Nonlinear equation root finding by the secant method.
# Inputs
# f        : nonlinear function
# x0, x1   : initial root bracket
# max_iter : maximum number of iterations performed
# tol      : numerical tolerance used to check for root
# Outputs
# x        : one-dimensional array containing estimates of root
# i        : number of iterations (number of times a new point is attempted to be estimated)
# e        : ExitFlag (enumeration)

def secant(f, x0, x1, max_iter, tol):
    x = [x0, x1]
    for i in range(max_iter):
        fx0, fx1 = f(x0), f(x1)
        if abs(fx1) < tol:
            return x, i, ExitFlag.Converged

        if fx1 - fx0 == 0:  # Avoid division by zero
            return x, i, ExitFlag.DivideByZero

        x_new = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        x.append(x_new)

        x0, x1 = x1, x_new

    return x, max_iter, ExitFlag.MaxIterations


# Nonlinear equation root finding by the Regula falsi method.
# Inputs
# f        : nonlinear function
# xl, xr   : initial root bracket
# max_iter : maximum number of iterations performed
# tol      : numerical tolerance used to check for root
# Outputs
# x        : one-dimensional array containing estimates of root
# i        : number of iterations (number of times a new point is attempted to be estimated)
# e        : ExitFlag (enumeration)

def regula_falsi(f, xl, xr, max_iter, tol):
    x = [xl, xr]
    fxl, fxr = f(xl), f(xr)

    for i in range(max_iter):
        x_new = xr - fxr * (xr - xl) / (fxr - fxl)
        x.append(x_new)
        fx_new = f(x_new)

        if abs(fx_new) < tol:
            return x, i, ExitFlag.Converged

        if fxl * fx_new < 0:
            xr, fxr = x_new, fx_new
        else:
            xl, fxl = x_new, fx_new

    return x, max_iter, ExitFlag.MaxIterations


# Nonlinear equation root finding by Newton's method
# Inputs
# f        : nonlinear function
# g        : nonlinear function derivative (gradient)
# x0       : initial root estimate
# max_iter : maximum number of iterations performed
# tol      : numerical tolerance used to check for root
# Outputs
# x        : one-dimensional array containing estimates of root
# i        : number of iterations (number of times a new point is attempted to be estimated)
# e        : ExitFlag (enumeration)

def newton(f, g, x0, max_iter, tol):
    x = [x0]
    for i in range(max_iter):
        fx = f(x0)
        if abs(fx) < tol:
            return x, i, ExitFlag.Converged

        gx = g(x0)
        if abs(gx) < 1e-12:  # or some small number close to zero
            return x, i, ExitFlag.DivideByZero  # or some appropriate exit flag indicating failure due to 0 derivative

        x_new = x0 - fx / gx
        x.append(x_new)

        x0 = x_new

    return x, max_iter, ExitFlag.MaxIterations


def newton_damped(f, df, x0, max_iter, tol, beta):
    x = [x0]
    for i in range(max_iter):
        fx = f(x0)
        if abs(fx) < tol:
            return x, i, ExitFlag.Converged

        dfx = df(x0)
        if abs(dfx) < 1e-12:
            return x, i, ExitFlag.Converged

        # Compute the damping factor
        delta = -fx / dfx
        alpha = 1 / (1 + beta * (abs(delta) ** 2))

        # Update the estimate using the damping factor
        x_new = x0 + alpha * delta
        x.append(x_new)

        x0 = x_new

    return x, max_iter, ExitFlag.MaxIterations




# Nonlinear equation root finding by the combined bisection/Newton's method
# Inputs
# f        : nonlinear function
# g        : nonlinear function derivative (gradient)
# xl, xr   : initial root bracket
# max_iter : maximum number of iterations performed
# tol      : numerical tolerance used to check for root
# Outputs
# x        : one-dimensional array containing estimates of root
# i        : number of iterations (number of times a new point is attempted to be estimated)
# e        : ExitFlag (enumeration)

def combined(f, g, xl, xr, max_iter, tol):
    x = [xl, xr, (xl + xr) / 2]
    x0 = x[-1]

    for i in range(max_iter):
        fx = f(x0)
        if abs(fx) < tol:
            return x, i, ExitFlag.Converged

        gx = g(x0)
        if gx == 0:  # Avoid division by zero
            return x, i, ExitFlag.DivideByZero
        x_new = x0 - fx / gx

        if xl < x_new < xr:
            x.append(x_new)
        else:
            x_new = (xl + xr) / 2
            x.append(x_new)

        if f(xl) * f(x_new) < 0:
            xr = x_new
        else:
            xl = x_new

        x0 = x_new

    return x, max_iter, ExitFlag.MaxIterations
