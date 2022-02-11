import numpy as np
import matplotlib.pyplot as plt
from filterLineSearchSuper import interior_pt 
from helperfunctions import grad_check
from problems import example2

if __name__ == "__main__":
    nx = 10
    A  = np.zeros((nx, nx))
    for i in range(nx):
        if i < int(nx/2):
            A[i,i] = -0.25
        else:
            A[i,i] = 1.
    problem = example2(nx, A)
    

    #---- gradient check

    F     = lambda x : problem.phi(x, 0.1)
    gradF = lambda x : problem.Dxphi(x, 0.1)
    
    x0 = 2.*np.ones(problem.n) + np.abs(np.random.randn(problem.n))
    xhat = np.random.randn(problem.n)
    grad_check(F, gradF, x0, xhat)

    solver  = interior_pt(problem, "direct")
    
    x0      = 1.0 * np.ones(problem.n)
    X0      = [x0, np.ones(problem.m), np.ones(problem.n2)]
    solver.initialize(X0)
    complementarity0 = np.linalg.norm(X0[2][:]*(x0[problem.n1:] - problem.rhol[:]), np.inf)
    feasibility0     = problem.theta(x0)
    optimality0      = np.linalg.norm(problem.DxL(X0), 2)


    mu0 = 1.e0
    tol = 1.e-10
    max_it = 35
    Xf, mu, E, Mus = solver.solve(tol, max_it, mu0)
    xf = Xf[0][:]
    complementarityf = np.linalg.norm(Xf[2][:]*(xf[problem.n1:] - problem.rhol[:]), np.inf)
    feasibilityf     = problem.theta(xf)
    optimalityf      = np.linalg.norm(problem.DxL(Xf), 2)
    print("complementarity error reduction = {0:1.3e}".format(complementarityf / complementarity0))
    print("feasibility error reduction = {0:1.3e}".format(feasibilityf / feasibility0))
    print("optimality error reduction = {0:1.3e}".format( optimalityf / optimality0))
    print("||lam||_1 = ", np.linalg.norm(Xf[1], 1))
    print("||z||_1   = ", np.linalg.norm(Xf[2], 1))

    xexpected = np.ones(nx)
    xexpected[1]  = 1. / (A[1,1] / A[0,0] + nx - 1.)
    xexpected[2:] = xexpected[1]
    xexpected[0]  = (A[1,1] / A[0,0]) * xexpected[1]
    lamexpected   = -A[1,1] * xexpected[1]
    print("computed optimal solution = ", Xf[0][:nx])
    print("expected optimal solution = ", xexpected)
    print("computed equality constraint multiplier = ", Xf[1][-1])
    print("expected equality constraint multiplier = ", lamexpected)
    print("computed bound-constraint multipliers   = ", Xf[2])
    Es = [np.array([E[i][j] for i in range(len(E))]) for j in range(len(E[0]))]
    titles = ["optimality", "feasibility", "complementarity"]
    for j in range(1,4):
        plt.plot(Es[j], "*")
        plt.title(titles[j-1] + " history")
        plt.yscale('log')
        plt.show()
