import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

import dolfin as dl
dl.parameters['reorder_dofs_serial']= False
import matplotlib.pyplot as plt
from hippylib import nb

class interior_pt:
    def __init__(self, problem, Vhs):
        self.problem = problem
        self.Vhs     = Vhs
        self.ml      = problem.ml
        
        # -------- dimensions of various function spaces
        self.n1   = Vhs[0].sub(0).dim()
        self.n    = Vhs[0].dim() # dimension of primal (state-parameter) state
        self.m    = Vhs[1].dim() # dimension of PDE equality constraint, adjoint
        self.N    = self.n + self.m + Vhs[2].dim() # dimension of discretized state,paraneter, adjoint and bound-constraint multiplier

        self.x    = dl.Function(Vhs[0])  # 0, primal
        self.lam  = dl.Function(Vhs[1])  # 1, dual (c(x) = 0)
        self.z    = dl.Function(Vhs[2])  # 6, (x >= 0)
        
        # -------- construct index set 
        self.idx0 = 0
        self.idx1 = self.idx0 + self.n
        self.idx2 = self.idx1 + self.m
        self.idx3 = self.idx2 + Vhs[2].dim()
        
        self.it   = 0
        self.X    = [dl.Function(Vhs[0]), dl.Function(Vhs[1]), dl.Function(Vhs[2])]
        self.Xhat = [dl.Function(Vhs[0]), dl.Function(Vhs[1]), dl.Function(Vhs[2])]
        

        # -------- parameters for interior point -------

        # ensure that large Lagrange multipliers
        # do not corrupt E
        self.smax = 1.e2
        self.sc   = 1.e2
        self.sd   = 1.e2
        # control deviation from primal dual Hessian
        self.kSig = 1.e10

        # control the minimum size of the fraction to boundary parameter
        self.tau_min = 5.e-1

        # backtracking constant
        self.eta = 1.e-4

        # control the allowable violation of the equality constraints
        self.theta_min = 1.e-4

        # constants in line-step A-5.4
        self.delta  = 1.#1.e-2
        self.stheta = 1. + 1.e-2
        self.sphi   = 1.

        # control the rate at which the penalty parameter is decreased
        self.kmu = 0.8
        self.thetamu = 1.5

        # the Filter
        self.F = []

        # maximum constraint violation
        self.thetamax = 1.e6

        # data for second order correction
        self.ck_soc = 0.
        self.k_soc  = 0.5

        # equation (18)
        self.gtheta = 1.e-5#0.5
        self.gphi   = 1.e-5#0.5

        #
        self.keps = 10.

        
    def initialize(self, X):
        self.X = X
    
    def split(self, X):
        return X[:]

    """
    Setup the IP-Newton system as needed in equations (11) and (26) of Wachter Biegler
    """
    def formH(self, X, mu):
        Wk = self.problem.DxxL(X)
        x, lam, z = X[:]
        """
        By eliminating the bound-constraint multiplier, there is an additional term
        that is added to the IP-Newton system matrix, that is dHmm
        """
        dHmm = self.problem.M.dot(sps.diags(z.vector().get_local() \
                             / (x.vector().get_local()[self.n1:] - self.ml.vector().get_local()[:])))
 
        r4    = z.vector().get_local() * (x.vector().get_local()[self.n1:] - self.ml.vector().get_local()[:]) - mu
        """
        By eliminating the bound-constraint multiplier, the rhs of the IP-Newton system is altered by dr
        """
        dr    = r4 / (x.vector().get_local()[self.n1:] - self.ml.vector().get_local()[:])
        dr    = self.problem.M.dot(dr)
        Dxc   = self.problem.Dxc(x)
        AT    = Dxc[:self.n1,:]
        CT    = Dxc[self.n1:,:]
        A     = AT.transpose()
        C     = CT.transpose()
        H   = sps.bmat([[Wk[:self.n1, :self.n1], Wk[:self.n1, self.n1:], AT], \
                        [Wk[self.n1:, :self.n1], Wk[self.n1:, self.n1:]+dHmm, CT],\
                        [A, C, None]], format="csr")
        return H, dr, r4
    """ 
    determine the solution
    of the perturbed KKT conditions
    no inertia-correction!
    """
    def pKKT_solve(self, X, mu):
        x, p, z = X[:]
        xhat      = dl.Function(self.Vhs[0])
        phat      = dl.Function(self.Vhs[1])
        zhat      = dl.Function(self.Vhs[2])

        Hk, dr, r4 = self.formH(X, mu)

        rk = np.zeros(self.n+self.m)
        rk[:self.n] = self.problem.DxL(X)
        rk[self.n:] = self.problem.c(x)
        rk[self.n1:self.n] += dr[:]
        
        solver = spla.splu(Hk)
        sol = solver.solve(-1.*rk)
        #sol = spla.spsolve(Hk, -1.*rk)
        print("KKT residual = {0:1.2e}".format(np.linalg.norm(Hk.dot(sol) + rk)))
        xhat.vector().set_local(sol[:self.n])
        phat.vector().set_local(sol[self.n:])
        
        zhatnp = -1.*(r4 + z.vector().get_local()[:] * xhat.vector().get_local()[self.n1:]) / \
                     (x.vector().get_local()[self.n1:] - self.ml.vector()[:])
        zhat.vector().set_local(zhatnp)
        return xhat, phat, zhat
    def soc_solve(self, X, mu):
        x, p, z = X[:]
                
        xhat      = dl.Function(self.Vhs[0])
        phat      = dl.Function(self.Vhs[1])
        zhat      = dl.Function(self.Vhs[2])        

        # set up reduced system
        Hk, dr, r4 = self.formH(X, mu)
        rk = np.zeros(self.n+self.m)
        rk[:self.n] = self.problem.DxL(X)
        rk[self.n:] = self.ck_soc[:]
        rk[self.n1:self.n] += dr[:]
        solver = spla.splu(Hk)
        sol = solver.solve(-1.*rk)
        #sol = spla.spsolve(Hk, -1.*rk)
        print("KKT residual = {0:1.2e}".format(np.linalg.norm(Hk.dot(sol) + rk)))
        xhat.vector().set_local(sol[:self.n])
        phat.vector().set_local(sol[self.n:])
        
        
        zhatnp = -1.*(r4 + z.vector()[:] * xhat.vector()[self.n1:]) / \
                     (x.vector()[self.n1:] - self.ml.vector()[:])
        zhat.vector().set_local(zhatnp)
        return xhat, phat, zhat

    def filter_check(self, theta, phi):
        in_filter_region = False
        for i in range(len(self.F)):
            if theta >= (self.F)[i][0] and phi >= (self.F)[i][1]:
                in_filter_region = True
        return in_filter_region

    def line_search(self, X, Xhat, mu, tau):
        # indicator of Filter Line Search success
        linesearch_success = True
        x   = dl.Function(self.Vhs[0])
        x.assign(X[0])
        lam = dl.Function(self.Vhs[1])
        lam.assign(X[1])
        z   = dl.Function(self.Vhs[2])
        z.assign(X[2])
        xhat   = dl.Function(self.Vhs[0])
        xhat.assign(Xhat[0])
        lamhat = dl.Function(self.Vhs[1])
        lamhat.assign(Xhat[1])
        zhat   = dl.Function(self.Vhs[2])
        zhat.assign(Xhat[2])        
        
        # BEGIN A-5.1.
        alpha_max = 1.
        alphaz   = 1.
        for i in range(self.n1, self.n):
            xhati = xhat.vector().vec().getValue(i)
            xi    = x.vector().vec().getValue(i)
            mli   = self.ml.vector().vec().getValue(i-self.n1)
            zi    = z.vector().vec().getValue(i-self.n1)
            zhati = z.vector().vec().getValue(i-self.n1)
            if abs(xhati) > 1.e-14:
                alpha_tmp = -1.*tau* (xi - mli) / xhati
                if 0. < alpha_tmp and alpha_tmp < 1.:
                    alpha_max = min(alpha_max, alpha_tmp)
            if abs(zhati) > 1.e-14:
                alphaz_tmp = -1.*tau*zi / zhati
                if 0. < alphaz_tmp and alphaz_tmp < 1.:
                    alphaz = min(alphaz, alphaz_tmp)
        alpha = alpha_max
        print("alpha = {0:1.3e}".format(alpha))
        print("alphaz = {0:1.3e}".format(alphaz))
        # END A-5.1.

        max_backtrack     = 20
        it_backtrack      = 0
        
        xtrial = dl.Function(self.Vhs[0])
        x_soc  = dl.Function(self.Vhs[0])
        # A-5.2--> A-5.10 --> A-5.2 --> ... loop
        while it_backtrack < max_backtrack:
            # ------ A-5.2.
            xtrial.assign(x)
            xtrial.vector().axpy(alpha, xhat.vector())
            # ------
            
            # ------ A.5.3
            # if not in filter region go to A.5.4, potential exit and then A.5.5
            # else go to A.5.5
            Dxphi_xhat = np.dot(self.problem.Dxphi(x, mu), xhat.vector().get_local())
            print("in filter region? ", self.filter_check(self.problem.theta(xtrial), self.problem.phi(xtrial, mu)))
            if not self.filter_check(self.problem.theta(xtrial), self.problem.phi(xtrial, mu)):
                # ------ A.5.4: Check sufficient decrease
                # CASE I
                print("A.5.4")
                print("theta(x) = {0:1.2e}".format(self.problem.theta(x)))
                print("descent direction? ", Dxphi_xhat < 0.)
                print("theta(x) < theta_min? ", self.problem.theta(x) <= self.theta_min)
                if self.problem.theta(x) <= self.theta_min and Dxphi_xhat < 0. \
                        and alpha*(-1.*Dxphi_xhat)**self.sphi > \
                            self.delta*(self.problem.theta(x))**self.stheta:
                        if self.problem.phi(xtrial, mu) <= self.problem.phi(x, mu) + \
                           self.eta*alpha*Dxphi_xhat:
                               return xtrial, xhat, lamhat, alpha, alphaz, True # return the trial step
                # CASE II
                else:
                    print("phi(xtrial) <= phi(x) - gphi theta(x) ? ", \
                          self.problem.phi(xtrial, mu) <= self.problem.phi(x,mu) - self.gphi *self.problem.theta(x))
                    if self.problem.theta(xtrial) <= (1.-self.gtheta)*self.problem.theta(x) or \
                        self.problem.phi(xtrial, mu) <= self.problem.phi(x, mu) - \
                        self.gphi*self.problem.theta(x):
                            # ACCEPT THE TRIAL STEP
                            return xtrial, xhat, lamhat, alpha, alphaz, True
            # A.5.5: Initialize the second order correction
            print("A.5.5")
            print("theta(xtrial) = {0:1.2e}, theta(x) = {1:1.2e}".format(self.problem.theta(xtrial), self.problem.theta(x)))
            if not (it_backtrack > 0 or self.problem.theta(xtrial) < self.problem.theta(x)):
                p = 1
                maxp = 4
                # equation (27)
                self.ck_soc = alpha*self.problem.c(x) + self.problem.c(xtrial)
                theta_old_sc = self.problem.theta(xtrial)
                # A-5.6--> A-5.9 --> A-5.6 --> ... loop
                while p < maxp:
                    print("A.5.6")
                    # A-5.6: Compute the second order correction
                    xhat_soc, lamhat_soc, zhat_soc = self.soc_solve(X, mu)
                    alpha_soc   = 1.
                    """
                    Since x is a stacked vector of the state and parameter,
                    we access the components corresponding to the parameter m
                    via the [n1, n1+1,...n) indices of x
                    """
                    for i in range(self.n1, self.n):
                        xhat_soci = xhat_soc.vector().vec().getValue(i)
                        xi        = x.vector().vec().getValue(i)
                        mli       = self.ml.vector().vec().getValue(i - self.n1)
                        if abs(xhat_soci) > 0.:
                            alpha_tmp = -1.*tau* (xi - mli) / xhat_soci
                            if 0. < alpha_tmp and alpha_tmp < 1.:
                                alpha_soc = min(alpha_soc, alpha_tmp)
                    print("alpha_soc = {0:1.3e}".format(alpha_soc))
                    x_soc.assign(x)
                    x_soc.vector().axpy(alpha_soc, xhat_soc.vector())

                    # end A-5.6 
                    # A-5.7: Check acceptability to the filter
                    # If in filter region, exit A.5.6 --> A.5.9 loop to return to A.5.10
                    if self.filter_check(self.problem.theta(x_soc), self.problem.phi(x_soc, mu)):
                        break
                    else:
                        print("not acceptable to the filter")
                    # otherwise continue in loop to step A.5.8
                    # A-5.8: Check sufficient decrease with respect to the current iterate (in SOC)
                    # CASE I
                    Dxphi_xhat_soc = np.dot(self.problem.Dxphi(x, mu), xhat_soc.vector().get_local())
                    if self.problem.theta(x) < self.theta_min and \
                            Dxphi_xhat_soc < 0. and \
                            alpha_max*(-1.*Dxphi_xhat_soc)**self.sphi > \
                            self.delta*(self.problem.theta(x))**self.stheta:
                        if self.problem.phi(x_soc, mu) < self.problem.phi(x, mu) + \
                                self.eta*alpha_soc*Dxphi_xhat_soc:
                            # ACCEPT THE SOC TRIAL STEP
                            print("backtracking successful")
                            linesearch_success = True
                            return x_soc, xhat_soc, lamhat_soc, alpha_soc, alphaz, linesearch_success
                    # CASE II
                    else:
                        if self.problem.theta(x_soc) <= (1.-self.gtheta)*self.problem.theta(x) or \
                                self.problem.phi(x_soc, mu) <= \
                                     self.problem.phi(x, mu) - self.gphi*self.problem.theta(x):
                            # ACCEPT THE SOC TRIAL STEP
                            return x_soc, xhat_soc, lamhat_soc, alpha_soc, alphaz, True
                    # A-5.9: Next second order correction
                    if p == maxp or self.problem.theta(x_soc) > self.k_soc*theta_old_sc:
                        break
                    else:
                        p += 1
                        self.ck_soc = alpha_soc*self.ck_soc + self.problem.c(x_soc)
                        theta_old_sc = self.problem.theta(x_soc)
            # A-5.10: Choose a new trial step size
            print("A-5.10")
            alpha *= 0.5
            it_backtrack += 1                        
        if it_backtrack == max_backtrack:
            linesearch_success = False
            print("FILTER LINE SEARCH FAILURE -- should continue to feasibility restoration A-9")
        return xtrial, xhat, lamhat, alpha, alphaz, linesearch_success

    def project_z(self, X, mu):
        x, lam, z = X[:]
        """
        Make copies of numpy 'np' copies of z, m, and ml
        Return a copy of the nodal DOF values of the projected multiplier z
        according to (16) of Wachter, Biegler
        """
        znp = z.vector().get_local()[:]
        mnp = x.sub(1, deepcopy=True).vector().get_local()[:]
        mlnp = self.ml.vector().get_local()[:]
        for i in range(self.n-self.n1):
            if abs(mnp[i] - mlnp[i]) > 1.e-14:
                znp[i] = max(min(znp[i], self.kSig*mu / (mnp[i] - mlnp[i])), mu / (self.kSig*(mnp[i]-mlnp[i])))
        return znp

    """
    Determine a minimizer of the objective, utilizing the IP method with filter-line search
    """
    def solve(self, tol, max_iterations, mu0):
        mu  = mu0
        self.it = 0
        self.X[0].vector().set_local(np.ones(self.Vhs[0].dim()))
        self.X[1].vector().set_local(np.ones(self.Vhs[1].dim()))
        self.X[2].vector().set_local(np.ones(self.Vhs[2].dim()))
        theta0 = self.problem.theta(self.X[0])
        self.theta_min = 1.e-4*max(1., theta0)
        self.thetamax  = 1.e4*max(1., theta0)
        Es = []
        Mus = []
        while self.it < max_iterations:
            print("it = {0:d}".format(self.it))
            tau = max(self.tau_min, 1. - mu)
            # initialize the filter for the given value of mu
            self.F = [[self.thetamax, -1.*np.inf]]
            
            # -------- Check for convergence to global problem
            Es.append(self.problem.E(self.X, 0., self.smax)[:])
            Mus.append(mu)
            if self.problem.E(self.X, 0., self.smax)[0] < tol:
                print("solved interior point problem")
                break
            # -------- Check for convergence of the barrier problem
            # A-3
            if self.problem.E(self.X, mu, self.smax)[0] < self.keps*mu:
                print("solved barrier problem (mu = {0:1.3e})".format(mu))
                # decrease barrier parameter 
                mu  = max(tol/10., min(self.kmu*mu, mu**self.thetamu))
                tau = max(self.tau_min, 1. - mu)
                # Re-initialize the filter
                self.F = [[self.thetamax, -1.*np.inf]]
            # -------- Obtain search direction from perturbed KKT system
            # A-4
            self.Xhat[0], self.Xhat[1], \
                self.Xhat[2] = self.pKKT_solve(self.X, mu)
            # -------- Determine appropriate step length
            # A-5 Backtracking line-search
            x, xhat, lamhat, alpha, alphaz, linesearch_success = self.line_search(self.X, self.Xhat, mu, tau)
            print("linesearch success? ", linesearch_success)
            # check if line search was successful if not we restore feasibility
            if linesearch_success:
                # check if either (19) or (20) do not hold
                #filter_augment = False
                Dx_phi_xhat = np.dot(self.problem.Dxphi(self.X[0], mu), xhat.vector()[:])
                if not (Dx_phi_xhat < 0. or alpha*(-Dx_phi_xhat)**self.sphi > \
                    self.delta*(self.problem.theta(self.X[0]))**self.stheta) \
                    or not (self.problem.phi(x, mu) <= self.problem.phi(self.X[0], mu) + \
                    self.eta*alpha*Dx_phi_xhat):
                    self.F.append([(1.-self.gtheta)*self.problem.theta(self.X[0]),\
                        self.problem.phi(self.X[0], mu) - self.gphi*self.problem.theta(self.X[0])])
                # A-6: Accept trial point
                self.X[0].assign(x)
                self.X[1].vector().axpy(alpha, lamhat.vector())
                self.X[2].vector().axpy(alphaz, self.Xhat[2].vector())
                self.X[2].vector().set_local(self.project_z(self.X, mu))
                nb.plot(self.X[0].sub(0, deepcopy=True))
                plt.show()
                nb.plot(self.X[0].sub(1, deepcopy=True))
                plt.show()
            else:
                print("feasibility restoration")
                if self.problem.theta(self.X[0]) < 1.e-10:
                    print("ATTEMPTING TO RESTORE FEASIBILITY WHEN ITERATE IS FEASIBLE")
                # A-9 
                # Augment the filter
                self.F.append([(1.-self.gtheta)*self.problem.theta(self.X[0]),\
                     self.problem.phi(self.X[0], mu) - self.gphi*self.problem.theta(self.X[0])])
                # restore feasibility by freezing the parameter and solving for the state
                # so that the PDE is satisfied
                self.X[0].vector().vec()[:self.n1] = self.problem.restore_feasibility(self.X[0])
            self.it += 1
        return self.X, mu, Es, Mus
