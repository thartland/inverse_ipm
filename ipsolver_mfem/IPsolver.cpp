#include "mfem.hpp"
#include "IPsolver.hpp"
#include "problems.hpp"
#include <fstream>
#include <iostream>
#include <cstdlib>

using namespace std;
using namespace mfem;



//interiorPtSolver::interiorPtSolver(inverseDiffusion * Problem) : problem(Problem)
interiorPtSolver::interiorPtSolver(optimizationProblem * Problem) : problem(Problem), block_offsetsumlz(5),
 block_offsetsuml(4), Hk(NULL), Hk1(NULL), Hk2(NULL), Wk(NULL), Wmm(NULL),
 Wkmm(NULL), Hk2mm(NULL), Hlogbar(NULL), Jk(NULL), JkT(NULL)
{
  sMax     = 1.e2;
  kSig     = 1.e10;   // control deviation from primal Hessian
  tauMin   = 0.8;     // control rate at which iterates can approach the boundary
  eta      = 1.e-4;   // backtracking constant
  thetaMin = 1.e-4;   // allowed violation of the equality constraints

  // constants in line-step A-5.4
  delta    = 1.0;
  sTheta   = 1.1;
  sPhi     = 2.3;

  // control the rate at which the penalty parameter is decreased
  kMu     = 0.2;
  thetaMu = 1.5;

  // TO DO -- include the filter

  thetaMax = 1.e6; // maximum constraint violation
  // data for the second order correction
  kSoc     = 0.99;

  // equation (18)
  gTheta = 1.e-5;
  gPhi   = 1.e-5;

  kEps   = 1.e1;

  dimU = problem->getdimU();
  dimM = problem->getdimM();
  ckSoc.SetSize(dimU);
  block_offsetsu  = problem->getblock_offsetsu();
  block_offsetsx  = problem->getblock_offsetsx();

  block_offsetsumlz[0] = 0;
  block_offsetsumlz[1] = dimU;
  block_offsetsumlz[2] = dimM;
  block_offsetsumlz[3] = dimU;
  block_offsetsumlz[4] = dimM;
  block_offsetsumlz.PartialSum();

  for(int i = 0; i < block_offsetsuml.Size(); i++)
  {
    block_offsetsuml[i] = block_offsetsumlz[i];
  }

  // lumped mass matrix
  Mmlumped.SetSize(dimM); Mmlumped = problem->getMmlumped();
  // lower-bound for the inequality constraint m >= ml
  ml.SetSize(dimM); ml = problem->getml();

  converged = false;
  outputVTK = false;

  
  MyRank = problem->GetMyRank();
  MyComm = problem->GetComm();
  iAmRoot = MyRank == 0 ? true : false;
}

double interiorPtSolver::computeAlphaMax(Vector &x, Vector &xl, Vector &xhat, double tau)
{
  double alphaMaxloc = 1.0;
  double alphaTmp;
  for(int i = 0; i < x.Size(); i++)
  {
    if( xhat(i) < 0. )
    {
      alphaTmp = -1. * tau * (x(i) - xl(i)) / xhat(i);
      alphaMaxloc = min(alphaMaxloc, alphaTmp);
    } 
  }

  // alphaMaxloc is the local maximum step size which is
  // distinct on each MPI process. Need to compute
  // the global maximum step size 
  double alphaMaxglb;
  MPI_Allreduce(&alphaMaxloc, &alphaMaxglb, 1, MPI_DOUBLE, MPI_MIN, MyComm);
  return alphaMaxglb;
}

double interiorPtSolver::computeAlphaMax(Vector &x, Vector &xhat, double tau)
{
  Vector zero(x.Size()); zero = 0.0;
  return computeAlphaMax(x, zero, xhat, tau);
}


void interiorPtSolver::solve(BlockVector &X0, BlockVector &Xf, double tol, int maxOptIt, double mu0)
{
  converged = false;
  if(iAmRoot)
  {
    IPNewtonSolveTimes.open("IPNewtonSolveTimes.dat", ios::out | ios::trunc);
    IPNewtonSysFormTimes.open("IPNewtonSysFormTimes.dat", ios::out | ios::trunc);
  }
  // For visualizing the IP-Newton iterates
  ParGridFunction u_gf(problem->Vhu);
  ParGridFunction m_gf(problem->Vhm);
  ParGridFunction l_gf(problem->Vhu);
  ParGridFunction zl_gf(problem->Vhu);
  ParGridFunction mml_gf(problem->Vhm); // m minus it's lower bound
  ParaViewDataCollection paraview_dc("IPiterates", problem->Vhu->GetMesh());
  if(outputVTK)
  {
    paraview_dc.SetPrefixPath("ParaView");
    paraview_dc.SetLevelsOfDetail(2);
    paraview_dc.SetDataFormat(VTKFormat::BINARY);
    paraview_dc.SetHighOrderOutput(false);
    paraview_dc.RegisterField("u", &u_gf);
    paraview_dc.RegisterField("m", &m_gf);
    paraview_dc.RegisterField("m - ml", &mml_gf);
    paraview_dc.RegisterField("l", &l_gf);
    paraview_dc.RegisterField("zl", &zl_gf);
  }

  BlockVector x(block_offsetsx), xhat(block_offsetsx); x = 0.0; xhat = 0.0;
  BlockVector X(block_offsetsumlz), Xhat(block_offsetsumlz); X = 0.0; Xhat = 0.0;
  BlockVector Xhatuml(block_offsetsuml); Xhatuml = 0.0;
  Vector l(dimU);         l = 0.0;
  Vector zl(dimM);       zl = 0.0;
  Vector zlhat(dimM); zlhat = 0.0;

  // initialize component data from X0 input
  x.GetBlock(0).Set(1.0, X0.GetBlock(0));
  x.GetBlock(1).Set(1.0, X0.GetBlock(1));
  l.Set(1.0 , X0.GetBlock(2));
  zl.Set(1.0, X0.GetBlock(3));
  for(int i = 0; i < dimM; i++)
  {
    zl(i) = 1.e1 * mu0 / (x(i+dimU) - ml(i));
  }

  /* set theta0 = theta(x0)
   *     thetaMin
   *     thetaMax
   * when theta(xk) < thetaMin and the switching condition holds
   * then we ask for the Armijo sufficient decrease of the barrier
   * objective to be satisfied, in order to accept the trial step length alphakl
   * 
   * thetaMax controls how the filter is initialized for each log-barrier subproblem
   * F0 = {(th, phi) s.t. th > thetaMax}
   * that is the filter does not allow for iterates where the constraint violation
   * is larger than that of thetaMax
   */
  double theta0 = problem->theta(x);
  thetaMin = 1.e-4 * max(1.0, theta0);
  thetaMax = 1.e8  * thetaMin; // 1.e4 * max(1.0, theta0)


  double zlnorm, Eeval, mu, maxBarrierSolves, Eevalmu0;
  
  mu = mu0;
  maxBarrierSolves = 10;

  std::ofstream convergenceData;

  convergenceData.open("./outerOptconvergenceData.dat", ios::out | ios::trunc);

  for(jOpt = 0; jOpt < maxOptIt; jOpt++)
  {
    if(iAmRoot) { cout << "interior-point solve step " << jOpt << endl; }
    if(outputVTK)
    {
      u_gf.SetFromTrueDofs(x.GetBlock(0));
      m_gf.SetFromTrueDofs(x.GetBlock(1));
      x.GetBlock(1).Add(-1.0, ml);
      mml_gf.SetFromTrueDofs(x.GetBlock(1));
      x.GetBlock(1).Add(1.0, ml);
      l_gf.SetFromTrueDofs(l);
      zl_gf.SetFromTrueDofs(zl);
      paraview_dc.SetTime(double(jOpt));
      paraview_dc.SetCycle(jOpt);
      paraview_dc.Save();
    }

    // A-2. Check convergence of overall optimization problem
    //Eevalmu0 = problem->E(x, l, zl, 0.0, sMax);
    Eevalmu0 = E(x, l, zl, 0.0);
    convergenceData << setprecision(30) << Eevalmu0 << "\n";
    if(Eevalmu0 < tol)
    {
      convergenceData.close();
      IPNewtonSolveTimes.close();
      IPNewtonSysFormTimes.close();
      converged = true;
      if(iAmRoot) { cout << "solved interior point problem!!!!!!!!\n"; }
      break;
    }
    
    if(jOpt > 0) { maxBarrierSolves = 1; }
    
    for(int i = 0; i < maxBarrierSolves; i++)
    {
      Eeval = E(x, l, zl, mu);
      //Eeval = problem->E(x, l, zl, mu, sMax);
      if(iAmRoot) { cout << "E = " << Eeval << endl;}
      // A-3. Check convergence of the barrier subproblem
      if(Eeval < kEps * mu)
      {
        if(iAmRoot) { cout << "solved barrier subproblem :), for mu = " << mu << endl; }
        // A-3.1. Recompute the barrier parameter
        mu  = max(tol / 10., min(kMu * mu, pow(mu, thetaMu)));
        // A-3.2. Re-initialize the filter
        F1.DeleteAll();
        F2.DeleteAll();
      }
      else
      {
        break;
      }
    }
    
    // A-4. Compute the search direction
    // solve for (uhat, mhat, lhat)
    if(iAmRoot) { cout << "\n** A-4. IP-Newton solve **\n"; }
    zlhat = 0.0; Xhatuml = 0.0;
    pKKTSolve(x, l, zl, zlhat, Xhatuml, mu, false); 

    // assign data stack, Xhat = (uhat, mhat, lhat, zlhat)
    Xhat = 0.0;
    for(int i = 0; i < 3; i++)
    {
      Xhat.GetBlock(i).Set(1.0, Xhatuml.GetBlock(i));
    }
    Xhat.GetBlock(3).Set(1.0, zlhat);

    // assign data stack, X = (u, m, l, zl)
    X = 0.0;
    X.GetBlock(0).Set(1.0, x.GetBlock(0));
    X.GetBlock(1).Set(1.0, x.GetBlock(1));
    X.GetBlock(2).Set(1.0, l);
    X.GetBlock(3).Set(1.0, zl);
    
    
    // A-5. Backtracking line search.
    if(iAmRoot) { cout << "\n** A-5. Linesearch **\n"; }
    lineSearch(X, Xhat, mu);

    if(lineSearchSuccess)
    {
      if(iAmRoot) { cout << "lineSearch successful :)\n"; }
      if(!switchCondition || !sufficientDecrease)
      {
        F1.Append( (1. - gTheta) * thx0);
        F2.Append( phx0 - gPhi * thx0);
      }
      // ----- A-6: Accept the trial point
      // print info regarding zl...
      x.GetBlock(0).Add(alpha, Xhat.GetBlock(0));
      x.GetBlock(1).Add(alpha, Xhat.GetBlock(1));
      l.Add(alpha, Xhat.GetBlock(2));
      zl.Add(alphaz, Xhat.GetBlock(3));
      projectZ(x, zl, mu);
    }
    else
    {
      if(iAmRoot)
      {
        cout << "lineSearch not successful :(\n";
        cout << "attempting feasibility restoration with theta = " << thx0 << endl;
      }
      //cout << "feasibility restoration!!! :( :( :(\n";
      problem->feasibilityRestoration(x, 1.e-12);
      //      break;
    }
    //
    if(jOpt + 1 == maxOptIt && iAmRoot) 
    { 
      cout << "maximum optimization iterations :(\n";
      IPNewtonSolveTimes.close();
      IPNewtonSysFormTimes.close();
      convergenceData.close();
    }
  }
  // done with optimization routine, just reassign data to Xf reference so
  // that the application code has access to the optimal point
  Xf = 0.0;
  Xf.GetBlock(0).Set(1.0, x.GetBlock(0));
  Xf.GetBlock(1).Set(1.0, x.GetBlock(1));
  Xf.GetBlock(2).Set(1.0, l);
  Xf.GetBlock(3).Set(1.0, zl);
}

void interiorPtSolver::formH(BlockVector & x, Vector & l, Vector &zl , double mu, BlockOperator &Ak)
{
  if(jOpt > 0) { delete Hk1; delete Hk2; }
  Hk1 = new BlockOperator(block_offsetsx, block_offsetsx);
  Hk2 = new BlockOperator(block_offsetsx, block_offsetsx);
  problem->DxxL(x, l, Hk1, Hk2);
  
  // for now we are guaranteed that Hk1 has a nonzero 2,2 block due to the regularization term
  Wkmm = dynamic_cast<HypreParMatrix *>(&(Hk1->GetBlock(1,1)));
  
  //  WARNING!!! This will likely not work in the future
  if(!(Hk2->IsZeroBlock(1,1)))
  {
    Hk2mm = dynamic_cast<HypreParMatrix *>(&(Hk2->GetBlock(1,1)));
    Wkmm->Add(1.0, *Hk2mm);
  }

  // construct the Hessian of the log-barrier term
  Vector diag(dimM); diag = 0.0;
  for(int ii = 0; ii < dimM; ii++)
  {
    diag(ii) = Mmlumped(ii) * zl(ii) / (x(ii+dimU) - ml(ii));
  }

  // as a SparseMatrix  
  if(jOpt > 0) { delete diagSparse; delete Hlogbar; Wmm; Wk; Hk; Jk; JkT;}
  diagSparse = new SparseMatrix(diag);
  // as a HypreParMatrix
  
  Hlogbar = new HypreParMatrix(MyComm, Wkmm->GetGlobalNumRows(), Wkmm->GetRowStarts(), diagSparse);

  /*
     Wk_uu = Hk_uu, Wk_um = Hk_um
     Wk_mu = Hk_mu, Wk_mm = Hk_mm + Hlogbar   
  */


  /* Wkmm is specially constructed to be a HypreParMatrix
   * other blocks of Wk, can be generic operators, since
   * those other blocks (Wk_uu, Wk_um, Wk_mu) are made up
   * of contributions from the Hessian (w.r.t. x) of
   * l^T c(x) (PDE constraint) and f(x) (objective function)
   * we design a generic BlockSumOperator that maintains the
   * block structure the operators which it sums and then 
   * we pass the desired blocks to Wk 
   */
  Wmm = ParAdd(Wkmm, Hlogbar);
  Wk = new BlockOperator(block_offsetsx, block_offsetsx);
  Wk->SetBlock(1, 1, Wmm);

  Hk = new SumOperatorBlock(Hk1, Hk2);
  for(int i = 0; i < Wk->NumRowBlocks(); i++)
  {
    for(int j = 0; j < Wk->NumColBlocks(); j++)
    {
      if(i == 1 && j == 1)
      {
        continue;
      }
      else
      {
        if(!(Hk->IsZeroBlock(i, j)))
        {
          Wk->SetBlock(i, j, &(Hk->GetBlock(i, j)));
        }
      }
    }
  }
  /*
   * prepare Jacobian and Jacobian transpose operators
   * the off-diagonal blocks of the IP-Newton system matrix A
   */
  Jk = new BlockOperator(block_offsetsu, block_offsetsx);
  problem->Dxc(x, Jk);

  JkT = new TransposeOperatorBlock(Jk);
  
  
  /* the IP-Newton system matrix
     Ak = [[ Wk    Jk^T]
           [ Jk      0 ]]
     but with respect to the 3x3 block partitioning
     Ak = [[Wk_uu  Wk_um   Ju^T]
           [Wk_mu  Wk_mm   Jm^T]
           [Ju     Jm       0  ]]
  */
  constructBlockOpFromBlockOp(Wk, JkT, Jk, Ak);
}


// perturbed KKT system solve
// determine the search direction
void interiorPtSolver::pKKTSolve(BlockVector &x, Vector &l, Vector &zl, Vector &zlhat, BlockVector &Xhat, double mu, bool socSolve)
{
  BlockVector gradphi(block_offsetsx), gradcTp(block_offsetsx); gradphi = 0.0; gradcTp = 0.0;
  problem->Dxphi(x, mu, &gradphi);
  problem->DxcTp(x, l, &gradcTp);

  Vector cx(dimU); cx = 0.0;  
  problem->c(x, &cx);
  BlockVector b(block_offsetsuml); b = 0.0;
  for(int ii = 0; ii < 2; ii++)
  {
    b.GetBlock(ii).Set(1.0, gradphi.GetBlock(ii));
    b.GetBlock(ii).Add(1.0, gradcTp.GetBlock(ii));
  }
  b.GetBlock(2).Set(1.0, cx);

  BlockOperator A(block_offsetsuml, block_offsetsuml);
  
  IPNewtonSysFormStopWatch.Clear();
  IPNewtonSysFormStopWatch.Start();
  formH(x, l, zl, mu, A);
  IPNewtonSysFormStopWatch.Stop();
  if(iAmRoot)
  {
    IPNewtonSysFormTimes << setprecision(30) << IPNewtonSysFormStopWatch.RealTime() << endl;
  }


  string saveFile = "GMRESdata/res" + to_string(jOpt) + ".dat";
  SaveDataSolverMonitor mymonitor(saveFile);
  GMRESSolver Asolver(MyComm);
  double precTol = 1.e-13;
  GSPreconditioner Aprec(&A, precTol);
  Asolver.SetOperator(A);
  Asolver.SetPreconditioner(Aprec);
  Asolver.SetRelTol(1.e-10);
  Asolver.SetAbsTol(1.e-14);
  Asolver.SetMaxIter(200);
  Asolver.SetKDim(50);
  Asolver.SetPrintLevel(2);
  Asolver.SetMonitor(mymonitor);
  b *= -1.0; Xhat = 0.0;



  if(iAmRoot) { cout << "IP-Newton solve, "; }
  IPNewtonSolveStopWatch.Clear();
  IPNewtonSolveStopWatch.Start();
  Asolver.Mult(b, Xhat);
  IPNewtonSolveStopWatch.Stop();
  if(iAmRoot)
  {
    IPNewtonSolveTimes << setprecision(30) << IPNewtonSolveStopWatch.RealTime() << endl;
  }
  // output the CGAMG solve data for each application of the GS preconditioner
  Aprec.saveA02applies("GMRESdata/A02applies" + to_string(jOpt) + ".dat");
  Aprec.saveA20applies("GMRESdata/A20applies" + to_string(jOpt) + ".dat");
  Aprec.saveA11applies("GMRESdata/A11applies" + to_string(jOpt) + ".dat");

  for(int ii = 0; ii < dimM; ii++)
  {
    zlhat(ii) = -1.*(zl(ii) + (zl(ii) * Xhat(ii + dimU) - mu) / (x(ii + dimU) - ml(ii)) );
  }
}

// here Xhat, X will be BlockVectors w.r.t. the 4 partitioning X = (u, m, l, zl)

void interiorPtSolver::lineSearch(BlockVector& X0, BlockVector& Xhat, double mu)
{
  double tau  = max(tauMin, 1.0 - mu);
  Vector u0   = X0.GetBlock(0);
  Vector m0   = X0.GetBlock(1);
  Vector l0   = X0.GetBlock(2);
  Vector z0   = X0.GetBlock(3);
  Vector uhat = Xhat.GetBlock(0);
  Vector mhat = Xhat.GetBlock(1);
  Vector lhat = Xhat.GetBlock(2);
  Vector zhat = Xhat.GetBlock(3);
  double alphaMax  = computeAlphaMax(m0, ml, mhat, tau);
  double alphaMaxz = computeAlphaMax(z0, zhat, tau);
  alphaz = alphaMaxz;

  BlockVector x0(block_offsetsx); x0 = 0.0;
  x0.GetBlock(0).Set(1.0, u0);
  x0.GetBlock(1).Set(1.0, m0);
  
  BlockVector xhat(block_offsetsx); xhat = 0.0;
  xhat.GetBlock(0).Set(1.0, uhat);
  xhat.GetBlock(1).Set(1.0, mhat);
  
  BlockVector xtrial(block_offsetsx); xtrial = 0.0;
  BlockVector Dxphi0(block_offsetsx); Dxphi0 = 0.0;
  int maxBacktrack = 20;
  alpha = alphaMax;


  Vector ck0(dimU); ck0 = 0.0;
  Vector zhatsoc(dimM); zhatsoc = 0.0;
  int p;
  double thetaold;
  BlockVector Xhatumlsoc(block_offsetsuml); Xhatumlsoc = 0.0;
  BlockVector xhatsoc(block_offsetsx); xhatsoc = 0.0;
  Vector uhatsoc(dimU); uhatsoc = 0.0;
  Vector mhatsoc(dimM); mhatsoc = 0.0;
  double alphasoc;


  problem->Dxphi(x0, mu, &Dxphi0);
  Dxphi0_xhat = InnerProduct(MyComm, Dxphi0, xhat);
  descentDirection = Dxphi0_xhat < 0. ? true : false;
  if(iAmRoot) { cout << "descent direction? " << descentDirection << endl; }
  thx0 = problem->theta(x0);
  phx0 = problem->phi(x0, mu);

  lineSearchSuccess = false;
  for(int i = 0; i < maxBacktrack; i++)
  {
    if(iAmRoot) { cout << "\n--------- alpha = " << alpha << " ---------\n"; }

    // ----- A-5.2. Compute trial point: xtrial = x0 + alpha_i xhat
    xtrial.Set(1.0, x0);
    xtrial.Add(alpha, xhat);

    // ------ A-5.3. if not in filter region go to A.5.4 otherwise go to A-5.5.
    thxtrial = problem->theta(xtrial);
    phxtrial = problem->phi(xtrial, mu);

    filterCheck(thxtrial, phxtrial);    
    if(!inFilterRegion)
    {
      if(iAmRoot) { cout << "not in filter region :)\n"; }
      // ------ A.5.4: Check sufficient decrease
      if(!descentDirection)
      {
        switchCondition = false;
      }
      else
      {
        switchCondition = (alpha * pow(abs(Dxphi0_xhat), sPhi) > delta * pow(thx0, sTheta)) ? true : false;
      }
      if(iAmRoot) 
      { 
        cout << "theta(x0) = "     << thx0     << ", thetaMin = "                  << thetaMin             << endl;
        cout << "theta(xtrial) = " << thxtrial << ", (1-gTheta) *theta(x0) = "     << (1. - gTheta) * thx0 << endl;
        cout << "phi(xtrial) = "   << phxtrial << ", phi(x0) - gPhi *theta(x0) = " << phx0 - gPhi * thx0   << endl;
      }
      // Case I      
      if(thx0 <= thetaMin && switchCondition)
      {
        sufficientDecrease = phxtrial <= phx0 + eta * alpha * Dxphi0_xhat ? true : false;
        if(sufficientDecrease)
        {
          if(iAmRoot) { cout << "A-5.4. Case I -- accepted step length.\n"; } 
          // accept the trial step
          lineSearchSuccess = true;
          break;
        }
      }
      else
      {
        if(thxtrial <= (1. - gTheta) * thx0 || phxtrial <= phx0 - gPhi * thx0)
        {
          if(iAmRoot) { cout << "A-5.4. Case II -- accepted step length.\n"; } 
          // accept the trial step
          lineSearchSuccess = true;
          break;
        }
      }
      // A-5.5: Initialize the second-order correction
      if((!(thx0 < thxtrial)) && i == 0)
      {
        if(iAmRoot) { cout << "second order correction\n"; }
        problem->c(xtrial, &ckSoc);
        problem->c(x0, &ck0);
        ckSoc.Add(alphaMax, ck0);
        thetaold = problem->theta(x0);
        p = 1;
        // A-5.6 Compute the second-order correction.
        pKKTSolve(x0, l0, z0, zhatsoc, Xhatumlsoc, mu, true);
        mhatsoc.Set(1.0, Xhatumlsoc.GetBlock(1));
        alphasoc = computeAlphaMax(m0, ml, mhatsoc, tau);
        //WARNING: not complete but currently solver isn't entering this region
      }
    }
    else
    {
      if(iAmRoot) { cout << "in filter region :(\n"; } 
    }
    // A.5.5: Initialize the second order correction
    //if(Mpi::Root()) { cout << "A.5.5 no second order correction implemented :( \n"; }

    // include more if needed
    alpha *= 0.5;

  } 
}


void interiorPtSolver::projectZ(Vector &x, Vector &z, double mu)
{
  double zi;
  double mudivmml;
  for(int i = 0; i < dimM; i++)
  {
    zi = z(i);
    mudivmml = mu / (x(i + dimU) - ml(i));
    z(i) = max(min(zi, kSig * mudivmml), mudivmml / kSig);
  }
}

void interiorPtSolver::filterCheck(double th, double ph)
{
  inFilterRegion = false;
  if(th > thetaMax)
  {
    inFilterRegion = true;
  }
  else
  {
    for(int i = 0; i < F1.Size(); i++)
    {
      if(th >= F1[i] && ph >= F2[i])
      {
        inFilterRegion = true;
        break;
      }
    }
  }
}

double interiorPtSolver::E(BlockVector &x, Vector &l, Vector &zl, double mu)
{
  Vector cx(dimU), Muinvcx(dimU); cx = 0.0; Muinvcx = 0.0;
  BlockVector gradL(block_offsetsx), MxinvgradL(block_offsetsx); gradL = 0.0; MxinvgradL = 0.0;
  
  problem->c(x, &cx);
  problem->MuSolveMult(cx, Muinvcx);
  E1 = sqrt(InnerProduct(MyComm, cx, Muinvcx));


  problem->DxL(x, l, zl, &gradL);
  problem->MxSolveMult(gradL, MxinvgradL);
  E2 = sqrt(InnerProduct(MyComm, gradL, MxinvgradL));

  Vector m = x.GetBlock(1);
  Vector w2(dimM); w2 = 0.0;
  w2.Set( 1.0, m );
  w2.Add(-1.0, ml);
  w2 *= zl;
  w2 -= mu;
  E3 = InnerProduct(MyComm, w2, Mmlumped);
  
  Vector z1(dimU); z1 = 0.0;
  problem->MuMult(l, z1);
  double lL2 = InnerProduct(MyComm, l, z1);
  
  Vector w1(dimM); w1 = 0.0;
  w1.Set(1.0, zl);
  w1 *= Mmlumped;
  double zL2 = InnerProduct(MyComm, w1, zl);

  sc = max(sMax, zL2) / sMax;
  sd = max(sMax, lL2 / 2. + zL2 / 2.) / sMax;
  if(iAmRoot)
  {
    cout << "stationarity measure = "    << E2      << endl;
    cout << "feasibility measure  = "    << E1 / sd << endl;
    cout << "complimentarity measure = " << E3 / sc << endl;
  }
  return max(max(E1 / sd, E2), E3 / sc);
}


bool interiorPtSolver::GetConverged() const
{
  return converged;
}

void interiorPtSolver::SetOutputVTK(bool x)
{
  outputVTK = x;
}


interiorPtSolver::~interiorPtSolver()
{
  delete diagSparse;
  delete Hk;
  delete Hk1;
  delete Hk2;
  delete Wk;
  delete Wmm;
  delete Jk;
  delete JkT;
  delete Hlogbar;
}