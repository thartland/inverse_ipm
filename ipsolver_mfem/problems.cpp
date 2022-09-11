//         GetNDofs                    Problem classes, which contain
//                             needed functionality for an
//                             interior-point filter-line search
//                             solver   
//                               
//
//


#include "mfem.hpp"
#include "problems.hpp"
#include "helper.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;






optimizationProblem::optimizationProblem(ParFiniteElementSpace *fes, \
                                         ParFiniteElementSpace* fesm, Vector &Ml, MemoryType Mt) :
Vhu(fes), Vhm(fesm), Mform(NULL), Muform(NULL), 
Pu(fes->GetProlongationMatrix()), Ru(fes->GetRestrictionOperator()),
Pm(fesm->GetProlongationMatrix()), Rm(fesm->GetRestrictionOperator())
{
  MyComm = Vhu->GetComm();
  MyRank = Vhu->GetMyRank();
  NRanks = Vhu->GetNRanks();
  iAmRoot = MyRank == 0 ? true : false;

  dimU    = Vhu->GetTrueVSize();
  dimM    = Vhm->GetTrueVSize();
  dimUdof = Vhu->GetVSize();
  dimMdof = Vhm->GetVSize();

  // assuming that the spaces are not variable order
  int orderu = Vhu->GetOrder(0);
  int orderm = Vhm->GetOrder(0);
  MFEM_VERIFY(orderu == orderm, "user must supply identical FE spaces Vhu, Vhm")


  z1.SetSize(dimU); z1 = 0.0;
  z2.SetSize(dimU); z2 = 0.0;
  z3.SetSize(dimU); z3 = 0.0;
  z4.SetSize(dimU); z4 = 0.0;
  z1dof.SetSize(dimUdof); z1dof = 0.0;
  z2dof.SetSize(dimUdof); z2dof = 0.0;
  w1.SetSize(dimM); w1 = 0.0;
  w2.SetSize(dimM); w2 = 0.0;
  w1dof.SetSize(dimMdof); w1dof = 0.0;
  w2dof.SetSize(dimMdof); w2dof = 0.0;
  ml = Ml; // lower bound
  
  // define the offsets for x = (u, m) 
  block_offsetsx.SetSize(3); // number of variables + 1
  block_offsetsx[0] = 0;
  block_offsetsx[1] = dimU;
  block_offsetsx[2] = dimM;
  block_offsetsx.PartialSum();

  // define the offsets for (x, l) 
  // needed for setting up the IP-Newton system
  block_offsetsxl.SetSize(3);
  block_offsetsxl[0] = 0;
  block_offsetsxl[1] = dimU + dimM;
  block_offsetsxl[2] = dimU;
  block_offsetsxl.PartialSum();

  // define offsets for u
  // allows us to construct Jacobian operator
  block_offsetsu.SetSize(2);
  block_offsetsu[0] = 0;
  block_offsetsu[1] = dimU;
  block_offsetsu.PartialSum();

  mt   = Mt;
  
  // mass bilinear form, matrix and solver for state
  Muform = new ParBilinearForm(Vhu);
  Muform->AddDomainIntegrator(new MassIntegrator);
  Muform->Assemble();
  Muform->Finalize();
  Muform->FormSystemMatrix(empty_tdof_list, Mu);
  
  
  //Musolver = new HyprePCG(MPI_COMM_WORLD);
  Musolver = new HyprePCG(MyComm);
  Musolver->SetOperator(Mu);
  Musolver->SetTol(1.e-12);
  Musolver->SetMaxIter(500);
  Musolver->SetPrintLevel(0);


  // mass bilinear form, matrix and solver for parameter
  Mform = new ParBilinearForm(Vhm);
  Mform->AddDomainIntegrator(new MassIntegrator);
  Mform->Assemble();
  Mform->Finalize();
  Mform->FormSystemMatrix(empty_tdof_list, Mm);

  
  // Vector rep of the lumped mass matrix Mm * 1
  w1 = 1.;
  Mmlumped.SetSize(dimM);
  Mm.Mult(w1, Mmlumped);

  Mx = new BlockOperator(block_offsetsx, block_offsetsx);
  Mx->SetBlock(0, 0, &Mu);
  Mx->SetBlock(1, 1, &Mm);
  Mxsolver = new CGSolver(MyComm);
  Mxsolver->SetRelTol(double(1.e-14));
  Mxsolver->SetAbsTol(double(0.0));
  Mxsolver->SetMaxIter(500);
  Mxsolver->SetPrintLevel(2);
  Mxsolver->SetOperator(*Mx);
}

double optimizationProblem::theta(BlockVector &x)
{
  Vector cx(dimU); cx = 0.0;
  c(x, &cx);
  return sqrt(InnerProduct(MyComm, cx, cx));
}

double optimizationProblem::thetaM(BlockVector &x)
{
  c(x, &z3);
  Musolver->Mult(z3, z4);
  return sqrt(InnerProduct(MyComm, z3, z4));
}


// log-barrier objective
double optimizationProblem::phi(BlockVector &x, double mu)
{
  double fx = f(x);
  w1 = 0.0;
  w1.Set( 1.0, x.GetBlock(1)); // parameter -- m
  w1.Add(-1.0, ml);
  for(int i = 0; i < dimM; i++)
  {
    w1(i) = log(w1(i));
  }
  return fx - mu * InnerProduct(MyComm, Mmlumped, w1);
}


// gradient of log-barrier objective with respect to x = (u, m)
void optimizationProblem::Dxphi(BlockVector &x, double mu, BlockVector *y)
{
  Dxf(x, y);  
  Vector ym(dimM);
  y->GetBlockView(1, ym);
  for(int i = 0; i < dimM; i++)
  {
    ym(i) -= mu * Mmlumped(i) / (x(i + dimU) - ml(i));
  }
}


// Lagrangian function evaluation
// L(x, l, zl) = f(x) + l^T c(x) - zl^T M (m - ml)
double optimizationProblem::L(BlockVector &x, Vector &l, Vector &zl)
{
  double fx = f(x);
  Vector cx(dimU);
  c(x, &cx);

  w2 = 0.0;
  w2.Set( 1.0, x.GetBlock(1));
  w2.Add(-1.0, ml);
  w2 *= Mmlumped;
  return (fx + InnerProduct(MyComm, cx, l) - InnerProduct(MyComm, w2, zl));
}

void optimizationProblem::DxL(BlockVector &x, Vector &l, Vector &zl, BlockVector *y)
{
  Vector u = x.GetBlock(0);
  Vector m = x.GetBlock(1);

  BlockVector gradf(block_offsetsx, mt);
  Dxf(x, &gradf);
  BlockVector JaccTl(block_offsetsx, mt);
  DxcTp(x, l, &JaccTl);
  y->Set(1.0, gradf);
  y->Add(1.0, JaccTl);

  Vector ym(dimM);
  y->GetBlockView(1, ym);
  for(int i = 0; i < dimM; i++)
  {
    ym(i) -= Mmlumped(i) * zl(i); 
  }
}

void optimizationProblem::DxxL(BlockVector &x, Vector &l, BlockOperatorSum *y)
{
  Dxxf(x, y->bOp1);
  Dxxcp(x, l, y->bOp2);
}

void  optimizationProblem::DxxL(BlockVector &x, Vector &l, BlockOperator *y1, BlockOperator *y2)
{
  Dxxf(x, y1);
  Dxxcp(x, l, y2);  
}


void optimizationProblem::MuSolveMult(const Vector &x, Vector &y) const
{
  Musolver->Mult(x, y);
}

void optimizationProblem::MuMult(const Vector &x, Vector &y) const
{
  Mu.Mult(x, y);
}

void optimizationProblem::MxSolveMult(const Vector &x, Vector &y) const
{
  Mxsolver->Mult(x, y);
}





double optimizationProblem::E(BlockVector &x, Vector &l, Vector &zl, double mu, double smax)
{
  Vector cx(dimU), Muinvcx(dimU); cx = 0.0; Muinvcx = 0.0;
  BlockVector gradL(block_offsetsx, mt), MxinvgradL(block_offsetsx, mt); gradL = 0.0; MxinvgradL = 0.0;
  
  c(x, &cx);
  MuSolveMult(cx, Muinvcx);
  //Musolver->Mult(cx, Muinvcx);
  double E1 = sqrt(InnerProduct(MyComm, cx, Muinvcx));


  DxL(x, l, zl, &gradL);
  MxSolveMult(gradL, MxinvgradL);
  //Mxsolver->Mult(gradL, MxinvgradL);
  double E2 = sqrt(InnerProduct(MyComm, gradL, MxinvgradL));

  Vector m = x.GetBlock(1);
  w2 = 0.0;
  w2.Set( 1.0, m );
  w2.Add(-1.0, ml);
  w2 *= zl;
  w2 -= mu;
  double E3 = InnerProduct(MyComm, w2, Mmlumped);
  
  Mu.Mult(l, z1);
  double lL2 = InnerProduct(MyComm, l, z1);
  
  w1 = 0.0;
  w1.Set(1.0, zl);
  w1 *= Mmlumped;
  double zL2 = InnerProduct(MyComm, w1, zl);

  double sc = max(smax, zL2) / smax;
  double sd = max(smax, lL2 / 2. + zL2 / 2.) / smax;
  if(Mpi::Root())
  {
    cout << "stationarity measure = "    << E2      << endl;
    cout << "feasibility measure  = "    << E1 / sd << endl;
    cout << "complimentarity measure = " << E3 / sc << endl;
  }
  return max(max(E1 / sd, E2), E3 / sc);
}

int optimizationProblem::getdimU() { return dimU; }

int optimizationProblem::getdimM() { return dimM; }

Vector optimizationProblem::getMmlumped() {return Mmlumped; }

Vector optimizationProblem::getml() {return ml; }


Array<int> optimizationProblem::getblock_offsetsx()  {return block_offsetsx ; }
Array<int> optimizationProblem::getblock_offsetsxl() {return block_offsetsxl; }
Array<int> optimizationProblem::getblock_offsetsu()  {return block_offsetsu; }

MPI_Comm optimizationProblem::GetComm() const {return MyComm; }

int optimizationProblem::GetNRanks() const {return NRanks; }
int optimizationProblem::GetMyRank() const {return MyRank; }




//               For reference, we consider the PDE-constrained
//               optimization problem
//
//               min_(u, m) f(u, m) = 0.5 * || u - u_d ||_(M_w)^2 + 0.5 * R(m, m)
//               s.t.
//                 -div(m grad(u)) + beta u = g, in Omega
//                  du / dn                 = 0, on boundary of Omega
//                  m >= ml,                     a.e. in Omega

inverseDiffusion::inverseDiffusion(
	ParFiniteElementSpace *fes, \
        ParFiniteElementSpace *fesm, \
	ParFiniteElementSpace *fesgrad, \
	ParFiniteElementSpace *fesL2,\
	ParGridFunction &ud_gf,\
	ParGridFunction &g_gf,\
        Vector &Ml,\
	ParGridFunction &w_gf,\
	Array<int> tdof,\
        double Beta,\
	double gamma1,\
	double gamma2,
  MemoryType Mt) : optimizationProblem(fes, fesm, Ml, Mt), 
  Vhgrad(fesgrad), VhL2(fesL2), Grad(NULL), Mwform(NULL), Rform(NULL)
{ 
  beta = new ConstantCoefficient(Beta);

  ud   = ud_gf.GetTrueDofs();
  g     = new ParGridFunction(Vhu);
  *g    = g_gf;

  // Relaxed Dirac comb mass weight function
  // allows us to construct a mass-matrix like object
  // Mw, for which ||u||_{Mw}^{2} \approx \sum_{i=1}^{m} u(x_i)^2
  // and ultimately a cost functional which approximates a pointwise
  // defined tracking-type cost functional
  GridFunctionCoefficient w_gfc(&w_gf);
  Mwform = new ParBilinearForm(Vhu);
  Mwform->AddDomainIntegrator(new MassIntegrator(w_gfc));
  Mwform->Assemble();
  Mwform->Finalize();
  Mwform->FormSystemMatrix(empty_tdof_list, Mw);

  // set up the FE matrix representation of the
  // regularization operator Rm,
  // (Rm)_(i, j) = R(\psi_i, \psi_j)
  ConstantCoefficient gamma_1(gamma1);
  ConstantCoefficient gamma_2(gamma2);
  Rform = new ParBilinearForm(Vhm);
  Rform->AddDomainIntegrator(new MassIntegrator(gamma_1));
  Rform->AddDomainIntegrator(new DiffusionIntegrator(gamma_2));
  Rform->Assemble();
  Rform->Finalize();
  Rform->FormSystemMatrix(empty_tdof_list, R);
  
  // Set up an operator that maps u --> grad(u) = \nabla u
  Grad = new DiscreteLinearOperator(Vhu, Vhgrad);
  Grad->AddDomainInterpolator(new GradientInterpolator);
  Grad->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  Grad->Assemble();
  Grad->Finalize();


  // map a Vector (L2 function rep) to another (Vhm dual function rep)
  ML2mform = new ParMixedBilinearForm(VhL2, Vhm);
  ML2mform->AddDomainIntegrator(new MassIntegrator);
  ML2mform->Assemble();
  ML2mform->Finalize();
  ML2m = ML2mform->SpMat();


  Juform = new ParBilinearForm(Vhu);
}

double inverseDiffusion::f(BlockVector & x)
{
  // data discrepancy
  Vector u = x.GetBlock(0);
  z1.Set(1.0, u);
  z1.Add(-1.0, *ud);
  Mw.Mult(z1, z2);
  
  // parameter
  Vector m = x.GetBlock(1);
  w1.Set(1.0, m);
  R.Mult(w1, w2);

  // f = u^T Mw u / 2 + m^T R m / 2
  return 0.5 * InnerProduct(MyComm, z1, z2) + 0.5 * InnerProduct(MyComm, w1, w2);
}

void inverseDiffusion::Dxf(BlockVector & x, BlockVector * y)
{
  // data discrepancy
  Vector u = x.GetBlock(0);
  z1.Set(1.0,   u);
  z1.Add(-1.0, *ud);

  // parameter
  Vector m = x.GetBlock(1);
  
  // g = Mw u + R m
  Vector yu(dimU);
  Vector ym(dimM);
  y->GetBlockView(0, yu);
  y->GetBlockView(1, ym);
  Mw.Mult(z1, yu);
  R.Mult(m,  ym);
}

void inverseDiffusion::Dxxf(BlockVector& x, BlockOperator* y)
{
  y->SetBlock(0, 0, &Mw);
  y->SetBlock(1, 1, &R);
}


void inverseDiffusion::c(BlockVector& x, Vector* y)
{
  Vector u = x.GetBlock(0);
  Vector m = x.GetBlock(1);
  
  ParGridFunction m_gf(Vhm);
  m_gf.SetFromTrueDofs(m);
  GridFunctionCoefficient m_gfc(&m_gf);
  
  ParBilinearForm * cform = new ParBilinearForm(Vhu);
  cform->AddDomainIntegrator(new DiffusionIntegrator(m_gfc));
  cform->AddDomainIntegrator(new MassIntegrator(*beta));
  cform->Assemble();
  cform->Finalize();
  
  Pu->Mult(u, z1dof);
  cform->Mult(z1dof, z2dof);
  delete cform;
  
  Pu->MultTranspose(z2dof, *y);
  Ru->Mult(*g, z1);
  Mu.Mult(z1, z2);
  y->Add(-1.0, z2);
}

void inverseDiffusion::Dxc(BlockVector& x, BlockOperator* y)
{
  Vector u = x.GetBlock(0);
  Vector m = x.GetBlock(1);
  ParGridFunction m_gf(Vhm);
  
  Pm->Mult(m, m_gf);
  GridFunctionCoefficient m_gfc(&m_gf);

  delete Juform;
  Juform = new ParBilinearForm(Vhu);
  Juform->AddDomainIntegrator(new DiffusionIntegrator(m_gfc));
  Juform->AddDomainIntegrator(new MassIntegrator(*beta));
  Juform->Assemble();
  Juform->Finalize();
  Juform->FormSystemMatrix(empty_tdof_list, Ju);
  y->SetBlock(0, 0, &Ju);
  
  ParGridFunction * ugrad = new ParGridFunction(Vhgrad);
  Pu->Mult(u, z1dof);
  Grad->Mult(z1dof, *ugrad); 
  VectorGridFunctionCoefficient ugrad_gfc(ugrad);

  ParBilinearForm * Jmform(new ParBilinearForm(Vhu));
  constexpr double alpha = 1.0;
  int skip_zeros = 0;
  Jmform->AddDomainIntegrator(new TransposeIntegrator(new ConvectionIntegrator(ugrad_gfc, alpha)));
  Jmform->Assemble(skip_zeros);
  Jmform->Finalize();
  Jmform->FormSystemMatrix(empty_tdof_list, Jm);

  delete ugrad;
  y->SetBlock(0, 1, &Jm);
}

void inverseDiffusion::DxcTp(BlockVector& x, Vector &l, BlockVector *y)
{
  Vector u = x.GetBlock(0);
  Vector m = x.GetBlock(1);
  ParGridFunction m_gf(Vhm);
  m_gf.SetFromTrueDofs(m);
  GridFunctionCoefficient m_gfc(&m_gf);

  ParBilinearForm * cuform(new ParBilinearForm(Vhu));
  cuform->AddDomainIntegrator(new DiffusionIntegrator(m_gfc));
  cuform->AddDomainIntegrator(new MassIntegrator(*beta));
  cuform->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  cuform->Assemble();

  Vector yu(dimU);
  Vector ym(dimM);
  y->GetBlockView(0, yu);
  y->GetBlockView(1, ym);
  Pu->Mult(l, z1dof);
  cuform->Mult(z1dof, z2dof);
  Pu->MultTranspose(z2dof, yu);
  delete cuform;

  GridFunction * lgrad = new GridFunction(Vhgrad);
  GridFunction * ugrad = new GridFunction(Vhgrad);
  Grad->Mult(z1dof, *lgrad);
  Pu->Mult(u, z2dof); 
  Grad->Mult(z2dof, *ugrad);
  VectorGridFunctionCoefficient lgrad_gfc(lgrad);

  DiscreteLinearOperator * Inner = new DiscreteLinearOperator(Vhgrad, VhL2);
  Inner->AddDomainInterpolator(new VectorInnerProductInterpolator(lgrad_gfc));
  Inner->Assemble();
  Inner->Finalize();
  delete lgrad;
  
  GridFunction * ugradTlgrad = new GridFunction(VhL2);
  Inner->Mult(*ugrad, *ugradTlgrad);
  delete ugrad;
  delete Inner;

  ML2m.Mult(*ugradTlgrad, w1dof);
  Pm->MultTranspose(w1dof, ym);
  delete ugradTlgrad;
}


void inverseDiffusion::Dxxcp(BlockVector& x, Vector &l, BlockOperator* y)
{
  Vector u = x.GetBlock(0);
  Vector m = x.GetBlock(1);

  GridFunction * lgrad = new GridFunction(Vhgrad);
  Pu->Mult(l, z1dof);
  Grad->Mult(z1dof, *lgrad);
  VectorGridFunctionCoefficient lgrad_gfc(lgrad);  
  
  constexpr double alpha = 1.0;
  int skip_zeros = 0;

  ParBilinearForm * Dmucform(new ParBilinearForm(Vhu));
  Dmucform->AddDomainIntegrator(new ConvectionIntegrator(lgrad_gfc, alpha));
  Dmucform->Assemble(skip_zeros);
  Dmucform->Finalize();

  Dmucform->FormSystemMatrix(empty_tdof_list, Jmu);
  Jum = *(Jmu.Transpose());
  y->SetBlock(1, 0, &Jmu);
  y->SetBlock(0, 1, &Jum);
  //delete Dmucform;
}

void inverseDiffusion::feasibilityRestoration(BlockVector &x, double tol)
{
  Vector u = x.GetBlock(0);
  Vector m = x.GetBlock(1);

  ParGridFunction m_gf(Vhm);
  m_gf.SetFromTrueDofs(m);
  GridFunctionCoefficient m_gfc(&m_gf);

  ParBilinearForm * cuform(new ParBilinearForm(Vhu));
  cuform->AddDomainIntegrator(new DiffusionIntegrator(m_gfc));
  cuform->AddDomainIntegrator(new MassIntegrator(*beta));
  cuform->Assemble();
  cuform->Finalize();
  HypreParMatrix A;
  cuform->FormSystemMatrix(empty_tdof_list, A);
  //delete cuform;

  HyprePCG * Asolver = new HyprePCG(MyComm);
  HypreBoomerAMG * Aprec = new HypreBoomerAMG;
  Aprec->SetPrintLevel(2);
  Asolver->SetTol(tol);
  Asolver->SetMaxIter(1000);
  Asolver->SetPreconditioner(*Aprec);
  Asolver->SetOperator(A);

  Vector uview(dimU);
  x.GetBlockView(0, uview);

  
  Ru->Mult(*g, z1);
  
  Mu.Mult(z1, z2);
  Asolver->Mult(z2, uview);

  delete Aprec;
  delete Asolver;
}



inverseDiffusion::~inverseDiffusion()
{
  // Do not free the memory that Vh points to.
  // There is other data that relies on the data contained
  // in the address that Vh points to.
  delete Juform;
  delete Rform;
  delete Mform;
  delete Muform; 
  delete Mwform;
  delete ML2mform;
  delete Grad; 
  delete Mxsolver;
  delete Musolver;
  delete g;
  delete beta;
}






//               For reference, we consider the PDE-constrained
//               optimization problem
//
//               min_(u, m) f(u, m) = 0.5 * || u - u_d ||_(M_w)^2 + 0.5 * R(m, m)
//               s.t.
//                 -div(m grad(u)) + beta u = g, in Omega
//                  du / dn                 = 0, on boundary of Omega
//                  m >= ml,                     a.e. in Omega

inverseDiffusion2::inverseDiffusion2(
  ParFiniteElementSpace *fes, \
  ParFiniteElementSpace *fesm, \
  ParFiniteElementSpace *fesgrad, \
  ParFiniteElementSpace *fesL2,\
  ParGridFunction &ud_gf,\
  ParGridFunction &g_gf,\
  Vector &Ml,\
  Array<int> tdof,\
  double Beta,\
  double gamma1,\
  double gamma2,
  MemoryType Mt) : optimizationProblem(fes, fesm, Ml, Mt), 
  Vhgrad(fesgrad), VhL2(fesL2), Grad(NULL), Mwform(NULL), Rform(NULL)
{ 
  beta = new ConstantCoefficient(Beta);

  ud   = ud_gf.GetTrueDofs();
  g     = new ParGridFunction(Vhu);
  *g    = g_gf;

  // Relaxed Dirac comb mass weight function
  // allows us to construct a mass-matrix like object
  // Mw, for which ||u||_{Mw}^{2} \approx \sum_{i=1}^{m} u(x_i)^2
  // and ultimately a cost functional which approximates a pointwise
  // defined tracking-type cost functional
  Mwform = new ParBilinearForm(Vhu);
  Mwform->AddDomainIntegrator(new MassIntegrator);
  Mwform->Assemble();
  Mwform->Finalize();
  Mwform->FormSystemMatrix(empty_tdof_list, Mw);

  // set up the FE matrix representation of the
  // regularization operator Rm,
  // (Rm)_(i, j) = R(\psi_i, \psi_j)
  ConstantCoefficient gamma_1(gamma1);
  ConstantCoefficient gamma_2(gamma2);
  Rform = new ParBilinearForm(Vhm);
  Rform->AddDomainIntegrator(new MassIntegrator(gamma_1));
  Rform->AddDomainIntegrator(new DiffusionIntegrator(gamma_2));
  Rform->Assemble();
  Rform->Finalize();
  Rform->FormSystemMatrix(empty_tdof_list, R);
  
  // Set up an operator that maps u --> grad(u) = \nabla u
  Grad = new DiscreteLinearOperator(Vhu, Vhgrad);
  Grad->AddDomainInterpolator(new GradientInterpolator);
  Grad->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  Grad->Assemble();
  Grad->Finalize();


  // map a Vector (L2 function rep) to another (Vhm dual function rep)
  ML2mform = new ParMixedBilinearForm(VhL2, Vhm);
  ML2mform->AddDomainIntegrator(new MassIntegrator);
  ML2mform->Assemble();
  ML2mform->Finalize();
  ML2m = ML2mform->SpMat();

  Juform = new ParBilinearForm(Vhu);
}

double inverseDiffusion2::f(BlockVector & x)
{
  // data discrepancy
  Vector u = x.GetBlock(0);
  z1.Set(1.0, u);
  z1.Add(-1.0, *ud);
  Mw.Mult(z1, z2);
  
  // parameter
  Vector m = x.GetBlock(1);
  w1.Set(1.0, m);
  R.Mult(w1, w2);

  // f = u^T Mw u / 2 + m^T R m / 2
  return 0.5 * InnerProduct(MyComm, z1, z2) + 0.5 * InnerProduct(MyComm, w1, w2);
}

void inverseDiffusion2::Dxf(BlockVector & x, BlockVector * y)
{
  // data discrepancy
  Vector u = x.GetBlock(0);
  z1.Set(1.0,   u);
  z1.Add(-1.0, *ud);

  // parameter
  Vector m = x.GetBlock(1);
  
  // g = Mw u + R m
  Vector yu(dimU);
  Vector ym(dimM);
  y->GetBlockView(0, yu);
  y->GetBlockView(1, ym);
  Mw.Mult(z1, yu);
  R.Mult(m,  ym);
}

void inverseDiffusion2::Dxxf(BlockVector& x, BlockOperator* y)
{
  y->SetBlock(0, 0, &Mw);
  y->SetBlock(1, 1, &R);
}


void inverseDiffusion2::c(BlockVector& x, Vector* y)
{
  Vector u = x.GetBlock(0);
  Vector m = x.GetBlock(1);
  
  ParGridFunction m_gf(Vhm);
  m_gf.SetFromTrueDofs(m);
  GridFunctionCoefficient m_gfc(&m_gf);
  
  ParBilinearForm * cform = new ParBilinearForm(Vhu);
  cform->AddDomainIntegrator(new DiffusionIntegrator(m_gfc));
  cform->AddDomainIntegrator(new MassIntegrator(*beta));
  cform->Assemble();
  cform->Finalize();
  
  Pu->Mult(u, z1dof);
  cform->Mult(z1dof, z2dof);
  delete cform;
  
  Pu->MultTranspose(z2dof, *y);
  Ru->Mult(*g, z1);
  Mu.Mult(z1, z2);
  y->Add(-1.0, z2);
}

void inverseDiffusion2::Dxc(BlockVector& x, BlockOperator* y)
{
  Vector u = x.GetBlock(0);
  Vector m = x.GetBlock(1);
  ParGridFunction m_gf(Vhm);
  Pm->Mult(m, m_gf);
  GridFunctionCoefficient m_gfc(&m_gf);

  delete Juform;
  Juform = new ParBilinearForm(Vhu);
  Juform->AddDomainIntegrator(new DiffusionIntegrator(m_gfc));
  Juform->AddDomainIntegrator(new MassIntegrator(*beta));
  Juform->Assemble();
  Juform->Finalize();
  Juform->FormSystemMatrix(empty_tdof_list, Ju);
  y->SetBlock(0, 0, &Ju);
  
  ParGridFunction * ugrad = new ParGridFunction(Vhgrad);
  Pu->Mult(u, z1dof);
  Grad->Mult(z1dof, *ugrad); 
  VectorGridFunctionCoefficient ugrad_gfc(ugrad);

  ParBilinearForm * Jmform(new ParBilinearForm(Vhu));
  constexpr double alpha = 1.0;
  int skip_zeros = 0;
  Jmform->AddDomainIntegrator(new TransposeIntegrator(new ConvectionIntegrator(ugrad_gfc, alpha)));
  Jmform->Assemble(skip_zeros);
  Jmform->Finalize();
  Jmform->FormSystemMatrix(empty_tdof_list, Jm);

  delete ugrad;
  y->SetBlock(0, 1, &Jm);
}

void inverseDiffusion2::DxcTp(BlockVector& x, Vector &l, BlockVector *y)
{
  Vector u = x.GetBlock(0);
  Vector m = x.GetBlock(1);
  ParGridFunction m_gf(Vhm);
  m_gf.SetFromTrueDofs(m);
  GridFunctionCoefficient m_gfc(&m_gf);

  ParBilinearForm * cuform(new ParBilinearForm(Vhu));
  cuform->AddDomainIntegrator(new DiffusionIntegrator(m_gfc));
  cuform->AddDomainIntegrator(new MassIntegrator(*beta));
  cuform->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  cuform->Assemble();

  Vector yu(dimU);
  Vector ym(dimM);
  y->GetBlockView(0, yu);
  y->GetBlockView(1, ym);
  Pu->Mult(l, z1dof);
  cuform->Mult(z1dof, z2dof);
  Pu->MultTranspose(z2dof, yu);
  delete cuform;

  GridFunction * lgrad = new GridFunction(Vhgrad);
  GridFunction * ugrad = new GridFunction(Vhgrad);
  Grad->Mult(z1dof, *lgrad);
  Pu->Mult(u, z2dof); 
  Grad->Mult(z2dof, *ugrad);
  VectorGridFunctionCoefficient lgrad_gfc(lgrad);

  DiscreteLinearOperator * Inner = new DiscreteLinearOperator(Vhgrad, VhL2);
  Inner->AddDomainInterpolator(new VectorInnerProductInterpolator(lgrad_gfc));
  Inner->Assemble();
  Inner->Finalize();
  delete lgrad;
  
  GridFunction * ugradTlgrad = new GridFunction(VhL2);
  Inner->Mult(*ugrad, *ugradTlgrad);
  delete ugrad;
  delete Inner;

  ML2m.Mult(*ugradTlgrad, w1dof);
  Pm->MultTranspose(w1dof, ym);
  delete ugradTlgrad;
}


void inverseDiffusion2::Dxxcp(BlockVector& x, Vector &l, BlockOperator* y)
{
  Vector u = x.GetBlock(0);
  Vector m = x.GetBlock(1);

  GridFunction * lgrad = new GridFunction(Vhgrad);
  Pu->Mult(l, z1dof);
  Grad->Mult(z1dof, *lgrad);
  VectorGridFunctionCoefficient lgrad_gfc(lgrad);
  
  constexpr double alpha = 1.0;
  int skip_zeros = 0;

  ParBilinearForm * Dmucform(new ParBilinearForm(Vhu));
  Dmucform->AddDomainIntegrator(new ConvectionIntegrator(lgrad_gfc, alpha));
  Dmucform->Assemble(skip_zeros);
  Dmucform->Finalize();

  Dmucform->FormSystemMatrix(empty_tdof_list, Jmu);
  Jum = *(Jmu.Transpose());
  y->SetBlock(1, 0, &Jmu);
  y->SetBlock(0, 1, &Jum);
}

void inverseDiffusion2::feasibilityRestoration(BlockVector &x, double tol)
{
  Vector u = x.GetBlock(0);
  Vector m = x.GetBlock(1);

  ParGridFunction m_gf(Vhm);
  m_gf.SetFromTrueDofs(m);
  GridFunctionCoefficient m_gfc(&m_gf);

  ParBilinearForm * cuform(new ParBilinearForm(Vhu));
  cuform->AddDomainIntegrator(new DiffusionIntegrator(m_gfc));
  cuform->AddDomainIntegrator(new MassIntegrator(*beta));
  cuform->Assemble();
  cuform->Finalize();
  HypreParMatrix A;
  cuform->FormSystemMatrix(empty_tdof_list, A);

  HyprePCG * Asolver = new HyprePCG(MyComm);
  HypreBoomerAMG * Aprec = new HypreBoomerAMG;
  Aprec->SetPrintLevel(2);
  Asolver->SetTol(tol);
  Asolver->SetMaxIter(1000);
  Asolver->SetPreconditioner(*Aprec);
  Asolver->SetOperator(A);

  Vector uview(dimU);
  x.GetBlockView(0, uview);

  
  Ru->Mult(*g, z1);
  
  Mu.Mult(z1, z2);
  Asolver->Mult(z2, uview);

  delete Aprec;
  delete Asolver;
}



inverseDiffusion2::~inverseDiffusion2()
{
  // Do not free the memory that Vh points to.
  // There is other data that relies on the data contained
  // in the address that Vh points to.
  delete Juform;
  delete Rform;
  delete Mform;
  delete Muform; 
  delete Mwform;
  delete Grad;
  delete Mxsolver;
  delete Musolver; 
}





























