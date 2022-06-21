//                               Reduced Hessian Class for
//                                  Poisson PDE with a 
//                                    rhs parameter
//                               
//
//
// Description:  This example code demonstrates the use of MFEM's
//               operator class, in order to create a child class
//               that have a custom .Mult(m1, m2) function that
//               takes m1 --> m2, where m2 is the action of
//               the reduced Hessian on m1.
//
//               For reference, we consider the PDE-constrained
//               optimization problem
//
//               min_(u, m) f(u, m) = 0.5 * || u - u_d ||_(M_w)^2 + 0.5 * R(m, m)
//               s.t.
//                 -div(m grad(u)) + beta u = g, in Omega
//                  du / dn                 = 0, on boundary of Omega
//                  ml <= m,                     a.e. in Omega
//


#include "mfem.hpp"
#include "problems.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;



inverseDiffusion::inverseDiffusion(
	ParFiniteElementSpace *fes, \
        ParFiniteElementSpace *fesm, \
	ParFiniteElementSpace *fesgrad, \
	ParFiniteElementSpace *fesL2,\
	ParGridFunction ud_gf,\
	ParGridFunction g_gf,\
        ParGridFunction ml_gf,\
	ParGridFunction w_gf,\
	Array<int> tdof,\
        double Beta,\
	double gamma1,\
	double gamma2) : \
	Vhm(fesm), Vhu(fes), Vhgrad(fesgrad), VhL2(fesL2), Mform(NULL), Muform(NULL), Grad(NULL),
	Mwform(NULL), Rform(NULL), Mm(NULL), Mu(NULL), Mw(NULL),
       	Rm(NULL)
{ 
  beta = new ConstantCoefficient(Beta);
  Vhu_size = Vhu->GetTrueVSize();
  Vhm_size = Vhm->GetTrueVSize();
  /*z1.SetSize(Vhu_size);
  z2.SetSize(Vhu_size);*/
  /*w1.SetSize(Vhm_size);
  w2.SetSize(Vhm_size);*/
  /*z1(Vh_size);
  z2(Vh_size);
  u    = new GridFunction(Vh);*/
  ud    = new ParGridFunction(Vhu);
  *ud   = ud_gf;

  g     = new ParGridFunction(Vhu);
  *g    = g_gf;

  ml    = new ParGridFunction(Vhm);
  *ml   = ml_gf;

  /*p    = new GridFunction(Vh);
  uhat = new GridFunction(Vh);
  phat = new GridFunction(Vh);
  m    = new GridFunction(Vhm);
  f    = new GridFunction(Vhm);
  mhat = new GridFunction(Vhm);
  lamhat = new GridFunction(Vhm);
  lam  = new GridFunction(Vhm);
  ml   = new GridFunction(Vhm);
  */

  
  
  /**f   = f_gf;
  *ml  = ml_gf;
  // essential dofs
  ess_tdof_list = tdof;

  
  b  = new LinearForm(Vh);
  bp = new LinearForm(Vh);*/

  // Mass Bilinear Form
  Mform = new ParBilinearForm(Vhm);
  Mform->AddDomainIntegrator(new MassIntegrator);
  Mform->Assemble();
  Mm = &(Mform->SpMat());

  //Vector Mmlumped;
  Mmlumped.SetSize(Mm->Size());
  Mm->GetRowSums(Mmlumped);
  Muform = new ParBilinearForm(Vhu);
  Muform->AddDomainIntegrator(new MassIntegrator);
  Muform->Assemble();
  Mu = &(Muform->SpMat());
  
  
  GridFunctionCoefficient w_gfc(&w_gf);
  Mwform = new ParBilinearForm(Vhu);
  Mwform->AddDomainIntegrator(new MassIntegrator(w_gfc));
  Mwform->Assemble();
  Mw = &(Mwform->SpMat());


  // set up the FE matrix representation of the
  // regularization operator Rm,
  // (Rm)_(i, j) = R(\psi_i, \psi_j)
  ConstantCoefficient gamma_1(gamma1);
  ConstantCoefficient gamma_2(gamma2);
  Rform = new ParBilinearForm(Vhm);
  Rform->AddDomainIntegrator(new MassIntegrator(gamma_1));
  Rform->AddDomainIntegrator(new DiffusionIntegrator(gamma_2));
  Rform->Assemble();
  Rm = &(Rform->SpMat());
  
  Grad = new DiscreteLinearOperator(Vhu, Vhgrad);
  Grad->AddDomainInterpolator(new GradientInterpolator);
  Grad->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  Grad->Assemble();

  MmL2form = new ParMixedBilinearForm(VhL2, Vhm);
  MmL2form->AddDomainIntegrator(new MassIntegrator);
  MmL2form->Assemble();
  MmL2 = &(MmL2form->SpMat());


 
  //Rm = &(Rform->SpMat());
  /*M_solver.SetRelTol(double(1.e-14));
  M_solver.SetAbsTol(double(0.));
  M_solver.SetMaxIter(500);
  M_solver.SetPrintLevel(0);
  M_solver.SetOperator(*Mm);

  Muform = new BilinearForm(Vh);
  Muform->AddDomainIntegrator(new MassIntegrator);
  Muform->Assemble();
  Mu = &(Muform->SpMat());

  // Regularization Bilinear Form
  ConstantCoefficient gamma(gamma_i);
  Rform = new BilinearForm(Vhm);
  Rform->AddDomainIntegrator(new MassIntegrator(gamma));
  Rform->Assemble();
  Rm = &(Rform->SpMat());
  
  // bilinearform for state/adjoint 
  aform   = new BilinearForm(Vh);
  aform->AddDomainIntegrator(new DiffusionIntegrator);
  aform->Assemble();

  // bilinear form for int m_trial p_test
  // for more complex problems C may depend
  // on the state u and so should be formed
  // in set_adj_parameters() function
  Cform = new MixedBilinearForm(Vh, Vhm);
  Cform->AddDomainIntegrator(new MassIntegrator);
  Cform->Assemble();
  Cform->Finalize();
  C = &(Cform->SpMat());
  Ct = Transpose(*C);*/
}

double inverseDiffusion::f(BlockVector & x)
{
  // data discrepancy
  Vector u = x.GetBlock(0);
  z1.SetSize(u.Size());
  z2.SetSize(u.Size());
  z1 *= 0.;
  z1.Add(1.0, u);
  z1.Add(-1.0, *ud);
  Mw->Mult(z1, z2);
  double misfit = 0.5 * InnerProduct(MPI_COMM_WORLD, z1, z2);
  
  // parameter
  Vector m = x.GetBlock(1);
  w1.SetSize(m.Size());
  w2.SetSize(m.Size());
  w1.Set(1.0, m);

  Rm->Mult(w1, w2);
  double reg = 0.5 * InnerProduct(MPI_COMM_WORLD, w1, w2);
  // f = u^T M u / 2 + m^T R m / 2
  return misfit + reg;
}

void inverseDiffusion::Dxf(BlockVector & x, BlockVector * y)
{
  // data discrepancy
  Vector u = x.GetBlock(0);
  z1.SetSize(u.Size());
  z1.Set(1.0, u);
  z1.Add(-1.0, *ud);
  // parameter
  Vector m = x.GetBlock(1);
  
  // g = M u + R m
  Vector yu(u.Size());
  Vector ym(m.Size());
  y->GetBlockView(0, yu);
  y->GetBlockView(1, ym);
  Mw->Mult(z1, yu);
  Rm->Mult(m, ym);
}

void inverseDiffusion::Dxxf(BlockVector& x, BlockOperator* y)
{
  y->SetBlock(0, 0, Mw);
  y->SetBlock(1, 1, Rm);
}


void inverseDiffusion::c(BlockVector& x, Vector* y)
{
  Vector u = x.GetBlock(0);
  Vector m = x.GetBlock(1);
  ParGridFunction m_gf(Vhm);
  m_gf = m;
  GridFunctionCoefficient m_gfc(&m_gf);
  
  ParBilinearForm * cform(new ParBilinearForm(Vhu));
  cform->AddDomainIntegrator(new DiffusionIntegrator(m_gfc));
  cform->AddDomainIntegrator(new MassIntegrator(*beta));
  cform->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  cform->Assemble();
  cform->Mult(u, *y);
  Mu->AddMult(*g, *y, -1.0);
  //delete cform;
}

void inverseDiffusion::Dxc(BlockVector& x, BlockOperator* y)
{
  Vector u = x.GetBlock(0);
  Vector m = x.GetBlock(1);
  ParGridFunction m_gf(Vhm);
  m_gf = m;
  GridFunctionCoefficient m_gfc(&m_gf);

  ParBilinearForm * Ducform(new ParBilinearForm(Vhu));
  Ducform->AddDomainIntegrator(new DiffusionIntegrator(m_gfc));
  Ducform->AddDomainIntegrator(new MassIntegrator(*beta));
  Ducform->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  Ducform->Assemble();
  y->SetBlock(0, 0, Ducform);
  
  ParGridFunction * ugrad = new ParGridFunction(Vhgrad);
  Grad->Mult(u, *ugrad); 
  VectorGridFunctionCoefficient ugrad_gfc(ugrad);

  ParBilinearForm * Dmcform(new ParBilinearForm(Vhu));
  constexpr double alpha = 1.0;
  int skip_zeros = 0;
  Dmcform->AddDomainIntegrator(new ConvectionIntegrator(ugrad_gfc, alpha));
  //Dmcform->SetAssemblyLevel(AssemblyLevel::ELEMENT);
  Dmcform->Assemble(skip_zeros);
  SparseMatrix *Jm;
  Jm = &(Dmcform->SpMat());
  y->SetBlock(0, 1, Jm);
}

/*void inverseDiffusion::DxcTp(BlockVector& x, Vector &p, BlockVector *y)
{
  Vector u = x.GetBlock(0);
  Vector m = x.GetBlock(1);
  ParGridFunction m_gf(Vhm);
  m_gf = m;
  GridFunctionCoefficient m_gfc(&m_gf);

  ParBilinearForm * Ducform(new ParBilinearForm(Vhu));
  Ducform->AddDomainIntegrator(new DiffusionIntegrator(m_gfc));
  Ducform->AddDomainIntegrator(new MassIntegrator(*beta));
  Ducform->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  Ducform->Assemble();
  
  Vector yu(u.Size());
  Vector ym(m.Size());
  y->GetBlockView(0, yu);
  y->GetBlockView(1, ym);
  Ducform->Mult(p, yu);
  y->SetBlock(0, 0, Ducform);
  
  ParGridFunction * ugrad = new ParGridFunction(Vhgrad);
  ParGridFunction * pgrad = new ParGridFunction(Vhgrad);
  Grad->Mult(u, *ugrad); 
  Grad->Mult(p, *pgrad);
  VectorGridFunctionCoefficient ugrad_gfc(ugrad);

  ParBilinearForm * Dmcform(new ParBilinearForm(Vhu));
  constexpr double alpha = 1.0;
  int skip_zeros = 0;
  Dmcform->AddDomainIntegrator(new ConvectionIntegrator(ugrad_gfc, alpha));
  //Dmcform->SetAssemblyLevel(AssemblyLevel::ELEMENT);
  Dmcform->Assemble(skip_zeros);
  SparseMatrix *Jm;
  Jm = &(Dmcform->SpMat());
  y->SetBlock(0, 1, Jm);
}*/


void inverseDiffusion::Dxxcp(BlockVector& x, Vector &p, BlockOperator* y)
{
  Vector u = x.GetBlock(0);
  Vector m = x.GetBlock(1);
  /*ParGridFunction m_gf(Vhm);
  m_gf = m;
  GridFunctionCoefficient m_gfc(&m_gf);*/

  GridFunction * pgrad = new GridFunction(Vhgrad);
  Grad->Mult(p, *pgrad); 
  VectorGridFunctionCoefficient pgrad_gfc(pgrad);
  
  constexpr double alpha = 1.0;
  int skip_zeros = 0;

  ParBilinearForm * Dumcform(new ParBilinearForm(Vhu));
  Dumcform->AddDomainIntegrator(new ConvectionIntegrator(pgrad_gfc, alpha));
  //Dmcform->SetAssemblyLevel(AssemblyLevel::ELEMENT);
  Dumcform->Assemble(skip_zeros);
  SparseMatrix *Jum;
  Jum = &(Dumcform->SpMat());
  y->SetBlock(0, 1, Jum);
  
  ParBilinearForm * Dmucform(new ParBilinearForm(Vhu));
  Dmucform->AddDomainIntegrator(new ConvectionIntegrator(pgrad_gfc, alpha));
  //Dmcform->SetAssemblyLevel(AssemblyLevel::ELEMENT);
  Dmucform->Assemble(skip_zeros);
  SparseMatrix *Jmu;
  Jmu = &(Dmucform->SpMat());
  y->SetBlock(1, 0, Jmu);
}

double inverseDiffusion::phi(BlockVector &x, double mu)
{
  Vector m = x.GetBlock(1);
  Vector help;
  help.SetSize(m.Size());
  help.Set(1.0, m);
  help.Add(-1.0, *ml);
  for(int i = 0; i < help.Size(); i++)
  {
    help(i) = log(help(i));
  }
  return f(x) - mu * InnerProduct(MPI_COMM_WORLD, Mmlumped, help);   
}


void inverseDiffusion::Dxphi(BlockVector &x, double mu, BlockVector *y)
{
  Dxf(x, y);
  Vector u = x.GetBlock(0);
  Vector m = x.GetBlock(1);
  Vector yu(u.Size());
  Vector ym(m.Size());
  y->GetBlockView(0, yu);
  y->GetBlockView(1, ym);
  for(int i = 0; i < ym.Size(); i++)
  {
    ym(i) -= mu * Mmlumped(i) / (m(i) - ml->Elem(i));
  }
}


double inverseDiffusion::L(BlockVector &x, Vector &l, Vector &z)
{
  Vector u = x.GetBlock(0);
  Vector m = x.GetBlock(1);
  Vector cx;
  cx.SetSize(u.Size());
  c(x, &cx);
  Vector w1;
  w1.SetSize(m.Size());
  w1.Set(1.0, m);
  w1 -= *ml;
  w1 *= Mmlumped;
  return (f(x) + InnerProduct(MPI_COMM_WORLD, cx, l) - InnerProduct(MPI_COMM_WORLD, w1, z));
}



inverseDiffusion::~inverseDiffusion()
{
  // Do not free the memory that Vh points to.
  // There is other data that relies on the data contained
  // in the address that Vh points to.
  delete Rform;
  delete Mform;
  delete Muform; 
  delete Mwform; 
  /*delete u;
  delete p;
  delete m;
  delete uhat;
  delete phat;
  delete mhat;
  delete lam;
  delete lamhat;
  delete ml;
  delete mu;
  delete d;
  delete f;
  delete b;
  delete bp;
  delete aform;
  delete Rform;
  delete Mform;
  delete Cform;
  delete Muform;*/
}


relaxedDirac::relaxedDirac(int Dim, Vector & ObsPts, double Sig)
{
  dim = Dim;
  npts = ObsPts.Size() / dim;
  obsPts.SetSize(npts * dim);
  obsPts.Set(1.0, ObsPts);
  sig = Sig;
}


double relaxedDirac::relaxedDiracFunEval(const Vector & p)
{
  double f = 0.0;
  double arg;
  double ftemp;
  double xij;
  for(int i = 0; i < npts; i++)
  {
    arg = 0.0;
    ftemp = 0.0;
    for(int j = 0; j < dim; j++)
    {
      xij = obsPts(i + j *npts);
      arg += pow((p(j) - xij) / sig, 2.);
    }
    ftemp = exp(-0.5*arg);
    for(int j = 0; j < dim; j++)
    {
      ftemp /= (sig * pow(2.*M_PI, 0.5));
    }
    for(int j = 0; j < dim; j++)
    {
      xij = obsPts(i + j * npts);
      if( xij < 1.e-12 || xij > 1.-1.e-12)
      {
        ftemp *= 2.;
      }
    }


    f += ftemp;   
  }
  return f;
}
relaxedDirac::~relaxedDirac()
{

}
