#include "mfem.hpp"
#include "problems.cpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace std::placeholders;
using namespace mfem;

// initalizers for parameter and state
// rhs of PDE
double mFun(const Vector &);
double mlFun(const Vector &);

double uFun(const Vector &);
double gFun(const Vector &);
double dFun(const Vector &);


int main(int argc, char *argv[])
{
  // Initialize MPI and HYPRE
  MPI::Init(argc, argv);
  int nProcs = Mpi::WorldSize();
  int rank   = Mpi::WorldRank();
  bool iAmRoot = (rank == 0);
  Hypre::Init();
	
  const char *mesh_file = "inline-tri.mesh";
  const char *simple_mesh_file = "inline-tri-coarse.mesh";
  int ref_levels = 1;
  int order = 1;
  int orderm = 1;
  double sigma = 0.01;
  const char *device_config = "cpu";
  OptionsParser args(argc, argv);
  args.AddOption(&ref_levels, "-r", "--refine",\
		  "Number of times to refine the mesh uniformly.");
  args.AddOption(&order, "-o", "--order",\
		  "Order (degree) of the finite elements (state).");
  args.AddOption(&orderm, "-om", "--orderm",\
		  "Order (degree) of the finite elements (parameter).");
  args.AddOption(&sigma, "-sig", "--sigma",\
		  "Width of the Gaussian support of the Dirac approximations.");

  args.Parse();
  if(!args.Good())
  {
    args.PrintUsage(cout);
    return 1;
  }
  else
  {
    if(iAmRoot)
    {  
      args.PrintOptions(cout);
    }
  }
  
  Mesh *serial_mesh;
  Mesh *serial_mesh_simple;
  serial_mesh = new Mesh(mesh_file, 1, 1);
  serial_mesh_simple = new Mesh(simple_mesh_file, 1, 1);

  int mesh_poly_deg     = 1;
  int ncomp             = 1;
  int dim = serial_mesh->Dimension(); // geometric dimension
  
  ParMesh *mesh;
  mesh = new ParMesh(MPI_COMM_WORLD, *serial_mesh);
  ParMesh *simplemesh;
  simplemesh = new ParMesh(MPI_COMM_WORLD, *serial_mesh_simple);
  // Mesh bounding box (for the full serial mesh).
  Vector pos_min, pos_max;
  MFEM_VERIFY(mesh_poly_deg > 0, "The order of the mesh must be positive.");
  MFEM_VERIFY(order > 0, "The elementwise polynomial degree of the finite element space for the state must be positive")
  serial_mesh->Clear();
  //serial_mesh_simple->Clear();
  
  for(int lev = 0; lev < ref_levels; lev++)
  {
    mesh->UniformRefinement();
  }

  // choose the abstract FE spaces
  FiniteElementCollection *fec     = new H1_FECollection(order, dim);
  FiniteElementCollection *fecm    = new H1_FECollection(orderm, dim);
  FiniteElementCollection *fecdata = new H1_FECollection(1, dim);
  FiniteElementCollection *fecgrad = new ND_FECollection(order, dim);
  FiniteElementCollection *fecL2   = new L2_FECollection(pow(order, 2), dim);
  
  // fix them over respective meshes
  ParFiniteElementSpace *Vhmesh = new ParFiniteElementSpace(mesh, fec, dim); // used to set the NodalFESpace of the mesh
  FiniteElementSpace    *Vhdatamesh = new FiniteElementSpace(serial_mesh_simple, fecdata, dim);
  mesh->SetNodalFESpace(Vhmesh);
  serial_mesh_simple->SetNodalFESpace(Vhdatamesh);
  
  
  ParFiniteElementSpace *Vhu    = new ParFiniteElementSpace(mesh, fec);
  ParFiniteElementSpace *Vhm    = new ParFiniteElementSpace(mesh, fecm);
  ParFiniteElementSpace *Vhgrad = new ParFiniteElementSpace(mesh, fecgrad);
  ParFiniteElementSpace *VhL2   = new ParFiniteElementSpace(mesh, fecL2);
  FiniteElementSpace *Vhdata = new FiniteElementSpace(serial_mesh_simple, fecdata);
  


  if(iAmRoot)
  {
    cout << "dim(state) = " << Vhu->GetTrueVSize() << endl;
    cout << "dim(parameter) = " << Vhm->GetTrueVSize() << endl;
  }
   
  // Determine the list of true (i.e. conforming) essential boundary dofs.
  Array<int> ess_tdof_list;
  if (mesh->bdr_attributes.Size())
  {
    Array<int> ess_bdr(mesh->bdr_attributes.Max());
    ess_bdr = 1;
    Vhu->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
  }
  
  // set up data
  // m    -- initial parameter
  // ml   -- parameter lower bound
  // mu   -- parameter upper bound
  // u    -- state (used to set Dirichlet conditions)
  // f    -- rhs of state equation
  // d    -- `observation' data
  FunctionCoefficient m_fc(mFun);
  FunctionCoefficient ml_fc(mlFun);
  FunctionCoefficient u_fc(uFun);
  FunctionCoefficient g_fc(gFun);
  FunctionCoefficient d_fc(dFun);

  
  ParGridFunction m_gf(Vhm);
  ParGridFunction ml_gf(Vhm);
  ParGridFunction u_gf(Vhu);
  ParGridFunction g_gf(Vhm);
  ParGridFunction d_gf(Vhu);
  GridFunction d1_gf(Vhdata);
  
  
  // fc --> gf
  ml_gf.ProjectCoefficient(ml_fc);
  m_gf.ProjectCoefficient(m_fc);
  u_gf.ProjectCoefficient(u_fc);
  g_gf.ProjectCoefficient(g_fc);
  d_gf.ProjectCoefficient(d_fc);
  d1_gf.ProjectCoefficient(d_fc);

  // Generate equidistant points in physical coordinates over the whole mesh.
  // Note that some points might be outside, if the mesh is not a box. Note
  // also that all tasks search the same points (not mandatory).
  int pts_cnt = serial_mesh_simple->GetNV();
  Vector interpolationPoints(pts_cnt * dim);
  if(iAmRoot)
  {
    cout << "---- Interpolation points ---- \n";
  }
  for(int i = 0; i < pts_cnt; i++)
  {
    interpolationPoints(i) = (serial_mesh_simple->GetVertex(i))[0];
    interpolationPoints(i + pts_cnt) = (serial_mesh_simple->GetVertex(i))[1];
    if(iAmRoot)
    {
      cout << "(x, y) = (" << interpolationPoints(i) << ", " << interpolationPoints(i + pts_cnt) << ")\n";
    }
  }
  
  // Find and Interpolate FE function values on the desired points.
  int vec_dim = ncomp;
  Vector d(pts_cnt*vec_dim);
  FindPointsGSLIB finder(MPI_COMM_WORLD);
  finder.Setup(*mesh);
  finder.Interpolate(interpolationPoints, d_gf, d);
  Array<unsigned int> code_out    = finder.GetCode();
  Array<unsigned int> task_id_out = finder.GetProc();
  Array<unsigned int> mfem_elem   = finder.GetElem();
  Vector dist_p_out = finder.GetDist();
  
  // Desire    --- corrupt observation data d
  // Challenge --- a global copy of d is on each
  //               MPI process, how to ensure that
  //               the same random noise is used
  //               to corrupt d so that each MPI process
  //               has the same noisy observation vector d
  d1_gf = d; 

  double noise_lvl = 0.01;
  Vector noise;
  noise.SetSize(d.Size());
  
  // noise is consistent on each process...
  // but how to do Gaussian sampling...
  int seed = 1;
  noise.Randomize(seed);
  for(int i = 0; i < 4; i++)
  {
    if(iAmRoot)
    {
    cout << "d(" << i << ") = " << d1_gf(i) << " (rank " << rank << ")\n";
    }
  }
  noise *= noise_lvl/d1_gf.Norml2();
  d1_gf += noise;
  for(int i = 0; i < 4; i++)
  {
    if(iAmRoot)
    {
    cout << "d(" << i << ") = " << d1_gf(i) << " (rank " << rank << ")\n";
    }
  }


  
  ParGridFunction ud(Vhu);

  const int NE = mesh->GetNE(),
             nsp = Vhu->GetFE(0)->GetNodes().GetNPoints(),
             tar_ncomp = ud.VectorDim();
  Vector vxyz;
  vxyz = *(mesh->GetNodes());
  Vector iterp_vals2;
  const int nodes_cnt2 = vxyz.Size() / dim;
  Vector interp_vals2(nodes_cnt2);
  FindPointsGSLIB finder2;
  finder2.Setup(*serial_mesh_simple);
  finder2.Interpolate(vxyz, d1_gf, interp_vals2);
  
  ud = interp_vals2;



  relaxedDirac w(dim, interpolationPoints, sigma);
  std::function<double (const Vector &)> wFun = [&](const Vector & p) -> double { return w.relaxedDiracFunEval(p); };


  FunctionCoefficient w_fc(wFun);
  ParGridFunction w_gf(Vhu);
  w_gf.ProjectCoefficient(w_fc);
  ParaViewDataCollection paraview_dc("DiracWeightFunction", mesh);
  paraview_dc.SetPrefixPath("ParaView");
  paraview_dc.SetLevelsOfDetail(order);
  paraview_dc.SetDataFormat(VTKFormat::BINARY);
  paraview_dc.SetHighOrderOutput(true);
  paraview_dc.SetCycle(0);
  paraview_dc.SetTime(0.0);
  paraview_dc.RegisterField("w", &w_gf);
  paraview_dc.Save();
  
  
  // Regularization Coefficients
  double gamma1 = 1.e-1;
  double gamma2 = 1.e-2;
  double beta   = 1.; 
  inverseDiffusion problem(Vhu, Vhm, Vhgrad, VhL2, ud, g_gf, ml_gf, w_gf, ess_tdof_list, beta, gamma1, gamma2);

  Array<int> block_offsetsx(3); // number of variables + 1
  block_offsetsx[0] = 0;
  block_offsetsx[1] = Vhu->GetVSize();
  block_offsetsx[2] = Vhm->GetVSize();
  block_offsetsx.PartialSum();

  Array<int> block_offsetsu(2);
  block_offsetsu[0] = 0;
  block_offsetsu[1] = Vhu->GetVSize();
  block_offsetsu.PartialSum();
 
    
  
  Device device(device_config);
  if(iAmRoot)
  {
    device.Print();
  }
  MemoryType mt = device.GetMemoryType();
  BlockVector x(block_offsetsx, mt);
  BlockVector y(block_offsetsx, mt);
  BlockVector xdir(block_offsetsx, mt);
  x *= 0.;
  Vector mView(Vhm->GetVSize());
  Vector uView(Vhu->GetVSize());
  x.GetBlockView(0, uView);
  x.GetBlockView(1, mView);
  uView += 1.;
  mView += 0.;

  double f0 = problem.f(x);
  if(iAmRoot)
  {
    std::cout << "f(x) = " << f0 << " \n";
  }
  std::cout << "f(x) = " << f0 << ", (MPI process " << rank << ")\n";
  
  problem.Dxf(x, &y);
  
  double ysqr = InnerProduct(MPI_COMM_WORLD, y, y);
  xdir.Randomize(seed);
  //xdir += 1.;
  

  BlockVector x1(block_offsetsx, mt);
  double eps = 0.5;
  double f1;

  std::ofstream epsdata;
  std::ofstream graderrdata;
  epsdata.open("./fd_data/eps.dat", ios::out | ios::trunc);
  graderrdata.open("./fd_data/graderr.dat", ios::out | ios::trunc);

  double yxdir = InnerProduct(MPI_COMM_WORLD, y, xdir);

  for (int i = 0; i < 40; i++)
  {
    x1 *= 0.;
    x1.Add(eps, xdir);
    x1.Add(1.0, x);
    f1 = problem.f(x1);
    if(iAmRoot)
    {
      std::cout << "[(f1 - f0) / (x1 - x0)] / <g, xhat> = " << ((f1- f0) / (eps)) / yxdir << "\n";
      epsdata << setprecision(30) << eps << "\n";
      graderrdata << setprecision(30) << abs((((f1-f0)/eps) - yxdir)) << "\n";
    }
    eps *= 0.5;
  }
  epsdata.close();
  graderrdata.close();

  
  Vector r0, r1, Jx0, z;
  r0.SetSize(Vhu->GetNDofs());
  r1.SetSize(Vhu->GetNDofs());
  Jx0.SetSize(Vhu->GetNDofs());
  z.SetSize(Vhu->GetNDofs());

  problem.c(x, &r0);



  BlockOperator J0(block_offsetsu, block_offsetsx);
  problem.Dxc(x, &J0);
  J0.Mult(xdir, Jx0);
  
  eps = 0.5;
  std::ofstream cgraderrdata;
  cgraderrdata.open("./fd_data/cgraderr.dat", ios::out | ios::trunc);
  double err;
  for (int i = 0; i < 40; i++)
  {
    x1 *= 0.;
    x1.Add(eps, xdir);
    x1.Add(1.0, x);
    problem.c(x1, &r1);
    z *= 0.;
    z.Add(1.0, r1);
    z.Add(-1.0, r0);
    z /= eps;
    z.Add(-1.0, Jx0);
    err = z.Norml2();
    if(iAmRoot)
    {
      cgraderrdata << setprecision(30) << err << "\n";
    }
    eps *= 0.5;
  }
  cgraderrdata.close();

  double mu = 0.1;
  double phi0, phi1;
  BlockVector Dxphi0(block_offsetsx, mt);
  problem.Dxphi(x, mu, &Dxphi0);
  phi0 = problem.phi(x, mu);
  eps  = 0.5;
  double Dxphixdir = InnerProduct(MPI_COMM_WORLD, Dxphi0, xdir);
  
  std::ofstream phigraderrdata;
  phigraderrdata.open("./fd_data/phigraderr.dat", ios::out | ios::trunc);
  for (int i = 0; i < 40; i++)
  {
    x1 *= 0.;
    x1.Add(eps, xdir);
    x1.Add(1.0, x);
    phi1 = problem.phi(x1, mu);
    if(iAmRoot)
    {
      phigraderrdata << abs(((phi1 - phi0)/eps - Dxphixdir)) << endl;
    }
    eps *= 0.5;
  }
  phigraderrdata.close();


  Vector p(Vhu->GetVSize());
  Vector zl(Vhm->GetVSize());
  p  += 1.;
  zl -= 1.;
  BlockOperator Dxxcp(block_offsetsx, block_offsetsx);
  problem.Dxxcp(x, p, &Dxxcp);

  cout << problem.L(x, p, zl) << endl; 





  MPI::Finalize();
  return 0;
}


double mFun(const Vector &p)
{
  return 0.;
}

double mlFun(const Vector &p)
{
  return -30.;
}

double uFun(const Vector &p)
{
  return 0.; 
}

double gFun(const Vector &p)
{
  return 0.;
}

double dFun(const Vector &p)
{
  return cos(M_PI*p(0))*cos(M_PI*p(1));
}
