#include "mfem.hpp"
#include "problems.hpp"
#include "IPsolver.hpp"

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
double pFun(const Vector &);
double mTrueFun(const Vector &);

int main(int argc, char *argv[])
{
  // Initialize MPI and HYPRE
  MPI::Init(argc, argv);
  int nProcs = Mpi::WorldSize();
  int rank   = Mpi::WorldRank();
  bool iAmRoot = (rank == 0);
  Hypre::Init();
	

  int ref_levels = 1;
  int ser_ref_levels =1 ;
  int order = 1;
  int orderm = 1;
  int gdim = 2;
  int outputVTK = 0;

  const char *device_config = "cpu";
  OptionsParser args(argc, argv);
  args.AddOption(&ser_ref_levels, "-sr", "--serialrefine",\
		  "Number of times to refine the serial mesh uniformly.");
  args.AddOption(&ref_levels, "-r", "--parallelrefine",\
		  "Number of times to refine the parallel mesh uniformly.");
  args.AddOption(&order, "-o", "--order",\
		  "Order (degree) of the finite elements (state).");
  args.AddOption(&orderm, "-om", "--orderm",\
		  "Order (degree) of the finite elements (parameter).");
  args.AddOption(&gdim, "-dim", "--dimension", \
      "Geometric dimension of the problem.");
  args.AddOption(&outputVTK, "-outputVTK", "--outputVTK", \
       "Save VTK files of Newton iterates.");


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
  MFEM_VERIFY(gdim == 2 || gdim == 3, "geometric dimension must be either 2 or 3");


  
  string mFile = "inline-";
  string mFileSimple = "inline-";
  if(gdim == 2)
  {
    mFile       = mFile       + "quad.mesh";
    mFileSimple = mFileSimple + "quad-coarse.mesh";
  }
  else
  {
    mFile       = mFile       + "hex.mesh";
    mFileSimple = mFileSimple + "hex-coarse.mesh";    
  }

  const char *mesh_file = mFile.data();
  const char *simple_mesh_file = mFileSimple.data();
  Mesh *serial_mesh = new Mesh(mesh_file, 1, 1);
  Mesh *serial_mesh_simple = new Mesh(simple_mesh_file, 1, 1);

  int mesh_poly_deg     = 1;
  int ncomp             = 1;
  int dim = serial_mesh->Dimension(); // geometric dimension
  
  // refine the mesh up to a certain number of dofs
  for(int lev = 0; lev < ser_ref_levels; lev++)
  {
    serial_mesh->UniformRefinement();
  }


  ParMesh *mesh = new ParMesh(MPI_COMM_WORLD, *serial_mesh);
  // Mesh bounding box (for the full serial mesh).
  Vector pos_min, pos_max;
  MFEM_VERIFY(mesh_poly_deg > 0, "The order of the mesh must be positive.");
  MFEM_VERIFY(order > 0, "The elementwise polynomial degree of the finite element space for the state must be positive")
  
	  
  
  for(int lev = 0; lev < ref_levels; lev++)
  {
    mesh->UniformRefinement();
  }
  double h_min, h_max, kappa_min, kappa_max;
  mesh->GetCharacteristics(h_min, h_max, kappa_min, kappa_max);
  double sigma;
  //sigma = 2. * h_min; // width of Gaussian supports for the Dirac comb used to define observation operator
  sigma = 0.075;



  // choose the abstract FE spaces
  FiniteElementCollection *fec     = new H1_FECollection(order, dim);
  FiniteElementCollection *fecm    = new H1_FECollection(orderm, dim);
  FiniteElementCollection *fecdata = new H1_FECollection(1, dim);
  FiniteElementCollection *fecgrad = new ND_FECollection(order, dim);
  FiniteElementCollection *fecL2   = new L2_FECollection(order, dim);
  
  // fix them over respective meshes
  ParFiniteElementSpace *Vhmesh = new ParFiniteElementSpace(mesh, fec, dim); // used to set the NodalFESpace of the mesh
  FiniteElementSpace    *Vhdatamesh = new FiniteElementSpace(serial_mesh_simple, fecdata, dim);
  mesh->SetNodalFESpace(Vhmesh);
  serial_mesh_simple->SetNodalFESpace(Vhdatamesh);

  
  
  ParFiniteElementSpace *Vhu    = new ParFiniteElementSpace(mesh, fec);
  ParFiniteElementSpace *Vhm    = new ParFiniteElementSpace(mesh, fecm);
  ParFiniteElementSpace *Vhgrad = new ParFiniteElementSpace(mesh, fecgrad);
  ParFiniteElementSpace *VhL2   = new ParFiniteElementSpace(mesh, fecL2);
  FiniteElementSpace *Vhdata    = new FiniteElementSpace(serial_mesh_simple, fecdata);

  const Operator * Ru = Vhu->GetRestrictionOperator();
  const Operator * Rm = Vhm->GetRestrictionOperator();
  const Operator * Pu = Vhu->GetProlongationMatrix();
  const Operator * Pm = Vhm->GetProlongationMatrix();  


  HYPRE_BigInt dimUGlobal = Vhu->GlobalTrueVSize();
  HYPRE_BigInt dimMGlobal = Vhm->GlobalTrueVSize();
  if(iAmRoot)
  {
    cout << "dim(state) = "     << dimUGlobal << endl;
    cout << "dim(parameter) = " << dimMGlobal << endl;
    cout << "my order = "       << Vhu->GetOrder(0) << endl;
    std::ofstream problemDimStream;
    problemDimStream.open("problemDim.dat", ios::out | ios::trunc);
    problemDimStream << setprecision(30) << dimUGlobal << endl;
    problemDimStream.close();
  }
   
  // Determine the list of true (i.e. conforming) essential boundary dofs.
  Array<int> ess_tdof_list;
  /*if (mesh->bdr_attributes.Size())
  {
    Array<int> ess_bdr(mesh->bdr_attributes.Max());
    ess_bdr = 0;
    Vhu->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
  }*/
  
  // set up data
  // m    -- initial parameter
  // ml   -- parameter lower bound
  // u    -- state (used to set Dirichlet conditions)
  // g    -- rhs of state equation
  // d    -- `observation' data
  FunctionCoefficient m_fc(mFun);
  FunctionCoefficient ml_fc(mlFun);
  FunctionCoefficient u_fc(uFun);
  FunctionCoefficient g_fc(gFun);
  FunctionCoefficient d_fc(dFun);
  FunctionCoefficient p_fc(pFun);
  FunctionCoefficient mtrue_fc(mTrueFun);

  
  ParGridFunction m_gf(Vhm);
  ParGridFunction ml_gf(Vhm);
  ParGridFunction u_gf(Vhu);
  ParGridFunction p_gf(Vhu);
  ParGridFunction g_gf(Vhm);
  ParGridFunction d_gf(Vhu);
  ParGridFunction mtrue_gf(Vhm);
  GridFunction d1_gf(Vhdata);
  
  
  // fc --> gf
  ml_gf.ProjectCoefficient(ml_fc);
  m_gf.ProjectCoefficient(m_fc);
  u_gf.ProjectCoefficient(u_fc);
  p_gf.ProjectCoefficient(p_fc);
  g_gf.ProjectCoefficient(g_fc);
  d_gf.ProjectCoefficient(d_fc);
  d1_gf.ProjectCoefficient(d_fc);
  mtrue_gf.ProjectCoefficient(mtrue_fc);

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
    for(int j = 0; j < dim; j++)
    {
      interpolationPoints(i + j*pts_cnt) = (serial_mesh_simple->GetVertex(i))[j];
    }
    /*if(iAmRoot)
    {
      switch(dim)
      {
        case 1:
          cout << "x = " << interpolationPoints(i) << ")\n";
          break;
        case 2:
          cout << "(x, y) = (" << interpolationPoints(i) << ", " << interpolationPoints(i + pts_cnt) << ")\n";
          break;
        case 3:
          cout << "(x, y, z) = (" << interpolationPoints(i) << ", " << interpolationPoints(i + pts_cnt) << ", "
                                  << interpolationPoints(i + 2*pts_cnt) << ")\n";
          break;
      }
    }*/
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
  /*for(int i = 0; i < 4; i++)
  {
    if(iAmRoot)
    {
    cout << "d(" << i << ") = " << d1_gf(i) << " (rank " << rank << ")\n";
    }
  }*/
  noise *= noise_lvl/d1_gf.Norml2();
  d1_gf += noise;
  /*for(int i = 0; i < 4; i++)
  {
    if(iAmRoot)
    {
    cout << "d(" << i << ") = " << d1_gf(i) << " (rank " << rank << ")\n";
    }
  }*/


  
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
  paraview_dc.RegisterField("ud (interpolated)", &ud);
  paraview_dc.RegisterField("ud", &d_gf);
  paraview_dc.RegisterField("mtrue", &mtrue_gf);
  paraview_dc.Save();
  
  Device device(device_config);
  if(iAmRoot)
  {
    device.Print();
  }
  MemoryType mt = device.GetMemoryType();
  
  // Regularization Coefficients
  double gamma1, gamma2, beta;
  if(gdim == 2)
  {
    gamma1 = 1.e-1;
    gamma2 = 1.e-2;
  }
  else
  {
    if(gdim == 3)
    {
      gamma1 = 1.e-1;
      gamma2 = 1.e-2;
    }
  }
  beta   = 1.e0;
  inverseDiffusion problem(Vhu, Vhm, Vhgrad, VhL2, ud, g_gf, ml_gf,\
                           w_gf, ess_tdof_list, beta, gamma1, gamma2, mt);
  interiorPtSolver solver(&problem);
  if(!(outputVTK == 0))
  {
    solver.SetOutputVTK(true);
  }  

  int dimU = problem.getdimU();
  int dimM = problem.getdimM();
  int dimUmaxGlb, dimUminGlb, dimMmaxGlb, dimMminGlb; //max and min local dofs over each MPI process 
  /* the following is bad practice, generally one should reduce the total number of Allreduce's in favor of sending larger messages */ 
  MPI_Allreduce(&dimU, &dimUmaxGlb, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&dimU, &dimUminGlb, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&dimM, &dimMmaxGlb, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&dimM, &dimMminGlb, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if(iAmRoot)
  {
    cout << "checking for potential load imbalance issues due to mesh partioning" << endl;
    cout << "max local dim(state) = "     << dimUmaxGlb << endl;
    cout << "min local dim(state) = "     << dimUminGlb << endl;
    cout << "max local dim(parameter) = " << dimMmaxGlb << endl;
    cout << "min local dim(parameter) = " << dimMminGlb << endl;
  }

  
  Array<int> block_offsetsx(3); // number of variables + 1
  block_offsetsx[0] = 0;
  block_offsetsx[1] = dimU;
  block_offsetsx[2] = dimM;
  block_offsetsx.PartialSum();
  
  Array<int> block_offsetsu(2);
  block_offsetsu[0] = 0;
  block_offsetsu[1] = dimU;
  block_offsetsu.PartialSum();

  Array<int> block_offsetsuml(4);
  block_offsetsuml[0] = 0;
  block_offsetsuml[1] = dimU;
  block_offsetsuml[2] = dimM;
  block_offsetsuml[3] = dimU;
  block_offsetsuml.PartialSum();

  Array<int> block_offsetsumlz(5);
  block_offsetsumlz[0] = 0;
  block_offsetsumlz[1] = dimU;
  block_offsetsumlz[2] = dimM;
  block_offsetsumlz[3] = dimU;
  block_offsetsumlz[4] = dimM;
  block_offsetsumlz.PartialSum();


  BlockVector x(block_offsetsx, mt); x = 0.0;
  Vector uView(dimU); uView = 0.0;
  Vector mView(dimM); mView = 0.0;
  Vector l(dimU); l = 0.0;
  Rm->Mult(m_gf, mView);
  Ru->Mult(u_gf, uView);
  x.GetBlock(0).Set(1.0, uView);
  x.GetBlock(1).Set(1.0, mView);
  Ru->Mult(p_gf, l);

  

  problem.feasibilityRestoration(x, 1.e-12);
  BlockVector X0opt(block_offsetsumlz), Xfopt(block_offsetsumlz); X0opt = 100.0; Xfopt = 0.0;
  X0opt.GetBlock(0).Set(1.0, x.GetBlock(0));
  X0opt.GetBlock(1).Set(1.0, x.GetBlock(1));
  X0opt.GetBlock(2).Set(1.0, l);

  double tolOpt = 1.e-6;
  int maxOptIt = 75;
  double mu0 = 1.0;

  StopWatch optStopWatch;
  optStopWatch.Clear();
  optStopWatch.Start();
  solver.solve(X0opt, Xfopt, tolOpt, maxOptIt, mu0);
  optStopWatch.Stop();
  MFEM_VERIFY(solver.GetConverged(), "Interior point solver did not converge.");
  if(iAmRoot)
  {
    cout << "Optimizer was computed in, " << optStopWatch.RealTime() << "[sec]." << endl;
    std::ofstream timerDataStream;
    timerDataStream.open("optTime.dat", ios::out | ios::trunc);
    timerDataStream << setprecision(30) << optStopWatch.RealTime() << endl;
    timerDataStream.close();
    cout << "MFEM_TIMER_TYPE = " << MFEM_TIMER_TYPE << endl;
  }


  delete Vhu;
  delete Vhm;
  delete Vhgrad;
  delete VhL2;
  delete Vhdata;
  delete Vhmesh;
  delete Vhdatamesh;
  delete mesh;
  delete serial_mesh;
  delete serial_mesh_simple;
  delete fec;
  delete fecm;
  delete fecdata;
  delete fecgrad;
  delete fecL2; 

  MPI::Finalize();
  return 0;
}


double mFun(const Vector &x)
{
  return 100.0;
}

double mlFun(const Vector &x)
{
  return 1.0;
}

double uFun(const Vector &x)
{
  return 0.; 
}

double gFun(const Vector &x)
{
  double val1 = 0.5;
  for(int i = 0; i < x.Size(); i++)
  {
    val1 += x(i);
  }
  val1 *= x.Size() * pow(M_PI, 2);
  val1 += 1.0;  
  for(int i = 0; i < x.Size(); i++)
  {
    val1 *= cos(M_PI * x(i));
  }
  double val2;
  for(int i = 0; i < x.Size(); i++)
  {
    val2 = 1.0;
    for(int j = 0; j < x.Size(); j++)
    {
      if(i == j)
      {
        val2 *= sin(M_PI * x(j));
      }
      else
      {
        val2 *= cos(M_PI*x(i));
      }
    }
    val1 += M_PI * val2;
  }
  return val1;  
  /*double val = 0.0;
  for(int i = 0; i < x.Size(); i++)
  {
    val += pow(M_PI, 2);
  }
  val *= (0.5 + x(0));
  val += 1.0;
  for(int i = 0; i < x.Size(); i++)
  {
    val *= cos(M_PI * x(i));
  }
  double val1 = 1.0;
  val1 = (-1.0) * sin(M_PI * x(0));
  for(int i = 1; i < x.Size(); i++)
  {
    val1 *= cos(M_PI * x(i));
  }
  val -= val1;
  return val;*/
}

double dFun(const Vector &x)
{
  double val = 1.0;
  for(int i = 0; i < x.Size(); i++)
  {
    val *= cos(M_PI * x(i));
  }
  return val;
}

double pFun(const Vector &x)
{
  return 0.;
}

double mTrueFun(const Vector &x)
{
  double val = 0.5;
  for(int i = 0; i < x.Size(); i++)
  {
    val += x(i);
  }
  return val;
}


