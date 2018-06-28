#include "pore.h"  //class definign pore geom
#include "connectivity.h"   //class definign connectviyt
#include "conductancev2.h"  //class definign cinductance
#include "myfun.h"          //
#include "myfun2.h"
#include "solve.h"
#include "stiffness.h"
//include "post_processing_randv3.h" 
//#include "post_processing_rand_SF.h"
#include "post_processing_rand_numerical.h"
//#include "pressure_viz.h"
#include "volume_viz.h"
#include<iostream>
#include "mpi.h"
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cctype>
#include"depth_dakota.h"
#include <Teuchos_DefaultMpiComm.hpp>
#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_XMLParameterListCoreHelpers.hpp"
#include "Teuchos_ConfigDefs.hpp"
#include "Teuchos_config.h"

enum var_t { X1, X2 };

int main (int argc,char ** argv)
 {

//Dakota stuff starts here
 std::ifstream fin(argv[1]);
  if (!fin) {
    std::cerr << "\nError: failure opening " << argv[1] << std::endl;
    exit(-1);
  }
  size_t i, j, num_vars, num_fns, num_deriv_vars;
  std::string vars_text, fns_text, dvv_text;

  // define the std::string to enumeration map
  std::map<std::string, var_t> var_t_map;
  var_t_map["x1"] = X1;
  var_t_map["x2"] = X2;
 fin >> num_vars >> vars_text;
  std::map<var_t, double> vars;
  std::vector<var_t> labels(num_vars);
  double var_i; std::string label_i; var_t v_i;
  std::map<std::string, var_t>::iterator v_iter;
  for (i=0; i<num_vars; i++) {
    fin >> var_i >> label_i;
    transform(label_i.begin(), label_i.end(), label_i.begin(),
              (int(*)(int))tolower);
    v_iter = var_t_map.find(label_i);
    if (v_iter == var_t_map.end()) {
      std::cerr << "Error: label \"" << label_i
                << "\" not supported in analysis driver." << std::endl;
      exit(-1);
    }
    else
      v_i = v_iter->second;
    vars[v_i] = var_i;
    labels[i] = v_i;
  }
fin >> num_fns >> fns_text;
  std::vector<short> ASV(num_fns);
  for (i=0; i<num_fns; i++) {
    fin >> ASV[i];
    fin.ignore(256, '\n');
  }

  // Compute and output responses
  bool least_sq_flag = (num_fns > 1) ? true : false;
  double x1 = vars[X1], x2 = vars[X2];











  using Teuchos::Array;
  using Teuchos::arcp;
  using Teuchos::ArrayRCP;
  using Teuchos::ArrayView;
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::Time;
  using Teuchos::TimeMonitor;
  using Teuchos::tuple;
  using Teuchos::outArg;
  using std::endl;
  using Teuchos::ParameterList;

  typedef Tpetra::Vector<>::scalar_type scalar_type;
  typedef Tpetra::Map<>::local_ordinal_type local_ordinal_type;
  typedef Tpetra::Map<>::global_ordinal_type global_ordinal_type;
  typedef Tpetra::CrsMatrix<> matrix_type;
  typedef Tpetra::Operator<scalar_type, local_ordinal_type, global_ordinal_type> op_type;
  typedef Tpetra::MultiVector<scalar_type, local_ordinal_type, global_ordinal_type> vec_type;
  typedef Tpetra::Map<local_ordinal_type, global_ordinal_type> map_type;
  typedef typename Kokkos::Details::ArithTraits<scalar_type>::val_type impl_scalar_type;
  typedef  Tpetra::CrsMatrix<>::node_type Node;
  typedef Tpetra::CrsMatrix<>::device_type device_type;
  typedef Tpetra::CrsMatrix<>:: execution_space execution_space;
  typedef Kokkos::CrsMatrix<impl_scalar_type, local_ordinal_type, execution_space> local_matrix_type;
  typedef local_matrix_type::values_type k_values;


   const global_ordinal_type indexBase = 0;

  MPI_Init(&argc,&argv);
  Kokkos::initialize(argc,argv);
  srand(time(0));
RCP<const Teuchos::Comm<int> > comm =
    Tpetra::DefaultPlatform::getDefaultPlatform().getComm();

RCP<Teuchos::ParameterList> parlist =rcp(new Teuchos::ParameterList("Parameters"));

std::string xmlInFileName ="pore_input.xml";
//std::string name="numHor";
if(xmlInFileName.length())
 {
  Teuchos::updateParametersFromXmlFile(xmlInFileName,parlist.ptr());
}

 const Tpetra::global_size_t m =  parlist->get<double>("numHor");
 const Tpetra::global_size_t n =  parlist->get<double>("numVer");

 




  double dt=0.05;
  double visc1=20*(1e-3);
  double visc2=3*(1e-3);
  double surf= 0
  //double surf_p=surf*2*(1e+6)/(16.5);
//  int m=50;
//  int n=50;
  int N[4]={0};
  double time =1;
  double numtimesteps=parlist->get<double>("time")+1;
  double dep_Dak=0;
  int numGlobalElements=m*n+2*m;
//  double sol[numGlobalElements][(int)numtimesteps];
   RCP<const map_type > map =
    rcp (new map_type (numGlobalElements, indexBase, comm ));
   ArrayView<const global_ordinal_type> myGlobalElements = map->getNodeElementList();
  local_ordinal_type numMyElements= map->getNodeNumElements();
  Kokkos::View<double*,Kokkos::LayoutRight> arr1D("arr1D",m*n,1);
  Kokkos::View<double **,Kokkos::LayoutRight> arr2D("arr2D",m,n);

  Pore pore;
  pore.setViscosity(3*(1e-3),x1);
  pore.setSurfaceTension(surf);

  Connectivity con;
  con.setSize(m,n);
  con.setConnectionMatrix(arr1D,arr2D);

  ArrayRCP<size_t> NumNz_k=arcp<size_t>(numMyElements);
  ArrayRCP<size_t> NumNz=arcp<size_t>(numMyElements);
   myfun(map,N,m,n,arr2D,NumNz_k,myGlobalElements);
   myfun2(map,N,m,n,arr2D,NumNz,myGlobalElements);
   RCP<matrix_type> L=rcp(new matrix_type (map,NumNz_k));
   RCP<matrix_type> R=rcp(new matrix_type (map,NumNz_k));
   
   pore.setupLengthMatrix(1,L,arr2D,myGlobalElements,map,m,n,N);
   pore.setupRadiusMatrix(1,R,arr2D,myGlobalElements,map,m,n,N);
 
  Conductance conductance;
  RCP<matrix_type> K = rcp (new matrix_type (map,NumNz_k));
  RCP<matrix_type> A = rcp (new matrix_type (map, NumNz));

  RCP<matrix_type> Vol_frac = rcp (new matrix_type (map,NumNz_k));
  RCP<matrix_type> Current = rcp (new matrix_type (map,NumNz_k));
  std::cout<<"rank"<<comm->getRank(); 
  for ( double time=0; time <numtimesteps; time ++)
  {
          

  conductance.fillConductanceMatrix(R,L,K, N, m, n,time,Vol_frac, myGlobalElements,arr2D,map,pore);




   stiffnessMatrix stm;
  stm.setupStiffnessMatrix(A,K,map,m,n,myGlobalElements,time);
  
  Kokkos::View<double *, Kokkos::LayoutRight> b("b",numGlobalElements,1); 
  
  Kokkos::DualView<double**, Kokkos::LayoutLeft> bcl ("bcl", numGlobalElements,1);
bcl.modify<Kokkos::DualView<double**, Kokkos::LayoutLeft>::t_dev::execution_space> ();
Kokkos::deep_copy (Kokkos::subview (bcl.d_view, Kokkos::ALL (), 0), b);

  stm.setupMultiVector(A,map,b,bcl,myGlobalElements,m,n);


   RCP<vec_type> X = rcp (new vec_type (A->getDomainMap (), 1)); // Set to zeros by default.

   RCP<vec_type> B = rcp (new vec_type (A->getRangeMap (), bcl));
   
   solve<vec_type, op_type> ( *X, *B, *A) ;
   
   X->template sync<Kokkos::HostSpace>();
   
   auto x_2d=X->template getLocalView<Kokkos::HostSpace> ();
   auto x_1d=Kokkos::subview (x_2d, Kokkos::ALL (), 0);

postProcessing pp;
pp.setupCurrentMatrix(L,Current,x_1d,map,myGlobalElements,K,Vol_frac,time,arr2D,N,m,n,dt,surf);   


        
//dump_volume(Vol_frac,map,myGlobalElements,time);
//if(time==97)
// dep_Dak=depth(Vol_frac,map,myGlobalElements,time,m);

}


 dep_Dak=depth(Vol_frac,map,myGlobalElements,time,m);


Kokkos::finalize();
MPI_Finalize();
std::ofstream fout(argv[2]);
  if (!fout) {
    std::cerr << "\nError: failure creating " << argv[2] << std::endl;
    exit(-1);
  }
  fout.precision(15); // 16 total digits
  fout.setf(std::ios::scientific);
  fout.setf(std::ios::right);

  if (least_sq_flag) {
    std::cout << "least_squre_flag :on " << std::endl;
    if (ASV[0] & 1) fout << "                     " << dep_Dak  << " f1\n";
  }
else {
    if (ASV[0] & 1) fout << "                     " << dep_Dak  << " f\n";
  }

//Dakota Stuff starts here


fout.flush();
  fout.close();

 }
