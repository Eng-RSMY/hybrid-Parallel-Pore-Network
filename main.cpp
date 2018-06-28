#include "pore.h"  //class definign pore geom
#include "connectivity.h"   //class definign connectviyt
#include "conductancev2.h"  //class definign cinductance
#include "myfun.h"          //
#include "myfun2.h"
#include "solve.h"
#include "stiffness.h"
//#include "post_processing_randv3.h" 
//#include "post_processing_rand_SF.h"
#include "post_processing_rand_numerical.h"
#include "volume_viz.h"
#include<iostream>
#include "mpi.h"
#include "depth_dakota.h"
#include <Teuchos_DefaultMpiComm.hpp>
#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_XMLParameterListCoreHelpers.hpp"
#include "Teuchos_ConfigDefs.hpp"
#include "Teuchos_config.h"

int main (int argc, char ** argv)
 {

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

   std::cout<<n;
  double dt=0.05;
  double visc1=parlist->get<double>("mu1");
  double visc2=parlist->get<double>("mu2");
  double sigma1=parlist->get<double>("surf1");
  double sigma2=parlist->get<double>("surf2");
  double sigma_p=(1e+2)*(sigma1+sigma2)/(16.5);
  

  double surf=0*1.55*(1e+3);
//  int m=50;
//  int n=50;
  int N[4]={0};
  double time =1;
  double numtimesteps=parlist->get<double>("time")+1;
  int numGlobalElements=m*n+2*m;
  double dep_Dak=0;
//  double sol[numGlobalElements][(int)numtimesteps];
   RCP<const map_type > map =
    rcp (new map_type (numGlobalElements, indexBase, comm ));
   ArrayView<const global_ordinal_type> myGlobalElements = map->getNodeElementList();
  local_ordinal_type numMyElements= map->getNodeNumElements();
  Kokkos::View<double*,Kokkos::LayoutRight> arr1D("arr1D",m*n,1);
  Kokkos::View<double **,Kokkos::LayoutRight> arr2D("arr2D",m,n);

  Pore pore;
  pore.setViscosity(visc1,visc2);
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

  stm.setupMultiVector(sigma_p,A,map,b,bcl,myGlobalElements,m,n);


   RCP<vec_type> X = rcp (new vec_type (A->getDomainMap (), 1)); // Set to zeros by default.

   RCP<vec_type> B = rcp (new vec_type (A->getRangeMap (), bcl));
   
   solve<vec_type, op_type> ( *X, *B, *A) ;
   
   X->template sync<Kokkos::HostSpace>();
   
   auto x_2d=X->template getLocalView<Kokkos::HostSpace> ();
   auto x_1d=Kokkos::subview (x_2d, Kokkos::ALL (), 0);

  for(size_t k=0;k<X->getLocalLength();k++)
  {

// std::cout<<myGlobalElements[k]<<":"<<" "<<"from process:"<<comm->getRank()<<" "<<x_1d(k)<<std::endl;
  }




postProcessing pp;
pp.setupCurrentMatrix(L,Current,x_1d,map,myGlobalElements,K,Vol_frac,time,arr2D,N,m,n,dt,surf);   

        
dump_volume(Vol_frac,map,myGlobalElements,time);

}

// dep_Dak=depth(Vol_frac,map,myGlobalElements,time,m);
//  std::cout<<dep_Dak<<std::endl;
         std::ofstream fout1("test_volume.txt");
  if (fout1.is_open())
{

  for(size_t i=0;i<map->getNodeNumElements();i++)

 {
      size_t num;
      Teuchos::Array<global_ordinal_type> l_ind(4);
      Teuchos::Array<scalar_type> l_val(4);
      R->getGlobalRowCopy(myGlobalElements[i],l_ind,l_val,num);
      
        for(int k=0;k<4;k++)
         {
                             
               fout1<<"value: "<<l_val[k]<<" "<< "indices:"<<l_ind[k];
                

        }
         
     fout1<<std::endl;



  } 
}



Kokkos::finalize();
MPI_Finalize();
 }
