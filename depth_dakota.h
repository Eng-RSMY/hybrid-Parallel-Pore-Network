#ifndef VOL_H
#define VOL_H
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_DefaultPlatform.hpp>
#include <Tpetra_Version.hpp>
#include <BelosTpetraAdapter.hpp>
#include <BelosSolverFactory.hpp>
//#include <Ifpack2_Factory.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <iostream>
#include <Kokkos_Core.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include<cmath>
#include<ctime>
#include<fstream>
#include <Kokkos_Array.hpp>
#include"time.h"

#define pi 3.14
 
template<class T1,class T2, class T3> 
 double depth( T1 &Vol_frac,T2 &map,T3 &myGlobalElements,double time,int N)
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



double node[N*N+2*N]={0};
double pen[N]={0};
double sum=0.0;
double avg_depth=0.0;      
//Teuchos::Array<scalar_type> node(N*N+2*N);
//Teuchos::Array<scalar_type> pen(N);


//ArrayRCP<size_t> node=arcp<size_t>(N*N+2*N);
//ArrayRCP<size_t> pen=arcp<size_t>(N);


  for(size_t i=0;i<map->getNodeNumElements();i++)

 {
 
     size_t num;
      Teuchos::Array<global_ordinal_type> l_ind(4);
      Teuchos::Array<scalar_type> l_val(4);
      Vol_frac->getGlobalRowCopy(myGlobalElements[i],l_ind,l_val,num);
    sum=0;
        for(int k=0;k<4;k++)
         {

               sum=sum+l_val[k];

         }

     sum=0.25*sum;
    node[i]=sum;


  }

  for(int i=0;i<N;i++)
  {
     for(int k=0;k<N*N;k=k+N)
{
    pen[i]=pen[i]+node[k+i];
}
  }


  for(int j=0;j<N;j++)
 {
     avg_depth+=pen[j];

 }

  avg_depth=avg_depth/N;

return(avg_depth);
}



#endif 
