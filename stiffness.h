#ifndef STIFFNESS_H
#define STIFFNESS_H

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



class stiffnessMatrix
{
  public:
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
   

 
  template<class T1, class T2, class T3, class T4>
  void setupStiffnessMatrix(T1 & A, T2 &K, T3 & map,int m, int n, T4 & myGlobalElements, double time)   
  {
   for(size_t i=0;i<map->getNodeNumElements();i++)
 {
  if(myGlobalElements[i]<m*n)
{

local_ordinal_type zero=1;
size_t five =5;
size_t num;
Teuchos::Array<global_ordinal_type> mat_ind(5);// This size will always remian five
Teuchos::Array<scalar_type>mat_val(5);

K->getGlobalRowCopy(myGlobalElements[i],mat_ind,mat_val,num);
size_t NumEntries;
 
  size_t countNZ=0;
    for(int j=0;j<5;j++)
      {
        if(mat_val[j]>0)
       {
        countNZ=countNZ+1;
       }
      }
NumEntries=countNZ;

Teuchos::Array<scalar_type> values(NumEntries);
Teuchos::Array<global_ordinal_type> indices(NumEntries);
double diagonal=0;
global_ordinal_type countN=0;
    for(int j=0;j<5;j++)
      {
        if(mat_val[j]>0)
       {

        diagonal=diagonal+mat_val[j];
        values[countN]=-mat_val[j];
        indices[countN]=mat_ind[j];

       countN=countN+1;
      }

      }
if (time!=0)
 {
    A->resumeFill();
    A->replaceGlobalValues (myGlobalElements[i], indices, values);
 

 }
else
{
 A->insertGlobalValues (myGlobalElements[i], indices, values);
}
if (time !=0)
 {
    A->resumeFill();
     A->replaceGlobalValues(myGlobalElements[i],Teuchos::tuple(myGlobalElements[i]),Teuchos::tuple(static_cast<scalar_type>(diagonal)));

 }
else
{
 A->insertGlobalValues(myGlobalElements[i],Teuchos::tuple(myGlobalElements[i]),Teuchos::tuple(static_cast<scalar_type>(diagonal)));
}

if(myGlobalElements[i]<m)
{

if (time!=0)
{
A->resumeFill();
A->replaceGlobalValues(myGlobalElements[i],Teuchos::tuple<global_ordinal_type>((myGlobalElements[i]+m*n)),Teuchos::tuple(static_cast<scalar_type>(1)));
}
else
{
A->insertGlobalValues(myGlobalElements[i],Teuchos::tuple<global_ordinal_type>((myGlobalElements[i]+m*n)),Teuchos::tuple(static_cast<scalar_type>(1)));
}

}

if(myGlobalElements[i]>=(m*n-m) && myGlobalElements[i]<m*n)
  {
       if (time !=0)
{
   A->resumeFill();
A->replaceGlobalValues(myGlobalElements[i],Teuchos::tuple<global_ordinal_type>((myGlobalElements[i]+2*m)),Teuchos::tuple(static_cast<scalar_type>(1)));

}
else
{
A->insertGlobalValues(myGlobalElements[i],Teuchos::tuple<global_ordinal_type>((myGlobalElements[i]+2*m)),Teuchos::tuple(static_cast<scalar_type>(1)));
}
}

}
else if(myGlobalElements[i]>=m*n && myGlobalElements[i]<m*n+m)
{

const scalar_type one =static_cast<scalar_type>(1);
  if (time !=0)
 {
   A->resumeFill();
A->replaceGlobalValues(myGlobalElements[i],Teuchos::tuple<global_ordinal_type>((myGlobalElements[i]-m*n)),Teuchos::tuple(one));
}
else
{
A->insertGlobalValues(myGlobalElements[i],Teuchos::tuple<global_ordinal_type>((myGlobalElements[i]-m*n)),Teuchos::tuple(one));
}


}
//ends here
else if(myGlobalElements[i]>=m*n+m)


{
const scalar_type one=static_cast<scalar_type>(1);
  if(time !=0)
{
A->resumeFill();
A->replaceGlobalValues(myGlobalElements[i],Teuchos::tuple<global_ordinal_type>((myGlobalElements[i]-2*m)),Teuchos::tuple(one));
}

else
{
A->insertGlobalValues(myGlobalElements[i],Teuchos::tuple<global_ordinal_type>((myGlobalElements[i]-2*m)),Teuchos::tuple(one));

}
}
//ends here
}

A->fillComplete();





  }//dont touch this shit

   template<class T1, class T2, class T3>
   void setupMultiVector(double x1,T1 & A, T2 & map, Kokkos::View<double*, Kokkos::LayoutRight> & b, Kokkos::DualView<double **, Kokkos::LayoutLeft> & bcl, T3 & myGlobalElements,int m, int n)
   {

    Kokkos::parallel_for(map->getNodeNumElements(),KOKKOS_LAMBDA(local_ordinal_type i){

  if(myGlobalElements[i]>=m*n && myGlobalElements[i]<m*n+m)

  {
    b(i)=100;
  }

});
bcl.modify<Kokkos::DualView<double**, Kokkos::LayoutLeft>::t_dev::execution_space> ();
Kokkos::deep_copy (Kokkos::subview (bcl.d_view, Kokkos::ALL (), 0), b);




  } //don't touch this crap




};
#endif
