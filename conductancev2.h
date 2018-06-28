#ifndef CONDUCTANCE_H
#define CONDUCTANCE_H
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
#include<cmath>
#define pi 3.14

class Conductance {

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
  template <class T1, class T2, class T3, class T4,class T5,class T6, class T7,class T8>  
  void fillConductanceMatrix(T8 & R,T7 &L,T1 & K, int *N,int m, int n,double time, T2 & Vol_frac, T3 & myGlobalElements_k, T5 &arr2D,T4 &map,T6 &pore)
 {
    for(size_t i=0;i<map->getNodeNumElements();i++)
{
  if(myGlobalElements_k[i]<m*n)
{
 N[0]=-1;
 N[1]=-1;
 N[2]=-1;
 N[3]=-1;

int q=myGlobalElements_k[i]/m;
int rem =myGlobalElements_k[i]%m;
  if(q>0)
 {
  N[0]=arr2D(q-1,rem);
 }
if(rem>0)
{
 N[1]=arr2D(q,rem-1);
}

if(q<m-1)
  {
   N[2]=arr2D(q+1,rem);
 }

if(rem<m-1)
  {
   N[3]=arr2D(q,rem+1);

 }

size_t NumEntries_k;


     size_t countNZ=0;
    for(int j=0;j<4;j++)
      {
        if(N[j]>=0)
       {
        countNZ=countNZ+1;
       }
      }
NumEntries_k=countNZ;


Teuchos::Array<scalar_type> values_k(NumEntries_k);
Teuchos::Array<global_ordinal_type> indices_k(NumEntries_k);

if(time==0)
 {
     global_ordinal_type countN=0;
      for(global_ordinal_type j=0;j<4;j++)
      {
        if(N[j]>=0)
       {

         indices_k[countN]=N[j];
         if(myGlobalElements_k[i]<m &&(( myGlobalElements_k[i]-N[j]==1)||( myGlobalElements_k[i]-N[j]==-1)) )
          {
            values_k[countN]=0;
          }
        else
   {    
        size_t num; 
        Teuchos::Array<scalar_type> length_val(NumEntries_k);
        Teuchos::Array<scalar_type> radius_val(NumEntries_k);
        Teuchos::Array<global_ordinal_type> length_ind(NumEntries_k);
        Teuchos::Array<global_ordinal_type> radius_ind(NumEntries_k);
        L->getGlobalRowCopy(myGlobalElements_k[i],length_ind,length_val,num);
        R->getGlobalRowCopy(myGlobalElements_k[i], radius_ind, radius_val,num);
          for (int k=0;k<NumEntries_k;k++)
         {
           if(N[j]==length_ind[k])
         values_k[countN]= static_cast<scalar_type> ((1e-3)*pow(radius_val[k],2)/pore.getDefendingViscosity()*length_val[k]);

         }
  }

        countN=countN+1;
       }
      }
}

 if(time!=0)
{
    global_ordinal_type countNN=0;
  for(global_ordinal_type j=0;j<4;j++)
  {
      if(N[j]>=0)
 {
    if(myGlobalElements_k[i]<m &&(( myGlobalElements_k[i]-N[j]==1)||( myGlobalElements_k[i]-N[j]==-1)) )
          {
            values_k[countNN]=0;
            indices_k[countNN]=N[j];           
          }

     else
{
     
 Teuchos::Array<global_ordinal_type> mat_ind_k(NumEntries_k); 
 Teuchos::Array<scalar_type>mat_val_k(NumEntries_k);
  Teuchos::Array<global_ordinal_type> length_ind(NumEntries_k);
  Teuchos::Array<global_ordinal_type> radius_ind(NumEntries_k);
 Teuchos::Array<scalar_type>length_val(NumEntries_k);
 Teuchos::Array<scalar_type>radius_val(NumEntries_k);
      size_t num;
      Vol_frac->getGlobalRowCopy(myGlobalElements_k[i],mat_ind_k,mat_val_k,num);
       L->getGlobalRowCopy(myGlobalElements_k[i],length_ind,length_val,num);
       R->getGlobalRowCopy(myGlobalElements_k[i],radius_ind,radius_val,num);
      size_t NumEntries;

      
   
  
   
//   global_ordinal_type countN=0;
     for(int k=0;k<NumEntries_k;k++)
   {

      if(mat_ind_k[k]==N[j])
       {
         values_k[countNN]=(1e-3)*length_val[k]*pow(radius_val[k],2)/(pore.getInvadingViscosity()*mat_val_k[k]+pore.getDefendingViscosity()*(length_val[k]-mat_val_k[k]));
         indices_k[countNN]=mat_ind_k[k];
         

       }



    }
}

countNN++;


}

}
  }


if(time !=0)
  {
 K->resumeFill();
 K->replaceGlobalValues (myGlobalElements_k[i], indices_k, values_k);
  }
  else
  {
  K->insertGlobalValues (myGlobalElements_k[i], indices_k, values_k);

  }

// Teuchos::Array<global_ordinal_type> mat_ind(NumEntries_k);
// Teuchos::Array<scalar_type>mat_val(NumEntries_k);
  

}

}

K->fillComplete();
/*
 Teuchos::Array<global_ordinal_type> mat_ind(4);
 Teuchos::Array<scalar_type>mat_val(4);
  size_t num;
  K->getGlobalRowCopy(myGlobalElements_k[3],mat_ind,mat_val,num);
      for(int k=0;k<4;k++)
      {
        if(mat_val[k]>0)
       {
        std::cout<<mat_val[k]<<" ";
       std::cout<<mat_ind[k]<<" ";
       }
      }

     std::cout<<std::endl;
*/
}


};

#endif
