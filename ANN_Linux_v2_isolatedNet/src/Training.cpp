/**
 *
 * Carlos III University of Madrid.
 *
 * Master Final Thesis: Heartbeat classifier based on ANN (Artificial Neural
 * Network).
 *
 * Software implementation in C++ for GNU/Linux x86 & Zynq's ARM platforms
 *
 * Author: Pedro Marcos Solórzano
 * Tutor: Luis Mengibar Pozo (Tutor)
 *
 *
 * Back-propagation training for feedforward ANN
 *
 * Source file
 *
 *
 */

#include "Training.h"



Training::Training(int numLayer, int *layerSize, double momentum,
		   double learnRate) :
		   ANN(numLayer, layerSize, randWandB(numLayer, layerSize))
{
  int i, j, k;

  /*
   * Free memory allocated in _randWandB by randWandB function
   */
  freeRandWeight(numLayer, layerSize);

  /*
   * Memory allocation.
   *
   * Delta error matrix
   */
  _deltaErr = new double*[numLayer];
  for(i=1; i<numLayer; ++i)
    {
      _deltaErr[i] = new double[layerSize[i]];
    }

  /*
   * Previous weights & bias matrix
   * Take into account the first layer's neurons (input) don't have weights.
   * Note the neurons have a weight for each previous neuron connected plus an
   * extra weight for its bias.
   */
  _preWandB = new double**[numLayer];
  for(i=1; i<numLayer; ++i)
    {
      _preWandB[i]=new double*[layerSize[i]];
    }
  for(i=1; i<numLayer; ++i)
    {
      for(j=0; j<layerSize[i]; ++j)
	{
	  _preWandB[i][j] = new double[layerSize[i-1]+1];
	}
    }

  /*
   * Previous weights & bias matrix initialization
   */
  for(i=1; i<numLayer; ++i)
    {
      for(j=0; j<layerSize[i]; ++j)
	{
	  for(k=0; k<layerSize[i-1]+1; ++k)
	    {
	      _preWandB[i][j][k] = (double)0.0;
	    }
	}
    }

  /*
   * Data copy
   */
  _learnRate = learnRate;

  _momentum = momentum;
}



Training::~Training ()
{
  int i, j;

  /*
   * Free all dynamic memory.
   * Note the ANN base is released automatically
   *
   * The random weights & bias matrix (if it's still allocated)
   */
  freeRandWeight(_numLayer, _layerSize);

  /*
   * previous weights & bias matrix
   */
  for(i=1; i<_numLayer; ++i)
    {
      for(j=0; j<_layerSize[i]; ++j)
	{
	  delete[] _preWandB[i][j];
	}
    }
  for(i=1; i<_numLayer; ++i)
    {
      delete[] _preWandB[i];
    }
  delete[] _preWandB;

  /*
   * delta errors matrix
   */
  for(i=1; i<_numLayer; ++i)
    {
      delete[] _deltaErr[i];
    }
  delete[] _deltaErr;
}



void Training::backpropagation(double *in,double *target)
{
  double sum;
  int i, j, k;

  /*
   * LACK OF COMMENTS
   *
   *
   *
   */
  //	update output values for each neuron
  feedforward(in);

  //	find delta for output layer
  for(i=0;i<_layerSize[_numLayer-1];++i)
    {
      _deltaErr[_numLayer-1][i]=_out[_numLayer-1][i]*
	  (1-_out[_numLayer-1][i])*(target[i]-_out[_numLayer-1][i]);
    }

  //	find delta for hidden layers
  for(i=_numLayer-2;i>0;--i)
    {
      for( j=0;j<_layerSize[i];++j)
	{
	  sum=0.0;
	  for(k=0;k<_layerSize[i+1];++k)
	    {
	      sum+=_deltaErr[i+1][k]*_WandB[i+1][k][j];
	    }
	  _deltaErr[i][j]=_out[i][j]*(1-_out[i][j])*sum;
	}
    }

  //	apply momentum ( does nothing if alpha=0 )
  for(i=1;i<_numLayer;++i)
    {
      for(j=0;j<_layerSize[i];++j)
	{
	  for(k=0;k<_layerSize[i-1];++k)
	    {
	      _WandB[i][j][k]+=_momentum*_preWandB[i][j][k];
	    }
	  _WandB[i][j][_layerSize[i-1]]+=_momentum*_preWandB[i][j][_layerSize[i-1]];
	}
    }

  //	adjust weights usng steepest descent
  for(i=1;i<_numLayer;++i)
    {
      for(j=0;j<_layerSize[i];++j)
	{
	  for(k=0;k<_layerSize[i-1];++k)
	    {
	      _preWandB[i][j][k]=_learnRate*_deltaErr[i][j]*_out[i-1][k];
	      _WandB[i][j][k]+=_preWandB[i][j][k];
	    }
	  _preWandB[i][j][_layerSize[i-1]]=_learnRate*_deltaErr[i][j];
	  _WandB[i][j][_layerSize[i-1]]+=_preWandB[i][j][_layerSize[i-1]];
	}
    }
}



double ***Training::randWandB(int numLayer, int *layerSize)
{
  int i, j, k;

  /*
   * memory allocation as the same way of the weights & bias matrix
   */
  _randWandB = new double**[numLayer];
  for(i=1; i<numLayer; ++i)
    {
      _randWandB[i]=new double*[layerSize[i]];
    }
  for(i=1; i<numLayer; ++i)
    {
      for(j=0; j<layerSize[i]; ++j)
	{
	  _randWandB[i][j] = new double[layerSize[i-1]+1];
	}
    }

  /*
   * Save random weights to the matrix
   */
  for(i=1; i<numLayer; ++i)
    {
      for(j=0; j<layerSize[i]; ++j)
	{
	  for(k=0; k<layerSize[i-1]+1; ++k)
	    {
	      _randWandB[i][j][k] = (double)(rand())/(RAND_MAX/2) - 1;
	    }
	}
    }

  /*
   * And return its pointer
   */
  return _randWandB;
}



void Training::freeRandWeight(int numLayer, int *layerSize)
{
  if (_randWandB)
    {
      int i, j;

      /*
       * Free _randWandB dynamic memory
       */
      for(i=1; i<numLayer; ++i)
	{
	  for(j=0; j<layerSize[i]; ++j)
	    {
	      delete[] _randWandB[i][j];
	    }
	}
      for(i=1; i<numLayer; ++i)
	{
	  delete[] _randWandB[i];
	}
      delete[] _randWandB;
    }
}



//double Training::squareErr(double *target) const
//{
//  /*
//   *
//   *
//   *
//   *
//   * LACK OF COMMENTS
//   */
//  double mse=0.0;
//  int i;
//  for(i=0;i<_layerSize[_numLayer-1];++i)
//    {
//      mse+=(target[i]-_out[_numLayer-1][i])*(target[i]-_out[_numLayer-1][i]);
//    }
//  return mse/2;
//}
double Training::netErr(double *target) const
{
  double err=0.0;
  int i;
  for(i=0;i<_layerSize[_numLayer-1];++i)
    {
 //     cout<<"["<<i<<"] => ideal="<<target[i]<<" Output="<<_out[_numLayer-1][i]<<endl;
      err+=abs(target[i]-_out[_numLayer-1][i]);
    }
  return err/_layerSize[_numLayer-1];
}
