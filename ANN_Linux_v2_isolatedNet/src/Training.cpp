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


Training::Training(int numLayer, int *layerSize):
ANN(numLayer, layerSize, randWandB(numLayer, layerSize))
{
  int i, j;

  /*
   * Free memory allocated in _randWandB by randWandB function
   */
  freeRandWeight(numLayer, layerSize);

  /*
   * Memory allocation.
   *
   * Delta error matrix
   */
  _grad = new double*[numLayer];
  for(i=1; i<numLayer; ++i)
    {
      _grad[i] = new double[layerSize[i]];
    }

  /*
   * Previous weights & bias matrix
   * Take into account the first layer's neurons (input) don't have weights.
   * Note the neurons have a weight for each previous neuron connected plus an
   * extra weight for its bias.
   */
  _delta = new double**[numLayer]();
  for(i=1; i<numLayer; ++i)
    {
      _delta[i]=new double*[layerSize[i]]();
    }
  for(i=1; i<numLayer; ++i)
    {
      for(j=0; j<layerSize[i]; ++j)
	{
	  _delta[i][j] = new double[layerSize[i-1]+1]();
	}
    }

  /*
   * Previous weights & bias matrix initialization
   */
  //  for(i=1; i<numLayer; ++i)
  //    {
  //      for(j=0; j<layerSize[i]; ++j)
  //	{
  //	  for(k=0; k<layerSize[i-1]+1; ++k)
  //	    {
  //	      _delta[i][j][k] = (double)0.0;
  //	    }
  //	}
  //    }

  _learnRate = START_LEARN_RATE;
  _momentum = MOMENTUM;
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
	  delete[] _delta[i][j];
	}
    }
  for(i=1; i<_numLayer; ++i)
    {
      delete[] _delta[i];
    }
  delete[] _delta;

  /*
   * delta errors matrix
   */
  for(i=1; i<_numLayer; ++i)
    {
      delete[] _grad[i];
    }
  delete[] _grad;
}



void Training::backpropagation(double *in, double *target)
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
  //  for(i=0;i<_layerSize[_numLayer-1];++i)
  //    {
  //      _grad[_numLayer-1][i]=_uOut[_numLayer-1][i]*
  //	  (1-_uOut[_numLayer-1][i])*(target[i]-_uOut[_numLayer-1][i]);
  //    }

  for(i=0;i<_layerSize[_numLayer-1];++i)
    {
      _grad[_numLayer-1][i]=(target[i]-_uOut[_numLayer-1][i]);
    }

  //	find delta for hidden layers
  for(i=_numLayer-2;i>0;--i)
    {
      for( j=0;j<_layerSize[i];++j)
	{

	  for(k=0, sum=0.0; k<_layerSize[i+1];++k)
	    {
	      sum+=_grad[i+1][k]*_WandB[i+1][k][j];
	    }
	  _grad[i][j]=_uOut[i][j]*(1-_uOut[i][j])*sum;
	}
    }

  //	apply momentum ( does nothing if momentum=0 )
  for(i=1; i<_numLayer && _momentum>0; ++i)
    {
      for(j=0;j<_layerSize[i];++j)
	{
	  for(k=0;k<_layerSize[i-1];++k)
	    {
	      _WandB[i][j][k]+=_momentum*_delta[i][j][k];
	    }
	  _WandB[i][j][_layerSize[i-1]]+=_momentum*_delta[i][j][_layerSize[i-1]];
	}
    }

  //	adjust weights
  for(i=1;i<_numLayer;++i)
    {
      for(j=0;j<_layerSize[i];++j)
	{
	  for(k=0;k<_layerSize[i-1];++k)
	    {
	      _delta[i][j][k]=_learnRate*_grad[i][j]*_uOut[i-1][k];
	      _WandB[i][j][k]+=_delta[i][j][k];
	    }
	  _delta[i][j][_layerSize[i-1]]=_learnRate*_grad[i][j];
	  _WandB[i][j][_layerSize[i-1]]+=_delta[i][j][_layerSize[i-1]];
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
  srand (time(NULL));
  for(i=1; i<numLayer; ++i)
    {
      for(j=0; j<layerSize[i]; ++j)
	{
	  for(k=0; k<layerSize[i-1]+1; ++k)
	    {
	      _randWandB[i][j][k] = (double)(rand())/(RAND_MAX) - 0.5;
	    }
	}
    }

  /*
   * And return its pointer
   */
  return _randWandB;
}


//
//double Training::train(double **in, double **target, int numRowTrain)
//{
//  int i;
//  double mcee;
//
//  for(i=0, mcee=0; i<numRowTrain; ++i)
//    {
//
//    }
//}



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



//double Training::meanSquErr(double *target) const
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
//      mse+=(target[i]-_uOut[_numLayer-1][i])*(target[i]-_uOut[_numLayer-1][i]);
//    }
//  return mse/2;
//}

/*
 *
 * DEBUG
 */
//double Training::netErr(double *target) const
//{
//  double err=0.0;
//  int i;
//  for(i=0;i<_layerSize[_numLayer-1];++i)
//    {
//      err+=abs(target[i]-_uOut[_numLayer-1][i]);
//    }
//  return err/_layerSize[_numLayer-1];
//}

double Training::CEE(double *target)
{
  double err = 0.0;
  int i;
  for(i=0; i<_layerSize[_numLayer-1]; ++i)
    {
      err += log(_uOut[_numLayer-1][i]) * target[i] * (-1);
    }
  return err/_layerSize[_numLayer-1];
}



void Training::updateLRandM(double currMCEE, double lastMCEE)
{
  if (currMCEE<lastMCEE)
    {
      if(_learnRate<MAX_LEARN_RATE)
	{
	  _learnRate*=INCRE_LEARN_RATE;
	}
      _momentum=MOMENTUM;
    }
  else if (currMCEE>lastMCEE*VARY_RATE)
    {
      if(_learnRate>MIN_LEARN_RATE)
	{
	  _learnRate*=DECRE_LEARN_RATE;
	}
      _momentum=0;
    }
}
