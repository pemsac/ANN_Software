/**
 *
 * Carlos III University of Madrid.
 *
 * Master's Final Thesis: Heart-beats classifier based on ANN (Artificial Neural
 * Network).
 *
 * Software implementation in C++ for GNU/Linux x86 & Zynq's ARM platforms
 *
 * Author: Pedro Marcos Solórzano
 * Tutor: Luis Mengibar Pozo
 *
 *
 * Feed-forward Artificial Neural Network with sigmoid activation & softmax
 * output functions.
 *
 * Source code
 *
 *
 */

#include "ANN.h"



ANN::ANN(int numLayer, int *layerSize, double ***WandB)
{
  int i, j, k;
  /*.
   * Number of Layers data copy
   */
  _numLayer=numLayer;

  /*
   * Layers Sizes Array memory allocation and data copy
   */
  _layerSize=new int[numLayer];

  for(i=0; i<numLayer; ++i)
    {
      _layerSize[i]=layerSize[i];
    }

  /*
   * Binary Network Outputs Array memory allocation and initialization to 0
   */
  _netOut=new int[layerSize[numLayer-1]]();

  /*
   * Outputs Matrix memory allocation and initialization to 0
   */
  _uOut=new double*[numLayer]();

  for(i=0; i<numLayer; ++i)
    {
      _uOut[i]=new double[layerSize[i]]();
    }

  /*
   * Weights & Bias Matrix memory allocation and data copy
   *
   * Take into account the first layer's neurons (input) doesn't have weights.
   * Note the neurons have a weight for each previous neuron connected plus its
   * bias' weight.
   */
  _WandB=new double**[numLayer];

  for(i=1; i<numLayer; ++i)
    {
      _WandB[i]=new double*[layerSize[i]];
    }
  for(i=1; i<numLayer; ++i)
    {
      for(j=0; j<layerSize[i]; ++j)
	{
	  _WandB[i][j]=new double[layerSize[i-1]+1];
	}
    }
  for(i=1; i<numLayer; ++i)
    {
      for(j=0; j<layerSize[i]; ++j)
	{
	  for(k=0; k<layerSize[i-1]+1; ++k)
	    {
	      _WandB[i][j][k]=WandB[i][j][k];
	    }
	}
    }
}



ANN::~ANN()
{
  int i, j;
  /*
   * Weights & Bias Matrix memory freeing
   */
  for(i=1; i<_numLayer; ++i)
    {
      for(j=0; j<_layerSize[i]; ++j)
	{
	  delete[] _WandB[i][j];
	}
    }
  for(i=1; i<_numLayer; ++i)
    {
      delete[] _WandB[i];
    }
  delete[] _WandB;

  /*
   * Outputs Matrix memory freeing
   */
  for(i=0; i<_numLayer; ++i)
    {
      delete[] _uOut[i];
    }
  delete[] _uOut;

  /*
   * Network Outputs Array memory freeing
   */
  delete[] _netOut;

  /*
   * Layers Sizes Array memory freeing
   */
  delete[] _layerSize;
}



void ANN::feedforward(double *in)
{
  double sum, sumsoft;
  int i, j, k, max;

  /*
   * 1º step: Assign content to input layer
   */
  for(i=0;i<_layerSize[0];++i)
    {
      _uOut[0][i]=in[i];
    }

  /*
   * 2º step: Forward-propagation through hidden layers. Get the outputs of each
   * neuron in the hidden layers applying SIGMOID activation function
   */
  for(i=1;i<_numLayer;++i)
    {
      for(j=0; j<_layerSize[i]; ++j)
	{
	  /*
	   * Sum all the neuron inputs weighted.
	   */
	  for(k=0, sum=0.0;k<_layerSize[i-1];++k)
	    {
	      sum+=_uOut[i-1][k]*_WandB[i][j][k];
	    }
	  /*
	   * Apply bias
	   */
	  sum+=_WandB[i][j][_layerSize[i-1]];
	  /*
	   * Get neuron output with SIGMOID activation function
	   */
	  _uOut[i][j]=1/(1+exp(-sum));
	}
    }

  /*
   * 3º step: Calculate outputs. Get the outputs of each neuron in the last
   * layer applying SOFTMAX activation function.
   */
  for(i=0, sumsoft=0.0; i<_layerSize[_numLayer-1]; ++i)
    {
      /*
       * Sum all the neuron inputs weighted.
       */
      for(j=0, sum=0.0; j<_layerSize[_numLayer-2]; ++j)
	{
	  sum+=_uOut[_numLayer-2][j] * _WandB[_numLayer-1][i][j];
	}
      /*
       * Apply bias
       */
      sum+=_WandB[_numLayer-1][i][_layerSize[_numLayer-2]];
      /*
       * SOFTMAX activation function 1º step
       */
      _uOut[_numLayer-1][i] = exp(sum);
      sumsoft+=_uOut[_numLayer-1][i];
    }
  /*
   * SOFTMAX activation function 2º step. Get neurons outputs.
   */
  for(i=0; i<_layerSize[_numLayer-1]; ++i)
    {
      _uOut[_numLayer-1][i]/=sumsoft;
    }

  /*
   * 4º step: get Binary Network Outputs Array applying Winer-Take-All rule.
   * The highest last layer output become 1.
   *
   * Note these outputs are NOT used for training.
   */
  for(i=1, max=0; i<_layerSize[_numLayer-1]; ++i)
    {
      if(_uOut[_numLayer-1][i] > _uOut[_numLayer-1][max])
	{
	  _netOut[max]=0;
	  max=i;
	}
      else
	{
	  _netOut[i]=0;
	}
    }
  _netOut[max]=1;
}



void ANN::getNetOut(int *out)
{
  int i;

  /*
   * Copy arrays
   */
  for(i=0; i<_layerSize[_numLayer-1]; ++i)
    {
      out[i]=_netOut[i];
    }
}
