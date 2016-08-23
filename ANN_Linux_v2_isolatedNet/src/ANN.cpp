/**
 *
 * Carlos III University of Madrid.
 *
 * Master's Final Thesis: Heartbeats classifier based on ANN (Artificial Neural
 * Network).
 *
 * Software implementation in C++ for GNU/Linux x86 & Zynq's ARM platforms
 *
 * Author: Pedro Marcos Solórzano
 * Tutor: Luis Mengibar Pozo (Tutor)
 *
 *
 * Feed-forward Artificial Neural Network.
 *
 * Source code
 *
 *
 */

#include "ANN.h"



ANN::ANN(int numLayer, int *layerSize, double ***WandB)
{
  int i, j, k;
  /*
   * memory allocation and data copy.
   *
   * number of layers
   */
  _numLayer = numLayer;

  /*
   * Layer sizes
   */
  _layerSize = new int[numLayer];

  for (i=0; i<numLayer; ++i)
    {
      _layerSize[i] = layerSize[i];
    }

  /*
   * Binary network outputs with Winer-Take-All rule
   */
  _netOut = new int[layerSize[numLayer-1]];

  /*
   * output matrix (only memory allocation)
   */
  _uOut = new double*[numLayer];

  for (i=0; i<numLayer; ++i)
    {
      _uOut[i] = new double[layerSize[i]];
    }

  /*
   * Weights and bias matrix
   * Take into account the first layer's neurons (input) don't have weights.
   * Note the neurons have a weight for each previous neuron connected plus an
   * extra weight for its bias.
   */
  _WandB = new double**[numLayer];

  for (i=1; i<numLayer; ++i)
    {
      _WandB[i] = new double*[layerSize[i]];
    }
  for (i=1; i<numLayer; ++i)
    {
      for (j=0; j<layerSize[i]; ++j)
	{
	  _WandB[i][j] = new double[layerSize[i-1]+1];
	}
    }
  for (i=1; i<numLayer; ++i)
    {
      for (j=0; j<layerSize[i]; ++j)
	{
	  for (k=0; k<layerSize[i-1]+1; ++k)
	    {
	      _WandB[i][j][k] = WandB[i][j][k];
	    }
	}
    }
}



ANN::~ANN()
{
  int i, j;
  /*
   * Free all dynamic memory
   *
   * weights and bias matrix
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
   * output matrix
   */
  for(i=0; i<_numLayer; ++i)
    {
      delete[] _uOut[i];
    }
  delete[] _uOut;

  /*
   * layer sizes matrix
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
   * 2º step: feedforward process in hidden layers.
   * Get the outputs of each neuron in the hidden layers applying
   * sigmoid activation function
   */
  for(i=1;i<_numLayer;++i)
    {
      for(j=0; j<_layerSize[i]; ++j)
	{
	  /*
	   * Sum all the neuron inputs applying weights
	   */
	  for(k=0, sum=0.0;k<_layerSize[i-1];++k)
	    {
	      sum+= _uOut[i-1][k]*_WandB[i][j][k];
	    }
	  /*
	   * Apply bias
	   */
	  sum+=_WandB[i][j][_layerSize[i-1]];
	  /*
	   * SIGMOID activation function
	   */
	  _uOut[i][j]=1/(1+exp(-sum));
	}
    }

  /*
   * 3º step: Calculate Softmax outputs.
   */
  for(i=0, sumsoft=0.0; i<_layerSize[_numLayer-1]; ++i)
    {
      /*
       * Sum all the neuron inputs applying weights
       */
      for(j=0, sum=0.0; j<_layerSize[_numLayer-2]; ++j)
	{
	  sum += _uOut[_numLayer-2][j] * _WandB[_numLayer-1][i][j];
	}
      /*
       * Apply bias
       */
      sum += _WandB[_numLayer-1][i][_layerSize[_numLayer-2]];
      /*
       * SOFTMAX activation function
       */
      _uOut[_numLayer-1][i] = exp(sum);
      sumsoft += _uOut[_numLayer-1][i];
    }
  /*
   * SOFTMAX activation function
   */
  for(i=0; i<_layerSize[_numLayer-1]; ++i)
    {
      _uOut[_numLayer-1][i] /= sumsoft;
    }

  /*
   * 4º step: Winer-Take-All
   * The highest network output become 1.
   */
  for(i=1, max=0; i<_layerSize[_numLayer-1]; ++i)
    {
      if(_uOut[_numLayer-1][i] > _uOut[_numLayer-1][max])
	{
	  _netOut[max]=0;
	  max = i;
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

  for(i=0; i<_layerSize[_numLayer-1]; ++i)
    {
      out[i] = _netOut[i];
    }
}
