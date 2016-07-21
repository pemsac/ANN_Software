/**
 *
 * Carlos III University of Madrid.
 *
 * Master Final Thesis: Heartbeat classifier based on ANN (Artificial Neural
 * Network).
 *
 * Author: Pedro Marcos Solórzano
 * Tutor: Luis Mengibar Pozo (Tutor)
 *
 *
 * Feed-forward Artificial Neuronal Network.
 * Source file
 *
 *
 */

#include "ANN.hpp"


ANN::ANN()
{
  /*
   * Empty
   */
}

ANN::ANN(int numLayer, int *layerSize, double ***weight)
{
  int i, j, k;
  /*
   * memory allocation and data copy
   * Take into account the first layer's neurons (input) don't have weights
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
   * output matrix (only allocation)
   */
  _output = new double*[numLayer];
  for (i=0; i<numLayer; ++i)
    {
      _output[i] = new double[layerSize[i]];
    }

  /*
   * Weights matrix
   * Note there are an extra weight per euron
   */
  _weight = new double**[numLayer];
  for (i=1; i<numLayer; ++i)
    {
      _weight[i] = new double*[layerSize[i]];
    }
  for (i=1; i<numLayer; ++i)
    {
      for (j=0; j<layerSize[i]; ++j)
	{
	  _weight[i][j] = new double[layerSize[i-1]+1];
	}
    }

  for (i=1; i<numLayer; ++i)
    {
      for (j=0; j<layerSize[i]; ++j)
	{
	  for (k=0; k<layerSize[i-1]+1; ++k)
	    {
	      _weight[i][j][k] = weight[i][j][k];
	    }
	}
    }
}




ANN::~ANN()
{
  int i, j;
  /*
   * Free all dynamic memory
   */
  for(i=1; i<_numLayer; ++i)
    {
      for(j=0; j<_layerSize[i]; ++j)
	{
	  delete[] _weight[i][j];
	}
    }
  for(i=1; i<_numLayer; ++i)
    {
      delete[] _weight[i];
    }
  delete[] _weight;

  for(i=0; i<_numLayer; ++i)
    {
      delete[] _output[i];
    }
  delete[] _output;

  delete[] _layerSize;
}



//void ANN::load(const char *dir)
//{
//  int i, j, k;
//  fstream fAnn;
//
//  /*
//   * Open ANN file
//   */
//  try
//  {
//      fAnn.open(dir, fstream::in);
//      fAnn.seekg(0, ios::beg);
//  }
//  catch(exception e)
//  {
//      string msg("Impossible to open ANN file :");
//      msg.append(e.what());
//      throw msg;
//  }
//
//  /*
//   * memory allocation and data copy
//   * Take into account the first layer's neurons (input) don't have weights
//   */
//  try
//  {
//      /*
//       * number of layers
//       */
//      fAnn>>_numLayer;
//
//      /*
//       * Layer sizes
//       */
//      _layerSize = new int[_numLayer];
//
//      for (i=0; i<_numLayer; ++i)
//	fAnn>>_layerSize[i];
//
//      /*
//       * output matrix (only allocation)
//       */
//      _output = new double*[_numLayer];
//      for (i=0; i<_numLayer; ++i)
//	_output[i] = new double[_layerSize[i]];
//
//      /*
//       * weights matrix
//       */
//      _weight = new double**[_numLayer];
//      for (i=1; i<_numLayer; ++i)
//	_weight[i] = new double*[_layerSize[i]];
//      for (i=1; i<_numLayer; ++i)
//	for (j=0; j<_layerSize[i]; ++j)
//	  _weight[i][j] = new double[_layerSize[i-1]+1];
//
//      for (i=1; i<_numLayer; ++i)
//	for (j=0; j<_layerSize[i]; ++j)
//	  for (k=0; k<_layerSize[i-1]+1; ++k)
//	    fAnn >>_weight[i][j][k];
//  }
//  catch(exception e)
//  {
//      string msg("Fail reading ANN file :");
//      msg.append(e.what());
//      throw msg;
//  }
//  /*
//   * Close the file
//   */
//  try
//  {
//      fAnn.close();
//  }
//  catch(exception e)
//  {
//      string msg("Fail closing ANN file :");
//      msg.append(e.what());
//      throw msg;
//  }
//}



double *ANN::feedforward(double *input)
{
  int i, j, k;
  double sum;

  /*
   * copy data to input layer
   */
  for (i=1; i<_numLayer; ++i)
    {
      _output[0][i] = input[i];
    }

  /*
   * Process each neuron output taking in account their weight until getting
   * the last output values, that are the result.
   */
  for (i=1; i<_numLayer; ++i)
    {
      for (j=0; j<_layerSize[i]; ++j)
	{
	  sum = 0.0;
	  /*
	   * Neuron's entries from each previous layer neuron
	   */
	  for (k=0; k<_layerSize[i-1]; ++k)
	    {
	      sum += _output[i-1][k] * _weight[i][j][k];
	    }
	  /*
	   * bias
	   */
	      sum += _weight[i][j][_layerSize[i-1]];

	  /*
	   * SIGMOID as neuron function
	   */
	  _output[i][j] = (double)(1/(1+exp(-sum)));
	}
    }
  /*
   * return last layer pointer
   */
  return _output[_numLayer - 1];
}

