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

#include "ANN.h"

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



void ANN::feedforward(double *in)
{
	double sum, sumsoft;
	int i, j, k;

	//	assign content to input layer
	for(i=0;i<_layerSize[0];++i)
	{
		_output[0][i]=in[i];  // output_from_neuron(i,j) Jth neuron in Ith Layer
	}

	//	assign output(activation) value
	//	to each neuron usng sigmoid func
	for(i=1;i<_numLayer-1;++i)
	{				// For each layer
		for(j=0;j<_layerSize[i];++j)
		{		// For each neuron in current layer
			sum=0.0;
			for(k=0;k<_layerSize[i-1];++k)
			{		// For input from each neuron in preceeding layer
				sum+= _output[i-1][k]*_weight[i][j][k];	// Apply weight to inputs and add to sum
			}
			sum+=_weight[i][j][_layerSize[i-1]];		// Apply bias
			_output[i][j]=1/(1+exp(-sum));			// Apply sigmoid function
		}
	}
	/*
	 * Softmax for output neurons
	 */
	sumsoft=0.0;
	for(i=0; i<_layerSize[_numLayer-1]; ++i)
	{
		sum=0.0;
		for(j=0;j<_layerSize[_numLayer-2];++j)
		{		// For input from each neuron in preceeding layer
			sum += _output[_numLayer-2][j] * _weight[_numLayer-1][i][j];	// Apply weight to inputs and add to sum
		}
		sum += _weight[_numLayer-1][i][_layerSize[_numLayer-2]];

		_output[_numLayer-1][i] = exp(sum);
		sumsoft += _output[_numLayer-1][i];
	}
	for(i=0; i<_layerSize[_numLayer-1]; ++i)
	{
		_output[_numLayer-1][i] /= sumsoft;
	}
}
