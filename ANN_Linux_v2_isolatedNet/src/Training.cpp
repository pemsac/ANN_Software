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
 * Back-propagation training for feedforward ANN
 * Source file
 *
 *
 */

#include "Training.h"



Training::Training(int numLayer, int *layerSize, double momentum,
		double learnRate) : ANN(numLayer, layerSize,
				randWeight(numLayer, layerSize))
{
	int i, j, k;

	freeRandWeight(numLayer, layerSize);
	/*
	 * Memory allocation.
	 * Take into account the first layer's neurons (input) don't have delta error
	 */
	_deltaErr = new double*[numLayer];
	for(i=1; i<numLayer; ++i)
	{
		_deltaErr[i] = new double[layerSize[i]];
	}

	_prevWeight = new double**[numLayer];
	for(i=1; i<numLayer; ++i)
	{
		_prevWeight[i]=new double*[layerSize[i]];
	}
	for(i=1; i<numLayer; ++i)
	{
		for(j=0; j<layerSize[i]; ++j)
		{
			_prevWeight[i][j] = new double[layerSize[i-1]+1];
		}
	}

	/*
	 * Previous weights initialization
	 * Note there are an extra weight per layer for the bias' training
	 */
	for(i=1; i<numLayer; ++i)
	{
		for(j=0; j<layerSize[i]; ++j)
		{
			for(k=0; k<layerSize[i-1]+1; ++k)
			{
				_prevWeight[i][j][k] = (double)0.0;
			}
		}
	}

	/*
	 * Data copy and untrained ANN creation.
	 */
	_learnRate = learnRate;

	_momentum = momentum;
}



Training::~Training ()
{
	int i, j;
	/*
	 * Free all dynamic memory
	 */
	for(i=1; i<_numLayer; ++i)
	{
		for(j=0; j<_layerSize[i]; ++j)
		{
			delete[] _prevWeight[i][j];
		}
	}
	for(i=1; i<_numLayer; ++i)
	{
		delete[] _prevWeight[i];
	}
	delete[] _prevWeight;

	for(i=1; i<_numLayer; ++i)
	{
		delete[] _deltaErr[i];
	}
	delete[] _deltaErr;
}



void Training::backpropagation(double *in,double *tgt)
{
	double sum;
	int i, j, k;

	//	update output values for each neuron
	feedforward(in);

	//	find delta for output layer
	for(i=0;i<_layerSize[_numLayer-1];++i)
	{
		_deltaErr[_numLayer-1][i]=_output[_numLayer-1][i]*
				(1-_output[_numLayer-1][i])*(tgt[i]-_output[_numLayer-1][i]);
	}

	//	find delta for hidden layers
	for(i=_numLayer-2;i>0;--i)
	{
		for( j=0;j<_layerSize[i];++j)
		{
			sum=0.0;
			for(k=0;k<_layerSize[i+1];++k)
			{
				sum+=_deltaErr[i+1][k]*_weight[i+1][k][j];
			}
			_deltaErr[i][j]=_output[i][j]*(1-_output[i][j])*sum;
		}
	}

	//	apply momentum ( does nothing if alpha=0 )
	for(i=1;i<_numLayer;++i)
	{
		for(j=0;j<_layerSize[i];++j)
		{
			for(k=0;k<_layerSize[i-1];++k)
			{
				_weight[i][j][k]+=_momentum*_prevWeight[i][j][k];
			}
			_weight[i][j][_layerSize[i-1]]+=_momentum*_prevWeight[i][j][_layerSize[i-1]];
		}
	}

	//	adjust weights usng steepest descent
	for(i=1;i<_numLayer;++i)
	{
		for(j=0;j<_layerSize[i];++j)
		{
			for(k=0;k<_layerSize[i-1];++k)
			{
				_prevWeight[i][j][k]=_learnRate*_deltaErr[i][j]*_output[i-1][k];
				_weight[i][j][k]+=_prevWeight[i][j][k];
			}
			_prevWeight[i][j][_layerSize[i-1]]=_learnRate*_deltaErr[i][j];
			_weight[i][j][_layerSize[i-1]]+=_prevWeight[i][j][_layerSize[i-1]];
		}
	}
}



double ***Training::randWeight(int numLayer, int *layerSize)
{
	int i, j, k;

	/*
	 * memory allocation
	 * Take into account the first layer's neurons (input) don't have weights
	 */
	_randWeight = new double**[numLayer];
	for(i=1; i<numLayer; ++i)
	{
		_randWeight[i]=new double*[layerSize[i]];
	}
	for(i=1; i<numLayer; ++i)
	{
		for(j=0; j<layerSize[i]; ++j)
		{
			_randWeight[i][j] = new double[layerSize[i-1]+1];
		}
	}

	/*
	 * Save random weights in the matrix
	 */
	for(i=1; i<numLayer; ++i)
	{
		for(j=0; j<layerSize[i]; ++j)
		{
			for(k=0; k<layerSize[i-1]+1; ++k)
			{
				_randWeight[i][j][k] = (double)(rand())/(RAND_MAX/2) - 1;
			}
		}
	}

	/*
	 * And return its pointer
	 */
	return _randWeight;
}

void Training::freeRandWeight(int numLayer, int *layerSize)
{
	int i, j;

	for(i=1; i<numLayer; ++i)
	{
		for(j=0; j<layerSize[i]; ++j)
		{
			delete[] _randWeight[i][j];
		}
	}
	for(i=1; i<numLayer; ++i)
	{
		delete[] _randWeight[i];
	}
	delete[] _randWeight;
}


double Training::mse(double *in) const
{
	double mse=0.0;
	int i;
	for(i=0;i<_layerSize[_numLayer-1];++i)
	{
		mse+=(in[i]-_output[_numLayer-1][i])*(in[i]-_output[_numLayer-1][i]);
	}
	return mse/2;
}
