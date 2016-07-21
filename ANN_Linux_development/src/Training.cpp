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

#include "Training.hpp"

/*
 * Training mode check
 */
#ifdef TRAINING_MODE

Training::Training ()
{
  /*
   * Empty
   */
}



Training::Training(int numLayer, int *layerSize, double momentum,
		   double learnRate) : ANN(numLayer, layerSize,
					   randomWeight(numLayer, layerSize))
{
  int i, j, k;
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
	      _prevWeight[i][j][k] = 0;
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



double Training::backpropagation(double **trainMat, int numRow)
{
  double sum, sqerr=0;
  int i, j, k, x;
  /*
   * Control the number of repeats made
   */
  for ( i=0; i<numRow; ++i)
    {
      /*
       * Process with the current ANN the next input array.
       * Get the result
       */
      double *result = feedforward((double*)trainMat[i]);

      /*
       * Get OUTPUT LAYER's DELTA ERRORS:
       */
      for(j=0; j<_layerSize[_numLayer-1]; ++j)
	{
	  /*
	   * Output Delta Error = Result * (1 - Result) * (Real Result - Result)
	   */
	  _deltaErr[_numLayer-1][j] =
	      result[j] *
	      (1 - result[j]) *
	      (trainMat[i][_layerSize[0]+j] - result[j]);
	}
      /*
       * Get HIDDEN LAYER's DELTA ERRORS:
       *
       * From the last layer...
       */
      for(j=_numLayer-2; j>0; --j)
	{
	  for(k=0; k<_layerSize[j]; ++k)
	    {
	      /*
	       *  summatory of (delta errors)*(weight) of the next layer
	       */
	      sum=0;
	      for(x=0; x<_layerSize[j+1]; ++x)
		{
		  sum += _deltaErr[j+1][x] * _weight[j+1][x][k];
		  cout<<"delta: "<< _deltaErr[j+1][x]<<endl;
		}
	      /*
	       * delta error = output * (1 - output) * summatory
	       */
	      _deltaErr[j][k] =
		  _output[j][k] * (1.0 - _output[j][k]) * sum;
	    }
	}
      /*
       * Apply momentum whether it's defined.
       */
      if(_momentum != 0)
	{
	  for(j=1; j<_numLayer; ++j)
	    {
	      for(k=0; k<_layerSize[j]; ++k)
		{
		  for(x=0; x<_layerSize[j-1]; ++x)
		    {
		      _weight[j][k][x] += _momentum * _prevWeight[j][k][x];
		    }
		  _weight[j][k][_layerSize[j-1]] +=
		      _momentum * _prevWeight[j][k][_layerSize[j-1]];
		}
	    }
	}
      /*
       * Apply the algorithm Gradient Descent to adjust the new weights
       * Check documentation for more information
       */
      for(j=1; j<_numLayer; ++j)
	{
	  for(k=0; k<_layerSize[j]; ++k)
	    {
	      for(x=0; x<_layerSize[j-1]; ++x)
		{

		  _prevWeight[j][k][x] =
		      _learnRate * _deltaErr[j][k] * _output[j-1][x];

		  _weight[j][k][x] += _prevWeight[j][k][x];

		}
	      _prevWeight[j][k][_layerSize[j-1]] = _learnRate * _deltaErr[j][k];

	      _weight[j][k][_layerSize[j-1]] +=
		  _prevWeight[j][k][_layerSize[j-1]];
	    }
	}

      /*
       * Check if the current Square Error has achieved the maximum error
       */

      sqerr += squareErr(&trainMat[i][_layerSize[0]]);
    }

  /*
   * End. Return the achieved square error.
   */
  cout<<"S error="<<sqerr/numRow<<endl;
  return sqerr/numRow;
}



double ***Training::randomWeight(int numLayer, int *layerSize)
{
  int i, j, k;
  double ***weight;

  /*
   * memory allocation
   * Take into account the first layer's neurons (input) don't have weights
   */
  weight = new double**[numLayer];
  for(i=1; i<numLayer; ++i)
    {
      weight[i]=new double*[layerSize[i]];
    }
  for(i=1; i<numLayer; ++i)
    {
      for(j=0; j<layerSize[i]; ++j)
	{
	  weight[i][j] = new double[layerSize[i-1]+1];
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
	      weight[i][j][k] = (double)(rand())/(RAND_MAX/2) - 1;
	    }
	}
    }

  /*
   * And return its pointer
   */
  return weight;
}


double Training::squareErr(double *ideal)
{
  double sum=0;
  int i;

  for(i=0; i<_layerSize[_numLayer-1]; ++i)
    {
      sum +=
	  (ideal[i]-_output[_numLayer-1][i]) *
	  (ideal[i]-_output[_numLayer-1][i]);
    }
  return sum/2;
}
//
#endif
