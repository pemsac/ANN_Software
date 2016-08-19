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
 * Back-propagation training for feed-forward ANN with momentum & gradient
 * descent optimization algorithm
 *
 * Header file with class definition
 * This class is derived from ANN class in order to get access to all the
 * network during the train.
 *
 *
 */


/*
 * Header guard
 */
#ifndef TRAINING_H_
#define TRAINING_H_

/*
 * Includes & namespace
 */
#include "ANN.h"
#include <stdlib.h>
#include <time.h>

using namespace std;

/*
 * Back-propagation training class derived from ANN class
 */
class Training : public ANN
{
private:
  /*
   * Private variables:
   * _learnRate:	parameter of gradient descent algorithm
   * _momentum:		parameter to stabilize the training  (optional)
   * _preWandB:		previous weights & bias of a neuron after adjusting them
   * _deltaErr:		delta error of each neuron
   * _randWandB:	random weights & bias to initialize a new ANN base.
   */
  double _learnRate, _momentum, ***_preWandB, **_deltaErr, ***_randWandB;

  /*
   * Private function to initialize and return the _randWandB matrix with
   * random weights and bias values. It's used to initialize the ANN base object
   *
   * In order to keep a good use of memory resources, the memory allocated by
   * _randWan
   */
  double ***randWandB(int numLayer, int *layerSize);

  /*
   * In order to keep a good use of memory resources, the _randWandB memory
   * allocated should be released after using randWandB function
   */
  void freeRandWeight(int numLayer, int *layerSize);

public:

  /*
   * Constructor method for new ANN
   *
   * It initializes the ANN base object with random weights.
   * Other ANN parameters must be set:
   * - number of layers, including input & output layers. (numLayer)
   * - number of neurons in each layer (layerSize)
   * The training parameters to be introduced are the momentum (optional) and
   * the learning rate.
   */
  Training(int numLayer, int *layerSize, double momentum,
	   double learnRate);

  /*
   * Virtual destructor to free all dynamic memory (including ANN base)
   */
  virtual ~Training();

  /*
   * Back-propagation training method.
   *
   * The training is performed introducing an array of inputs (in) and their
   *
   * *******LACK OF COMMENTS
   */
  void backpropagation(double *in,double *target);

  /*
   * Method to get the current Mean Squared Error
   *
   * *********** LACK OF COMMENTS
   */
  //  double squareErr(double *target) const;
  double netErr(double *target) const;
};

#endif
