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
 * Feed-forward Artificial Neural Network with sigmoid & softmax activation
 * functions.
 *
 * Header file with the class definition.
 *
 *
 */

/*
 * Header guard
 */
#ifndef ANN_H_
#define ANN_H_

/*
 * Includes
 */
#include <cmath>

/*
 * Feed-forward Artificial Neural Network class
 */
class ANN
{
protected:
  /*
   * Private variables:
   * _WandB:	 weights and bias of each neuron
   * _uOut:	 output values of each neuron
   * _numLayer:	 number of layers including input and output layers
   * _layerSize: number of neurons in each layer.
   */
  double ***_WandB, **_uOut;
  int _numLayer, *_layerSize, *_netOut;

public:
  /*
   * Main constructor method.
   *
   * Create a new ANN introducing weights, number of layers and layers' sizes.
   */
  ANN(int numLayer, int *layerSize, double ***WandB);

  /*
   * Virtual destructor to free all dynamic memory
   */
  virtual ~ANN();

  /*
   * Feed-forward process.
   * The neurons have Sigmoid activation functions and Softmax output functions
   *
   * Check documentation to get more information
   */
  void feedforward(double *in);

  /*
   * Getter for network's outputs (outputs of last layer's neurons)
   */
  void getNetOut(int *out);
};

#endif
