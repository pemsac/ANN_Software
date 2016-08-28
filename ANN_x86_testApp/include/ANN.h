/**
 *
 * Carlos III University of Madrid.
 *
 * Master's Final Thesis: Heart-beats classifier based on ANN (Artificial Neural
 * Network).
 *
 * Software implementation in C++ for GNU/Linux x86 platform.
 *
 * Author: Pedro Marcos Solórzano
 * Tutor: Luis Mengibar Pozo
 *
 *
 * Feed-forward Artificial Neural Network with sigmoid activation & softmax
 * output functions.
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
   * _WandB:		Weights & Bias Matrix.
   * _uOut:		Outputs Matrix. Output values of each neuron
   * _numLayer:		Number of Layers. It includes input and output layers
   * _layerSize:	Layers Sizes Array. Number of neurons in each layer.
   * _netOut:		Binary Network Outputs Array. Normalized in binary
   */
  double ***_WandB, **_uOut;
  int _numLayer, *_layerSize;
  bool *_netOut;



public:
  /*
   * Main constructor method.
   *
   * Create a new ANN introducing, number of layers, layers sizes and Weight &
   * Bias Matrix.
   */
  ANN(int numLayer, int *layerSize, double ***WandB);

  /*
   * Virtual destructor to free all dynamic memory
   */
  virtual ~ANN();

  /*
   * Feed-forward method.
   *
   * There're 4 steps:
   * 1- Introduce new inputs
   * 2- Forward-propagation through hidden layers with Sigmoid activation
   * functions.
   * 3- Output layer operations with Softmax activation functions.
   * 4- Apply Winer-Take-All rule to get binary outputs and save them in _netOut
   *
   * Check documentation for more information
   */
  void feedforward(double *in);

  /*
   * Getter for _netOut.
   *
   * Get the binary netnetwork's outputs normalized by Winer-Take-All rule
   */
  void getNetOut(bool *out);
};

#endif
