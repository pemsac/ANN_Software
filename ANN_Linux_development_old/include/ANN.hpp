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
 * Header file with the class definition.
 *
 *
 */

/*
 * Header guard
 */
//#ifndef TRAINING_MODE
//#define TRAINING_MODE
//#endif
#ifndef ANN_H_
#define ANN_H_

/*
 * Includes
 */
#include <math.h>
//#include <fstream>
//
//using namespace std;

/*
 * Feed-forward Artificial Neuronal Network class
 */
class ANN{

protected:
  /*
   * Private variables:
   * weight & output value of each neuron
   * number of layers and of neurons in each one.
   */
  double ***_weight, **_output;
  int _numLayer, *_layerSize;

public:
  /*
   * Empty constructor
   */
  ANN();

  /*
   * Main constructor method.
   *
   * Create a new ANN setting:
   * - number of layers, including input & output layers. (numLayer)
   * - number of neurons in each layer (layerSize)
   * - weights of each neuron connection.
   */
  ANN(int numLayer, int *layerSize, double ***weight);

//  /*
//   * method to load the ANN form a file.
//   *
//   * Similar than the main constructor, gets all data from a fstream
//   * instance;
//   */
//  void load(const char *dir);

  /*
   * Destructor
   */
  virtual ~ANN();

  /*
   * Process a new input and return a pointer to the result array
   * (last layer outputs)
   *
   * Check Feedforward Artificial Neuronal Network's documentation for more
   * information about the performance
   */
  double *feedforward(double *input);

  /*
   * Get the number of outputs & inputs (last layer size)
   */
  int getNumOut(){return _layerSize[_numLayer-1];}
  int getNumIn(){return _layerSize[0];}

  /*
   * Getters & Setters
   */
//  int getNumLayer(){return _numLayer;}
//  int getLayerSize(int i){return _layerSize[i];}
//  double getOutput(int i, int j){return _output[i][j];}
//  double getWeight(int i, int j, int k){return _weight[i][j][k];}
};

#endif
