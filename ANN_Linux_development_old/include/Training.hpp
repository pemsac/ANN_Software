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
 * Header file
 *
 *
 */
//#ifndef TRAINING_MODE
//#define TRAINING_MODE
//#endif
/*
 * Training mode check
 */
//#ifdef TRAINING_MODE

/*
 * Header guard
 */
#ifndef TRAINING_H_
#define TRAINING_H_

/*
 * Includes
 */
#include "ANN.hpp"
#include <stdlib.h>

/*
 * Back-propagation training class
 * This class is derived from ANN class
 */
class Training : public ANN
{
private:
  /*
   * Private variables:
   *
   * The training needs some parameters such a learning Rate and a momentum.
   *
   * On the other hand, the training needs to save information about the last
   * neurons' weights used (_prevWeight) and their Delta Error (deltaErr)
   *
   * This class is derived from ANN class to be able to create an ANN object
   * with access to all its protected variables.
   */
  double **_deltaErr, ***_prevWeight, _learnRate, _momentum;

  /*
   * Private function to create an weights matrix with random values. This is
   * used to initialized a new ANN before training it.
   */
  double ***randomWeight(int numLayer, int *layerSize);
  /*
   * Private function to calculate the current square error while training
   */
  double squareErr(double *result);

public:
  /*
   * Empty constructor
   */
  Training();

  /*
   * Constructor method for new ANN
   *
   * It initializes an own ANN with initial random weights.
   * Other ANN parameters must be set:
   * - number of layers, including input & output layers. (numLayer)
   * - number of neurons in each layer (layerSize)
   *
   * The training parameters are:
   * - momentum for the training (momentum).
   * - learning rate (learnRate)
   */
  Training(int numLayer, int *layerSize, double momentum,
           double learnRate);

  /*
   * Constructor method to train again and improve an existing ANN
   *
   * It retrains a provided ANN. These parameters must be set:
   * - The said ANN parameters to be copied (numLayer, layerSize & weight)
   * - momentum for the training (OPTIONAL)
   * - learning rate (learnRate)
   */
  Training(int numLayer, int *layerSize, double ***weight, double momentum,
           double learnRate);

  /*
   * Destructor
   */
  virtual ~Training();

  /*
   * Training method.
   *
   * The training is performed introducing a Matrix (trainMat) with possible
   * inputs beside their expected outputs. The matrix format must be:
   * ( In  , In  , ... , ... , Out , Out )
   * ( In  , In  , ... , ... , Out , Out )
   * ( ... , ... , ... , ... , ... , ... )
   *
   * where numInputs is the number of rows.
   *
   * The training goal is to achieve the desired ANN Squared Error defined in
   * maxSquareErr, but it must be limited to a maximum number of interactions
   * (maxInter)
   *
   * Check Back-Propagation training's documentation for more information
   * about the performance
   */
  double backpropagation(double **trainMat, int numRow, int maxRepeat,
                         double maxSquareErr);

  //  /*
  //   * Function to save the just trained ANN to a binary file.
  //   */
  //  void saveToFile(fstream &fAnn);

  /*
   * ANN Getters
   */
  int getNumLayer () const   {return _numLayer;}

  int getLayerSize (int i) const {return _layerSize[i];}

  double getWeight (int i, int j, int k) const   {return _weight[i][j][k];}
};

#endif

//#endif
