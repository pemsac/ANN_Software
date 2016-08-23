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
 * Main code to test training and ANN performance
 *
 * Header file
 *
 *
 */

/*
 * Header guard
 */
#ifndef INCLUDE_MAIN_H_
#define INCLUDE_MAIN_H_

/*
 * Includes & name spaces
 */
#include <iostream>
#include <fstream>
#include "Training.h"

using namespace std;

/*
 * Files' directories
 */
#define TARGET_FILE_DIR		"example_4/target.dat"
#define ANN_FILE_DIR		"example_4/ANN.ann"
#define TRAIN_FILE_DIR		"example_4/Training.ann"
#define IN_FILE_DIR		"example_4/input.dat"

#define CODEC_MIN 		0
#define CODEC_MAX		1


#endif /* INCLUDE_MAIN_H_ */
