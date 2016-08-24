/**
 *
 * Carlos III University of Madrid.
 *
 * Master's Final Thesis: Heart-beats classifier based on ANN (Artificial Neural
 * Network).
 *
 * Software implementation in C++ for GNU/Linux x86 & Zynq's ARM platforms
 *
 * Author: Pedro Marcos Solórzano
 * Tutor: Luis Mengibar Pozo
 *
 *
 * Main code to train, validate and test a feed-forward ANN
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
 * Includes & name space
 */
#include <iostream>
#include <fstream>
#include "Training.h"

using namespace std;

/*
 * Files' directories
 */
#define TARGET_FILE_DIR		"example_3/target.dat"
#define ANN_FILE_DIR		"example_3/ANN.ann"
#define TRAIN_FILE_DIR		"example_3/Training.ann"
#define IN_FILE_DIR		"example_3/input.dat"

/*
 * Input codify parameters
 */
#define CODEC_MIN 		0
#define CODEC_MAX		1

/*
 * Function declaration
 */
int main(int argc, char *argv[]);

#endif
