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
 * Main code to train, validate and test some feed-fordware ANN examples.
 *
 * Source code
 *
 *
 */

#include "main.h"



int main(int argc, char *argv[])
{
  int i,j, k, numLayer, *layerSize, numRowTrain, numRowVal, numRowTest, numRow,
  ite, maxIte, minIte, numOut, numIn, *goodOut, *badOut, goodOutTotal, netOut;
  double **dIn, **dTarget, mcee, minMcee, lastMcee, thMcee, *maxIn, *minIn;
  bool bad;
  fstream fAnn, fTarget, fIn, fTrain;
  clock_t time1, time2;

  /*
   * Introduction
   */
  cout<<endl<<endl<<endl
      <<"ANN test Application"<<endl
      <<"####################"<<endl<<endl;

  /*
   * Open configuration and data files
   */
  cout<<"Loading data... "<<flush;
  try
  {
      fAnn.open(ANN_FILE_DIR, fstream::in);
      fAnn.seekg(0, ios::beg);
      fTarget.open(TARGET_FILE_DIR, fstream::in);
      fTarget.seekg(0, ios::beg);
      fIn.open(IN_FILE_DIR, fstream::in);
      fIn.seekg(0, ios::beg);
      fTrain.open(TRAIN_FILE_DIR, fstream::in);
      fTrain.seekg(0, ios::beg);
  }
  catch (fstream::failure &e)
  {
      cerr<<endl<<"Error opening some file:"<<endl;
      cerr<<e.what()<<endl;
      return 1;
  }


  /*
   * ANN configuration
   *
   * Load ANN parameters such as number of layers and layers' sizes from their
   * file. Allocate required memory to save these data and other variables
   *
   * Check documentation for more information about file format
   */
  try
  {
      /*
       * Load Number of Layers
       */
      fAnn>>numLayer;

      /*
       * Allocate Layers Sizes Array and load them
       */
      layerSize = new int[numLayer];

      for (i=0; i<numLayer; ++i)
	{
	  fAnn>>layerSize[i];
	}

      /*
       * Get number of network outputs & inputs
       */
      numOut = layerSize[numLayer-1];
      numIn = layerSize[0];

      /*
       * Allocate and initialize to 0 statistical variables of ANN test
       */
      goodOut = new int[numOut]();
      badOut = new int[numOut]();

      /*
       * Allocate and initialize input coding variables
       */
      maxIn = new double[numIn];
      minIn = new double[numIn];
      for(i=0; i<numIn; ++i)
	{
	  maxIn[i] = CODEC_MIN;
	  minIn[i] = CODEC_MAX;
	}

      /*
       * Close file
       */
      fAnn.close();
  }
  catch(fstream::failure &e)
  {
      cerr<<endl<<"Error reading ANN configuration file:"<<endl;
      cerr<<e.what()<<endl;
      return 1;
  }
  catch (exception &e)
  {
      cerr<<endl<<"Error setting up ANN:"<<endl;
      cerr<<e.what()<<endl;
      cerr<<"Are all parameters correct?"<<endl;
      return 1;
  }


  /*
   * Training configuration
   *
   * Load training parameters such as threshold mcee or max iterations.
   *
   * Check documentation for more information about file format
   */
  try
  {
      /*
       * Load training parameters
       */
      fTrain>>thMcee>>maxIte>>numRowTrain>>numRowVal>>numRowTest;

      /*
       * Calculate number of total samples
       */
      numRow = numRowTrain + numRowVal + numRowTest;

      /*
       * Close the file
       */
      fTrain.close();
  }
  catch (fstream::failure &e)
  {
      cerr<<endl<<"Error reading training configuration file:"<<endl;
      cerr<<e.what()<<endl<<endl;
      return 1;
  }


  /*
   * Input data
   *
   * Load the input data for the network and find maximum and minimum values of
   * each type of input to codify all data later
   *
   * Check documentation for more information about file format
   */
  try
  {
      /*
       * Allocate input buffer
       */
      dIn = new double*[numRow];
      for(i=0; i<numRow; ++i)
	{
	  dIn[i] = new double[numIn];
	}

      /*
       * Load all data and find maximums and minimums
       */
      for(i=0; i<numRow; ++i)
	{
	  for(j=0; j<numIn; ++j)
	    {
	      fIn>>dIn[i][j];

	      if(dIn[i][j]>maxIn[j])
		{
		  maxIn[j] = dIn[i][j];
		}
	      if(dIn[i][j]<minIn[j])
		{
		  minIn[j] = dIn[i][j];
		}
	    }
	}

      /*
       * CLose file
       */
      fIn.close();
  }
  catch (exception &e)
  {
      cerr<<endl<<"Error reading Input data file:"<<endl;
      cerr<<e.what()<<endl<<endl;
      return 1;
  }


  /*
   * Target data
   *
   * Load the target outputs of each input sample data
   *
   * Check documentation for more information about file format
   */
  try
  {
      /*
       * Allocate target buffer
       */
      dTarget = new double*[numRow];
      for(i=0; i<numRow; ++i)
	{
	  dTarget[i] = new double[numOut];
	}

      /*
       * Load all data
       */
      for(i=0; i<numRow; ++i)
	{
	  for(j=0; j<numOut; ++j)
	    {
	      fTarget>>dTarget[i][j];
	    }
	}

      /*
       * Close file
       */
      fTarget.close();
  }
  catch (exception &e)
  {
      cerr<<endl<<"Error reading Target configuration file:"<<endl;
      cerr<<e.what()<<endl<<endl;
      return 1;
  }

  cout<<"DONE!"<<endl;


  /*
   * Codify the inputs data and normalize all of them to 1.
   *
   * Check documentation for more information
   */
  cout<<"Coding input data..."<<flush;
  for(i=0; i<numIn; ++i)
    {
      /*
       * Calculate Slope
       */
      double a = (CODEC_MAX-CODEC_MIN)/(maxIn[i]-minIn[i]);

      /*
       * Calculate y-intercept
       */
      double b = CODEC_MIN - a*minIn[i];

      /*
       * Apply the Straight Line equation to this type of input data
       */
      for(j=0; j<numRow; ++j)
	{
	  dIn[j][i] = dIn[j][i]*a + b;
	}
    }
  /*
   * Free coding variables memory.
   */
  delete[] maxIn;
  delete[] minIn;

  cout<<"DONE!"<<endl;

  /*
   * Start to measure training time and create the Training instance
   * (It will create an own ANN to be trained)
   *
   */
  cout<<"Training a new feed-forward Neural Network..."<<flush;
  time1 = clock();
  Training trainIns(numLayer, layerSize);

  /*
   * layerSize isn't still useful
   */
  delete[] layerSize;

  /*
   * TRAINING PROCESS:
   *
   * Main steps summary:
   * 1- Back-propagation
   * 2- Validation
   * 3- Results check
   * 4- Update training parameters
   */
  ite=0;
  minIte=0;
  minMcee=999;
  lastMcee=999;
  while(1)
    {
      /*
       * iterations counter
       */
      ++ite;

      /*
       * 1º Step. Back-propagation
       *
       * Apply Back-propagation with each training sample
       */
      for(i=0; i<numRowTrain; ++i)
	{
	  trainIns.backpropagation(dIn[i], dTarget[i]);
	}

      /*
       * 2º Step. Validation
       *
       * Validate the training. Calculate the Mean Cross Entropy Error (MCEE)
       * with some validation samples.
       */
      mcee=0;
      for(i=numRowTrain; i<numRowTrain+numRowVal; ++i)
	{
	  trainIns.feedforward(dIn[i]);
	  mcee+=trainIns.CEE(dTarget[i]);
	}
      mcee /= numRowVal;

      /*
       * Save the lower MCEE achieved
       */
      if(mcee<minMcee)
	{
	  minMcee = mcee;
	  minIte = ite;
	}

      /*
       * 3º Step: Results check
       *
       * Check if minimum MCEE or maximum iterations are achieved and stop
       * training if someone is. Print the result after finishing training.
       *
       *Stop time if the train is over
       */
      if(mcee<=thMcee)
	{
	  time2=clock();
	  cout<<"DONE!"<<endl<<endl
	      <<"##  TRAINING RESULTS:"<<endl
	      <<"# Threshold Mean Cross Entropy Error achieved in "<<ite
	      <<" iterations"<<endl;
	  break;
	}
      if(ite>=maxIte)
	{
	  time2=clock();
	  cout<<"DONE!"<<endl<<endl
	      <<"##  TRAINING RESULTS:"<<endl
	      <<"# Threshold Mean Cross Entropy Error not achieved"<<endl
	      <<"# Minimum Validation MCEE found at iteration No "<<minIte
	      <<" with a value of "<<minMcee<<endl;
	  break;
	}

      /*
       * 4º Step: Update training parameters
       *
       * If the training runs again, update the Learning Rate and Momentum
       */
      trainIns.updateLRandM(mcee,lastMcee);
      lastMcee=mcee;
    }
  /*
   * last training information to print
   */
  cout<<"# Validation MCEE = "<<mcee<<endl;
  if((time2-time1)<CLOCKS_PER_SEC)
    {
      cout<<"# Training time = "<<(double)(time2-time1)*1000000/CLOCKS_PER_SEC
	  <<" microsec"<<endl<<endl;
    }
  else
    {
      cout<<"# Training time = "<<(double)(time2-time1)/CLOCKS_PER_SEC
	  <<" sec"<<endl<<endl;
    }
  /*
   * TESTING PROCESS
   */
  cout<<"Testing ANN..."<<flush;
  /*
   * Start timing and test
   */
  time1 = clock();
  ite=0;
  mcee=0;
  k=0;
  goodOutTotal=0;
  for(i=numRowTrain+numRowVal; i<numRow; ++i)
    {
      /*
       * iteration counter
       */
      ++ite;

      /*
       * Feed-forward sample
       */
      trainIns.feedforward(dIn[i]);

      /*
       * Check whether the classification has been correctly done
       */
      netOut = trainIns.getNetOut();

      bad=false;
      for(j=0; j<numOut; ++j)
	{
	  if(dTarget[i][j])
	    {
	      k=j;
	      if(netOut!=j)
		{
		  bad=true;
		}
	    }
	}

      /*
       * Count bad and good results.
       */
      if(bad)
	{
	  ++badOut[k];
	}
      else
	{
	  ++goodOut[k];
	  ++goodOutTotal;
	}

      /*
       * Calculate MCEE of all the test
       */
      mcee+=trainIns.CEE(dTarget[i]);
    }
  /*
   * Finish test
   */
  mcee = mcee/numRowTest;
  time2=clock();

  /*
   * Print the test results
   */
  cout<<"DONE!"<<endl<<endl
      <<"##  TEST RESULTS:"<<endl;
  for(i=0; i<numOut; ++i)
    {
      cout<<"# Class No "<<i<<" => Good = "<<goodOut[i]<<" Bad = "
	  <<badOut[i]<<" ("<<goodOut[i]*100/(badOut[i]+goodOut[i])<<"%)"<<endl;
    }
  cout<<"# Test MCEE = "<<mcee<<endl;
  cout<<"# Classification accuracy = "<<goodOutTotal*100/numRowTest<<"%"<<endl;
  if((time2-time1)*1000000<CLOCKS_PER_SEC)
    {
      cout<<"# Test time = "<<(double)(time2-time1)*1000000000/CLOCKS_PER_SEC
	  <<" nanosec."<<endl<<endl;
    }
  else
    {
      cout<<"# Test time = "<<(double)(time2-time1)*1000000/CLOCKS_PER_SEC
	  <<" microsec."<<endl<<endl;
    }

  /*
   * END
   *
   * Free all dynamic memory (the object will be destroyed automatically)
   */
  cout<<"Ending program..."<<flush;

  delete[] goodOut;
  delete[] badOut;
  for(i=0; i<numRow; ++i)
    {
      delete[] dIn[i];
      delete[] dTarget[i];
    }
  delete[] dIn;
  delete[] dTarget;
  cout<<"DONE!"<<endl;
  return 0;
}



