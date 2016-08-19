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
 * Source code
 *
 *
 */

#include "main.h"



int main(int argc, char* argv[])
{
  int i,j, numLayer, *layerSize, numRowTrain, numRowVal, numRowTest, numRow,
  count, maxInte;
  double **dIn, **dTarget, err, beta, alpha, thErr;
  fstream fAnn, fTarget, fIn, fTrain;

  /*
   * Open files and check errors
   */
  cout<<"Loading data from... "<<flush;
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
  catch (exception &e)
  {
      cerr<<endl<<"Error opening some file:"<<endl;
      cerr<<e.what()<<endl<<endl;
      return 1;
  }

  /*
   * Load ANN data such as number of layers and layers' sizes. Allocate memory
   */
  cout<<"ANN configuration file... "<<flush;
  try
  {
      fAnn>>numLayer;

      layerSize = new int[numLayer];

      for (i=0; i<numLayer; ++i)
	{
	  fAnn>>layerSize[i];
	}
      fAnn.close();
  }
  catch (exception &e)
  {
      cerr<<endl<<"Error reading ANN configuration file:"<<endl;
      cerr<<e.what()<<endl<<endl;
      return 1;
  }

  /*
   * Load the training configuration.
   */
  try
  {
      fTrain>>beta>>alpha>>thErr>>maxInte>>numRowTrain>>numRowVal>>numRowTest;
      numRow = numRowTrain + numRowVal + numRowTest;
  }
  catch (exception &e)
  {
      cerr<<endl<<"Error reading training configuration file:"<<endl;
      cerr<<e.what()<<endl<<endl;
      return 1;
  }
  fTrain.close();

  /*
   * Load the input data for the network. Get the number of entries and allocate
   * memory
   */
  cout<<"network's inputs file... "<<flush;
  try
  {
      dIn = new double*[numRow];
      for(i=0; i<numRow; ++i)
	{
	  dIn[i] = new double[layerSize[0]];
	}

      for(i=0; i<numRow; ++i)
	{
	  for(j=0; j<layerSize[0]; ++j)
	    {
	      fIn>>dIn[i][j];
	    }
	}
      fIn.close();
  }
  catch (exception &e)
  {
      cerr<<endl<<"Error reading Input data file:"<<endl;
      cerr<<e.what()<<endl<<endl;
      return 1;
  }

  /*
   * Load the target data (ideal outputs for the previous entries). Check if the
   * number of these entries is the same of the inputs.
   */
  cout<<"network's target file... "<<flush;
  try
  {
      dTarget = new double*[numRow];
      for(i=0; i<numRow; ++i)
	{
	  dTarget[i] = new double[layerSize[numLayer-1]];
	}

      for(i=0; i<numRow; ++i)
	{
	  for(j=0; j<layerSize[numLayer-1]; ++j)
	    {
	      fTarget>>dTarget[i][j];
	    }
	}
      fTarget.close();
  }
  catch (exception &e)
  {
      cerr<<endl<<"Error reading Target configuration file:"<<endl;
      cerr<<e.what()<<endl<<endl;
      return 1;
  }
  cout<<"OK!"<<endl<<"Now, a new ANN will be trained..."<<endl;

  /*
   * Create the Training instance (It will create an own ANN to be trained)
   */
  Training *trainIns = new Training(numLayer, layerSize, alpha, beta);

  /*
   * Training process:
   */
  count=0;
  while(1)
    {
      count++;
      /*
       * Apply blackpropagation for each training sample
       */
      for (i=0; i<numRowTrain; i++)
	{
	  trainIns->backpropagation(dIn[i], dTarget[i]);
	}
      /*
       * Calculate the Network Error with the validation samples and print it
       */
      err=0;
      for(i=numRowTrain; i<numRowTrain+numRowVal; i++)
	{
	  trainIns->feedforward(dIn[i]);
	  err+=trainIns->netErr(dTarget[i]);
	}
      err = err/numRowVal;
      cout<<"MEAN ERROR: "<<err<<endl;
      /*
       * Check if minimum error or maximum interations are achieved
       */
      if(err<=thErr)
	{
	  cout<<endl<<"Aim error value achieved with a ";
	  break;
	}
      if(count>=maxInte)
	{
	  cout<<endl<<"Max interations exceeded with a ";
	  break;
	}
    }
  cout<<"validation error of "<<err<<endl;

  /*
   * Run a test and print the error gotten
   */
  cout<<endl<<"TEST RESULT:"<<endl;
  err=0;
  count=0;
  for(i=numRowTrain+numRowVal; i<numRow; i++)
    {
      count++;
      trainIns->feedforward(dIn[i]);
      /*
       * Print the result
       */
      cout<<"Test No"<<count<<":"<<endl;
      cout<<"In: [";
      for(j=0; j<layerSize[0]; j++)
	{
	  cout<<" "<<dIn[i][j];
	}
      cout<<" ] ==> Out: [";
      for(j=0; j<layerSize[numLayer-1]; j++)
	{
	  cout<<" "<<trainIns->getOut(j);
	}
      cout<<" ] vs. Real Out: [";
      for(j=0; j<layerSize[numLayer-1]; j++)
	{
	  cout<<" "<<dTarget[i][j];
	}
      cout<<" ]"<<endl;
      /*
       * Sum the error to calculate the mean
       */
      err+=trainIns->netErr(dTarget[i]);
    }
  err = err/numRowTest;
  cout<<endl<<"TEST DONE with a mean error of "<<err<<endl;

  /*
   * Free all dynamic memory
   */

  delete[] layerSize;

  for(i=0; i<numRow; ++i)
    {
      delete[] dIn[i];
      delete[] dTarget[i];
    }
  delete[] dIn;
  delete[] dTarget;


  return 0;
}



