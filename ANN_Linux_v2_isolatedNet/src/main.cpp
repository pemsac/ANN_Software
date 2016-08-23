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
  int i,j, k, numLayer, *layerSize, numRowTrain, numRowVal, numRowTest, numRow,
  ite, maxIte, *netOut, numOut, *goodOut, *badOut;
  double **dIn, **dTarget, mcee, lastMcee, thMcee, *maxIn, *minIn;
  bool bad;
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

      for (i=0; i<numLayer; i++)
	{
	  fAnn>>layerSize[i];
	}

      numOut = layerSize[numLayer-1];

      netOut = new int[numOut];

      maxIn = new double[layerSize[0]];
      minIn = new double[layerSize[0]];

      goodOut = new int[numOut]();
      badOut = new int[numOut]();

      for(i=0; i<layerSize[0]; i++)
	{
	  maxIn[i] = CODEC_MIN;
	  minIn[i] = CODEC_MAX;
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
      fTrain>>thMcee>>maxIte>>numRowTrain>>numRowVal>>numRowTest;
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
      for(i=0; i<numRow; i++)
	{
	  dIn[i] = new double[layerSize[0]];
	}

      for(i=0; i<numRow; i++)
	{
	  for(j=0; j<layerSize[0]; j++)
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
      for(i=0; i<numRow; i++)
	{
	  dTarget[i] = new double[numOut];
	}

      for(i=0; i<numRow; i++)
	{
	  for(j=0; j<numOut; j++)
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

  /*
   * Codify the inputs
   */
  cout<<"Coding entries..."<<flush;

  for(i=0; i<layerSize[0]; i++)
    {
      double a = (CODEC_MAX-CODEC_MIN)/(maxIn[i]-minIn[i]);
      double b = CODEC_MIN - a*minIn[i];
      for(j=0; j<numRow; j++)
	{
	  dIn[j][i] = dIn[j][i]*a + b;
	}
    }
  delete[] maxIn;
  delete[] minIn;

  cout<<"OK!"<<endl<<"Now, a new ANN will be trained..."<<endl;

  /*
   * Create the Training instance (It will create an own ANN to be trained)
   */
  Training *trainIns = new Training(numLayer, layerSize);

  /*
   * Training process:
   */
  ite=0;
  lastMcee=999;
  while(1)
    {
      ++ite;
      /*
       * Apply blackpropagation for each training sample
       */
      for(i=0; i<numRowTrain; ++i)
	{
	  trainIns->backpropagation(dIn[i], dTarget[i]);
	}
      /*
       * Calculate the Network Error with the validation samples and print it
       */
      for(i=numRowTrain, mcee=0; i<numRowTrain+numRowVal; i++)
	{
	  trainIns->feedforward(dIn[i]);
	  mcee+=trainIns->CEE(dTarget[i]);
	  trainIns->getNetOut(netOut);
	}
      mcee /= numRowVal;
      cout<<"VALIDATION: MCEE = "<<mcee<<endl;
      /*
       * Check if minimum error or maximum iterations are achieved
       */
      if(mcee<=thMcee)
	{
	  cout<<"DONE!"<<endl;
	  break;
	}
      if(ite>=maxIte)
	{
	  cout<<"WARNING Target error wasn't achieved"<<endl;
	  break;
	}
      trainIns->updateLRandM(mcee,lastMcee);
      lastMcee=mcee;
    }

  /*
   * Run a test and print the error gotten
   */

  cout<<endl<<"Testing..."<<endl;

  for(i=numRowTrain+numRowVal, ite=0, mcee=0, k=0; i<numRow; i++)
    {
      ite++;
      trainIns->feedforward(dIn[i]);

      /*
       * Verify the result
       */
      trainIns->getNetOut(netOut);
      for(j=0, bad=false; j<numOut; ++j)
	{
	  if(netOut[j]!=dTarget[i][j])
	    {
	      bad=true;
	    }
	  if(dTarget[i][j]==1)
	    {
	      k=j;
	    }
	}
      if(bad)
	{
	  badOut[k]++;
	}
      else
	{
	  goodOut[k]++;
	}

      /*
       * Sum the error to calculate the mean
       */
      mcee+=trainIns->CEE(dTarget[i]);
    }
  mcee = mcee/numRowTest;
  cout<<"TEST DONE Results:"<<endl;
  cout<<"MCEE = "<<mcee<<endl;
  for(i=0; i<numOut; ++i)
    {
      cout<<"Outputs "<<i<<" Good = "<<goodOut[i]<<" Bad = "<<badOut[i]<<endl;
    }

  /*
   * Free all dynamic memory
   */

  delete[] layerSize;

  delete[] netOut;

  delete[] goodOut;

  delete[] badOut;

  for(i=0; i<numRow; i++)
    {
      delete[] dIn[i];
      delete[] dTarget[i];
    }
  delete[] dIn;
  delete[] dTarget;


  return 0;
}



