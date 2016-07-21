
#include <fstream>

#include "BackProp.h"

/*
 * FILE SYSTEM DIRECTORIES
 */
#define FS_PATH			"0:/"
#define TRAIN_FILE_DIR		"files_b/training.ann"
#define ANN_FILE_DIR		"files_b/ANN.ann"
#define IN_FILE_DIR		"files_b/input.ann"
#define OUT_FILE_DIR		"output.ann"

int main(int argc, char* argv[])
{
  long i,j;

  double **data, *testData, sqerr, maxsqerr;
  int numLayers, *lSz, numRow, count;

  fstream fAnn, fTrain, fInput, fOutput;


  // Learing rate - beta
  // momentum - alpha
  // Threshhold - thresh (value of target mse, training stops once it is achieved)
  double beta = 0.05, alpha = 0.6, Thresh =  0.001;


  /*
   * Open files
   */
  cout<<"Loading data..."<<endl;
  try
  {
      fAnn.open(ANN_FILE_DIR, fstream::in);
      fAnn.seekg(0, ios::beg);
      fTrain.open(TRAIN_FILE_DIR, fstream::in);
      fTrain.seekg(0, ios::beg);
      fInput.open(IN_FILE_DIR, fstream::in);
      fInput.seekg(0, ios::beg);
      fOutput.open(OUT_FILE_DIR, fstream::out);
      fOutput.seekg(0, ios::beg);
  }
  catch (exception &e)
  {
      cerr<<"Error opening file:"<<endl;
      cerr<<e.what()<<endl<<endl;
      return 1;
  }

  /*
   * Load data
   */
  fAnn>>numLayers;

  lSz = new int[numLayers];

  for (i=0; i<numLayers; ++i)
    {
      fAnn>>lSz[i];
    }

  fTrain>>numRow;

  data = new double*[numRow];
  for(i=0; i<numRow; ++i)
    {
      data[i] = new double[lSz[0]+lSz[numLayers-1]];
    }

  for(i=0; i<numRow; ++i)
    {
      for(j=0; j<lSz[0]+lSz[numLayers-1]; ++j)
	{
	  fTrain>>data[i][j];
	}
    }
  fTrain.close();
  fAnn.close();
  // maximum no of iterations during training
  long num_iter = 100000;

  testData = new double[lSz[0]];


  // Creating the net
  CBackProp *bp = new CBackProp(numLayers, lSz, beta, alpha);

  cout<< endl <<  "Now training the network...." << endl;

  //  for ( i=0; i<num_iter ; i++)
  count=0;
  do
    {
      sqerr=0.0;
      for (i=0; i<numRow ; i++)
	{
	  bp->bpgt(data[i], &data[i][lSz[0]]);
	  sqerr += bp->mse(&data[i][lSz[0]]);
	}
      ++count;
      sqerr = sqerr / numRow;
      if (count%100 == 0)
	{
	  cout<<  endl <<  "MSE:  " << sqerr << "... Training..." << endl;
	}
    }
  while((sqerr > Thresh) && (count<num_iter));

	  cout << endl << "Network Trained in " << count << " iterations." << endl;
	  cout << "MSE:  " << sqerr  <<  endl <<  endl;

  cout<< "Now using the trained network to make predctions on test data...." << endl << endl;

  count=0;
  while ( !fInput.eof() )
    {
      ++count;
      cout<<"Entry No "<<count<<":"<<endl;
      fOutput<<"Entry No "<<count<<":"<<endl;
      for (i=0; i<lSz[0]; ++i)
	{
	  fInput>>testData[i];
	  //	  cout << testData[i]<< "  ";
	}

      bp->ffwd(testData);
      cout<<"out: ";
      for(i=0; i<lSz[numLayers-1]; ++i)
	{
	  cout<<bp->Out(i)<<"  ";
	}
      cout<<endl;
    }

  fInput.close();
  fOutput.close();
  delete[] testData;

  delete[] lSz;

  for(i=0; i<numRow; ++i)
    {
      delete[] data[i];
    }
  delete data;


  return 0;
}



