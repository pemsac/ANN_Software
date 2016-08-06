
#include "main.h"




int main(int argc, char* argv[])
{
	long i,j;

	double **data, *testData, sqerr, maxsqerr;
	int numLayers, *lSz, numRow, count;

	fstream fAnn, fTrain, fInput, fOutput;


	// Learing rate - beta
	// momentum - alpha
	// Threshhold - thresh (value of target mse, training stops once it is achieved)
	double beta = 0.2, alpha = 0.1, Thresh =  0.00001;
	// maximum no of iterations during training
	long num_iter = 1000000;

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


	testData = new double[lSz[0]];


	// Creating the net
	Training *bp = new Training(numLayers, lSz, alpha, beta);

	cout<< endl <<  "Now training the network...." << endl;

	//  for ( i=0; i<num_iter ; i++)
	count=0;
	do
	{
		sqerr=0.0;
		for (i=0; i<numRow ; i++)
		{
			bp->backpropagation(data[i], &data[i][lSz[0]]);
			sqerr += bp->mse(&data[i][lSz[0]]);
		}
		++count;
		sqerr = sqerr / numRow;
		if (count%10 == 0)
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

		bp->feedforward(testData);
		cout<<"out: ";
		for(i=0; i<lSz[numLayers-1]; ++i)
		{
			cout<<bp->getOut(i)<<"  ";
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



