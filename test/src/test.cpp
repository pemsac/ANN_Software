//============================================================================
// Name        : test.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <ctime>
using namespace std;

int main() {
  clock_t time1, time2;
  int count=1;

  time1 = clock();
  while(1)
    {
      time2 = clock();
      if(time2-time1>CLOCKS_PER_SEC*count)
	{
	  cout<<count<<" segundos"<<endl;
	  count++;
	}
    }

  return 0;
}
