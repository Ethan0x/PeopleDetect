#include <iostream>
#include "pd_header.h"

using namespace std;

int main (int argc, const char **argv[])
{
	//TrainSimple();
	TrainHard();
	//DetectMutiObjects();
	//DetectOneObject();

	int c = waitKey(0) & 255;
	return 0;
}