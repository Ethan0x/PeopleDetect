#ifndef _PD_HEADER_H
#define _PD_HEADER_H

#include <iostream>
#include <string>
#include <stdio.h>
#include <ctype.h>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "pd_parameters.h"
#include "pd_blockcache.h"
#include "pd_hog.h"

using namespace std;
using namespace cv;

int iTrainType = CvSVM::RBF;                        
Size  sPadding = Size(16,24);                       
double fGroupThreshold = 0.7;	//original is 0.6                       
double fScale = 1.05;                               
int iTrainPos = 2416;	//the number of Pos training samples;                                
int iTrainNeg = 5000;	//the number of Neg training samples;
int iDetect = 925;		//the number of Test samples;
Point pRoi = Point(15,15);								
char *cpDelectImg = "00000002a.png";     
char *cpTrainPos = "D:\\Database\\Train\\Train_Pos\\train_pos.lst";
char *cpTrainNeg = "D:\\Database\\Train\\Train_Neg\\train_neg.lst";
char *cpTestPos = "D:\\Database\\Train\\Test_Pos\\test_pos.lst";
char *cpTestNeg = "D:\\Database\\Train\\Test_Neg\\test_neg.lst";
//char *cpDetect = "D:\\Database\\Train\\Train_Pos\\train_pos.lst";
//char *cpDetect = "D:\\Database\\Train\\Test_Pos\\test_pos.lst";
//char *cpDetect = "D:\\Database\\Train\\Test_Neg\\test_neg.lst";
//char *cpDetect = "D:\\Database\\Train\\Detect\\detect.txt";
char *cpDetect = "D:\\Database\\Train\\Train_Neg\\train_neg.lst";
char *cpLinearFeatures = "D:\\featuresL.xml";        
char *cpLWeightV = "D:\\WeightVL.xml";               
char *cpNLinearFeatures = "D:\\featuresNL_test.xml";     
char *cpPCAModeName = "D:\\PCA.xml";      
int iNumPrinciples = 50; 
bool bUsePCA = false; 

void TrainSimple()
{
	PD_HOG cHOG;
	cHOG.usePCA(bUsePCA, iNumPrinciples);

	double t = (double)getTickCount();
	cHOG.HOGTraining(cpTrainPos, cpTrainNeg, iTrainType == CvSVM::LINEAR ? cpLinearFeatures : cpNLinearFeatures, iTrainType, Size(0,0), Size(0,0), iTrainPos, iTrainNeg, pRoi);

	t = (double)getTickCount() - t;
	printf("the train time = %gms\n", t*1000./cv::getTickFrequency());
	if(iTrainType == CvSVM::LINEAR)
	{
		printf("transform to weight vector.........\n");
		vector<float> vWeightVector;
		PD_HOG::SVMModeToWeightV(cpLinearFeatures, vWeightVector);
		PD_HOG::SaveWeightV(cpLWeightV, vWeightVector);
	}
	if(bUsePCA)
	{
		printf("saving PCA......\n");
		cHOG.savePCAMode(cpPCAModeName);
	}
	printf("train over!\n");
}

void TrainHard()
{
	//First train simple
	TrainSimple();
	//detectMulti
	printf("start to detect the negative sample for hard samples!\n");
	//waitKey(2000);
	PD_HOG cHOG;
	vector<int> vResults;
	char *cFilename = iTrainType==CvSVM::LINEAR ? cpLWeightV : cpNLinearFeatures;
	cHOG.setSVM(cFilename, iTrainType);
	//cHOG.SetDefaultSVM(); //only Linear kernel is supported

	namedWindow("cHOG", 1);
	double t = (double)getTickCount();
	//void Detect(const Mat &mImg, vector<Rect> &vFoundRect,S ize sPaddingTL, Size sPaddingBR, unsigned int iMaxLayers = MAXLAYERS, double dHitTheshold = 0.0, double dPyrScale = 1.05, double dGroupRectThreshold = 0.5);
	//cHOG.Detect(img, found, sPadding, sPadding, iTrainType, 50, 0.0, fScale, fGroupThreshold);
	//vector<int> vResults;
	//void PD_HOG :: DetectBatch(const char * cFileName, vector<int> &vResults,Size sPaddingTL, Size sPaddingBR,
	//	int iTrainType,unsigned int iMaxDetect,unsigned int iMaxLayers, double dHitTheshold, double dPyrScale)
	cHOG.DetectBatch(cpTrainNeg, vResults, sPadding, sPadding, iTrainType, iTrainNeg, 50, 0.4, fScale);
	t = (double)getTickCount() - t;
	printf("the multidetection time = %gms\n", t*1000./cv::getTickFrequency());

	//Hard train
	printf("start hard train!\n");
	//waitKey(2000);
	t = (double)getTickCount();
	cHOG.HOGHardTraining(cpTrainPos, cpTrainNeg, iTrainType == CvSVM::LINEAR ? cpLinearFeatures : cpNLinearFeatures, vResults, iTrainType, Size(0,0), Size(0,0), iTrainPos, iTrainNeg, pRoi);
	t = (double)getTickCount() - t;
	printf("the hard train time = %gms\n", t*1000./cv::getTickFrequency());

	if(iTrainType == CvSVM::LINEAR)
	{
		printf("transform to weight vector.........\n");
		vector<float> vWeightVector;
		PD_HOG::SVMModeToWeightV(cpLinearFeatures,vWeightVector);
		PD_HOG::SaveWeightV(cpLWeightV,vWeightVector);
	}
	printf("hard train over!\n");
	return;
}

void DetectMutiObjects()
{
	PD_HOG cHOG;
	cHOG.usePCA(bUsePCA,iNumPrinciples);
	if(bUsePCA)
	{
		cHOG.setPCAMode(cpPCAModeName);
	}
	char *cFilename = iTrainType == CvSVM::LINEAR ? cpLWeightV : cpNLinearFeatures;
	cHOG.setSVM(cFilename, iTrainType);
	//only Linear kernel is supported
	//cHOG.SetDefaultSVM(); 

	namedWindow("cHOG", 1);
	double t = (double)getTickCount();
	//void Detect(const Mat &mImg, vector<Rect> &vFoundRect, Size sPaddingTL, Size sPaddingBR, unsigned int iMaxLayers = MAXLAYERS, 
	//	double dHitTheshold = 0.0, double dPyrScale = 1.05, double dGroupRectThreshold = 0.5);
	//cHOG.Detect(img, found, sPadding, sPadding, iTrainType, 50, 0.0, fScale, fGroupThreshold);
	vector<int> vResults;
	//void PD_HOG :: DetectBatch(const char * cFileName, vector<int> &vResults,Size sPaddingTL, Size sPaddingBR,
	//	int iTrainType,unsigned int iMaxDetect,unsigned int iMaxLayers, double dHitTheshold, double dPyrScale)
	cHOG.DetectBatch(cpDetect, vResults, sPadding, sPadding, iTrainType, (unsigned int)15000, 50, 0.1, fScale);
	t = (double)getTickCount() - t;
	printf("multi detection time = %gms\n", t*1000./cv::getTickFrequency());
}

void DetectOneObject()
{
	char *cFileName = cpDelectImg;
	Mat img;
	img = imread(cFileName);
	if(img.empty())
	{
		printf("can't not open the file %s\n",cFileName);
		return ;
	}

	printf("loading XML....\n");

	PD_HOG cHOG;
	cHOG.usePCA(bUsePCA,iNumPrinciples);
	if(bUsePCA)
	{
		cHOG.setPCAMode(cpPCAModeName);
	}
	char *cFilename = iTrainType == CvSVM::LINEAR ? cpLWeightV : cpNLinearFeatures;
	cHOG.setSVM(cFilename,iTrainType);
	//cHOG.SetDefaultSVM(); // only Linear kernel is supported

	namedWindow("cHOG", 1);
	vector<Rect> found;
	printf("detecting....\n");
	double t = (double)getTickCount();
	//void Detect(const Mat &mImg, vector<Rect> &vFoundRect, Size sPaddingTL, Size sPaddingBR, unsigned int iMaxLayers = MAXLAYERS, double dHitTheshold = 0.0, double dPyrScale = 1.05, double dGroupRectThreshold = 0.5);
	cHOG.Detect(img, found, sPadding, sPadding, iTrainType, 50, 0.1, fScale, fGroupThreshold);
	t = (double)getTickCount() - t;
	printf("detection time = %gms\n", t*1000./cv::getTickFrequency());
	unsigned int i ;
	std::cout << found.size() << std::endl;
	for (unsigned int i = 0; i <found.size(); i ++)
	{
		std::cout << i << " " << found[i].x << " " << found[i].y << " " << found[i].width << " " << found[i].height << std::endl;
	}
	for (i = 0; i < found.size(); i++ )
	{
		Rect r = found[i];
		//the HOG detector returns slightly larger rectangles than the real objects.
		//so we slightly shrink the rectangles to get a nicer output.
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += cvRound(r.height*0.01);
		r.height = cvRound(r.height*0.85);
		rectangle(img, r.tl(), r.br(), cv::Scalar(0,255,0), 2);

	}
	imshow("people detector", img);
	return;
}

#endif //_PD_HEADER_H