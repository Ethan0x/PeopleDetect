#ifndef _PD_HOG_H
#define _PD_HOG_H

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"

#include "pd_parameters.h"
#include "pd_blockcache.h"

using namespace cv; 

class PD_HOG
{
public:
	PD_HOG(Size sWinSize = Size(WINSIZE_W, WINSIZE_H), Size sBlockSize = Size(BLOCKSIZE_X, BLOCKSIZE_Y), 
		Size sBlockStride =  Size(BLOCKSTRIDE_X, BLOCKSTRIDE_Y), Size sWinStride = Size(WINSTRIDE_X, WINSTRIDE_Y), Size sCellSize =  Size(CELLSIZE_X, CELLSIZE_Y), 
		int iBins = BINS, enGamacorrect enGammaCorStyle = GamaSqrt, enBlockNormStyle enNormStyle = L2_Hys, float fGaussSigma = GUASSSIGMA, double dBlockThreshold = THRESHOLD);

	enErorr CheckParameter() const;

	void Detect(const Mat &mImg, vector<Rect>&vFoundRect, Size sPaddingTL, Size sPaddingBR, int iTrainType = CvSVM::LINEAR, unsigned int iMaxLayers = MAXLAYERS, 
		double dHitTheshold = 0.0, double dPyrScale = 1.05, double dGroupRectThreshold = 0.5);

	void DetectBatch(const char *cFileName, vector<int> &vResults, Size sPaddingTL, Size sPaddingBR, int iTrainType = CvSVM::LINEAR, unsigned int iMaxDetect = MAXDETECT,
		unsigned int iMaxLayers = MAXLAYERS, double dHitTheshold = 0.0, double dPyrScale = 1.05);

	void DetectLinearPyrLayer(const Mat &mImg, vector<Point>&vFoundPoint,Size sPaddingTL, Size sPaddingBR,double dHitTheshold = 0.0);

	void DetectNLinearPyrLayer(const Mat &mImg, vector<Point>&vFoundPoint, Size sPaddingTL, Size sPaddingBR, double dHitTheshold = 0.0);

	virtual void computeGradient(const Mat& mImg, Mat& mGrad, Mat& mAngle, Size sPaddingTL = Size(PADDING_W, PADDING_H), Size sPaddingBR = Size(PADDING_W, PADDING_H)) const;

	unsigned int getWDescriptorSize()const;

	void setSVM(const char *cFileName, const int iTrainType = CvSVM::LINEAR);

	static vector<float> getDefaultPeopleDetector();

	static void GroupRect(vector<Rect>&vRects, double dGroupThreshold);

	void HOGTraining(char *cFileNamePos, char *cFileNameNeg, char *cFileNameStore, int iTrainType = CvSVM::LINEAR, Size sPaddingTL = Size(PADDING_W, PADDING_H),
		Size sPaddingBR = Size(PADDING_W, PADDING_H), unsigned int iPosCount = TRAINCOUNT, unsigned int iNegCount = TRAINCOUNT, Point pRoi = Point(ROIOFFSET, ROIOFFSET));

	void HOGHardTraining(char *cFileNamePos, char *cFileNameNeg, char *cFileNameStore, vector<int>& vResults, int iTrainType = CvSVM::LINEAR, Size sPaddingTL = Size(PADDING_W, PADDING_H),
		Size sPaddingBR=Size(PADDING_W, PADDING_H),unsigned int iPosCount = TRAINCOUNT, unsigned int iNegCount = TRAINCOUNT, Point pRoi = Point(ROIOFFSET, ROIOFFSET));

	void HOGFeatureCompute(const Mat &mImg, vector<float> &vDescriptor, Size sPaddingTL = Size(PADDING_W, PADDING_H),
		Size sPaddingBR = Size(PADDING_W, PADDING_H));
	static void SVMModeToWeightV(char *cModeFileName, vector<float> &vWeightVector);
	static void SaveWeightV(char *cFileName, const vector<float> &vWeightVector);
	void GetSVMWeightV(const char * cFileName);

	void SetDefaultSVM();

	void calcPCA(Mat &mFeature, Mat &mMean, int iNumPrincipals, int flag);
	void pcaProject(PCA &pca, Mat&mFeatureIn, Mat &mFeatureOut);
	void setPCAMode(char *cPCAModeFileName);
	void savePCAMode(char *cPCAModeFileName);
	void usePCA(bool bUsePca, int iNumPrinciples = 30);


protected:

private:
	Mat m_mGrad;	//gradient of img
	Mat m_mAngle;	//gradient's orientation
	Size sWinSize;	//size of windows 
	Size sBlockSize;	//size of block 
	Size sBlockStride;	//moving stride of block,containing x and y orientantion
	Size sWinStride;	//moving stride of window,containing x and y orientantion
	Size sCellSize;	//size of cell
	int iBins;	//numbers of orientation bins 
	CvSVM cvSvm;
	enBlockNormStyle enNormStyle;	//the norm style to block
	double dBlockThreshold;	//the threshold of block norm style, maxium is 0.2
	enGamacorrect enGammaCorStyle;	//style of Gammacorrect 
	float m_fGaussSigma;
	vector<float> m_vSVMDetector;
	PCA pca;	//to reduce the dimensions of HOG feature
	bool bUsePCA;
	int iNumPrinciples;

};

#endif //_PD_HOG_H


