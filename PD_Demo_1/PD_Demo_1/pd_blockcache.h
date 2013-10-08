#ifndef _PD_BLOCK_CACHE_H
#define _PD_BLOCK_CACHE_H

#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "pd_parameters.h"

using namespace cv; 

class PD_Pixel
{
public:
	size_t m_iGradOfs, m_iAngleOfs;
	float m_iGausWeight;
	int m_aVoteCellOfs[4];
	float m_aVoteCellWeight[4];
};

class PD_Block
{
public:
	int m_iHistOfsWin;
	Point m_pBOfsWin;
};

class PD_BlockCache
{
public:
	PD_BlockCache( Mat &mGrad, Mat &mAngle, Size sCellSize, Size sBlockSize,
		unsigned int iBins, Size sWinSize, Size sBlockStride, Size sCacheStride, float fGaussSigma = GUASSSIGMA);
	float *getBlock(Point pPosInImg, float * fHistBlock);   
	Point getWindow(Size sImagsize, Size sWinStride, int iIndex) const;
	const Mat &getGausWeight(const float sigma);
	void iniPixelInB();
	void iniBlockInW(const Size sBlockSizeInWin, const int iHistsInB);
	void normBlock(float *hist, const int iHistInBlock, const enBlockNormStyle enStyle, float fThreshold = THRESHOLD);
	Mat *m_mGrad;
	Mat *m_mAngle;
	Size m_sCacheStride;
	Size m_sBlockStride;
	Size m_sGradSize;				//the padded img size
	vector<PD_Pixel> m_vPixelInB;
	vector<PD_Block> m_vBlockInW;		
	Mat_<float> m_mBlockHistCache;  //store all the hist in a cache  
	Mat_<float> m_mGauSweight;
	Mat_<uchar> m_ucCacheFlags;
	vector<int> m_vYCache;			//when a new row is ready to cached, an old row in the m_mblockHistCache should be removed  
	Size m_sCellSize;
	Size m_sBlockSize;
	Size m_sWinSize;
	unsigned int m_iBins;
	unsigned int m_iHistInBlock;
	int m_iSituation1, m_iSituation2, m_iSituation3;
};

#endif //_PD_BLOCK_CACHE_H