#include "pd_blockcache.h"
#include <iostream>

//Class Constructor 
PD_BlockCache ::PD_BlockCache(Mat &mGrad, Mat &mAngle, Size sCellSize, Size sBlockSize,
	unsigned int iBins, Size sWinSize, Size sBlockStride, Size sCacheStride, float fGaussSigma)
{
	m_mGrad = &mGrad;
	m_mAngle = &mAngle;
	m_sCellSize = sCellSize;
	m_sBlockSize = sBlockSize;
	m_iBins = iBins;
	m_sWinSize = sWinSize;
	m_sBlockStride = sBlockStride;
	m_sCacheStride = sCacheStride;
	m_sGradSize = mGrad.size();

	const int iPixelsInB = m_sBlockSize.height * m_sBlockSize.width;
	const Size sBlockSizeInWin = Size((m_sWinSize.width - m_sBlockSize.width)/m_sBlockStride.width + 1,
		(m_sWinSize.height - m_sBlockSize.height)/m_sBlockStride.height + 1); //block size in windows
	m_vPixelInB.resize(iPixelsInB * 3); // the pixels in block will be partitioned into three parts, details will be stated follow
	m_vBlockInW.resize(sBlockSizeInWin.width * sBlockSizeInWin.height);
	m_iHistInBlock = iPixelsInB/(m_sCellSize.height * m_sCellSize.width) * m_iBins;
	// the sCacheSize is the caches when the windows move from the most left to most right along the padded img 
	Size sCacheSize = Size((m_sGradSize.width - m_sBlockSize.width) / m_sCacheStride.width + 1, m_sWinSize.height /  m_sCacheStride.height+1 );
	//so the m_mBlockHistCache store all the hists in a cache
	m_mBlockHistCache.create(sCacheSize.height, sCacheSize.width * m_iHistInBlock);
	//records if a block has been computed
	m_ucCacheFlags.create(sCacheSize);
	m_vYCache.resize(sCacheSize.height);
	//initialize the m_vYcache
	for (int i = 0; i < sCacheSize.height; i++)
		m_vYCache[i] = -1;
	// compute the gausweight
	getGausWeight(fGaussSigma);
	iniPixelInB();
	iniBlockInW(sBlockSizeInWin,m_iHistInBlock);
}

//Get the the Gauss weight according to the pixel's position in a block
//(x0,y0) is the center position of the block 
//GausWeight(i,j) = exp(((xi-x0)2+(yi-y0)2)/temp)
const Mat &PD_BlockCache ::getGausWeight(const float sigma )
{
	Mat_<float>mWeight(m_sBlockSize);
	m_mGauSweight = mWeight;
	float temp = -1/(2 * sigma * sigma);
	int x0 = m_sBlockSize.width/2;
	int y0 = m_sBlockSize.height/2;
	int i,j;
	for (i = 0; i < m_sBlockSize.width; i++)
	{
		for(j = 0; j < m_sBlockSize.height; j++)
		{
			m_mGauSweight(i,j) = exp(((i - x0 )*(i - x0 ) + (j - y0)*(j - y0))*temp);
		}
	}
	return m_mGauSweight;
}

//size_t m_iGradOfs, m_iAngleOfs;
//m_iGausWeight;
//m_aVoteCellOfs[4];
//m_aVoteCellWeight[4];
//m_iGradOfs,m_iAngleOfs represents the offset from the first grad or angle in a block
//m_iGausweight is the weight to the block
//a specific pixel votes at most four cells due to its position in a block, at lest one
//m_aVoteCellOfs[4] separately present the voted cell's offset of a block hist
//and m_aVoteCellWeight[4] is corresponding weight to each cell. 
//the weight is caculated due to it's x offset and y offset to each cell 
//|_0_|_1_|_2_|_3_|
//|_4_|_5_|_6_|_7_|
//|_8_|_9_|_10|_11|
//|_12|_13|_14|_15| 

void PD_BlockCache :: iniPixelInB()
{
	int i , j;
	// three situations, 
	int iPixelsInB = m_sBlockSize.height * m_sBlockSize.width;
	int iPixelsInB1 = iPixelsInB * 2;
	Size sCellsInB = Size(m_sBlockSize.width/m_sCellSize.width, m_sBlockSize.height/m_sCellSize.height);
	//fCellXOfs,fCellYOfs;reprents the offset which a pixel to a cell 
	float fCellXOfs,fCellYOfs;
	int iCellX0,iCellY0;
	PD_Pixel *pPixelData = NULL;
	//every block have divided four part, part size is ucSizeBlockPiece
	uchar ucSizeBlockPiece = sCellsInB.width * sCellsInB.height/4;
	Size ucHalfBlockSize(m_sBlockSize.width/2,m_sBlockSize.height/2);
	m_iSituation1 = 0; 
	m_iSituation2 = 0; 
	m_iSituation3 = 0; 
	for (j = 0; j < m_sBlockSize.width ; j++)
	{
		for( i = 0; i < m_sBlockSize.height; i++)
		{    
			//+0.5because float number in computer is not accurate
			fCellXOfs = (j + 0.5f)/ucHalfBlockSize.width - 0.5f;
			fCellYOfs = (i + 0.5f)/ucHalfBlockSize.height - 0.5f;
			iCellX0 = cvFloor(fCellXOfs);
			iCellY0 = cvFloor(fCellYOfs);
			//iCellY1 = iCellY0 + 1;
			//iCellX1 = iCellX0 + 1;
			fCellXOfs -= iCellX0;
			fCellYOfs -= iCellY0;
			// the "if" means the pixel is in the area of 1 2 5 6 9 10 13 14
			if (iCellX0 ==0)
			{
				//the "if" means the pixel is in the area of 5 6 9 10
				if(iCellY0 ==0)
				{
					//the iCellX0 iCellY0 are 0 and iCellX1 and iCellY1 are 1
					pPixelData = &m_vPixelInB[iPixelsInB1 + (m_iSituation3++)];
					pPixelData->m_aVoteCellOfs[0] = 0;// (iCellX0 + iCellY0) * m_iBins;
					pPixelData->m_aVoteCellWeight[0] = (1- fCellXOfs) *  (1- fCellYOfs);
					pPixelData->m_aVoteCellOfs[1] =  2 * ucSizeBlockPiece * m_iBins;
					pPixelData->m_aVoteCellWeight[1] = fCellXOfs *  (1- fCellYOfs);
					pPixelData->m_aVoteCellOfs[2] =  ucSizeBlockPiece * m_iBins;
					pPixelData->m_aVoteCellWeight[2] =(1- fCellXOfs) *  fCellYOfs ;
					pPixelData->m_aVoteCellOfs[3] = 3 * ucSizeBlockPiece * m_iBins;
					pPixelData->m_aVoteCellWeight[3] = fCellXOfs *  fCellYOfs ;

				}
				//the "else" means the pixel is in the area of 1 2 13 14
				else
				{
					pPixelData = &m_vPixelInB[iPixelsInB + (m_iSituation2++)];
					//the "if" means the pixel is in the area of 13 14
					if(iCellY0 == 1)
					{
						pPixelData->m_aVoteCellOfs[0] =  ucSizeBlockPiece * m_iBins;
						pPixelData->m_aVoteCellWeight[0] = (1- fCellXOfs) * (1- fCellYOfs);
						pPixelData->m_aVoteCellOfs[1] = 3 * ucSizeBlockPiece * m_iBins;
						pPixelData->m_aVoteCellWeight[1] =fCellXOfs *  (1- fCellYOfs);
						pPixelData->m_aVoteCellOfs[2] = pPixelData->m_aVoteCellOfs[3] =0;
						pPixelData->m_aVoteCellWeight[2] = pPixelData->m_aVoteCellWeight[3] = 0;  

					}
					//the "else" means the pixel is in the area of 1 2
					else
					{       
						pPixelData->m_aVoteCellOfs[0] = 0;// (iCellX0 + iCellY0) * m_iBins;
						pPixelData->m_aVoteCellWeight[0] =(1- fCellXOfs) *  fCellYOfs;
						pPixelData->m_aVoteCellOfs[1] = 2 * ucSizeBlockPiece * m_iBins;
						pPixelData->m_aVoteCellWeight[1] = fCellXOfs *  fCellYOfs;
						pPixelData->m_aVoteCellOfs[2] = pPixelData->m_aVoteCellOfs[3] =0;
						pPixelData->m_aVoteCellWeight[2] = pPixelData->m_aVoteCellWeight[3] = 0;  
					}

				}
			}
			//the "else" means the pixel is in the area of 0 4 8 2 3 7 11 15
			else
			{
				// the "if" means the pixel is in the area of 4 8 7 11
				if(iCellY0 == 0)
				{
					pPixelData = &m_vPixelInB[iPixelsInB + (m_iSituation2++)];
					// the "if" means the pixel is in the area of 4 8 
					if(iCellX0 == -1)
					{
						//pPixelData = &m_vPixelInB[iPixelsInB1 + (m_iSituation2++)];
						pPixelData->m_aVoteCellOfs[0] = 0;
						pPixelData->m_aVoteCellWeight[0] = fCellXOfs *  (1- fCellYOfs);
						pPixelData->m_aVoteCellOfs[1] = ucSizeBlockPiece * m_iBins;
						pPixelData->m_aVoteCellWeight[1] = fCellXOfs *  fCellYOfs;
						pPixelData->m_aVoteCellOfs[2] = pPixelData->m_aVoteCellOfs[3] =0;
						pPixelData->m_aVoteCellWeight[2] = pPixelData->m_aVoteCellWeight[3] = 0;  
					}
					//the "else" means the pixel is in the area of 7 11
					else
					{
						pPixelData->m_aVoteCellOfs[0] =2 * ucSizeBlockPiece * m_iBins;
						pPixelData->m_aVoteCellWeight[0] = (1-fCellXOfs) *  (1- fCellYOfs);
						pPixelData->m_aVoteCellOfs[1] = 3 * ucSizeBlockPiece * m_iBins;
						pPixelData->m_aVoteCellWeight[1] = (1-fCellXOfs) *  fCellYOfs;
						pPixelData->m_aVoteCellOfs[2] = pPixelData->m_aVoteCellOfs[3] =0;
						pPixelData->m_aVoteCellWeight[2] = pPixelData->m_aVoteCellWeight[3] = 0;  
					}
				}
				//the "else" means the pixel is in the area of 0 3 12 15
				else
				{
					pPixelData = &m_vPixelInB[m_iSituation1++];
					if(iCellX0 == -1 && iCellY0 == -1)
					{
						pPixelData->m_aVoteCellOfs[0] = 0;
						pPixelData->m_aVoteCellWeight[0] = fCellXOfs *  fCellYOfs;
					}
					else if(iCellX0 == 1 && iCellY0 == -1)
					{
						pPixelData->m_aVoteCellOfs[0] = 2* ucSizeBlockPiece * m_iBins;
						pPixelData->m_aVoteCellWeight[0] = (1 -fCellXOfs) *  fCellYOfs;


					}
					else if (iCellX0 == -1 && iCellY0 == 1)
					{
						pPixelData->m_aVoteCellOfs[0] = ucSizeBlockPiece * m_iBins;
						pPixelData->m_aVoteCellWeight[0] = fCellXOfs * (1 - fCellYOfs);
					}
					else
					{
						pPixelData->m_aVoteCellOfs[0] = 3 * ucSizeBlockPiece * m_iBins;
						pPixelData->m_aVoteCellWeight[0] = (1 - fCellXOfs) * (1 - fCellYOfs);
					}
					pPixelData->m_aVoteCellOfs[1] = pPixelData->m_aVoteCellOfs[2] = pPixelData->m_aVoteCellOfs[3] =0;
					pPixelData->m_aVoteCellWeight[1] = pPixelData->m_aVoteCellWeight[2] = pPixelData->m_aVoteCellWeight[3] = 0;  
				}

			}
			pPixelData->m_iGradOfs = (m_sGradSize.width * i + j) * 2;
			pPixelData->m_iAngleOfs = pPixelData->m_iGradOfs;
			pPixelData->m_iGausWeight = m_mGauSweight(i,j);
			//for test. there is nothing different
			/* std::cout<<"("<<i<<","<<j<<") "<<pPixelData->m_iGradOfs<<" "<<pPixelData->m_iGausWeight<<" "<<" ("
			<<pPixelData->m_aVoteCellOfs[0]<<","<< pPixelData->m_aVoteCellWeight[0]<<") "<<"("
			<<pPixelData->m_aVoteCellOfs[1]<<","<< pPixelData->m_aVoteCellWeight[1]<<") "<<"("
			<<pPixelData->m_aVoteCellOfs[2]<<","<< pPixelData->m_aVoteCellWeight[2]<<") "<<"("
			<<pPixelData->m_aVoteCellOfs[3]<<","<< pPixelData->m_aVoteCellWeight[3]<<") "<<std::endl;*/
		}

	}

}

//Initialize all the blocks in a window
//the m_iHistOfsWin and the m_pBofsWin
//the parameter sBlockSizeInwin stores the numbers of Blocks in windows both in width and in height
//the other parameter iHistsInB stores the total Histograms in a Block
void PD_BlockCache ::  iniBlockInW(const Size sBlockSizeInWin, const int iHistsInB)
{
	//int i,j;
	for(int i = 0; i < sBlockSizeInWin.width; i++)
	{
		for(int j = 0; j < sBlockSizeInWin.height; j++)
		{
			PD_Block &cbCurrBlock = m_vBlockInW[i * sBlockSizeInWin.height + j];
			//cbCurrBlock.m_iHistOfsWin = (i * sBlockSizeInWin.height + j) * iHistsInB;
			cbCurrBlock.m_pBOfsWin = Point(i * m_sBlockStride.width, j * m_sBlockStride.height);
			//for test  there is nothing different
			//std::cout<< "("<<i<<","<<j<<") "<<cbCurrBlock.m_iHistOfsWin <<" "<<cbCurrBlock.m_pBOfsWin<<std::endl;
		}
	}
}

//Normalize the block, the enBlockNormStyle is define in the pd_parameters.h
//enum enBlockNormStyle{L2norm,L2_Hys,L1norm,L1sqrt};
//the different styles are detail described in paper
void PD_BlockCache ::  normBlock(float *hist, const int iHistInBlock, const enBlockNormStyle enStyle, float fThreshold)
{
	switch (enStyle)
	{
		// the L2_Hys,is L2_norm followed by clipping(limiting the maximum values of v to 0.2) and renormalizing
	case L2_Hys:
		{
			float *fFirstHistInB = & hist[0];
			int i = 0;
			float sum = .0;
			//the ||v||2
			for(i; i < iHistInBlock; i++)
				sum += fFirstHistInB[i] * fFirstHistInB[i];
			//for test
			//std::cout<<sum<<"   "<<fFirstHistInB[0]<<std::endl;
			//we always add a "e" avoiding that the divisor is equal to 0
			//float scale = 1/(std::sqrt(sum) + 0.1f);
			float scale = 1/(std::sqrt(sum)+m_iHistInBlock*0.1f );
			sum  = .0;
			for (i = 0; i < iHistInBlock; i++)
			{
				//first normalize and clipping 
				fFirstHistInB[i] = std::min(fFirstHistInB[i]* scale, fThreshold);
				sum += fFirstHistInB[i] * fFirstHistInB[i];
			}
			// renormalize
			//scale = 1/(std::sqrt(sum) + 0.1f);
			scale = 1/(std::sqrt(sum)+1e-3f);
			for (i = 0; i < iHistInBlock; i++)
				fFirstHistInB[i] = fFirstHistInB[i]* scale;
			break;
		}

	case L2norm:
		{
			float *fFirstHistInB = &hist[0];
			int i = 0;
			float sum = .0;
			//the ||v||2
			for(i = 0; i < iHistInBlock; i++)
				sum += fFirstHistInB[i] * fFirstHistInB[i];		
			// we alway add a "e" avoiding that the divisor is equal to 0
			float scale = 1/(std::sqrt(sum)+m_iHistInBlock*0.1f );
			for (i = 0; i < iHistInBlock; i++)
				fFirstHistInB[i] = fFirstHistInB[i]* scale;
			break;
		}

	case L1norm:
		break;
	case L1sqrt:
		break;
	default:
		{

		}
	}
}

//Return a float pointer which points to the buffer of a specific block histogram according to the position
//the space the fhistBlock points to will be covered by the really Hist information
float *PD_BlockCache :: getBlock(Point pPosInImg, float *fHistBlock)
{
	//assert the parameter's validity
	assert(m_mGrad != NULL && m_mAngle != NULL &&
		(unsigned)pPosInImg.x <= (unsigned)(m_mGrad->cols - m_sBlockSize.width) &&
		(unsigned)pPosInImg.y <= (unsigned)(m_mGrad->rows - m_sBlockSize.height) );
	assert(pPosInImg.x % m_sCacheStride.width == 0 && pPosInImg.y % m_sCacheStride.height == 0);
	
	//pCachIdx is the position which block is in a cache
	//when the cache moves down, the computing need the operation "%"
	Point pCachIdx(pPosInImg.x/m_sCacheStride.width, (pPosInImg.y/m_sCacheStride.height) % m_mBlockHistCache.rows);
	//for test  there is no different
	//std::cout<<pCachIdx.x<<" "<<pCachIdx.y<<std::endl;
	if(pPosInImg.y != m_vYCache[pCachIdx.y])
	{
		//for test there is no different
		//std::cout<<pPosInImg.y <<" "<<m_vYCache[pCachIdx.y]<< "i am invoked"<<std::endl;
		m_ucCacheFlags.row(pCachIdx.y) = (uchar)0;
		m_vYCache[pCachIdx.y] = pPosInImg.y; 
	}

	fHistBlock = &m_mBlockHistCache[pCachIdx.y][pCachIdx.x * m_iHistInBlock];
	//for test
	//std::cout<<pCachIdx.y<<" "<<pCachIdx.x * m_iHistInBlock<< " "<<*fHistBlock<<std::endl;
	uchar &ucFlag = m_ucCacheFlags(pCachIdx.y, pCachIdx.x);
	if(ucFlag != 0)
	{
		//for test there is no different
		//std::cout<<pPosInImg.y <<" "<<ucFlag<< "i am invoked"<<std::endl;
		return fHistBlock;
	}
	else
		ucFlag = (uchar)1; // if has not cached, then set the flag as 1 before computing
	////initialize the fhistBlock
	for (unsigned int i = 0; i < m_iHistInBlock; i++)
		fHistBlock[i] = 0.0f;

	//locate the address of the current block's grad and angle information
	const float *fpGradPtr = (const float*)(m_mGrad->data + m_mGrad->step * pPosInImg.y) + pPosInImg.x*2;
	const uchar *ucpAnglePtr = m_mAngle->data + m_mAngle->step*pPosInImg.y + pPosInImg.x*2;
	int k ;
	// cpFirstPixInB points to the first Pixel in a block
	const PD_Pixel *cppFirstPixInB = &m_vPixelInB[0];
	int iPixelsInB = m_sBlockSize.height * m_sBlockSize.width;
	int iPixelsInB1 = iPixelsInB * 2;
	//situation 1;
	for(k = 0; k < m_iSituation1; k++)
	{
		const PD_Pixel *cppCurrPixel = &cppFirstPixInB[k];
		const float *fpGard =  fpGradPtr + cppCurrPixel->m_iGradOfs;
		const uchar *ucpAngle = ucpAnglePtr + cppCurrPixel->m_iAngleOfs;
		//the neighbouring bins(in fact the histograms)
		int iHistIdx1 = ucpAngle[0];
		int iHistIdx2 = ucpAngle[1];
		float * fCurrHistInB = fHistBlock + cppCurrPixel->m_aVoteCellOfs[0];
		float fWeight = cppCurrPixel->m_iGausWeight * cppCurrPixel->m_aVoteCellWeight[0];
		fCurrHistInB[iHistIdx1] += fpGard[0] *  fWeight;
		fCurrHistInB[iHistIdx2] += fpGard[1] * fWeight;
		//for test
		//std::cout<<k<<" "<<(int)ucpAnglePtr[0]<<" "<<pPosInImg.y<<" "<<pPosInImg.x<<" "<<iHistIdx1 <<" "<<iHistIdx2<<" " <<fCurrHistInB[iHistIdx1] <<" "<<fCurrHistInB[iHistIdx2]<<std::endl;
	}
	//for test
	//int itest ;
	//std::cin >>itest;

	//situation 2;
	for(k = 0; k < m_iSituation2; k++)
	{

		const PD_Pixel *cppCurrPixel = &cppFirstPixInB[iPixelsInB +k];
		const float *fpGard =  fpGradPtr + cppCurrPixel->m_iGradOfs;
		const uchar *ucpAngle = ucpAnglePtr + cppCurrPixel->m_iAngleOfs;
		//the neighbouring bins(in fact the histograms)
		int iHistIdx1 = ucpAngle[0];
		int iHistIdx2 = ucpAngle[1];
		float *fCurrHistInB = fHistBlock + cppCurrPixel->m_aVoteCellOfs[0];
		float fWeight = cppCurrPixel->m_iGausWeight * cppCurrPixel->m_aVoteCellWeight[0];
		fCurrHistInB[iHistIdx1] += fpGard[0] *  fWeight;
		fCurrHistInB[iHistIdx2] += fpGard[1] * fWeight;

		fCurrHistInB = fHistBlock + cppCurrPixel->m_aVoteCellOfs[1];
		fWeight = cppCurrPixel->m_iGausWeight * cppCurrPixel->m_aVoteCellWeight[1];
		fCurrHistInB[iHistIdx1] += fpGard[0] *  fWeight;
		fCurrHistInB[iHistIdx2] += fpGard[1] * fWeight;

	}
	//situation 3;
	for(k = 0; k < m_iSituation3; k++)
	{
		const PD_Pixel *cppCurrPixel = &cppFirstPixInB[iPixelsInB1 + k];
		const float *fpGard =  fpGradPtr + cppCurrPixel->m_iGradOfs;
		const uchar *ucpAngle = ucpAnglePtr + cppCurrPixel->m_iAngleOfs;
		//the neighbouring bins(in fact the histograms)
		int iHistIdx1 = ucpAngle[0];
		int iHistIdx2 = ucpAngle[1];
		float *fCurrHistInB = fHistBlock + cppCurrPixel->m_aVoteCellOfs[0];
		float fWeight = cppCurrPixel->m_iGausWeight * cppCurrPixel->m_aVoteCellWeight[0];
		fCurrHistInB[iHistIdx1] += fpGard[0] *  fWeight;
		fCurrHistInB[iHistIdx2] += fpGard[1] * fWeight;

		fCurrHistInB = fHistBlock + cppCurrPixel->m_aVoteCellOfs[1];
		fWeight = cppCurrPixel->m_iGausWeight * cppCurrPixel->m_aVoteCellWeight[1];
		fCurrHistInB[iHistIdx1] += fpGard[0] *  fWeight;
		fCurrHistInB[iHistIdx2] += fpGard[1] * fWeight;

		fCurrHistInB = fHistBlock + cppCurrPixel->m_aVoteCellOfs[2];
		fWeight = cppCurrPixel->m_iGausWeight * cppCurrPixel->m_aVoteCellWeight[2];
		fCurrHistInB[iHistIdx1] += fpGard[0] *  fWeight;
		fCurrHistInB[iHistIdx2] += fpGard[1] * fWeight;

		fCurrHistInB = fHistBlock + cppCurrPixel->m_aVoteCellOfs[3];
		fWeight = cppCurrPixel->m_iGausWeight * cppCurrPixel->m_aVoteCellWeight[3];
		fCurrHistInB[iHistIdx1] += fpGard[0] *  fWeight;
		fCurrHistInB[iHistIdx2] += fpGard[1] * fWeight;

	}

	return fHistBlock;
}

//Compute the position of a window due to its iIndex 
Point PD_BlockCache ::  getWindow(Size sImagsize, Size sWinStride, int iIndex) const
{
	//compute how many windows when window scans from the most left to the most right along the img
	int iWindowsX = (sImagsize.width - m_sWinSize.width)/sWinStride.width + 1;
	//rows means height
	int j = iIndex/iWindowsX;
	int i = iIndex - j * iWindowsX;
	//rect(x,y,z1,z2); (x,y)this the coordinate of the upleft of the retangle, and z1,z2 is its with and height 
	//return Rect(i * m_sWinSize.width, j * m_sWinSize.height, m_sWinSize.width, m_sWinSize.height );
	return Point(i * sWinStride.width, j * sWinStride.height);
}
