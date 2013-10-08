// the resouce.h covers all the variables all are shared in all the documents
#ifndef _PD_PARAMETERS_H
#define _PD_PARAMETERS_H

enum enBlockNormStyle{L2norm, L2_Hys, L1norm, L1sqrt};
enum enGamacorrect{GamaSqrt, NoGama};
enum enErorr{PARAERORR, UNKNOWN};

#define  PI	3.1415926
#define  WINSIZE_W	64
#define  WINSIZE_H	128
#define  BLOCKSTRIDE_X 8
#define  BLOCKSTRIDE_Y 8
#define  WINSTRIDE_X 8
#define  WINSTRIDE_Y 8
#define  CELLSIZE_X 8
#define  CELLSIZE_Y 8
#define  BLOCKSIZE_X 16
#define  BLOCKSIZE_Y 16
#define  BINS 9
#define  PADDING_W 16
#define  PADDING_H 24
#define  THRESHOLD 0.2f
#define	 MAXLAYERS 50
#define	 GUASSSIGMA 4
#define  TRAINCOUNT 100
#define  MAXDETECT 20000
#define  ROIOFFSET 16
#define  RANDOMNEG 10
#define  TRAINCOUNTNEG 1
#define  MAXFILENAME 1024

#endif //_PD_PARAMETERS_H