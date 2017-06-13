 #include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
 #include <math.h>
#define PI 3.14
#define BIN_SIZE 20
#define BIN_NVM 9
#define NORM_WIDTH 130
#define NORM_HEIGHT 82
#define CELL_SIZE 8
#define BLOCK_SIZE 2
#define PIC_CELL_WH 50
#define CELL_W_NVM  ((NORM_WIDTH-2) / CELL_SIZE)
#define CELL_H_NVM  ((NORM_HEIGHT-2) / CELL_SIZE)
#define BLOCK_W_NVM  (CELL_W_NVM - BLOCK_SIZE + 1)
#define BLOCK_H_NVM  (CELL_H_NVM - BLOCK_SIZE + 1)
#define CELL_NVM (CELL_W_NVM * CELL_H_NVM)
#define BLOCK_NVM (BLOCK_W_NVM * BLOCK_H_NVM)
#define ARRAY_ALL (BLOCK_W_NVM * BLOCK_H_NVM * BLOCK_SIZE * BLOCK_SIZE * BIN_NVM)


void  func(int i_x, int i_y, int i_w, IplImage* Img_in, float* fbin)
{
	memset(fbin, 0, 9*sizeof(float));
	float f_x = 0.0f, f_y = 0.0f, f_Nvm = 0.0f, f_theta = 0.0f;
	for (int ii = i_y; ii < i_y + i_w; ii++)
	{
		for (int jj = i_x; jj < i_x + i_w; jj++)
		{
			uchar* pData = (uchar*)(Img_in->imageData + ii * Img_in->widthStep + jj);
			f_x = pData[1] - pData[-1];
			f_y = pData[Img_in->widthStep]- pData[-Img_in->widthStep];
			f_Nvm = pow( f_x*f_x + f_y*f_y,  0.5f); 

			float fAngle = 90.0f;
			if (f_x == 0.0f)
			{
				if (f_y > 0)
				{
					fAngle = 90.0f;
				}
			}
			else if (f_y == 0.0f)
			{
				if (f_x > 0)
				{
					fAngle == 0.0f;
				}
				else if (f_x < 0)
				{
					fAngle == 180.0f;
				}
			}
			else
			{
				f_theta = atan(f_y/f_x); //// atan() 范围为 -Pi/2 到 pi/2 所有9个bin范围是 0~180°
				fAngle = (BIN_SIZE*BIN_NVM * f_theta)/PI;
			}

			if (fAngle < 0)
			{
				fAngle += 180;
			}

			int iWhichBin = fAngle/BIN_SIZE;
			fbin[iWhichBin] += f_Nvm;
		}
	}
}
 
 void main()
 {
 	IplImage* img = cvLoadImage("./003.jpg");
 	IplImage *img1 = cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
 	IplImage *img2 = cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
 	CvMat* mat = cvCreateMat(img->width, img->height,CV_32FC1);

	//// 灰度图
 	cvCvtColor(img,img1,CV_BGR2GRAY); 
 	cvNamedWindow("GrayImage",CV_WINDOW_AUTOSIZE);
 	cvShowImage("GrayImage",img1); 

	//// gamma校正
 	uchar* uData  = (uchar*)(img1->imageData);
 	float* fMat = mat->data.fl;
 
 	for (int ii = 0; ii < img1->imageSize; ii++)
 	{
 		fMat[ii] = pow( uData[ii],  0.5f); 
 		((uchar*)img2->imageData)[ii] = (uchar)(fMat[ii]);
 	} 

	//// 缩放原有图片
	IplImage* img3 = 0;
	CvSize dst_cvsize;
	dst_cvsize.width = NORM_WIDTH;
	dst_cvsize.height = NORM_HEIGHT;
	img3 = cvCreateImage(dst_cvsize, IPL_DEPTH_8U,1 );
	cvResize(img2, img3, CV_INTER_LINEAR);

	//// 计算每个cell每个梯度的大小和方向
	int i_binNvm = 0;
	float f_bin_out[CELL_NVM][BIN_NVM];
	float i_AllbinNvm[][BLOCK_SIZE*BLOCK_SIZE*BIN_NVM] = {0.0f};
	int ii_nvm1 = 0, ii_nvm2 = 0;
	for (int ii = 1; ii + CELL_SIZE < img3->height; ii+=CELL_SIZE)
	{
		for (int jj = 1; jj + CELL_SIZE < img3->width; jj+=CELL_SIZE)
		{
			func(jj, ii, CELL_SIZE, img3, f_bin_out[i_binNvm++]);
		}
	}
	 
	//// 创建了一个img, 画每个cell的9个bin的方向大小, 注意 X正方形向右, Y正方形向下
	CvSize pic_cvsize;
	pic_cvsize.width = CELL_W_NVM * PIC_CELL_WH;
	pic_cvsize.height = CELL_H_NVM * PIC_CELL_WH;
	CvScalar cc;
	cc.val[0] = 255;
	cc.val[1] = 255;
	cc.val[2] = 255;
	IplImage *img4 = cvCreateImage(pic_cvsize, IPL_DEPTH_8U,3);
	for (int ii = 1; ii < CELL_W_NVM; ii++)
	{
		cvLine(img4, cvPoint(ii*PIC_CELL_WH, 0), cvPoint(ii*PIC_CELL_WH, img4->height) , cc);
	}
	for (int ii = 1; ii < CELL_H_NVM; ii++)
	{
		cvLine(img4, cvPoint(0,ii*PIC_CELL_WH), cvPoint(img4->width, ii*PIC_CELL_WH) , cc);
	}


	//// 此图范围是0~360°
	//// [0,40) (第一个bin)范围的画在0°方向上, [40,80) 画在40°方向上,以此类推
	CvScalar ss;
	ss.val[0] = 0;
	ss.val[1] = 0;
	ss.val[2] = 255;
	int iWhichCell = 0;
	float f_cell_max = 0.0f;
	float f_Pic[(NORM_WIDTH/CELL_SIZE)*(NORM_HEIGHT/CELL_SIZE)][BIN_NVM];
	//// 因为有些数字太大,将他们都归到画图的img上每个cell宽的一半
	//// 比如 10个cell, 每个cell 都是用50*50画9个bin的图,那么就归到0~25范围内
	for (int ii = 0; ii < CELL_H_NVM; ii++)
	{
		for (int jj = 0; jj < CELL_W_NVM; jj++)
		{
			for (int kk = 0; kk < BIN_NVM; kk++)
			{
				f_cell_max = (f_bin_out[ii*CELL_W_NVM+jj][kk] > f_cell_max) ? f_bin_out[ii*CELL_W_NVM+jj][kk] : f_cell_max;
			}

	  		for (int kk = 0; kk < BIN_NVM; kk++)
			{
				if(f_cell_max == 0.0f)
				{
					f_Pic[ii*CELL_W_NVM+jj][kk] = 0.0f;
				}
				else
				{
					f_Pic[ii*CELL_W_NVM+jj][kk] = f_bin_out[ii*CELL_W_NVM+jj][kk] * (PIC_CELL_WH/2) /  f_cell_max; 
				}  
			}

			f_cell_max = 0.0f;
		}
	}

	for (int ii = 0; ii < CELL_H_NVM; ii++)
	{
		for (int jj = 0; jj < CELL_W_NVM; jj++)
		{
			for (int kk = 0; kk < BIN_NVM; kk++)
			{
				CvPoint stStart = cvPoint( (PIC_CELL_WH/2) + jj*PIC_CELL_WH, (PIC_CELL_WH/2) + ii*PIC_CELL_WH);
				CvPoint stStop;
				float fAngle = kk*40*PI/180;
				float fSize = f_Pic[ii*CELL_W_NVM+jj][kk];
				stStop.x = fSize * cos(fAngle) + stStart.x;
				stStop.y = fSize * sin(fAngle) + stStart.y;
				cvLine(img4, stStart, stStop , ss);
			}
		}
	}

	cvNamedWindow("LineShow",CV_WINDOW_AUTOSIZE);
	cvShowImage("LineShow",img4);

	cvWaitKey(0);

	//// 归一化每个block 并输出一个表示特征的大数组
	int iBlockWhichCell = 0;
	int uu = 0;
	float  f_max = 0.0f;
	float f_Ether_Block[BLOCK_SIZE*BLOCK_SIZE][BIN_NVM];
	float f_Last_Array[ARRAY_ALL];
	for (int ii = 0; ii < BLOCK_W_NVM; ii++ )
	{
		for (int jj = 0; jj < BLOCK_H_NVM; jj++)
		{
			for (int kk = 0; kk < BIN_NVM; kk++ )
			{
				f_Ether_Block[0][kk] = f_bin_out[ii*CELL_W_NVM+jj][kk];
				f_Ether_Block[1][kk] = f_bin_out[ii*CELL_W_NVM+jj+1][kk];
				f_Ether_Block[2][kk] = f_bin_out[ii*CELL_W_NVM+jj+ CELL_W_NVM][kk];
				f_Ether_Block[3][kk] = f_bin_out[ii*CELL_W_NVM+jj+ CELL_W_NVM+1][kk];
			}

			for (int ss = 0; ss < BLOCK_SIZE * BLOCK_SIZE; ss++ )
			{
				for (int mm = 0; mm < BIN_NVM; mm++)
				{
					f_max = (f_Ether_Block[ss][mm] > f_max) ? f_Ether_Block[ss][mm] : f_max;
				}
			}

			for (int ss = 0; ss < BLOCK_SIZE * BLOCK_SIZE; ss++ )
			{
				for (int mm = 0; mm < BIN_NVM; mm++)
				{
					f_Ether_Block[ss][mm] /= f_max;
					f_Last_Array[uu++] = f_Ether_Block[ss][mm];
				}
			}

		}
	}
 
 	cvReleaseImage(&img);
 	cvReleaseImage(&img1);
	cvReleaseImage(&img2);
	cvReleaseImage(&img3);
	cvReleaseImage(&img4);
 	cvReleaseMat(&mat);
	cvDestroyWindow("GrayImage");
	cvDestroyWindow("LineShow");
 }

