#include "myopencv.h"
#include<algorithm>
void myopencv::HistEqual(cv::Mat inputImg, cv::Mat outputImg) {
	//直方图均衡化-->图像增强
	float hist[256] = { 0 };
	if (!inputImg.empty()) {
		for (int i = 0; i < inputImg.rows; i++) {
			for (int j = 0; j < inputImg.cols; j++) {
				hist[inputImg.at<uchar>(i, j)]++;
			}
		}
		int sum = inputImg.rows * inputImg.cols;
		for (int i = 0; i < 256; i++) {
			hist[i] = hist[i] / sum;
			if (i > 0) hist[i] += hist[i - 1];
		}
		for (int i = 0; i < inputImg.rows; i++) {
			for (int j = 0; j < inputImg.cols; j++) {
				outputImg.at<uchar>(i, j) = hist[inputImg.at<uchar>(i, j)] * 255;
			}
		}
	}


}
//彩色图转化为灰度图
//取均值 
void myopencv::ColorToGray(cv::Mat inputImg, cv::Mat outputImg) {
	if (inputImg.channels() != 1) {
		outputImg = cv::Mat::zeros(inputImg.size(), CV_8UC1);
		for (int i = 0; i < inputImg.rows; i++) {
			for (int j = 0; j < inputImg.cols; j++) {
				cv::Vec3b& pix = *(inputImg.ptr<cv::Vec3b>(i, j));//彩色图   
				uchar& pix2 = *(outputImg.ptr<uchar>(i, j));//灰度图 
				pix2 = (pix[0] + pix[1] + pix[2]) / 3;
			}
		}
	}
}

void myopencv::GrayToBinary(cv::Mat inputImg, cv::Mat outputImg)
{
	for (int i = 0; i < inputImg.rows; i++) {
		for (int j = 0; j < inputImg.cols; j++) {
			if (inputImg.at<uchar>(i, j) > 180)
				outputImg.at<uchar>(i, j) = 255;
			else
				outputImg.at<uchar>(i, j) = 0;
		}
	}
}

void myopencv::Median(cv::Mat inputImg, cv::Mat outputImg)
{
	if (!inputImg.empty()) {
		int Map[9];
		for (int i = 1; i < inputImg.rows - 1; i++) {
			for (int j = 1; j < inputImg.cols - 1; j++) {
				//获得邻域
				Map[0] = inputImg.at<uchar>(i - 1, j - 1);
				Map[1] = inputImg.at<uchar>(i - 1, j);
				Map[2] = inputImg.at<uchar>(i - 1, j + 1);
				Map[3] = inputImg.at<uchar>(i, j - 1);
				Map[4] = inputImg.at<uchar>(i, j);
				Map[5] = inputImg.at<uchar>(i, j + 1);
				Map[6] = inputImg.at<uchar>(i + 1, j - 1);
				Map[7] = inputImg.at<uchar>(i + 1, j);
				Map[8] = inputImg.at<uchar>(i + 1, j + 1);
				std::sort(Map, Map + 9);
				//取中值
				outputImg.at<uchar>(i, j) = Map[4];
			}

		}
	}
}

void myopencv::GrayToBinary_OSTU(cv::Mat inputImg, cv::Mat outputImg) {
	int hist[256] = { 0 };
	double P[256] = { 0 };//概率
	double PK[256] = { 0 };//概率累加和
	double MK[256] = { 0 };//灰度值的累加均值
	int T = 0;//阈值
	if (!inputImg.empty()) {
		//求直方图
		for (int i = 0; i < inputImg.rows; i++) {
			for (int j = 0; j < inputImg.cols; j++) {
				hist[inputImg.at<uchar>(i, j)]++;
			}
		}
		int sum = inputImg.rows * inputImg.cols;

		//计算概率累加和
		for (int i = 0; i < 256; i++) {
			P[i] = (double)hist[i] / (double)sum;
			if (i > 0) {
				PK[i] = P[i] + PK[i - 1];
				MK[i] = i * P[i] + MK[i - 1];
			}
			else {
				PK[i] = P[i];
				MK[i] = i * P[i];
			}

		}
		//确定阈值
		double tmp = 0;//记录类间方差过程值
		double curretTmp = 0;//记录计算过程值
		for (int i = 0; i < 256; i++) {
			curretTmp = (MK[255] * PK[i] - MK[i]) * (MK[255] * PK[i] - MK[i]) / ((1 - PK[i]) * (PK[i]));
			if (curretTmp >= tmp) {
				T = i;
				tmp = curretTmp;
			}
		}

		//std::cout << "T=" << T << std::endl;

		//分割
		for (int i = 0; i < inputImg.rows; i++) {
			for (int j = 0; j < inputImg.cols; j++) {
				if (inputImg.at<uchar>(i, j) > T) outputImg.at<uchar>(i, j) = 255;
				else  outputImg.at<uchar>(i, j) = 0;
			}
		}


	}
}

void myopencv::GetEdge(cv::Mat inputImg, cv::Mat outputImg)
{
	//拉普拉斯算法
	outputImg = 0;
	if (!inputImg.empty()) {
		for (int i = 0; i < inputImg.rows; i++) {
			for (int j = 0; j < inputImg.cols; j++) {
				uchar tmp = 0;
				int index_x = i - 1;
				int index_y = j - 1;
				for (int m = 0; m < 3; m++, index_x++) {
					for (int n = 0; n < 3; n++, index_y++) {
						if (index_x >= 0 && index_x < inputImg.rows && index_y >= 0 && index_y < inputImg.cols) {
							tmp += (inputImg.at <uchar>(index_x, index_y) * Laplace2[m][n]);
						}
					}
				}
				if (tmp <= 20) outputImg.at<uchar>(i, j) = 0;
				else outputImg.at<uchar>(i, j) = 255;

			}
		}
	}
}

void myopencv::mask(cv::Mat inputImg, cv::Mat outputImg)
{
	int width = inputImg.cols;
	int height = inputImg.rows;
	cv::Mat mask = cv::Mat::zeros(inputImg.size(), inputImg.type());
	cv::Point pts[4] = {
		cv::Point(width*0.1, height*0.95),
		cv::Point(width*0.1, height*0.3),
		cv::Point(width*0.95, height*0.3),
		cv::Point(width*0.95,height*0.95)
	};
	
	cv::fillConvexPoly(mask, pts, 4, cv::Scalar(255, 0, 0));

	// Multiply the edges image and the mask to get the output
	cv::bitwise_and(inputImg, mask, outputImg);
	
}

std::vector<MYLINE> myopencv::GetLines(cv::Mat& inputImg, cv::Mat& outputImg)
{
	
	int thetaMin = 20;
	int thetaMax = 170;
	int RMax = 1000;
	int theta;
	int R;
	int width = inputImg.cols;
	int height = inputImg.rows;
	float fRate = (float)(PI / 180);
	int max = 0;
	int AccuArrLength = (thetaMax - thetaMin + 1) * (RMax + 1);//长度
	for (theta = thetaMin; theta < thetaMax; theta++) {
		//if (theta == 70) theta = 120;
		int cosV = (int)(cos(theta * fRate) * 2048);
		int sinV = (int)(sin(theta * fRate) * 2048);
		//遍历寻找每条直线经过的点
		for (int i = 400; i < inputImg.rows - 1; i++) {
			for (int j = 200; j < inputImg.cols - 1; j++) {
				if (inputImg.at<uchar>(i, j)) {
					R = (i* cosV + j* sinV) >> 11;
					float k = (float)(-1.0 / tan(theta * fRate));
					if (abs(R) < RMax) {
						Count[theta][R]++;
						if (Count[theta][R] > max) max = Count[theta][R];
					}
				}
			}
		}

		
	}
	
	outputImg = 0;
	std::vector < MYLINE > mylines;
	MYLINE myline;
	//寻找每组直线中最优的直线：
	for (int i = thetaMin; i < thetaMax; i++) {
		for (int j = 0; j < RMax; j++) {
			if (Count[i][j] > max/3) {
				theta = i;
				R = j;
				float k = (float)(-1.0 / tan(theta * fRate));
				float b = R / sin(theta * fRate);
				myline.k = k;
				myline.b = b;
				myline.count = Count[i][j];
				bool similar = false;
				int count = 0;
				for (auto l : mylines) {
					if (fabs(l.k - myline.k) < 10&&(fabs(l.b-myline.b)/sqrt(1+l.k*myline.k))<150)
					{
						similar = true;
						if (myline.count < l.count) {
							myline = l;
						}
						
						mylines.erase(mylines.begin() + count);
						count--;
					}
					count++;
				}

				mylines.push_back(myline);



			}
		}
	}
	
	for (auto x : mylines) {
		for (int i = height*0.4; i < inputImg.rows; i++) {
			for (int j = 0; j < inputImg.cols; j++) {
				if (fabs(j - x.k * i- x.b) < 5.0) {
					outputImg.at<uchar>(i, j) = 255;
				}
			}
		}
	}
	return mylines;
}

void myopencv::ShowLines(cv::Mat& Img_orgin, cv::Mat& Img_lines)
{
	for (int i = 450; i < Img_orgin.rows; i++) {
		for (int j = 0; j < Img_orgin.cols; j++) {
			if (Img_lines.at<uchar>(i, j) > 1) {
				cv::Vec3b& pix = *(Img_orgin.ptr<cv::Vec3b>(i, j));
				pix = { 0,0,255 };

			}
		}
	}
}

void myopencv::erode(cv::Mat inputImg, cv::Mat outputImg) {
	if (!inputImg.empty()) {
		outputImg = 0;
		for (int i = 1; i < inputImg.rows - 2; i++) {
			for (int j = 1; j < inputImg.cols - 2; j++) {
				if (inputImg.at<uchar>(i, j) != 0) {
					int index_x = i - 1;
					bool flag = false;
					for (int m = 0; m < 3; m++, index_x++) {
						int index_y = j - 1;
						for (int n = 0; n < 3; n++, index_y++) {
							if (!inputImg.at<uchar>(index_x, index_y) && erode_kernel[m][n]) {
								flag = true;
								break;
							}
						}
						if (flag) break;
					}
					if (flag) outputImg.at<uchar>(i, j) = 0;
					else outputImg.at<uchar>(i, j) = 255;
				}
			}
		}
	}
}
