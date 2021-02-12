#pragma once
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include<math.h>
#include<vector>
//灰度图变换
#define PI 3.1415926
struct MYLINE {
	float k;
	float b;
	int count;
};

class myopencv
{
private:
	int Count[250][1000] = { 0 };	//累加数组
	int Laplace1[3][3] = { 0, -1, 0, -1, 4, -1, 0, -1, 0 };//laplace模板,4邻域  
	int Laplace3[3][3] = { -1, 0, -1, 0, 4, 0, -1, 0, -1 };//laplace模板,4邻域  
	int Laplace2[3][3] = { -1, -1, -1, -1, 8, -1, -1, -1, -1 };//laplace模板,8邻域 
	int erode_kernel[3][3] = { 0,1,0,1,1,1,0,1,0 };//腐蚀核
public:
	//直方图均衡化-->图像增强
	void HistEqual(cv::Mat inputImg, cv::Mat outputImg);
	//彩色图到灰度图
	void ColorToGray(cv::Mat inputImg, cv::Mat outputImg);
	//中指滤波-->去噪
	void Median(cv::Mat inputImg, cv::Mat outputImg);
	//灰度图到二值图
	void GrayToBinary(cv::Mat inputImg, cv::Mat outputImg);
	void GrayToBinary_OSTU(cv::Mat inputImg, cv::Mat outputImg);
	//提取边缘
	void GetEdge(cv::Mat inputImg, cv::Mat outputImg);
	//获取ROI（图片下半部分）
	void mask(cv::Mat inputImg, cv::Mat outputImg);
	//提取主要直线（车道线） 返回结构体向量，存储直线信息
	std::vector<MYLINE> GetLines(cv::Mat& inputImg, cv::Mat& outputImg);
	//可视化车道线
	void ShowLines(cv::Mat& inputImg, cv::Mat& outputImg);
	//腐蚀操作
	void erode(cv::Mat inputImg, cv::Mat outputImg);
};

