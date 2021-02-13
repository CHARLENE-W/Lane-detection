

# 车道线检测实验


## 一、实验目的

检测给定图片中的车道线（主要针对四条车道线），并标注。评价程序时，用测试集中的图片检测准确率。要求不能用神经网络的方法，并且在图像处理过程中不使用opencv库中的函数。

## 二、实验内容

- 测试集分析

  车道线是道路中的一部分，具有线性、不交叉、均匀分布等特点，测试集中的数据均为车前视角，故可以认为车道均在画面中间。数据均有此特性：上方为天空画面，左右为沿路景色，均为无效信息。

- 检测思路

  首先将图像转化为灰度图，通过中值滤波减弱图片畸变区域的影响。然后二值处理，其中通过OSTU方法确定阈值，再提取边缘。然后通过掩膜处理获得我们感兴趣的范围即图片下方部分区域ROI,最后在所得的ROI图像中通过霍夫变换，筛选出满足条件的直线，即可在原图中标注出检测到的车道线位置。 

- 程序评测

  通过阅读题目中的论文，我们知道一般通过检测真值点存在检测出的直线上的概率描述模型准确度。在此，我通过计算真值点在一定误差范围内存在于检测出的直线上的概率来描述。

## 三、实验步骤

#### 1. 彩色图转化为灰度图

由于待检测的车道线为黄、白两色，所以就采用计算红绿蓝三通道均值的方法获取灰度值，该函数将Mat格式的三通道彩色inputImg转换为一通道灰度图outputImg：

```cpp
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
```

转化效果示例如下：

![image-20210212222245541](C:\Users\万ql\AppData\Roaming\Typora\typora-user-images\image-20210212222245541.png)

#### 2.中值去噪

去噪原理为：g（x,y）=med{f(x-k,y-l),(k,l∈W)} ，其中W为去噪模板，在本程序中采用3*3模板，中值滤波的优点是可以很好的过滤掉椒盐噪声，在一定程度上可以过去路面上的颗粒噪声。

```cpp
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
```


#### 3.二值化（OSTU法）

显然将灰度图转化为二值图能够更高效的进行后续处理，在确定阈值时采用OSTU算法，这种算法假设一副图像由前景色和背景色组成，通过统计学的方法来选取一个阈值，使得这个阈值可以将前景色和背景色尽可能的分开。阈值判优的依据是最大类间方差，

类间方差定义为
$$
ICV=P_A∗(M_A−M)^2+P_B∗(M_B−M)^2
$$
其中，灰度均值为M，任选灰度值t,则可以将所有点分为灰度值大于和小于等于t的两部分，各自平均值为MA、MB，PA、Pb则为A、B部分像素数所占比例，最佳的阈值tt 就是使得 ICVICV 最大的那个值。

```cpp
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
```

计算过程中，采用了ICV的变形公式
$$
\sigma^2=(M_G*P_t-M_t)^2/((1-P_t)*P_t)
$$
 来计算,MG为全局累加均值，P为概率值。


#### 4.腐蚀处理

为了消除二值图像中游离点的的影响，以及细化车道线，所以进行一次3*3模板的腐蚀。再次腐蚀核为erode_kernel[3] [3]= { 0,1,0,1,1,1,0,1,0 }

```cpp
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
```


#### 5.边缘提取及ROI获取

采用高斯-拉普拉斯算子进行边缘检测，即用算子模板与原图求卷积，检查每个像素的领域。

```cpp
//卷积部分的操作，其中Laplace2[3][3] = { -1, -1, -1, -1, 8, -1, -1, -1, -1 };
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
```

根据数据集中数据的大概分布，设定图像兴趣范围如下：

```cpp
cv::Point pts[4] = {
		cv::Point(width*0.1, height*0.95),
		cv::Point(width*0.1, height*0.3),
		cv::Point(width*0.95, height*0.3),
		cv::Point(width*0.95,height*0.95)
	};
```

在边缘提取时，由于路面并不平坦，所以存在较多的噪点，这也导致了提取的边缘中有较多无效的曲线信息


#### 6.基于霍夫变换的直线检测

步骤如下:

1）针对每个像素点（x，y），使得theta从20度到170度，使用公式p = xcos(theta) + ysin(theta) 计算得到共150组（p，theta）代表着霍夫空间的270条直线。将这270组值存储到H中。

如果一组点共线，则这组点中的每个值，都会使得H（p，theta）加1。

(2) 因此找到H（p，theta）值最大的直线，就是共线的点最多的直线，H（p，theta）值次大的，是共线点次多的直线。可以根据一定的阈值，将比较明显的线全部找出来。

```cpp
//遍历寻找每条直线经过的点
		for (int i = 400; i < inputImg.rows - 1; i++) {
			for (int j = 200; j < inputImg.cols - 1; j++) {
				if (inputImg.at<uchar>(i, j)) {
					R = (j * cosV + i * sinV) >> 11;
					float k = (float)(-1.0 / tan(theta * fRate));
					if (abs(R) < RMax) {
						Count[theta][R]++;
					}
				}
			}
		}
```

通过遍历，所有直线信息就存储在Count数组中，通过一定阈值，就可以筛选出多条直线。再从每组相似直线中筛选出经过点最多即最优的直线，就是我们需要的几条直线了。

## 四、程序评测
**1.详细输出**
通过groundtruth.json中的数据进行检测,计算出每张图片有效的真值点（count）占总的点数(total count)的比例，最后计算100张图片的均值，结果如下：

命中率在30%左右，效果并不算理想；另外可以看出命中率（acc）分布很不均匀，部分图片可达到70%~80%，部分图片命中率在10%左右
**2.总评**
测试程序搬运链接：[https://](https://github.com/TuSimple/tusimple-benchmark)[github.com/TuSimple/tusimple-benchmark](https://github.com/TuSimple/tusimple-benchmark)
测试过程:在lane_demo.ipynb中把json文件改成自己的文件就好了,json_pred保存预测值，json_gt保留真实值
```py
json_pred = [json.loads(line) for line in open('E://pred.json').readlines()]
json_gt = [json.loads(line) for line in open('E://gt.json')]
```
**注意**：
1. 计算过程中，会以gt中的数据为索引，在pred中定位，所以要注意数据范围的一致性（一个偷懒的方式：在输出pred.json时，重新输出一个真值文件，数据与预测值数据对应）
2. LaneEval.bench(pred_lanes, gt_lanes, y_samples, run_time)中最后有一个runtime参数，根据评测程序中的readme文件可知，该参数不影响具体计算，只是有个判定（跑的太慢了，准确率视作0...）,所以可以直接在评测程序中将该参数写为常值
	
  ## 五、附录

基本操作接口如下,详细定义在源码文件夹中。

```cpp
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
```



  

