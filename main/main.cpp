//头文件
#include<opencv2/objdetect/objdetect.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <iostream>
#include <vector>
#include "face.h"

//using namespace cv;
using namespace std;
using namespace dlib;
cv::CascadeClassifier faceCascade;
int main(int argc, char** argv)
{
	//cv::Mat image = cv::imread("D:/output/real2.jpg");
#ifdef _DEBUG
	cv::Mat image = cv::imread("D:/output/nresult.jpg");
#endif // _DEBUG
//#ifdef _RELEASE
	cv::VideoCapture cap;
	cap.open(0);   //打开摄像头
	if (!cap.isOpened())
	{
		return 1;
	}
	faceCascade.load("D:/OPENCV/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml");   //加载分类器，注意文件路径
	std::vector<cv::Rect> faces;
	cv::Mat image;
	int c = 0;
	cap >> image;
// #endif // _RELEASE

	//image process
	while (true)
	{

		if (!image.empty())
		{
			
			cv::Mat rot;
			rot = cv::Mat(image, cv::Rect(150, 150, 320, 300));
			cv::imwrite("D:/output/rot.jpg", rot);
			cv::imshow("提取", rot);
			


			faceCascade.detectMultiScale(image, faces, 1.2, 6, 0, cv::Size(120, 120));   //检测人脸
			//cv::imshow("faces", faces);

			cv::Mat faceROI;
			if (faces.empty())
			{
				cout << "cant dectect any faces" << endl;
				cap >> image; continue;
			}
			for (int i = 0; i < faces.size(); i++)
			{
				cv::Rect roi;
				roi.x = faces[static_cast<int>(i)].x + 10;
				roi.y = faces[static_cast<int>(i)].y -40 ;
				roi.width = faces[static_cast<int>(i)].width + 10;
				roi.height = faces[static_cast<int>(i)].height +40;
				faceROI = image(roi);
				//人脸所在的感兴趣区域
				cv::rectangle(image, roi, cv::Scalar(0, 255, 0), 1, 8, 0);

				//将人脸图像的大小统一调整为150*150
				resize(faceROI, faceROI, cv::Size(200, 200));
				/*char q = cv::waitKey(1);
				if (q == 27)
				{
					imwrite("D:/output/real2.jpg", faceROI);
					std::cout << "captured" << std::endl;
					break;
				}*/

				cv::imshow("color_with_face", image);
				cv::imshow("原始图人脸图", faceROI);
				cv::Mat age;
				gener_recongnition(image);
				Face f;
				f.Deal_face(faceROI);
				//Deal_face(image);
				c = cv::waitKey();
				image.release();
			}
		}
		cap >> image;
	}




	return 0;
}









