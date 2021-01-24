//ͷ�ļ�
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
	cap.open(0);   //������ͷ
	if (!cap.isOpened())
	{
		return 1;
	}
	faceCascade.load("D:/OPENCV/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml");   //���ط�������ע���ļ�·��
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
			cv::imshow("��ȡ", rot);
			


			faceCascade.detectMultiScale(image, faces, 1.2, 6, 0, cv::Size(120, 120));   //�������
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
				//�������ڵĸ���Ȥ����
				cv::rectangle(image, roi, cv::Scalar(0, 255, 0), 1, 8, 0);

				//������ͼ��Ĵ�Сͳһ����Ϊ150*150
				resize(faceROI, faceROI, cv::Size(200, 200));
				/*char q = cv::waitKey(1);
				if (q == 27)
				{
					imwrite("D:/output/real2.jpg", faceROI);
					std::cout << "captured" << std::endl;
					break;
				}*/

				cv::imshow("color_with_face", image);
				cv::imshow("ԭʼͼ����ͼ", faceROI);
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









