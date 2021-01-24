#pragma once
//头文件
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2\imgproc\types_c.h>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <iostream>
#include <vector>


//using namespace cv;
using namespace std;
using namespace dlib;
//人脸检测的类



class Face
{
public:
	
	void Deal_face(cv::Mat &image);
	void breatful(cv::Mat &src, cv::Mat &dst);
	void saturability(cv::Mat &src);
	void adjust_bright(cv::Mat &src, cv::Mat &dst);
	void outline_extraction(cv::Mat &src, cv::Mat &dst);
	void mark_special_point(cv::Mat &src);
	void get_face(cv::Mat& src, cv::Mat& dst, cv::Mat &gcl);
	void remove_background(cv::Mat& src, cv::Mat &dst);

};


void gener_recongnition(cv::Mat &src);








