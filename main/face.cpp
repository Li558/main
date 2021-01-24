#include "face.h"
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
#include <opencv2/opencv.hpp>
#include<iostream>
#include<opencv2/dnn.hpp>
using namespace std;
using namespace dlib;
using namespace cv::dnn;

//void breatful(cv::Mat &src, cv::Mat &dst);
//void adjust_bright(cv::Mat &src, cv::Mat &dst);
//void mark_special_point(cv::Mat &src);
//void get_face(cv::Mat &src, cv::Mat &dst, cv::Mat &gcl);
//void remove_background(cv::Mat &src, cv::Mat &dst);
//void saturability(cv::Mat &src);




//�Ա�ʶ��
void gener_recongnition(cv::Mat &src)
{
	const size_t width = 300;
	const size_t height = 300;
	cv::String model_bin = "D:/OPENCV/opencv/sources/samples/dnn/face_detector/dd/opencv_face_detector_uint8.pb";
	cv::String config_text = "D:/OPENCV/opencv/sources/samples/dnn/face_detector/opencv_face_detector.pbtxt";

	cv::String ageProto = "D:/OPENCV/opencv/sources/samples/dnn/face_detector/age/deploy_age.prototxt";
	cv::String ageModel = "D:/OPENCV/opencv/sources/samples/dnn/face_detector/age/age_net.caffemodel";

	cv::String genderProto = "D:/OPENCV/opencv/sources/samples/dnn/face_detector/age/deploy_gender.prototxt";
	cv::String genderModel = "D:/OPENCV/opencv/sources/samples/dnn/face_detector/age/gender_net .caffemodel";

	//String  ageList[] = { "(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)" };
	cv::String genderList[] = { "Male", "Female" };
	cv::namedWindow("input image", cv::WINDOW_AUTOSIZE);
	imshow("input image", src);

	Net net = readNetFromTensorflow(model_bin, config_text);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);

	Net ageNet = readNet(ageModel, ageProto);
	Net genderNet = readNet(genderModel, genderProto);

	cv::Mat blobImage = blobFromImage(src, 1.0,
		cv::Size(300, 300),
		cv::Scalar(104.0, 177.0, 123.0), false, false);

	net.setInput(blobImage, "data");
	cv::Mat detection = net.forward("detection_out");
	/*vector<double> layersTimings;
	double freq = getTickFrequency() / 1000;
	double time = net.getPerfProfile(layersTimings) / freq;
	printf("execute time : %.2f ms\n", time);*/
	int padding = 20;
	cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
	float confidence_threshold = 0.5;
	for (int i = 0; i < detectionMat.rows; i++) {
		float confidence = detectionMat.at<float>(i, 2);
		if (confidence > confidence_threshold) {
			size_t objIndex = (size_t)(detectionMat.at<float>(i, 1));
			float tl_x = detectionMat.at<float>(i, 3) * src.cols;
			float tl_y = detectionMat.at<float>(i, 4) * src.rows;
			float br_x = detectionMat.at<float>(i, 5) * src.cols;
			float br_y = detectionMat.at<float>(i, 6) * src.rows;

			cv::Rect object_box((int)tl_x, (int)tl_y, (int)(br_x - tl_x), (int)(br_y - tl_y));

			cv::Rect roi;
			roi.x = max(0, object_box.x - padding);
			roi.y = max(0, object_box.y - padding);
			roi.width = min(object_box.width + padding, src.cols - 1);
			roi.height = min(object_box.height + padding, src.rows - 1);
			cv::Mat face = src(roi);
			cv::Mat faceblob = blobFromImage(face, 1.0, cv::Size(227, 227), cv::Scalar(78.4263377603, 87.7689143744, 114.895847746), false, false);
			ageNet.setInput(faceblob);
			genderNet.setInput(faceblob);
			cv::Mat agePreds = ageNet.forward();
			cv::Mat genderPreds = genderNet.forward();

			cv::Mat probMat = agePreds.reshape(1, 1);
			cv::Point classNumber;
			double classProb;
			minMaxLoc(probMat, NULL, &classProb, NULL, &classNumber);
			int classidx = classNumber.x;
			//String age = ageList[classidx];

			probMat = genderPreds.reshape(1, 1);
			minMaxLoc(probMat, NULL, &classProb, NULL, &classNumber);
			classidx = classNumber.x;
			cv::String gender = genderList[classidx];
			cv::rectangle(src, object_box, cv::Scalar(0, 0, 255), 2, 8, 0);
			putText(src, cv::format("gender:%s", gender.c_str()), object_box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 1, 8);
			cout << gender.c_str() << endl;
		}
	}
	imshow("ssd-face-detection", src);
	

}





void Face::Deal_face(cv::Mat &image)
{
	cv::Mat pot, dst1, img, result, nresult,rrt;
	breatful(image, pot);
	saturability(pot);
	adjust_bright(pot, dst1);
	outline_extraction(dst1, rrt);
	cv::Mat img1 = dst1.clone();
	mark_special_point(dst1);
	get_face(dst1, img1, result);
	remove_background(result, nresult);
	cv::waitKey();

}

//��ͼ�����˫���˲�
void Face::breatful(cv::Mat &src, cv::Mat &dst)
{
	int value1 = 3, value2 = 1;     //ĥƤ�̶���ϸ�ڳ̶ȵ�ȷ��
	int dx = value1 * 5;    //˫���˲�����֮һ  
	double fc = value1 * 12.5; //˫���˲�����֮һ  
	int p = 50; //͸����  
	cv::Mat temp1, temp2, temp3, temp4;
	//˫���˲�  
	bilateralFilter(src, temp1, dx, fc, fc);
	temp2 = (temp1 - src + 128);
	//��˹ģ��  
	GaussianBlur(temp2, temp3, cv::Size(2 * value2 - 1, 2 * value2 - 1), 0, 0);
	temp4 = src + 2 * temp3 - 255;
	dst = (src*(100 - p) + temp4 * p) / 100;
	dst.copyTo(src);
	//cv::imshow("˫���˲����ͼ��", src1);

}

//���ڱ��Ͷ�
void Face::saturability(cv::Mat &src)
{
	int saturation = 0;
	const int max_increment = 200;
	float increment = (saturation - 80) * 1.0 / max_increment;

	for (int col = 0; col < src.cols; col++)
	{
		for (int row = 0; row < src.rows; row++)
		{
			// R,G,B �ֱ��Ӧ�������±�� 2,1,0
			uchar r = src.at<cv::Vec3b>(row, col)[2];
			uchar g = src.at<cv::Vec3b>(row, col)[1];
			uchar b = src.at<cv::Vec3b>(row, col)[0];

			float maxn = max(r, max(g, b));
			float minn = min(r, min(g, b));

			float delta, value;
			delta = (maxn - minn) / 255;
			value = (maxn + minn) / 255;

			float new_r, new_g, new_b;

			if (delta == 0)		 // ��Ϊ 0 ��������������ԭ���ص�
			{
				src.at<cv::Vec3b>(row, col)[0] = new_b;
				src.at<cv::Vec3b>(row, col)[1] = new_g;
				src.at<cv::Vec3b>(row, col)[2] = new_r;
				continue;
			}

			float light, sat, alpha;
			light = value / 2;

			if (light < 0.5)
				sat = delta / value;
			else
				sat = delta / (2 - value);

			if (increment >= 0)
			{
				if ((increment + sat) >= 1)
					alpha = sat;
				else
				{
					alpha = 1 - increment;
				}
				alpha = 1 / alpha - 1;
				new_r = r + (r - light * 255) * alpha;
				new_g = g + (g - light * 255) * alpha;
				new_b = b + (b - light * 255) * alpha;
			}
			else
			{
				alpha = increment;
				new_r = light * 255 + (r - light * 255) * (1 + alpha);
				new_g = light * 255 + (g - light * 255) * (1 + alpha);
				new_b = light * 255 + (b - light * 255) * (1 + alpha);
			}
			src.at<cv::Vec3b>(row, col)[0] = new_b;
			src.at<cv::Vec3b>(row, col)[1] = new_g;
			src.at<cv::Vec3b>(row, col)[2] = new_r;
		}
	}
	cv::imshow("���Ͷ�", src);
	cv::imwrite("D:/output/fist_photo.png", src);

}

//��ͼ��������ȵ���
void Face::adjust_bright(cv::Mat &src, cv::Mat &dst)
{
	int height = src.rows;//���src�ĸ�
	int width = src.cols;//���src�Ŀ�
	dst = cv::Mat::zeros(src.size(), src.type());  //������Ҫ������һ����ԭͼһ����С�Ŀհ�ͼƬ              
	float alpha = 1.1;//�����Աȶ�Ϊ1.5
	float beta = 40;//�������ȼ�50
	//ѭ������������ÿһ�У�ÿһ�е�Ԫ��
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			if (src.channels() == 3)//�ж��Ƿ�Ϊ3ͨ��ͼƬ
			{
				//�������õ���ԭͼ����ֵ�����ظ�����b,g,r
				float b = src.at<cv::Vec3b>(row, col)[0];//nlue
				float g = src.at<cv::Vec3b>(row, col)[1];//green
				float r = src.at<cv::Vec3b>(row, col)[2];//red
				//��ʼ�������أ��Ա���b,g,r���ı���ٷ��ص��µ�ͼƬ��
				dst.at<cv::Vec3b>(row, col)[0] = cv::saturate_cast<uchar>(b*alpha + beta);
				dst.at<cv::Vec3b>(row, col)[1] = cv::saturate_cast<uchar>(g*alpha + beta);
				dst.at<cv::Vec3b>(row, col)[2] = cv::saturate_cast<uchar>(r*alpha + beta);
			}
			else if (src.channels() == 1)//�ж��Ƿ�Ϊ��ͨ����ͼƬ
			{

				float v = src.at<uchar>(row, col);
				dst.at<uchar>(row, col) = cv::saturate_cast<uchar>(v*alpha + beta);
			}
		}
	}
	//cv::imshow("��ͼ��������ȵ���", src1);
}

//��ȡ

void Face::outline_extraction(cv::Mat &src, cv::Mat &dst)
{

	cv::Mat mask = cv::Mat::zeros(src.size(), CV_8UC1);
	cv::Rect rect(0, 10, 150, 170);
	cv::Mat bgdmodel = cv::Mat::zeros(1, 65, CV_64FC1);
	cv::Mat fgdmodel =cv:: Mat::zeros(1, 65, CV_64FC1);
	grabCut(src, mask, rect, bgdmodel, fgdmodel, 5, cv::GC_INIT_WITH_RECT);
	cv::Mat result;
	for (int row = 0; row < mask.rows; row++) {
		for (int col = 0; col < mask.cols; col++) {
			int pv = mask.at<uchar>(row, col);
			if (pv == 1 || pv == 3) {
				mask.at<uchar>(row, col) = 255;
			}
			else {
				mask.at<uchar>(row, col) = 0;
			}
		}
	}
	bitwise_and(src, src, result, mask);
	imshow("grabcut result", result);

	if (result.channels() != 4)
	{
		cv::cvtColor(result, dst, cv::COLOR_BGR2BGRA);


		for (int y = 0; y < dst.rows; ++y)
		{
			for (int x = 0; x < dst.cols; ++x)
			{
				cv::Vec4b & pixel = dst.at<cv::Vec4b>(y, x);
				if (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0)
				{
					pixel[0] = 255;
					pixel[1] = 255;
					pixel[2] = 255;
					pixel[3] = 0;
				}

			}
		}
	}
	else
	{
		dst = result.clone();
		for (int y = 0; y < dst.rows; ++y)
		{
			for (int x = 0; x < dst.cols; ++x)
			{
				cv::Vec4b & pixel = dst.at<cv::Vec4b>(y, x);
				if (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0)
				{
					pixel[0] = 255;
					pixel[1] = 255;
					pixel[2] = 255;
					pixel[3] = 0;
				}

			}
		}
	}
	cv::imshow("ȥ��ͷ��", dst);
}
//-----�������������֮��������Щ������
void Face::mark_special_point(cv::Mat &src)
{

	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor pose_model;
	deserialize("D:\\OPENCV\\dlib19.17_install\\shape_predictor_68_face_landmarks.dat") >> pose_model;

	// Grab and process frames until the main window is closed by the user.
	// Grab a frame
	cv_image<bgr_pixel> cimg(src);
	// Detect faces 
	std::vector<rectangle> faces = detector(cimg);
	// Find the pose of each face.
	std::vector<full_object_detection> shapes;
	for (unsigned long i = 0; i < faces.size(); ++i)
	{
		shapes.push_back(pose_model(cimg, faces[i]));
	}
	if (!shapes.empty())
	{

		//������ݣ�����������
		for (int i = 0; i < 16; i++)
		{
			cv::line(src, cvPoint(shapes[0].part(i).x(), shapes[0].part(i).y()), cvPoint(shapes[0].part(i + 1).x(), shapes[0].part(i + 1).y()), cv::Scalar(255, 255, 255), 1);

		}
		for (int i = 17; i < 19; i++)
		{
			//������"-20"��������Ҫ������ͼ������
			cv::line(src, cvPoint(shapes[0].part(i).x(), shapes[0].part(i).y() - 20), cvPoint(shapes[0].part(i + 1).x(), shapes[0].part(i + 1).y() - 20), cv::Scalar(255, 255, 255), 1);
		}

		cv::line(src, cvPoint(shapes[0].part(19).x(), shapes[0].part(19).y() - 20), cvPoint(shapes[0].part(20).x(), shapes[0].part(20).y() - 10 - 20), cv::Scalar(255, 255, 255), 1);
		//�˴���"-10"Ҫ�ѵ�20�� 21����"-20"�Ļ��������������죬��20������10��21������20 �Դ˱Ƚ�Բ��
		cv::line(src, cvPoint(shapes[0].part(20).x(), shapes[0].part(20).y() - 10 - 20), cvPoint(shapes[0].part(21).x(), shapes[0].part(21).y() - 20 - 20), cv::Scalar(255, 255, 255), 1);
		cv::line(src, cvPoint(shapes[0].part(21).x(), shapes[0].part(21).y() - 20 - 20), cvPoint(shapes[0].part(22).x(), shapes[0].part(22).y() - 20 - 20), cv::Scalar(255, 255, 255), 1);
		//�˴���"-10"Ҫ�ѵ�22�� 23����"-20"�Ļ��������������죬��22������20��21������10 �Դ˱Ƚ�Բ��
		cv::line(src, cvPoint(shapes[0].part(22).x(), shapes[0].part(22).y() - 20 - 20), cvPoint(shapes[0].part(23).x(), shapes[0].part(23).y() - 10 - 20), cv::Scalar(255, 255, 255), 1);
		cv::line(src, cvPoint(shapes[0].part(23).x(), shapes[0].part(23).y() - 10 - 20), cvPoint(shapes[0].part(24).x(), shapes[0].part(24).y() - 20), cv::Scalar(255, 255, 255), 1);
		for (int i = 24; i < 26; i++)
		{
			cv::line(src, cvPoint(shapes[0].part(i).x(), shapes[0].part(i).y() - 20), cvPoint(shapes[0].part(i + 1).x(), shapes[0].part(i + 1).y() - 20), cv::Scalar(255, 255, 255), 1);
		}
		cv::line(src, cvPoint(shapes[0].part(0).x(), shapes[0].part(0).y()), cvPoint(shapes[0].part(17).x(), shapes[0].part(17).y() - 20), cv::Scalar(255, 255, 255), 1);
		cv::line(src, cvPoint(shapes[0].part(16).x(), shapes[0].part(16).y()), cvPoint(shapes[0].part(26).x(), shapes[0].part(26).y() - 20), cv::Scalar(255, 255, 255), 1);

	}
	////Display it all on the screen
	//cv::imshow("Dlib������", src);


}

//�õ���������ͼ��
void Face::get_face(cv::Mat& src, cv::Mat& dst, cv::Mat &gcl)
{
	//ȥ�����������ı�������
		//1.��һ����dst1һ���ߴ�ĺ�ɫ��Ĥ��Ȼ���ڸ���Ĥ�ϰ�dst1������1-16��17-26�ĵ���沢����
		//2.������ˮ����㷨,������������Ϊ��ɫ
		//3.����ͼ����imgͼ����һ�������㣬�õ������������ڵ�ͼ�񣬵���ʱ�����Ǻ�ɫ��
		//4.����ɫ����ͨ�����ص�ĸ���ȥ��Ϊ��ɫ


		//����һ�ź�ԭͼͬ����С��ȫ��ͼ�񣬻��ϸ���һ��ͬ�����ߣ��ٰ�������ȫ��Ϊ��ɫ��
		//�ڴ���ͼ���ϻ��ߣ�
	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor pose_model;
	deserialize("D:\\OPENCV\\dlib19.17_install\\shape_predictor_68_face_landmarks.dat") >> pose_model;
	cv_image<bgr_pixel> cimg(src);
	// Detect faces 
	std::vector<rectangle> faces = detector(cimg);
	// Find the pose of each face.
	std::vector<full_object_detection> shapes;
	for (unsigned long i = 0; i < faces.size(); ++i)
	{
		shapes.push_back(pose_model(cimg, faces[i]));
	}
	cv::Mat M_mask = src.clone();
	cv::Mat black = cv::Mat::zeros(M_mask.size(), M_mask.type());

	if (!shapes.empty())
	{
		//������ݣ�����������
		for (int i = 0; i < 16; i++)
		{
			cv::line(black, cvPoint(shapes[0].part(i).x(), shapes[0].part(i).y()), cvPoint(shapes[0].part(i + 1).x(), shapes[0].part(i + 1).y()), cv::Scalar(255, 255, 255), 1);

		}
		for (int i = 17; i < 19; i++)
		{
			cv::line(black, cvPoint(shapes[0].part(i).x(), shapes[0].part(i).y() - 20), cvPoint(shapes[0].part(i + 1).x(), shapes[0].part(i + 1).y() - 20), cv::Scalar(255, 255, 255), 1);
		}

		cv::line(black, cvPoint(shapes[0].part(19).x(), shapes[0].part(19).y() - 20), cvPoint(shapes[0].part(20).x(), shapes[0].part(20).y() - 10 - 20), cv::Scalar(255, 255, 255), 1);
		cv::line(black, cvPoint(shapes[0].part(20).x(), shapes[0].part(20).y() - 10 - 20), cvPoint(shapes[0].part(21).x(), shapes[0].part(21).y() - 20 - 20), cv::Scalar(255, 255, 255), 1);
		cv::line(black, cvPoint(shapes[0].part(21).x(), shapes[0].part(21).y() - 20 - 20), cvPoint(shapes[0].part(22).x(), shapes[0].part(22).y() - 20 - 20), cv::Scalar(255, 255, 255), 1);
		cv::line(black, cvPoint(shapes[0].part(22).x(), shapes[0].part(22).y() - 20 - 20), cvPoint(shapes[0].part(23).x(), shapes[0].part(23).y() - 10 - 20), cv::Scalar(255, 255, 255), 1);
		cv::line(black, cvPoint(shapes[0].part(23).x(), shapes[0].part(23).y() - 10 - 20), cvPoint(shapes[0].part(24).x(), shapes[0].part(24).y() - 20), cv::Scalar(255, 255, 255), 1);
		for (int i = 24; i < 26; i++)
		{
			cv::line(black, cvPoint(shapes[0].part(i).x(), shapes[0].part(i).y() - 20), cvPoint(shapes[0].part(i + 1).x(), shapes[0].part(i + 1).y() - 20), cv::Scalar(255, 255, 255), 1);
		}
		cv::line(black, cvPoint(shapes[0].part(0).x(), shapes[0].part(0).y()), cvPoint(shapes[0].part(17).x(), shapes[0].part(17).y() - 20), cv::Scalar(255, 255, 255), 1);
		cv::line(black, cvPoint(shapes[0].part(16).x(), shapes[0].part(16).y()), cvPoint(shapes[0].part(26).x(), shapes[0].part(26).y() - 20), cv::Scalar(255, 255, 255), 1);

	}
	//cv::imshow("�ڰ��ͼ��black", black);

	//��ˮ����㷨

	//���������óɰ�ɫ������cvPoint�������ڵĵ㼴�ɣ�CV_RGB�ǰ�ɫ���ɣ�
	cv::floodFill(
		black,
		cvPoint(shapes[0].part(30).x(), shapes[0].part(30).y()),
		CV_RGB(255, 255, 255),
		0,
		cvScalar(20, 30, 40, 0),
		cvScalar(20, 30, 40, 0)
	);
	//cv::imshow("��ɫblack", black);
	//������
	cv::bitwise_and(dst, black, gcl);
	//cv::imshow("�������", gcl);
}

//ȥ������ͼ ������ͼ��
void Face::remove_background(cv::Mat& src, cv::Mat &dst)
{
	if (src.channels() != 4)
	{
		cv::cvtColor(src, dst, cv::COLOR_BGR2BGRA);


		for (int y = 0; y < dst.rows; ++y)
		{
			for (int x = 0; x < dst.cols; ++x)
			{
				cv::Vec4b & pixel = dst.at<cv::Vec4b>(y, x);
				if (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0)
				{
					pixel[0] = 255;
					pixel[1] = 255;
					pixel[2] = 255;
					pixel[3] = 0;
				}

			}
		}
	}
	else
	{
		dst = src.clone();
		for (int y = 0; y < dst.rows; ++y)
		{
			for (int x = 0; x < dst.cols; ++x)
			{
				cv::Vec4b & pixel = dst.at<cv::Vec4b>(y, x);
				if (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0)
				{
					pixel[0] = 255;
					pixel[1] = 255;
					pixel[2] = 255;
					pixel[3] = 0;
				}

			}
		}
	}
	cv::imshow("����ͼ", dst);
	cv::imwrite("D:/output/src6.png", dst);
	//cv::imwrite("D:/output/freal1.png", src1);
}

