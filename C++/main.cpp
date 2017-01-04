#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
using namespace std;
using namespace cv;

void showErika();
void seeVideo();

int main(){
	// showErika();
	seeVideo();
	destroyAllWindows();
	return 0;
}

void showErika(){
	Mat img = imread("erika.jpg");
	if(!img.data){
		std::cout << "No Image Found" << std::endl;
	}
	namedWindow("Erika", WINDOW_AUTOSIZE);
	imshow("Erika", img);
	waitKey(0);
}

void seeVideo(){
	VideoCapture stream(0);
	for(int x = 0; x < 120; x++){
		stringstream ss;
		ss << x;
		string str = ss.str();
		Mat frame;
		stream.read(frame);
		putText(frame, str, cvPoint(0,480), 1, 5, 5);
		imshow("output", frame);
		waitKey(1);
	}
}
void trackMeBby(){
	Rect2d roi;
	Mat frame;
	Ptr<Tracker> tracker = Tracker::create("KCF");
	VideoCapture stream(0);
	cap >> frame;
	roi=selectROI("tracker", frame);
	if(roi.width==0 || roi.height==0){
		return 0;
	}
	for(;;){
		cap >> frame;
		if(frame.rows==0 || frame.cols==0){
			break;
		}
		tracker->update(frame,roi);
		rectangle(frame, roi, Scalar(255, 0, 0), 2, 1);
		imshow("tracker", frame);
	}
	if(waitKey(1)==27)break;
}
}
