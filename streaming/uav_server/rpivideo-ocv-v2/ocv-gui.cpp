#include "ocv-gui.h"
#include "ocv-utils.h"


// global variables
IplImage* mwFLIRImage = 0;	// stores the image read from the FLIR sensor
pthread_mutex_t mutex_FLIRImage; // controls the concurrent access to FLIR data
IplImage* mwImage = 0;		// image used to show/scale FLIR image on the main window
int mwHeight = 100;			// main window heigth
int mwWidth = 100;			// main window width
int app_exit = 0;			// controls when application is terminated


void buildGUI() {
	// creating main window
	cvNamedWindow(W_MAIN, CV_WINDOW_AUTOSIZE); 
	cvMoveWindow(W_MAIN, 100, 100);
//	cvResizeWindow(W_MAIN, mwWidth, mwHeight);
}






void updateWindow() {
	if (mwImage != NULL)
		cvReleaseImage(&mwImage);

	if (mwFLIRImage != NULL) {
//		pthread_mutex_lock(&mutex_FLIRImage); // entering critical region
		mwImage = cvCreateImage(cvSize(mwWidth, mwHeight), mwFLIRImage->depth, mwFLIRImage->nChannels);
		cvResize(mwFLIRImage, mwImage, cv::INTER_CUBIC);
//		pthread_mutex_unlock(&mutex_FLIRImage); // leaving critical region
	}
	
	if (mwImage != NULL) {
		cvShowImage(W_MAIN, mwImage);
		// wait 1ms to alleviate the CPU processing
		SLEEP_1_MS;
	}
#ifdef DEBUG
	printf("New image size WxH = %d x %d\n", mwImage->width, mwImage->height);
#endif
}







void increaseSizeMainWindow() {
	mwWidth *= 2;
	mwHeight *= 2;
//	updateWindow();
}

void decreaseSizeMainWindow() {
	mwWidth /= 2;
	mwHeight /= 2;
//	updateWindow();
}






void mySetRGB(IplImage* img, int x, int y, char red, char green, char blue) {
	int height = img->height;
	int width = img->width;
	int step = img->widthStep/sizeof(uchar);
	int channels = img->nChannels;
	uchar* data = (uchar *)mwFLIRImage->imageData;

	data[y*step+x*channels] = blue;
	data[y*step+x*channels+1] = green;
	data[y*step+x*channels+2] = red;
}






void loadImage(char * file_name) {
	mwFLIRImage = cvLoadImage(file_name);
	mwWidth = mwFLIRImage->width;
	mwHeight = mwFLIRImage->height;
}






void createFLIRImage() {
	mwWidth = 80;
	mwHeight = 60;
	mwFLIRImage = cvCreateImage(cvSize(mwWidth, mwHeight), IPL_DEPTH_8U, 3);
//	loadImage("maw.png");
	for(int y=mwHeight-1; y >= 0; y--)
		for(int x=mwWidth-1; x >= 0; x--) {
//			mySetRGB(mwFLIRImage, x, y, 100, 0, 0);
			RgbImage  imgA(mwFLIRImage);
			imgA[y][x].b = 0;
			imgA[y][x].g = 111;
			imgA[y][x].r = 0;
		}
}
