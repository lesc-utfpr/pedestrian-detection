#ifndef OCV_RGB_IMAGE_TEMPLATE
#define OCV_RGB_IMAGE_TEMPLATE

// template class to facilitate the access to RGB pixels
// source: http://www.cs.iit.edu/~agam/cs512/lect-notes/opencv-intro/
template<class T> class Image
{
	private:
		IplImage* imgp;
 	public:
		Image(IplImage* img=0) {imgp=img;}
		~Image(){imgp=0;}
		void operator=(IplImage* img) {imgp=img;}
		inline T* operator[](const int rowIndx) {
			return ((T *)(imgp->imageData + rowIndx*imgp->widthStep));
		}
};

typedef struct{
  unsigned char b,g,r;
} RgbPixel;

typedef struct{
  float b,g,r;
} RgbPixelFloat;

typedef Image<RgbPixel>		RgbImage;
typedef Image<RgbPixelFloat>	RgbImageFloat;
typedef Image<unsigned char>	BwImage;
typedef Image<float>			BwImageFloat;

/*
*** EXAMPLES ***
**For a single-channel byte image:
IplImage* img=cvCreateImage(cvSize(640,480),IPL_DEPTH_8U,1);
BwImage imgA(img);
imgA[i][j] = 111;

** For a multi-channel byte image:
IplImage* img=cvCreateImage(cvSize(640,480),IPL_DEPTH_8U,3);
RgbImage  imgA(img);
imgA[i][j].b = 111;
imgA[i][j].g = 111;
imgA[i][j].r = 111;

** For a multi-channel float image:
IplImage* img=cvCreateImage(cvSize(640,480),IPL_DEPTH_32F,3);
RgbImageFloat imgA(img);
imgA[i][j].b = 111;
imgA[i][j].g = 111;
imgA[i][j].r = 111;
*/

#endif
