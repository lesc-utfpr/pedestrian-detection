//#include "caffe/staticSaliencyFineGrained.h"

#include <iostream>
#include <sys/time.h>

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <math.h>

#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/ml/ml.hpp"
#include <opencv2/saliency.hpp>

//#define REDUCE

//#define WID 128
//#define HEI 128
#define WID 200
#define HEI 200

#define DEBUG


using namespace std;
using namespace cv;
using namespace cv::ml;
using namespace saliency;

int main(int argc, char* argv[])
{
	string strSVM = argv[2];

	// Faz loading do SVM
    Ptr<SVM> svm = StatModel::load<SVM>(strSVM);
Ptr<Saliency> saliencyAlgorithm;

	Mat img;
    Mat image;
    Mat saliencyMap;

    string file = argv[1];
    ifstream readFile;
    ofstream writeFile;
    string line;
    readFile.open(argv[1]);

	string file2 = string(file, 0, file.length()-4)+"_resultado_saliency_svm.csv";
    const char* file3 = file2.c_str();

    writeFile.open(file3);
    cout << "Processando lista de arquivos: " << file << endl;
    cout << endl << file2 << endl;

    double soma = 0;
    int countObjs = 0;
    vector<double> tempoObjs;
    writeFile << "Imagem;tempo saliency;media objs;desvio padrao;tempo total" << endl;

	if(readFile.is_open())
    {
        while(getline(readFile,line))
        {
			istringstream ss(line);
			string strImg = "",
			rotulo = "";

			getline(ss,strImg,';'); //Pega caminho da imagem

			Mat img = imread(strImg, CV_LOAD_IMAGE_GRAYSCALE);
#ifdef REDUCE
			resize(img, img, Size(), 0.5, 0.5, INTER_LINEAR);
#endif
			image = img;
			bool resulthaar, resultlbp;
			cout << "imagem: " << strImg << endl;
			//StaticSaliencyFineGrained saliencyGenerator;

			saliencyAlgorithm = StaticSaliencyFineGrained::create();

			double t = (double)getTickCount();
			double s = (double)getTickCount();
			
			//if( saliencyGenerator.computeSaliencyImpl(image, saliencyMap) )
			if( saliencyAlgorithm->computeSaliency(	image, saliencyMap) )		
			{
				s = (double)getTickCount() - s;
				//imshow("Saliency Ori",saliencyMap);
				threshold(saliencyMap,saliencyMap,0,255,CV_THRESH_BINARY | CV_THRESH_OTSU);
				//imshow("Pos threshold!",saliencyMap);
				erode(saliencyMap,saliencyMap,Mat(), Point(-1,-1),2);

				vector<vector<Point> > contours;
				vector<Vec4i> hierarchy;

				Canny(saliencyMap,saliencyMap,100,200,3);
				dilate(saliencyMap, saliencyMap, Mat(), Point(-1,-1),3);
				imshow("Canny",saliencyMap);
				findContours( saliencyMap, contours, hierarchy, RETR_EXTERNAL , CV_CHAIN_APPROX_NONE, Point(0, 0) );

				vector<vector<Point> > contours_poly( contours.size() );
				vector<Rect> boundRect( contours.size() );
				//imshow("Find",saliencyMap);
				cout << "num de objetos: " << contours.size() << endl;
				Mat drawing = Mat::zeros( saliencyMap.size(), CV_8UC1 );
				for( int i = 0; i< contours.size(); i++ )
				{
					approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
					boundRect[i] = boundingRect( Mat(contours_poly[i]) );

					double area = contourArea(contours[i]);
					cout << i << "-Area: " << area << endl;
					if( area >=1000 && area < 10000)
					{
						double c = (double)getTickCount();
						//cout << "Area: " << area << endl;
						drawContours( drawing, contours, i, Scalar(255,255,255), CV_FILLED, 8, hierarchy, 0, Point() );
						if(boundRect[i].width < WID)
						{
							int diff = WID - boundRect[i].width;
							boundRect[i].width = WID;

							if(boundRect[i].x + boundRect[i].width > drawing.size().width)
								boundRect[i].x = drawing.size().width - WID;
							else if (boundRect[i].x - (diff/2) < 0)
								boundRect[i].x = 0;
							else
								boundRect[i].x = boundRect[i].x - (diff/2);
						}
						if(boundRect[i].height < HEI)
						{
							int diff = HEI - boundRect[i].height;
							boundRect[i].height = HEI;

							if(boundRect[i].y + boundRect[i].height > drawing.size().height)
								boundRect[i].y = drawing.size().height - HEI;
							else if (boundRect[i].y - (diff/2) < 0)
								boundRect[i].y = 0;
							else
								boundRect[i].y = boundRect[i].y - (diff/2);
						}
						Mat mini = image(boundRect[i]);
                        //cvtColor(mini, mini, CV_RGB2GRAY);
                        //cout << mini.cols << " x " << mini.rows << endl;
                        resize(mini,mini,cvSize(128,128));
						HOGDescriptor hog (cvSize(128,128),cvSize(16,16), cvSize(8,8),cvSize(8,8),9);
                        std::vector<float> descriptor;
                        hog.compute(mini,descriptor,cv::Size(),cv::Size());

						float response = svm->predict(descriptor);

                        rectangle(image,Point(boundRect[i].x,boundRect[i].y),Point(boundRect[i].x+boundRect[i].width,boundRect[i].y+boundRect[i].height),Scalar(0,0,0),2,8,0);

						if (response > 0)
                            rectangle(image,Point(boundRect[i].x,boundRect[i].y),Point(boundRect[i].x+boundRect[i].width,boundRect[i].y+boundRect[i].height),Scalar(255,255,255),2,8,0);

						c = (double)getTickCount() - c;
						soma = soma + (c/((double)getTickFrequency()));
						tempoObjs.push_back((c/((double)getTickFrequency())));
						countObjs++;
					}


				}
				t=(double)getTickCount() - t;
				double media = soma / countObjs;
				double desvio = 0;

				//Calcula desvio padrao
				for(int l=0; l<tempoObjs.size(); l++)
					desvio = pow((tempoObjs[l]-media),2) + desvio;
				desvio = sqrt(desvio/countObjs);

				writeFile << strImg << ";" << s/((double)getTickFrequency())  << ";" << media << ";" << desvio << ";" << t/((double)getTickFrequency()) << endl;

				countObjs = 0;
				tempoObjs.clear();
				soma = 0;

				cout << "Tempo total: " << t/((double)getTickFrequency()) << "s" << endl;
#ifdef DEBUG
				 cv::imshow( "Original Image", image );
                                        //cv::imshow( "Saliency Map", saliencyMap );
                                        //cv::imshow( "Objetos", drawing);
                                        cv::waitKey( 0 );
#endif

			}
		}


	}
	writeFile.close();
	return 0;
}

