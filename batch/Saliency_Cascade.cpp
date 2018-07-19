//#include "caffe/staticSaliencyFineGrained.h"
#include <opencv2/saliency.hpp>
#include <iostream>
#include <sys/time.h>

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <math.h>

#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define WID 180
#define HEI 180

//#define DEBUG

using namespace std;
using namespace cv;

String str_Cas;
CascadeClassifier haarCas, lbpCas;

int main(int argc, char* argv[])
{
	str_Cas = argv[2];
    int opt = atoi(argv[3]);
	
	Mat img;
	cv::Mat image;
    cv::Mat saliencyMap;
	Ptr<Saliency> saliencyAlgorithm;

    string file = argv[1];
    ifstream readFile;
    ofstream writeFile;
    string line;
    readFile.open(argv[1]);
	
	string file2 = "";
		
	if(opt == 0)
	{
		file2 = string(file, 0, file.length()-4)+"_resultado_saliency_cascade_haar.csv";
		if( !haarCas.load( str_Cas ) )
		{
			cout << "Erro ao carregar " << str_Cas << endl ;
			return -1;
		}
		else
			cout << str_Cas << " carregado!" << endl;
	}
	else
	{
		file2 = string(file, 0, file.length()-4)+"_resultado_saliency_cascade_lbp.csv";
		if( !lbpCas.load( str_Cas ) )
		{
			cout << "Erro ao carregar " << str_Cas << endl ;
			return -1;
		}
		else
			cout << str_Cas << " carregado!" << endl;
	}
    
    
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
			image = img;
			bool resulthaar, resultlbp;
			cout << "imagem: " << strImg << endl;
			//StaticSaliencyFineGrained saliencyGenerator;
			saliencyAlgorithm = StaticSaliencyFineGrained::create();

			double t = (double)getTickCount();
			double s = (double)getTickCount();
			
			if( saliencyAlgorithm->computeSaliency(	image, saliencyMap)  )
			{
				s = (double)getTickCount() - s;
				//imshow("Saliency Ori",saliencyMap);	
				threshold(saliencyMap,saliencyMap,0,255,CV_THRESH_BINARY | CV_THRESH_OTSU);
				//imshow("Pos threshold!",saliencyMap);
				erode(saliencyMap,saliencyMap,Mat(), Point(-1,-1),2);

				vector<vector<Point> > contours;
				vector<Vec4i> hierarchy;	

				Canny(saliencyMap,saliencyMap,100,200,3);
				dilate(saliencyMap, saliencyMap, Mat(), Point(-1,-1));
				//imshow("Canny",saliencyMap);
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
					//cout << i << "-Area: " << area << endl;
					if( area >=1000 && area < 5000)
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
						
						std::vector<Rect> objetos;
						double escala = image.rows/(image.rows-0.5);
						
						if(opt == 0)
							haarCas.detectMultiScale(mini, objetos ,escala, 2, 0|CV_HAAR_SCALE_IMAGE, Size(50, 100) );
						else
							lbpCas.detectMultiScale(mini, objetos ,escala, 15, 0, Size(50, 100) );
						rectangle(image,Point(boundRect[i].x,boundRect[i].y),Point(boundRect[i].x+boundRect[i].width,boundRect[i].y+boundRect[i].height),Scalar(0,0,0),2,8,0);
						if(objetos.size() >=1)
						{
								rectangle(image,Point(boundRect[i].x,boundRect[i].y),Point(boundRect[i].x+boundRect[i].width,boundRect[i].y+boundRect[i].height),Scalar(255,255,255),2,8,0);
				
						}
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

