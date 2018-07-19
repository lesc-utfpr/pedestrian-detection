#include <iostream>
//#include <cv.h>
//#include <highgui.h>

#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sys/time.h>
#include <math.h>

#define WID 220
#define HEI 220

#define MOVP 45
#define MOVT 30


using namespace cv;
using namespace std;

//#define DEBUG
//#define DEBUG2

String str_Cas;
CascadeClassifier haarCas, lbpCas;

int main(int argc, char** argv)
{
	// Definindo arquivos
	string file = argv[1];
	ifstream readFile;
	ofstream writeFile;
	string line;
	readFile.open(argv[1]);
	
	str_Cas = argv[2];	
	int option = atoi(argv[3]);
	
	// Define arquivo de saída
	string file2;
	if(option == 0)
		file2 = string(file, 0, file.length()-4)+"_resultado_thermal_cascade_haar.csv";
	else
		file2 = string(file, 0, file.length()-4)+"_resultado_thermal_cascade_lbp.csv";
		
	writeFile.open(file2.c_str());
	
	// Verifica se é possível carregar os arquivos
	if(option == 0)
	{
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
		if( !lbpCas.load( str_Cas ) )
		{
			cout << "Erro ao carregar " << str_Cas << endl ;
				return -1;
		}
		else
			cout << str_Cas << " carregado!" << endl;
	}

	double soma = 0;
	int countObjs = 0;
	vector<double> tempoObjs;
	writeFile << "Imagem;tempo saliency;media objs;desvio padrao;tempo total" << endl;

	if(readFile.is_open())
	{
		while(getline(readFile,line))
		{
			istringstream ss(line);
            		string strRgb = "",
            		strFlir = "";
			
			getline(ss,strRgb,';'); //Pega caminho da imagem
            		getline(ss,strFlir,';'); //Pega rótulo
				
			Mat imgRgb = imread(strRgb,CV_LOAD_IMAGE_GRAYSCALE);
			Mat imgFlir = imread(strFlir,-1);
			Mat binFlir;
			//cvtColor(imgRgb,imgRgb,CV_BGR2GRAY);
			double t = (double)getTickCount();
			double s = (double)getTickCount();			
						
			// Faz o processamento da Flir pra achar objeto
			inRange(imgFlir,Scalar(89,89,89),Scalar(100,100,100),binFlir);

#ifdef DEBUG
			namedWindow("Teste",CV_WINDOW_AUTOSIZE);
			imshow("Teste",binFlir);
			waitKey(0);
#endif

			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;
			std::vector<Rect> objetos;    				
				
			//Canny(binFlir,binFlir,100,200,3);
			//dilate(binFlir, binFlir, Mat(), Point(-1,-1));
			findContours( binFlir, contours, hierarchy, RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

			s = (double)getTickCount() - s;			

			vector<vector<Point> > contours_poly( contours.size() );
                        vector<Rect> boundRect( contours.size() );			
				                     	 
						
			for( int i = 0; i< contours.size(); i++ )
			{				
				if(contours[i].size() > 2)
				{
					double c = (double)getTickCount();

					approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
            			
					boundRect[i] = boundingRect( Mat(contours_poly[i]) );
				
					drawContours( binFlir, contours, i, Scalar(255,255,255), CV_FILLED, 8, hierarchy, 0, Point() );
				
					boundRect[i].x = boundRect[i].x * ((1280/80));
					boundRect[i].y = boundRect[i].y * ((720/60)+2);
				
					if(boundRect[i].width < WID)
            				{
                				int diff = WID - boundRect[i].width;
	                			boundRect[i].width = WID;
	
 		               			if(boundRect[i].x + boundRect[i].width > imgRgb.size().width)
                	    				boundRect[i].x = imgRgb.size().width - WID;
                				else if (boundRect[i].x - (diff/2) < 0)
                    					boundRect[i].x = 0;
       		         			else
                	    				boundRect[i].x = boundRect[i].x - MOVP;
            				}
           	 			if(boundRect[i].height < HEI)
            				{
                				int diff = HEI - boundRect[i].height;
                				boundRect[i].height = HEI;

                				if(boundRect[i].y + boundRect[i].height > imgRgb.size().height)
                    					boundRect[i].y = imgRgb.size().height - HEI;
      	         				else if (boundRect[i].y - (diff/2) < 0)
        	            				boundRect[i].y = 0;
                				else
                    					boundRect[i].y = boundRect[i].y - MOVT;
            				}
					Mat mini = imgRgb(boundRect[i]);
					double escala = mini.rows/(mini.rows-0.5);
            		
            		if(option == 0)
						haarCas.detectMultiScale(mini, objetos ,escala, 2, 0|CV_HAAR_SCALE_IMAGE, Size(50, 100) );
					else
						lbpCas.detectMultiScale(mini, objetos ,escala, 15, 0, Size(50,100) );	
				
					for(int k=0; k<objetos.size(); k++)
					{
						rectangle(imgRgb,objetos[k].tl(),objetos[k].br(),Scalar(0,0,0),2,8,0);
					}
				
					          			
					rectangle(imgRgb,Point(boundRect[i].x,boundRect[i].y),Point(boundRect[i].x+boundRect[i].width,boundRect[i].y+boundRect[i].height),Scalar(255,255,255),2,8,0);
					

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

			writeFile << strRgb << ";" << s/((double)getTickFrequency())  << ";" << media << ";" << desvio << ";" << t/((double)getTickFrequency()) << endl;

			countObjs = 0;
			tempoObjs.clear();
			soma = 0;

			cout << "Tempo: " << t/((double)getTickFrequency()) << "ms" << endl;
#ifdef DEBUG2
			namedWindow("Teste",CV_WINDOW_AUTOSIZE);
			imshow("Teste",imgRgb);
			waitKey(0);
#endif
								
		}		
	}
	
}
