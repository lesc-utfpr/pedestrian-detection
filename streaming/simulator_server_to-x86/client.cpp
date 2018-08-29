/**
 * OpenCV video streaming over TCP/IP
 * Client: Receives video from server and display it
 * by Steve Tuenkam
 */

#include "opencv2/opencv.hpp"
#include <sys/socket.h> 
#include <arpa/inet.h>
#include <unistd.h>

using namespace cv;


int main(int argc, char** argv)
{

    //--------------------------------------------------------
    //networking stuff: socket , connect
    //--------------------------------------------------------
    int         sokt, sokt2;
    char*       serverIP;
    int         serverPort;

    if (argc < 3) {
           std::cerr << "Usage: cv_video_cli <serverIP> <serverPort> " << std::endl;
           return -1;
    }

    serverIP   = argv[1];
    serverPort = atoi(argv[2]);

    struct  sockaddr_in serverAddr, serverAddr2;
    socklen_t           addrLen = sizeof(struct sockaddr_in);

    if ((sokt = socket(PF_INET, SOCK_STREAM, 0)) < 0) {
        std::cerr << "socket() failed" << std::endl;
    }
    
    if ((sokt2 = socket(PF_INET, SOCK_STREAM, 0)) < 0) {
        std::cerr << "socket() failed" << std::endl;
    }

    serverAddr.sin_family = PF_INET;
    serverAddr.sin_addr.s_addr = inet_addr(serverIP);
    serverAddr.sin_port = htons(serverPort);
    
    serverAddr2.sin_family = PF_INET;
    serverAddr2.sin_addr.s_addr = inet_addr(serverIP);
    serverAddr2.sin_port = htons(4098);

    if (connect(sokt, (sockaddr*)&serverAddr, addrLen) < 0) {
        std::cerr << "connect() failed!" << std::endl;
    }
    
    sleep(1);
    
    if (connect(sokt2, (sockaddr*)&serverAddr2, addrLen) < 0) {
        std::cerr << "connect() failed!" << std::endl;
    }



    //----------------------------------------------------------
    //OpenCV Code
    //----------------------------------------------------------

    Mat img;
    img = Mat::zeros(720 , 1280, CV_8UC3);    
    Mat img2 = Mat::zeros(60 , 80, CV_8UC3);    
    int imgSize = img.total() * img.elemSize();
    int imgSize2 = img2.total() * img2.elemSize();
    
    uchar *iptr = img.data;
    uchar *iptr2 = img2.data;
    int bytes = 0, bytes2 = 0;
    int key;

    //make img continuos
    if ( ! img.isContinuous() ) { 
          img = img.clone();
    }
    
    if ( ! img2.isContinuous() ) { 
          img2 = img2.clone();
    }
        
    std::cout << "Image Size:" << imgSize << std::endl;
	std::cout << "Image Size 2:" << imgSize2 << std::endl;

    namedWindow("CV Video Client",1);
	namedWindow("CV Video Client2",1);
    while (key != 'q') {
		std::cout << "recebendo..." << std::endl;
        if ((bytes = recv(sokt, iptr, imgSize , MSG_WAITALL)) == -1) {
            std::cerr << "recv failed, received bytes = " << bytes << std::endl;
            close(sokt);
			exit(1);
        }
        std::cout << "recebendo2..." << std::endl;
        //sleep(1);
        if ((bytes2 = recv(sokt2, iptr2, imgSize2 , MSG_WAITALL)) == -1) {
            std::cerr << "recv failed, received bytes = " << bytes2 << std::endl;
            close(sokt2);
			exit(1);
        }
        
        cv::imshow("CV Video Client", img);
        
        //if (key = cv::waitKey(1) >= 0) break;
        cv::imshow("CV Video Client2", img2); 
      
        if (key = cv::waitKey(1) >= 0) break;
    }   

    close(sokt);

    return 0;
} 
