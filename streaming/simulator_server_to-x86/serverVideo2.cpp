#include "opencv2/opencv.hpp"
#include <iostream>
#include <sys/socket.h> 
#include <arpa/inet.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <unistd.h> 
#include <string.h>

using namespace cv;

void *display(void *);
void *displayFlir(void *);

int capDev = 0;

VideoCapture cap("rgb_sample.avi"); // open the default camera
VideoCapture cap2("flir_sample.avi"); // open the default camera

int localSocket, localSocketFlir,
        remoteSocket, remoteSocketFlir, 
        port = 4097,
        portFlir = 4098;                               

    struct  sockaddr_in localAddr, localAddrFlir,
                        remoteAddr, remoteAddrFlir;
    pthread_t thread_id, thread_id2;
    int addrLen;

int main(int argc, char** argv)
{   

    //--------------------------------------------------------
    //networking stuff: socket, bind, listen
    //--------------------------------------------------------
    
    
           
    addrLen = sizeof(struct sockaddr_in);

       
    if ( (argc > 1) && (strcmp(argv[1],"-h") == 0) ) {
          std::cerr << "usage: ./cv_video_srv [port] [capture device]\n" <<
                       "port           : socket port (4097 default)\n" <<
                       "capture device : (0 default)\n" << std::endl;

          exit(1);
    }

    if (argc == 2) port = atoi(argv[1]);

    localSocket = socket(AF_INET , SOCK_STREAM , 0);
    if (localSocket == -1){
         perror("socket() call failed!!");
    }
    
    localSocketFlir = socket(AF_INET , SOCK_STREAM , 0);
    if (localSocketFlir == -1){
         perror("socket() call failed!!");
    }        

    localAddr.sin_family = AF_INET;
    localAddr.sin_addr.s_addr = INADDR_ANY;
    localAddr.sin_port = htons( port );
    
    localAddrFlir.sin_family = AF_INET;
    localAddrFlir.sin_addr.s_addr = INADDR_ANY;
    localAddrFlir.sin_port = htons( portFlir );

    if( bind(localSocket,(struct sockaddr *)&localAddr , sizeof(localAddr)) < 0) {
         perror("Can't bind() socket");
         exit(1);
    }
    
    if( bind(localSocketFlir,(struct sockaddr *)&localAddrFlir , sizeof(localAddrFlir)) < 0) {
         perror("Can't bind() socket");
         exit(1);
    }
    
    //Listening
    listen(localSocket , 1);
    listen(localSocketFlir , 1);
    
    std::cout <<  "Waiting for connections...\n"
              <<  "Server Port:" << port << std::endl;

    pthread_create(&thread_id,NULL,display,NULL);
	pthread_create(&thread_id2,NULL,displayFlir,NULL);
    
    //accept connection from an incoming client
    while(1){
		
		 sleep(1);
    }
    pthread_join(thread_id,NULL);
    pthread_join(thread_id2,NULL);
    close(remoteSocket);
    close(remoteSocketFlir);

    return 0;
}

void *display(void *ptr){
    remoteSocket = accept(localSocket, (struct sockaddr *)&remoteAddr, (socklen_t*)&addrLen);
    
    //int socket = *(int *)ptr;
    //OpenCV Code
    //----------------------------------------------------------

    Mat img, imgGray;
    img = Mat::zeros(720 , 1280, CV_8UC3);   
     //make it continuous
    if (!img.isContinuous()) {
        img = img.clone();
    }

    int imgSize = img.total() * img.elemSize();
    int bytes = 0;
    int key;
    

    //make img continuos
    if ( ! img.isContinuous() ) { 
          img = img.clone();
          imgGray = img.clone();
    }
        
    std::cout << "Image Size:" << imgSize << std::endl;
	int count = 0;
    while(cap.isOpened()) {
                
		/* get a frame from camera */
		cap >> img;
	
		//do video processing here 
		//cvtColor(img, imgGray, CV_BGR2GRAY);

		//send processed image
		if ((bytes = send(remoteSocket, img.data, imgSize, 0)) < 0){
			 std::cerr << "bytes = " << bytes << std::endl;
			 break;
		}
		usleep(100000); // deixando taxa proxima de 10 fps
		count ++;
    }
    std::cout << "count: " << count << std::endl;
    //exit(1);

}



void *displayFlir(void *ptr){
	remoteSocketFlir = accept(localSocketFlir, (struct sockaddr *)&remoteAddrFlir, (socklen_t*)&addrLen);  
    
    //int socket = *(int *)ptr;
    //OpenCV Code
    //----------------------------------------------------------

    Mat img, imgGray;
    img = Mat::zeros(60 , 80, CV_8UC3);   
     //make it continuous
    if (!img.isContinuous()) {
        img = img.clone();
    }

    int imgSize = img.total() * img.elemSize();
    int bytes = 0;
    int key;
    

    //make img continuos
    if ( ! img.isContinuous() ) { 
          img = img.clone();
          imgGray = img.clone();
    }
        
    std::cout << "Image Size:" << imgSize << std::endl;
	int count = 0;
    while(cap2.isOpened()) {
                
		/* get a frame from camera */
		cap2 >> img;
	
		//do video processing here 
		//cvtColor(img, imgGray, CV_BGR2GRAY);

		//send processed image
		if ((bytes = send(remoteSocketFlir, img.data, imgSize, 0)) < 0){
			 std::cerr << "bytes = " << bytes << std::endl;
			 break;
		}
		usleep(100000); // deixando taxa proxima de 10 fps
		count ++;
    }
    std::cout << "count: " << count << std::endl;
    //exit(1);

}
