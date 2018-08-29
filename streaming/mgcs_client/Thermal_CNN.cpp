#include <iostream>


#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sys/time.h>

#include <caffe/caffe.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <math.h>

#include <sys/socket.h> 
#include <arpa/inet.h>
#include <unistd.h>

//#define BATCH

#define WID 180
#define HEI 180
#define MOVP 150
//#define MOVP 45
#define MOVT 0

#define THUP 200
#define THDOWN 160

//#define DEBUG3
#define DEBUG

using namespace cv;
using namespace std;
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

#define DEBUG2

typedef std::pair<string, float> Prediction;
string model_file;
string trained_file;
string mean_file;
string label_file;

class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file,
             const string& mean_file,
             const string& label_file);

  std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

 private:
  void SetMean(const string& mean_file);

  std::vector<float> Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
};

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& label_file) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file);

  /* Load labels. */
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));

  Blob<float>* output_layer = net_->output_blobs()[0];
  CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
  std::vector<float> output = Predict(img);

  N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }

  return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->ForwardPrefilled();

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, CV_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, CV_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, CV_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, CV_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

int name_counter = 0;

int main(int argc, char** argv)
{
	string model_file;
	string trained_file;
	string mean_file;
	string label_file;

	if (argc != 8)
	{
    		std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " mean.binaryproto labels.txt ip port_rgb port_flir" << std::endl;
    		return 1;
  	}
  	
	::google::InitGoogleLogging(argv[0]);

  	model_file   = argv[1];
  	trained_file = argv[2];
  	mean_file    = argv[3];
  	label_file   = argv[4];

	int         soktRgb, soktFlir;
    char*       serverIP;
    int         serverPortRgb, serverPortFlir;
	
	serverIP   = argv[5];
    serverPortRgb = atoi(argv[6]);
    serverPortFlir = atoi(argv[7]);


	// Verifica se é possível carregar os arquivos
	Classifier classifier(model_file, trained_file, mean_file, label_file);


	// Cria socket
	struct  sockaddr_in serverAddrRgb, serverAddrFlir;
    socklen_t           addrLen = sizeof(struct sockaddr_in);

    if ((soktRgb = socket(PF_INET, SOCK_STREAM, 0)) < 0) {
        cerr << "socket() rgb falhou" << endl;
        return 0;
    }
    
    if ((soktFlir = socket(PF_INET, SOCK_STREAM, 0)) < 0) {
        cerr << "socket() flir falhou" << endl;
        return 0;
    }
    
    serverAddrRgb.sin_family = PF_INET;
    serverAddrRgb.sin_addr.s_addr = inet_addr(serverIP);
    serverAddrRgb.sin_port = htons(serverPortRgb);

    if (connect(soktRgb, (sockaddr*)&serverAddrRgb, addrLen) < 0) {
        cerr << "connect() rgb falhou!" << endl;
        return 0;
    }
    
    serverAddrFlir.sin_family = PF_INET;
    serverAddrFlir.sin_addr.s_addr = inet_addr(serverIP);
    serverAddrFlir.sin_port = htons(serverPortFlir);

    if (connect(soktFlir, (sockaddr*)&serverAddrFlir, addrLen) < 0) {
        cerr << "connect() flir falhou!" << endl;
        return 0;
    }
    
    
    cv::Mat imageRgb = Mat::zeros(720 , 1280, CV_8UC3); 		// Cria uma imagem RGB zerada de 1280x720x3
    int imgSizeRgb = imageRgb.total() * imageRgb.elemSize(); // Tamanho da imagem que vai ser lido
	uchar *iptrRgb = imageRgb.data;
    int bytesRgb = 0;
    
    cv::Mat imageFlir = Mat::zeros(60 , 80, CV_8UC3); 		// Cria uma imagem Flir zerada
    int imgSizeFlir = imageFlir.total() * imageFlir.elemSize(); // Tamanho da imagem que vai ser lido
	uchar *iptrFlir = imageFlir.data;
    int bytesFlir = 0;
    
    
    int key;
    
    // Cria uma imagem continua
    if ( ! imageRgb.isContinuous() ) 
    { 
          imageRgb = imageRgb.clone();
    }
    
    if ( ! imageFlir.isContinuous() ) 
    { 
          imageFlir = imageFlir.clone();
    }
    
    namedWindow("Thermal CNN",1); // Cria janela de exibicao
        
    while (key != 'q') // Enquanto não apertar Q processa... 
    {
		// Recebe imagem RGB
        if ((bytesRgb = recv(soktRgb, iptrRgb, imgSizeRgb , MSG_WAITALL)) == -1)
        {
            std::cerr << "recv failed, received bytes = " << bytesRgb << std::endl;
        }
        
        // Recebe imagem Flir
        if ((bytesFlir = recv(soktFlir, iptrFlir, imgSizeFlir, MSG_WAITALL)) == -1)
        {
            std::cerr << "recv failed, received bytes = " << bytesRgb << std::endl;
        }
        
        if(bytesRgb > 0 && bytesFlir > 0)
        {
			// Faz o processamento da Flir pra achar objeto
			Mat binFlir;
			inRange(imageFlir,Scalar(THDOWN,THDOWN,THDOWN),Scalar(THUP,THUP,THUP),binFlir);
			
			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;
			std::vector<Rect> objetos;

			findContours( binFlir, contours, hierarchy, RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

			vector<vector<Point> > contours_poly( contours.size() );
                        vector<Rect> boundRect( contours.size() );

			// Verifica todos os objetos
			for( int i = 0; i< contours.size(); i++ )
			{
				if(contours[i].size() >= 1)
				{
					approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
					boundRect[i] = boundingRect( Mat(contours_poly[i]) );

					// Calcula posição dos objetos na img rgb
					boundRect[i].x = boundRect[i].x * ((1280/80));
					boundRect[i].y = boundRect[i].y * ((720/60)+2);
					
					// Verifica dimensões
					if(boundRect[i].width < WID)
					{
						int diff = WID - boundRect[i].width;
							boundRect[i].width = WID;
						if(boundRect[i].x + boundRect[i].width > imageRgb.size().width)
								boundRect[i].x = imageRgb.size().width - WID;
						else if (boundRect[i].x - (diff/2) < 0)
								boundRect[i].x = 0;
						else
								boundRect[i].x = boundRect[i].x - MOVP;
								
						if(boundRect[i].x < 0)
							boundRect[i].x = 0;
					}
					if(boundRect[i].height < HEI)
					{
						int diff = HEI - boundRect[i].height;
						boundRect[i].height = HEI;

						if(boundRect[i].y + boundRect[i].height > imageRgb.size().height)
								boundRect[i].y = imageRgb.size().height - HEI;
						else if (boundRect[i].y - (diff/2) < 0)
								boundRect[i].y = 0;
						else
								boundRect[i].y = boundRect[i].y - MOVT;
								
						if(boundRect[i].y < 0)
							boundRect[i].y = 0;
					}
					
					Mat mini = imageRgb(boundRect[i]);
					
					// Classifica ROI
					vector<Prediction> predictions = classifier.Classify(mini,2);
					Prediction p = predictions[0];
					
					if(p.first == "pos pessoa")
					{
						rectangle(imageRgb,Point(boundRect[i].x,boundRect[i].y),Point(boundRect[i].x+boundRect[i].width,boundRect[i].y+boundRect[i].height),Scalar(255,255,255),2,8,0);
					}
					
				}
			}
			//
			cv::imshow("Thermal CNN", imageRgb); 
      
			if ( (key = cv::waitKey(10)) >= 0) break;
			
		}
		
	}
	close(soktRgb);
	close(soktFlir);
	
	return 0;
}
