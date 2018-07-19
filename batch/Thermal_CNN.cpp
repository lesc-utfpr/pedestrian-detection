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
	int achados = 0;


	if (argc != 6)
	{
    		std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " mean.binaryproto labels.txt img.jpg" << std::endl;
    		return 1;
  	}
	::google::InitGoogleLogging(argv[0]);

  	model_file   = argv[1];
  	trained_file = argv[2];
  	mean_file    = argv[3];
  	label_file   = argv[4];

	// Definindo arquivos
	string file = argv[5];
	ifstream readFile;
	ofstream writeFile;
	string line;
	readFile.open(file.c_str());


	// Define arquivo de saída
	string file2 = string(file, 0, file.length()-4)+"_resultado_thermal_cnn.csv";

	writeFile.open(file2.c_str());

	// Verifica se é possível carregar os arquivos
	Classifier classifier(model_file, trained_file, mean_file, label_file);

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
			cout << strRgb << endl;
			Mat imgOri = imread(strRgb /*, CV_LOAD_IMAGE_GRAYSCALE*/ , CV_LOAD_IMAGE_COLOR);
			Mat imgRgb; // = imread(strRgb /*, CV_LOAD_IMAGE_GRAYSCALE*/ , CV_LOAD_IMAGE_COLOR);
			Mat imgFlir = imread(strFlir,-1);
			Mat binFlir;
			cvtColor(imgOri,imgRgb,CV_BGR2GRAY);


			double t = (double)getTickCount();
			double s = (double)getTickCount();
			//gettimeofday(&inicio,NULL);
			// Faz o processamento da Flir pra achar objeto
			inRange(imgFlir,Scalar(THDOWN,THDOWN,THDOWN),Scalar(THUP,THUP,THUP),binFlir);

#ifdef DEBUG
			Mat temp;
			resize(binFlir, temp, temp.size(), 5, 5);
			namedWindow("Binario",CV_WINDOW_AUTOSIZE);
			imshow("Binario",temp);
			//waitKey(0);
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
				cout << i <<  " size " << contours[i].size() << endl;
				if(contours[i].size() >= 1)
				{
					double c = (double)getTickCount();
					approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );

					boundRect[i] = boundingRect( Mat(contours_poly[i]) );

					drawContours( binFlir, contours, i, Scalar(255,255,255), CV_FILLED, 8, hierarchy, 0, Point() );

					boundRect[i].x = boundRect[i].x * ((1280/80));
					boundRect[i].y = boundRect[i].y * ((720/60)+2);

					cout << boundRect[i].x << "," << boundRect[i].y << " - " << boundRect[i].width << "," << boundRect[i].height << endl;
					if(boundRect[i].width < WID)
					{
						int diff = WID - boundRect[i].width;
							boundRect[i].width = WID;
						cout << "calc " << boundRect[i].x + boundRect[i].width << " " << imgRgb.size().width << endl;
						if(boundRect[i].x + boundRect[i].width > imgRgb.size().width)
								boundRect[i].x = imgRgb.size().width - WID;
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

						if(boundRect[i].y + boundRect[i].height > imgRgb.size().height)
								boundRect[i].y = imgRgb.size().height - HEI;
						else if (boundRect[i].y - (diff/2) < 0)
								boundRect[i].y = 0;
						else
								boundRect[i].y = boundRect[i].y - MOVT;
								
						if(boundRect[i].y < 0)
							boundRect[i].y = 0;
					}
					cout << boundRect[i].x << "," << boundRect[i].y << " - " << boundRect[i].width << "," << boundRect[i].height << endl;
					//Mat mini = imgRgb(boundRect[i]);
					Mat mini = imgOri(boundRect[i]);
					//resize(mini,mini,cvSize(128,128));
					
					namedWindow("mini",CV_WINDOW_AUTOSIZE);
					imshow("mini",mini);

					vector<Prediction> predictions = classifier.Classify(mini,2);
					Prediction p = predictions[0];
					cout << "T1" << i << ": " << p.first << " - " << p.second << endl;
					rectangle(imgRgb,Point(boundRect[i].x,boundRect[i].y),Point(boundRect[i].x+boundRect[i].width,boundRect[i].y+boundRect[i].height),Scalar(0,0,0),2,8,0);
					
					char file_sample[150];
					cout << "pre-printf" << endl;
					sprintf(file_sample, "%s_thermal_%s_%d.png", file.c_str(), p.first.c_str(), name_counter);
					cout << "Arquivo: " << file_sample << endl;
					cv::imwrite(file_sample, mini);
					name_counter++;
						
					/*
					Mat mini2;
					flip(mini, mini2, 0);
					vector<Prediction> predictions2 = classifier.Classify(mini2,2);
					Prediction p2 = predictions[0];
					cout << "T2" << i << ": " << p2.first << " - " << p2.second << endl;
					*/
					//namedWindow("T1",CV_WINDOW_AUTOSIZE);
					//imshow("T1",mini2);
					
					
					if(p.first == "pos pessoa")
					{
						rectangle(imgRgb,Point(boundRect[i].x,boundRect[i].y),Point(boundRect[i].x+boundRect[i].width,boundRect[i].y+boundRect[i].height),Scalar(255,255,255),2,8,0);
						achados++;
						//cout << "Achou!" << endl;
					}

					c = (double)getTickCount() - c;
					soma = soma + (c/((double)getTickFrequency()));
					tempoObjs.push_back((c/((double)getTickFrequency())));
					countObjs++;
				}
			}
			//cout << "Achados: " << achados << endl;
#ifdef DEBUG2
			
			resize(imgRgb,imgRgb,imgRgb.size(), 0.5, 0.5, INTER_LINEAR);
			namedWindow("Teste",CV_WINDOW_AUTOSIZE);
			//cv::resizeWindow("Teste", 50, 50);
			imshow("Teste",imgRgb);
			waitKey(1);
#endif

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

#ifdef DEBUG3
				cv::imshow( "Original Image", imgRgb );
				//cv::imshow( "Saliency Map", saliencyMap );
				//cv::imshow( "Objetos", drawing);
				cv::waitKey( 1 );
#endif
		}
	}
	cout << "Achados: " << achados << endl;
}
