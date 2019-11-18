#include "Classifier.h"
#include "SequentalClassifier.h"
#include "db.h"

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

Mat readFileToMat(string filename) {
	ifstream file(filename);
	if (!file)
	{
		cout << "Cannot open the file!\n";
		return Mat();
	}

	int featuresCount = 0;
	int dataSize = 0;

	file >> featuresCount >> dataSize;

	cout << "featuresCount = " << featuresCount << ", dataSize = " << dataSize << endl;

	Mat featuresMat(dataSize, featuresCount, CV_32F);
	for (int j = 0; j < featuresCount; ++j) {
		for (int i = 0; i < dataSize; ++i) {
			file >> featuresMat.at<float>(i, j);
		}
	}

	file.close();

	return featuresMat;
}

//#define INFO

int main()
{
	string pathFeatures = "\lfw\resnet\imageFeatures_corr.txt";
	
	MapOfFaces totalImages;
	loadFaces(pathFeatures, totalImages);

	PCA pca;

#ifdef INFO
	for (float i = 0.5; i < 1.0; i += 0.05) {
		trainPCA(totalImages, pca, i);
	}
#else
	double threshold_def = 0.8;
	//trainPCA(totalImages, pca, 1);

	vector<Classifier*> classifiers;
	classifiers.push_back(new SequentialClassifier(pca, threshold_def, MAX_COMPONENTS_DEF));

	for (Classifier* classifier : classifiers) {
		test_recognition_method(pca, classifier, totalImages);
	}
	for (Classifier* classifier : classifiers) {
		delete classifier;
	}
#endif
	return 0;
}