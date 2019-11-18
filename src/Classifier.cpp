#include "Classifier.h"
#include "FaceImage.h"
#include "db.h"
#include "SequentalClassifier.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <iterator>
#include <fstream>
#include <sstream>
#include <set>
#include <map>
#include <unordered_map>
#include <chrono>
#include <sstream>
#include <algorithm>
#include <numeric>

using namespace cv;
using namespace std;

int MAX_COMPONENTS_DEF;

void trainPCA(MapOfFaces& totalImages, PCA& pca, double threshold) {
	int total_images_count = 0;
	for (auto& person : totalImages) {
		total_images_count += person.second.size();
	}
	Mat mat_features(total_images_count, FEATURES_COUNT, CV_32F);
	int ind = 0;
	for (auto& person : totalImages) {
		for (const FaceImage* face : person.second) {
			for (int j = 0; j < FEATURES_COUNT; ++j) {
				mat_features.at<float>(ind, j) =
					face->getFeatures()[j];
			}
			++ind;
		}
	}
	cout << "People count = " << ind << endl;
	cout << "Before pca train: images = " << mat_features.rows << " features = " << mat_features.cols << endl;
	pca(mat_features, Mat(), PCA::DATA_AS_ROW, threshold);
	Mat db_projection_result;
	pca.project(mat_features, db_projection_result);
	cout << "After pca train: images = " << db_projection_result.rows << " features = " << db_projection_result.cols  << " threshold = " << threshold << endl;
	MAX_COMPONENTS_DEF = db_projection_result.cols;
	cout << "* End training pca\n";
}

float get_distance(const float* lhs, const float* rhs, int start, int end) {
	float d = 0;
	//chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();
	for (int fi = start; fi < end; ++fi) {
		d += (lhs[fi] - rhs[fi])*(lhs[fi] - rhs[fi]);
	}

	//chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
	//auto t = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();

	return d;
}

#define TEST_COUNT 1
//#define lfw_ytf 1
void test_recognition_method(PCA& pca, Classifier* classifier, MapOfFaces& totalImages) {
	srand(17);
	MapOfFaces faceImages, testImages;
	float total_accuracy = 0.0, total_time = 0.0;

	for (int t = 0; t < TEST_COUNT; ++t) {
		vector<int> histo;
		for (int i = 0; i < 64; i++) //4, 32, 64
			histo.push_back(0);

#ifdef lfw_ytf
		faceImages = totalImages;

		loadFaces_lfw_ytf("\Aggregation_features\feature_aggreg_feature.txt", testImages, faceImages);
#else
		getTrainingAndTestImages(totalImages, faceImages, testImages, true, 1);
#endif 
		trainPCA(faceImages, pca, 1);

		classifier->train(faceImages);

		int errorsCount = 0, totalVideos = 0, totalFrames = 0;
		unordered_map<string, int> class_errors;

		//auto t1 = chrono::high_resolution_clock::now();
		std::chrono::nanoseconds total (0);

		for (auto& iter : testImages) {
			if (iter.second.size() != 0) {
				int pos = classifier->get_correct_class_pos(iter.second, iter.first, histo, total);

				if (pos != 0) {
					std::cout << "\rorig=" << std::setw(35) << iter.first << " pos=" << std::setw(4) << pos << flush;
					++errorsCount;
					++class_errors[iter.first];
				}
				++totalVideos;
				totalFrames += iter.second.size();
			}
		}

		//auto t2 = chrono::high_resolution_clock::now();
		//float total = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
		double recall = 0;
		for (auto& iter : totalImages) {
			if (class_errors.find(iter.first) == class_errors.end())
				recall += 100;
			else
				recall += 100 - 100.0*class_errors[iter.first] / iter.second.size();
		}

		float total_time_milli = std::chrono::duration_cast<std::chrono::milliseconds>(total).count();

		cout << endl << " video error rate=" << (100.0*errorsCount / totalVideos) << " (" << errorsCount << " out of "
			<< totalVideos << ") recall=" << (recall / totalImages.size()) << " avg time=" << (total_time_milli / totalFrames) << "\n";

		for (int j = 0; j < histo.size(); ++j)
			cout << histo[j] << " ";
		cout << endl;

		total_accuracy += 100.0*(totalVideos - errorsCount) / totalVideos;
		total_time += total_time_milli / totalFrames;
	}

	cout << " average accuracy=" << total_accuracy / TEST_COUNT <<
		" total time (ms)=" << total_time / TEST_COUNT << "\n\n";
}

