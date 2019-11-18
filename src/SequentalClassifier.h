#include "db.h"
#include "Classifier.h"

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <unordered_map>

class SequentialClassifier:public Classifier {
public:
	SequentialClassifier(cv::PCA& pca, double th = 0.7, int feat_count = 32);
	void train(MapOfFaces& totalImages);
	virtual int get_correct_class_pos(const std::vector<FaceImage*>& video, std::string correctClassName, std::vector<int>& histo, std::chrono::nanoseconds& total);

	virtual int processVideo(std::vector<std::unordered_map<std::string, double>>& videoClassDistances, std::string correctClassName);
private:
	double threshold;
	int reduced_features_count;

	cv::PCA& pca;
	std::vector<FaceImage*> new_database;
	std::vector<int> class_indices;
	int class_count;

	void recognize_frame(const float* test_features, std::vector<double>& distances, std::vector<int>& end_feature_indices, std::vector<int>& classes_to_check, std::vector<int>& histo, std::chrono::nanoseconds& total);
};