#pragma once
#include "db.h"
#include "FaceImage.h"

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

void trainPCA(MapOfFaces& totalImages, cv::PCA& pca, double threshold);
float get_distance(const float* lhs, const float* rhs, int start, int end);
extern int MAX_COMPONENTS_DEF;

class Classifier {
public:
	Classifier() :pDbImages(nullptr) {}
	virtual ~Classifier() {}

	virtual void train(MapOfFaces& pDb) { pDbImages = &pDb; }
	virtual int get_correct_class_pos(const std::vector<FaceImage*>& video, std::string correctClassName, std::vector<int>& histo, std::chrono::nanoseconds& total) = 0;
	std::string get_name() { return name; }
private:
	std::string name;

protected:
	void set_name(std::string n) { name = n; }
	MapOfFaces* pDbImages;
};

void test_recognition_method(cv::PCA& pca, Classifier* classifier, MapOfFaces& totalImages);

