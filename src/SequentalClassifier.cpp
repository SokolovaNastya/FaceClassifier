#include "SequentalClassifier.h"
#include "FaceImage.h"
#include "db.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <iterator>
#include <fstream>
#include <sstream>
#include <set>
#include <map>
#include <chrono>
#include <sstream>
#include <algorithm>
#include <numeric>

using namespace cv;
using namespace std;

SequentialClassifier::SequentialClassifier(PCA& p, double th, int feat_count) : pca(p), threshold(1.0 / th), reduced_features_count(feat_count)
{
	ostringstream os;
	os << " threshold=" << threshold << " feat_count=" << reduced_features_count;
}

void SequentialClassifier::train(MapOfFaces& totalImages) {
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

	Mat mat_projection_result = pca.project(mat_features);

	int images_no = 0;
	class_count = 0;
	new_database.resize(total_images_count);
	class_indices.resize(total_images_count);
	for (auto& person : totalImages) {
		for (FaceImage* face : person.second) {
			std::vector<float> features(FEATURES_COUNT);
			for (int j = 0; j < mat_projection_result.cols; ++j) {
				features[j] = mat_projection_result.at<float>(images_no, j);
			}

			new_database[images_no] = new FaceImage(face->fileName, face->personName, features, false);
			class_indices[images_no] = class_count;
			++images_no;
		}
		++class_count;
	}
}

int SequentialClassifier::get_correct_class_pos(const vector<FaceImage*>& video, string correctClassName, vector<int>& histo, std::chrono::nanoseconds& total)
{	
	int frames_count = video.size();
	//pca transform
	vector<vector<float>> test_features(frames_count);

	Mat queryMat(frames_count, FEATURES_COUNT, CV_32FC1);
	for (int ind = 0; ind < frames_count; ++ind)
	{
		for (int fi = 0; fi < FEATURES_COUNT; ++fi) {
			queryMat.at<float>(ind, fi) = video[ind]->getFeatures()[fi];
		}
	}
	Mat pcaMat = pca.project(queryMat);

	for (int ind = 0; ind < frames_count; ++ind)
	{
		test_features[ind].resize(MAX_COMPONENTS_DEF);
		for (int fi = 0; fi < MAX_COMPONENTS_DEF; ++fi) {
			test_features[ind][fi] = pcaMat.at<float>(ind, fi);
		}
	}

	vector<unordered_map<string, double>> videoClassDistances(video.size());
	vector<vector<double>> distances(frames_count);

	if (MAX_COMPONENTS_DEF > 0) {
		vector<vector<int>> end_feature_indices(frames_count);
		vector<int> classes_to_check(class_count), total_classes_to_check(class_count);

		for (int ind = 0; ind < frames_count; ++ind)
		{
			fill(classes_to_check.begin(), classes_to_check.end(), 1);
			recognize_frame(&test_features[ind][0], distances[ind], end_feature_indices[ind], classes_to_check, histo, total);
			for (int c = 0; c < class_count; ++c)
				total_classes_to_check[c] += classes_to_check[c];
		}
		for (int ind = 0; ind < frames_count; ++ind)
		{
			unordered_map<string, double>& frameDistances = videoClassDistances[ind];
			for (int j = 0; j < new_database.size(); ++j) {
				if (!total_classes_to_check[class_indices[j]])
					continue;

				distances[ind][j] += get_distance(&test_features[ind][0], new_database[j]->getFeatures(),
					end_feature_indices[ind][j], MAX_COMPONENTS_DEF);
				distances[ind][j] /= MAX_COMPONENTS_DEF;

				string class_name = new_database[j]->personName;
				bool classNameExists = frameDistances.find(class_name) != frameDistances.end();
				if (!classNameExists || (classNameExists && frameDistances[class_name] > distances[ind][j])) {
					frameDistances[class_name] = distances[ind][j];
				}
			}
		}
	}
	else {
		for (int ind = 0; ind < frames_count; ++ind)
		{
			unordered_map<string, double>& frameDistances = videoClassDistances[ind];
			for (int j = 0; j < new_database.size(); ++j) {

				float dist = get_distance(&test_features[ind][0], new_database[j]->getFeatures(),
					0, -MAX_COMPONENTS_DEF) / (-MAX_COMPONENTS_DEF);

				string class_name = new_database[j]->personName;
				bool classNameExists = frameDistances.find(class_name) != frameDistances.end();
				if (!classNameExists || (classNameExists && frameDistances[class_name] > dist)) {
					frameDistances[class_name] = dist;
				}
			}
		}
	}
	return processVideo(videoClassDistances, correctClassName);
}

#undef max;
void SequentialClassifier::recognize_frame(const float* test_features, vector<double>& distances, vector<int>& end_feature_indices, vector<int>& classes_to_check, vector<int>& histo, std::chrono::nanoseconds& total) {
	distances.resize(new_database.size());
	end_feature_indices.resize(new_database.size());
	vector<double> class_min_distances(class_count);
	int last_feature = FEATURES_COUNT;

	vector<int> components = { 36, 75, 128, 207, 394, last_feature }; // 2048

	int plusVar = 64;
	//last_feature = 64;

	int cur_features = 0;
	int i = 0;
	for (; cur_features < last_feature; ) {

		plusVar = components[i] - cur_features;

		double bestDist = 100000;
		fill(class_min_distances.begin(), class_min_distances.end(), numeric_limits<float>::max());
		
		for (int j = 0; j < new_database.size(); ++j) {
			if (!classes_to_check[class_indices[j]])
				continue;
			float tmp = 0.0;
			auto t1 = chrono::high_resolution_clock::now();
			tmp = get_distance(test_features, new_database[j]->getFeatures(), cur_features, cur_features + plusVar);
			auto t2 = chrono::high_resolution_clock::now();
			distances[j] += tmp;
			total += t2 - t1;
			end_feature_indices[j] = cur_features + plusVar;

			if (distances[j] < class_min_distances[class_indices[j]])
				class_min_distances[class_indices[j]] = distances[j];
			if (distances[j] < bestDist) {
				bestDist = distances[j];
			}
		}
		
		int num_of_variants = 0;
		double dist_threshold = bestDist * threshold;

		for (int c = 0; c < classes_to_check.size(); ++c) {
			if (classes_to_check[c]) {
				if (class_min_distances[c] > dist_threshold)
					classes_to_check[c] = 0;
				else
					++num_of_variants;
			}
		}

		if (num_of_variants == 1) {
			histo[i] += 1;
			break;
		}

		if (i == components.size() - 1 || cur_features == last_feature - 64)
			histo[i] += 1;

		cur_features = cur_features + plusVar;
		++i;
	}
}

int SequentialClassifier::processVideo(vector<unordered_map<string, double>>& videoClassDistances, string correctClassName) {
	int frames_count = videoClassDistances.size();
	int class_count = videoClassDistances[0].size();
	unordered_map<string, double> avg_frames, current_frame;
	vector<pair<string, double>> distanceToClass(class_count);

	avg_frames = videoClassDistances[0];
	for (int i = 1; i < frames_count; ++i) {
		for (auto person : videoClassDistances[i]) {
			avg_frames[person.first] += person.second;
		}
	}

	int res = 0;

	string bestClass = "";
	float bestDist = numeric_limits<float>::max();
	for (auto& class_dist : avg_frames) {
		if (class_dist.second < bestDist) {
			bestDist = class_dist.second;
			bestClass = class_dist.first;
		}
	}
	res = (bestClass != correctClassName);

	return res;
}