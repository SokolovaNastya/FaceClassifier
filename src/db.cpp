#include "db.h"

#include <map>
#include <iostream>
#include <fstream>
#include <algorithm>
using namespace std;

#include "windows.h"

void getTrainingAndTestImages(const MapOfFaces& totalImages, MapOfFaces& faceImages, MapOfFaces& testImages, bool randomize, float fraction) {
	const int INDICES_COUNT = 400;
	int indices[INDICES_COUNT];
	for (int i = 0; i < INDICES_COUNT; ++i)
		indices[i] = i;
	if (randomize)
		std::random_shuffle(indices, indices + INDICES_COUNT);

	faceImages.clear();
	testImages.clear();

	int trainCount = 0;
	int testCount = 0;

	for (MapOfFaces::const_iterator iter = totalImages.begin(); iter != totalImages.end(); ++iter) {
		string class_name = iter->first;
		int currentFaceCount = iter->second.size();
		float size_f = currentFaceCount * fraction;
		int db_size = (int)size_f;
		if (rand() & 1)
			db_size = (int)ceil(size_f);
		if (db_size == currentFaceCount)
			db_size = currentFaceCount - 1;
		if (db_size == 0)
			db_size = 1;
		faceImages.insert(std::make_pair(class_name, std::vector<FaceImage*>()));
		if (db_size < currentFaceCount)
			testImages.insert(std::make_pair(class_name, std::vector<FaceImage*>()));

		int ind = 0;
		for (int i = 0; i < INDICES_COUNT; ++i) {
			if (indices[i] < currentFaceCount) {
				if (ind < db_size) {
					faceImages[class_name].push_back(iter->second[indices[i]]);
					trainCount++;
				}
				else {
					testImages[class_name].push_back(iter->second[indices[i]]);
					testCount++;
				}
				++ind;
			}
		}
	}
	cout << "Train count: " << trainCount << endl;
	cout << "Test count: " << testCount << endl;
}

void loadFaces(string fileFeature, MapOfFaces& dbImages) {
	ifstream fileFeatures(fileFeature);
	if (!fileFeatures)
	{
		cout << "Cannot open features file!\n";
		return;
	}

	ifstream fileLabels("\Aggregation_features\feature_aggreg_labels.txt");
	if (!fileLabels)
	{
		cout << "Cannot open labels file!\n";
		return;
	}

	int featuresCount = 0;
	int dataSize = 0;
	int total_count = 0;

	fileFeatures >> dataSize >> featuresCount;

	cout << "dataSize = " << dataSize << ", featuresCount = " << featuresCount << endl;
	//dataSize = 15000;
	for (int i = 0; i < dataSize; ++i) {
		string person_name;
		//fileFeatures >> person_name;
		fileLabels >> person_name;

		std::vector<float> features(featuresCount);
		for (int j = 0; j < featuresCount; ++j) {
			fileFeatures >> features[j];
		}

		if (dbImages.find(person_name) == dbImages.end()) {
			dbImages.insert(std::make_pair(person_name, std::vector<FaceImage*>()));
		}

		std::vector<FaceImage*>& currentDirFaces = dbImages[person_name];
		currentDirFaces.push_back(new FaceImage(person_name, person_name, features));
		++total_count;
	}

	cout << "total size=" << dbImages.size() << " totalImages=" << total_count << endl;

	fileFeatures.close();
	fileLabels.close();
}

void loadFaces_lfw_ytf(string fileFeature, MapOfFaces& dbImages, MapOfFaces& lfwImages) {
	ifstream fileFeatures(fileFeature);
	if (!fileFeatures)
	{
		cout << "Cannot open features file!\n";
		return;
	}

	ifstream fileLabels("\Aggregation_features\feature_aggreg_labels.txt");

	if (!fileLabels)
	{
		cout << "Cannot open labels file!\n";
		return;
	}

	int featuresCount = 0;
	int dataSize = 0;
	int total_count = 0;

	fileFeatures >> dataSize >> featuresCount;

	cout << "dataSize = " << dataSize << ", featuresCount = " << featuresCount << endl;
	for (int i = 0; i < dataSize; ++i) {
		string person_name;
		fileLabels >> person_name;

		std::vector<float> features(featuresCount);
		for (int j = 0; j < featuresCount; ++j) {
			fileFeatures >> features[j];
		}

		if (lfwImages.find(person_name) != lfwImages.end()) {

			if (dbImages.find(person_name) == dbImages.end()) {
				dbImages.insert(std::make_pair(person_name, std::vector<FaceImage*>()));
			}

			std::vector<FaceImage*>& currentDirFaces = dbImages[person_name];
			currentDirFaces.push_back(new FaceImage(person_name, person_name, features));
			++total_count;
		}
	}

	cout << "total size LFW-YTF=" << dbImages.size() << " totalImages=" << total_count << endl;

	fileFeatures.close();
	fileLabels.close();
}