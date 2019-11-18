#ifndef FACEIMAGE_H
#define FACEIMAGE_H


#include <Windows.h>

#include <string>
#include <vector>
#include <map>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

//#define USE_EXTRA_FEATURES
#define USE_DNN_FEATURES

const int FEATURES_COUNT = 2048;


enum class ImageTransform { NONE, FLIP, NORMALIZE };

class FaceImage
{
public:

	FaceImage(std::string fileName, std::string personName, std::vector<float>& features, bool normalize = true);

	std::string personName;
	std::string fileName;

	const float* getFeatures()const {
		return &featureVector[0];
	}
	
	std::vector<float>& getFeatureVector() {
		return featureVector;
	}

	int blockSize;
	friend class FacesDatabase;

	cv::Mat pixelMat;
private:
	std::vector<float> featureVector;
};


#if defined(_M_X64)
#define fast_sqrt(x) sqrt(x)
#else
float inline __declspec (naked) __fastcall fast_sqrt(float n)
{
	_asm fld dword ptr[esp + 4]
		_asm fsqrt
	_asm ret 4
}
#endif
#endif // FACEIMAGE_H