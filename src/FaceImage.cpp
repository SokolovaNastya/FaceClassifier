#include "FaceImage.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <vector>
#include <algorithm>

using namespace cv;

//#define ORTHO_SERIES_HISTOS
#ifdef ORTHO_SERIES_HISTOS
#define ORTHO_SERIES_CORRECT
#endif

#ifdef USE_DNN_FEATURES
FaceImage::FaceImage(std::string fn, std::string pn, std::vector<float>& features, bool normalize) :
	fileName(fn), personName(pn), featureVector(FEATURES_COUNT)
{
	float sum = 1;
	if (normalize) {
		sum = 0;
#if DISTANCE!=EUC
		for (int i = 0; i < FEATURES_COUNT; ++i)
			sum += features[i];
#else
		for (int i = 0; i < FEATURES_COUNT; ++i)
			sum += features[i] * features[i];
		sum = sqrt(sum);
#endif
	}
	//std::cout << "sum=" << sum << std::endl;
	for (int i = 0; i < FEATURES_COUNT; ++i)
		featureVector[i] = features[i] / sum;
}
#endif


