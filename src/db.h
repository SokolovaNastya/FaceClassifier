#ifndef __DB_H__
#define __DB_H__

#include "FaceImage.h"

#include <string>
#include <map>
#include <vector>

const float FRACTION = 0.5;

using MapOfFaces = std::map<std::string, std::vector<FaceImage*> >;
void loadFaces(std::string fileFeature, MapOfFaces& dbImages);
void loadFaces_lfw_ytf(std::string fileFeature, MapOfFaces& dbImages, MapOfFaces& lfwImages);
void getTrainingAndTestImages(const MapOfFaces& totalImages, MapOfFaces& faceImages, MapOfFaces& testImages, bool randomize = true, float fraction = FRACTION);

#endif //__DB_H__