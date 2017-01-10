// File Discription
// Author: crcrpar

#include "tone_manipulation.h"

#include <cmath>
#include <algorithm>

using namespace std;

ToneManipulation::ToneManipulation(double alpha, double beta)
  : alpha_(alpha), beta_(beta) {}
ToneManipulation::ToneManipulation(cv::Mat image, double alpha, double beta)  : image_{image}, alpha_(alpha_), beta(beta) {}
ToneManipulation::~ToneManipulation() {}

void ToneManipulation::Manipulate(const cv::Mat)
