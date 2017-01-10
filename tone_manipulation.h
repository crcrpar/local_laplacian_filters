// File Discription
// Author: crcrpar

#ifndef TONE_MANIPULATION_H
#define TONE_MANIPULATION_H

#include "remapping_function.h"

#include <opencv2/opencv.hpp>
#include <cmath>

class ToneManipulation : RemappingFunction {

public:
  void Manipulate(const cv::Vec3d& input, const cv::Vec3d& reference,
   double sigma_r, cv::Vec3d& output);

private:
  void Intensity(cv::Mat& image);
  cv::Vec3d ColorRatio(cv::Mat& image);
  cv::Mat image;
  cv::Mat intensity_image;
};

inline void Intensity(cv::Mat& image) {
  cv::Mat tmp = image.clone();
  std::vector<cv::Mat> channels;
  cv::split(tmp, channels); // channels = [B, G, R]
  double coef = {1 / 61.0, 40 / 61.0, 20 / 61.0};
  for (int i=0; i<3; i++) {
    channels[i] *= coef[i];
  }
  cv::merge(channels, intensity_image);
}

inline cv::Vec3d ColorRatio(cv::Mat& image)
