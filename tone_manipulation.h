// File Discription
// Author: crcrpar

#ifndef TONE_MANIPULATION_H
#define TONE_MANIPULATION_H

#include "remapping_function.h"

#include <opencv2/opencv.hpp>
#include <cmath>

class ToneManipulation : RemappingFunction {

public:
  ToneManipulation(cv::Mat image, double alpha, double beta);

public:
  void Manipulate(const cv::Vec3d& input, const cv::Vec3d& reference,
   double sigma_r, cv::Vec3d& output);

  void set_image(cv::Mat input) {
    image_ = input;
  }

private:
  cv::Mat Intensity(cv::Mat& image);
  cv::Vec3d ColorRatio(cv::Mat& image);
  cv::Mat image_;
  cv::Mat intensity_image_;
  std::vector<double> color_ratio_;
};

inline cv::Mat Intensity(cv::Mat& image) {
  cv::Mat tmp = image.clone();
  std::vector<cv::Mat> channels;
  cv::split(tmp, channels); // channels = [B, G, R]
  double coef = {1 / 61.0, 40 / 61.0, 20 / 61.0};
  for (int i=0; i<3; i++) {
    channels[i] *= coef[i];
  }
  return cv::merge(channels, intensity_image_);
}

inline cv::Vec3d ColorRatio(cv::Mat& image) {
  std::vector<cv::Mat> channels;
  cv::split(image, channels);
  cv::Mat intensity_image = Intensity(image);
  for (int i=0; i<3; i++) {
    color_ratio_[i] = channels[i] / intensity_image;
  }
}
