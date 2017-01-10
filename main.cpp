// File Description
// Author: Philip Salvaggio

#include "gaussian_pyramid.h"
#include "laplacian_pyramid.h"
#include "opencv_utils.h"
#include "remapping_function.h"
#include <iostream>
#include <sstream>

#define OPENCV_VERSION(a,b,c) (((a) << 16) + ((b) << 8) + (c))

using namespace std;

// get extension from full-path
inline string GetExtension(const string &path) {
  string ext;
  size_t pos1 = path.rfind('.');
  if(pos1 != string::npos){
    ext = path.substr(pos1+1, path.size()-pos1);
    string::iterator itr = ext.begin();
    while(itr != ext.end()){
        *itr = tolower(*itr);
        itr++;
    }
    itr = ext.end()-1;
    while(itr != ext.begin()){
      if(*itr == 0 || *itr == 32) ext.erase(itr--);
      else itr--;
    }
  }
  return ext;
}

// get file name from full-path
inline string GetFileName(const string &path) {
  size_t pos1;
  pos1 = path.rfind('\\');
  if(pos1 != string::npos){
      return path.substr(pos1+1, path.size()-pos1-1);
  }
  pos1 = path.rfind('/');
  if(pos1 != string::npos){
      return path.substr(pos1+1, path.size()-pos1-1);
  }
  return path;
}

// get minimum and maximum value of image.
inline void showMinMax(cv::Mat image) {
  double _min, _max;
  cv::minMaxIdx(image, &_min, &_max);
  cout << "# min: " << _min << ", max: " << _max << endl;
}

// display data type of image
inline void showType(cv::Mat image) {
  string dtype = GetMatDataType(image);
  cout << "# data type: " << dtype << endl;
}

// calculate :alpha: and :beta:
void calcParams(cv::Mat image, double *alpha, double *beta) {
  double min_, max_;
  cv::minMaxLoc(image, &min_, &max_);
  cout << "image pixels' min value is " << min_ <<
  ", max value is " << max_ << endl;
  double tmp_diff = abs(max_ - min_);
  *alpha = tmp_diff / 255.0;
  if (*alpha < 1.0) {
    *alpha = 1.0 / *alpha;
  }
  *beta = *alpha * abs(min_);
}

void OutputBinaryImage(const std::string& filename, cv::Mat image) {
  FILE* f = fopen(filename.c_str(), "wb");
  for (int x = 0; x < image.cols; x++) {
    for (int y = 0; y < image.rows; y++) {
      double tmp = image.at<double>(y, x);
      fwrite(&tmp, sizeof(double), 1, f);
    }
  }
  fclose(f);
}

// Perform Local Laplacian filtering on the given image.
//
// Arguments:
//  input    The input image. Can be any type,
//           but will be converted to double for computation.
//  alpha    Exponent for the detail remapping function. (< 1 for detail
//           enhancement, > 1 for detail suppression)
//  beta     Slope for edge remapping function
//           (< 1 for tone mapping, > 1 for　inverse tone mapping)
//  sigma_r  Edge threshold (in image range space).
template<typename T>
cv::Mat LocalLaplacianFilter(const cv::Mat& input,
  double alpha, double beta, double sigma_r)
{
  RemappingFunction r(alpha, beta);

  int num_levels =
  LaplacianPyramid::GetLevelCount(input.rows, input.cols, 30);
  cout << "Number of levels: " << num_levels << endl;

  const int kRows = input.rows;
  const int kCols = input.cols;

  GaussianPyramid gauss_input(input, num_levels);

  // Construct the unfilled Laplacian pyramid of the output.
  // Copy the residual over from the top of the Gaussian pyramid.
  LaplacianPyramid output(kRows, kCols, input.channels(), num_levels);
  gauss_input[num_levels].copyTo(output[num_levels]);

  // Calculate each level of the ouput Laplacian pyramid.
  for (int l = 0; l < num_levels; l++) {
    int subregion_size = 3 * ((1 << (l + 2)) - 1);
    int subregion_r = subregion_size / 2;
    // ref: paper section 4, $K = 3(2^{l_0+2}-1)$

    for (int y = 0; y < output[l].rows; y++) {
      // Calculate the y-bounds of the region in the full-res image.
      int full_res_y = (1 << l) * y;
      int roi_y0 = full_res_y - subregion_r;
      int roi_y1 = full_res_y + subregion_r + 1;
      cv::Range row_range(max(0, roi_y0), min(roi_y1, kRows));
      int full_res_roi_y = full_res_y - row_range.start;

      for (int x = 0; x < output[l].cols; x++) {
        // Calculate the x-bounds of the region in the full-res image.
        int full_res_x = (1 << l) * x;
        int roi_x0 = full_res_x - subregion_r;
        int roi_x1 = full_res_x + subregion_r + 1;
        cv::Range col_range(max(0, roi_x0), min(roi_x1, kCols));
        int full_res_roi_x = full_res_x - col_range.start;

        // Remap the region around the current pixel.
        cv::Mat r0 = input(row_range, col_range);
        cv::Mat remapped;
        r.Evaluate<T>(r0, remapped, gauss_input[l].at<T>(y, x), sigma_r);

        // Construct the Laplacian pyramid for the remapped region and
        // copy the coefficient over to the ouptut Laplacian pyramid.
        LaplacianPyramid tmp_pyr(remapped, l + 1,
            {row_range.start, row_range.end - 1,
             col_range.start, col_range.end - 1});
        output.at<T>(l, y, x) = tmp_pyr.at<T>(l,
          full_res_roi_y >> l, full_res_roi_x >> l);
      }
      cout << "Level " << (l+1) << " (" << output[l].rows << " x "
      << output[l].cols << "), footprint: " << subregion_size
      << "x" << subregion_size << " ... "
      << round(100.0 * y / output[l].rows) << "%\r";
      cout.flush();
    }
    stringstream ss;
    ss << "level" << l << ".png";
    cv::imwrite(ss.str(), ByteScale(cv::abs(output[l])));
    cout << endl;
  }

  return output.Reconstruct();
}
/*******************************main********************************/
int main(int argc, char** argv) {
  cout << "# version: " << CV_VERSION << endl;
  const double kSigmaR = 0.3;
  const double kAlpha = 0.25;
  const double kBeta = 0;

  if (argc != 2) {
    cerr << "Usage: " << argv[0] << " image_file" << endl;
    return 1;
  }

  string filename = argv[1];
  string ext = GetExtension(filename);
  cv::Mat input = cv::imread(filename, cv::IMREAD_UNCHANGED);
  if (input.data == NULL) {
    cerr << "Could not read input image." << endl;
    return 1;
  }
  showMinMax(input);
  imwrite("original.png", input);
  // check whether the data type of `input` is not changed.
  showType(input);
  input.convertTo(input, CV_64F, 1 / 255.0);

  cout << "# Input image: " << GetFileName(filename) << endl
  << "# Size: " << input.cols << " x " << input.rows << endl
  << "# Channels: " << input.channels() << endl;

  cv::Mat output;
  if (input.channels() == 1) {
    output = LocalLaplacianFilter<double>(input, kAlpha, kBeta, kSigmaR);
  } else if (input.channels() == 3) {
    output = LocalLaplacianFilter<cv::Vec3d>(input, kAlpha, kBeta, kSigmaR);
  } else {
    cerr << "Input image must have 1 or 3 channels." << endl;
    return 1;
  }

  output *= 255;
  output.convertTo(output, input.type());

  imwrite("output.png", output);

  return 0;
}
