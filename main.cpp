// File Description
// Author: Philip Salvaggio
// Modification: crcrpar
// so as to tone-manipulation

#include "gaussian_pyramid.h"
#include "laplacian_pyramid.h"
#include "opencv_utils.h"
#include "remapping_function.h"
#include <iostream>
#include <sstream>
#include <cmath>
#include <vector>

#define OPENCV_VERSION(a,b,c) (((a) << 16) + ((b) << 8) + (c))

using namespace std;

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

//  Tone Mapping: calculate the intensity image and color ratio,
//                then, perform local laplacian filtering.
//
// Arguments
// input     The input image, only hdr.
// alpha     The same as local laplacian filter
// beta      The same as local laplacian filter
// sigma_r   The same as local laplacian filter
template<typename T>
cv::Mat ToneManipulation(const cv::Mat& input, double alpha,
  double beta, double sigma_r)
{
  cout << "### Tone Manipulation ###" << endl;
  cv::Mat tmp_img = input.clone(), intensity=cv::Mat(input.size(), CV_64FC1);
  vector<cv::Mat> channels;
  cv::split(tmp_img, channels);
  float coefficients[] = {1/61.0, 40/61.0, 20/61.0};
  for (int i=0; i<3; i++) {
    intensity += channels[i] * coefficients[i];
  }
  // cv::merge(channels, intensity);
  vector<cv::Mat> color_ratio;
  // color_ratio: (\rho_r, \rho_g, \rho_b) = (I_r, I_g, I_b) / I_i
  for (int i=0; i<3; i++) {
    cv::Mat tmp = channels[i].clone();
    if (tmp.size() == intensity.size() &&
    tmp.channels() == intensity.channels())
    {
      // cout << "Division" << endl;
      // cout << "tmp "; showType(tmp);
      // cout << "intensity: "; showType(intensity);
      cv::divide(tmp, intensity, tmp);
      cout << "divided" << endl;
      // http://docs.opencv.org/3.2.0/d2/de8/group__core__array.html#ga6db555d30115642fedae0cda05604874
      // http://stackoverflow.com/questions/20975420/divide-two-matrices-in-opencv
      color_ratio.push_back(tmp);
    }
  }
  cout << "color_ratio size: " << color_ratio.size() << endl;
  cv::Mat compressed = LocalLaplacianFilter<double>(tmp_img, alpha, beta, sigma_r);
}

/****************************** main *******************************/
int main(int argc, char** argv) {
  cout << "*** tone mapping ***" << endl;
  cout << "# opencv version: " << CV_VERSION << endl;
  const double kSigmaR = log(0.25);
  const double kAlpha = 0.25;
  const double kBeta = 0.01;

  if (argc != 2) {
    cerr << "Usage: " << argv[0] << " image_file" << endl;
    return 1;
  }

  string filename = argv[1];
  string ext = GetExtension(filename);
  cout << "# image file type: " << ext << endl;
  cv::Mat input = cv::imread(filename, cv::IMREAD_UNCHANGED);
  if (input.data == NULL) {
    cerr << "Could not read input image." << endl;
    return 1;
  }
  showMinMax(input); // show min and max of input image.
  cv::Mat tmp = input.clone();
  tmp.convertTo(tmp, CV_8U, 255);
  imwrite("original.png", tmp);
  // check whether the data type of `input` is not changed.
  showType(input);
  input.convertTo(input, CV_64F, 1.0, 0.0);
  showMinMax(input);
  // test of ToneManipulation
  cv::Mat output = ToneManipulation<cv::Vec3d>(input, kAlpha, kBeta, kSigmaR);

  cout << "# Input image: " << GetFileName(filename) << endl
  << "# Size: " << input.cols << " x " << input.rows << endl
  << "# Channels: " << input.channels() << endl;


  output *= 255;
  output.convertTo(output, input.type());

  imwrite("output.png", output);

  return 0;
}
