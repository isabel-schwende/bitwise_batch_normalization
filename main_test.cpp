#include <iostream>
#include <Eigen/Dense>
#include <random>
// copied the Eigen folder to /usr/local/include/ or put symlink

// run this program as
// g++ main_test.cpp -std=c++11 -o test
using namespace Eigen;
using namespace std;

void batch_normalize_conv_inference (
	const float eps,
        MatrixXf& dest,
        const MatrixXf& target,
        const float gamma, 
        const float beta,
        const float running_mean,
        const float running_variance
)
{
  
  const float invstd = 1.0f/std::sqrt(running_variance + eps);
  for (int i = 0; i < target.rows(); i++)
  	{
  	for (int j = 0; j < target.cols(); j++)
        	{
                dest(i,j) = gamma*(target(i,j) - running_mean)*invstd + beta;
                }
	}
  cout << "destination matrix =" << endl << dest << endl;
}

void batch_normalize_conv_inference (
	const int eps,
        MatrixXi& dest,
        const MatrixXi& target,
        const int gamma, 
        const int beta,
        const int running_mean,
        const int running_variance
)
{
  
  const float invstd = 1.0f/std::sqrt(running_variance + eps);
  for (int i = 0; i < target.rows(); i++)
  	{
  	for (int j = 0; j < target.cols(); j++)
        	{
                dest(i,j) = gamma*(target(i,j) - running_mean)*invstd + beta;
                }
	}
  cout << "destination matrix =" << endl << dest << endl;
}

float get_variance_of_matrix (
	const MatrixXi& target,
        const int mean
)
{
  float var = 0;
  const float total_num = target.rows()*target.cols();
  for (int i = 0; i < target.rows(); i++)
  	{
  	for (int j = 0; j < target.cols(); j++)
        	{
                var = var + (float)target(i,j)*target(i,j);
                }
	}
  var = var / total_num - mean * mean;
  return var;
}


int main()
{
  // TODO: extend for multiple channel inputs -> would need Tensor instead of Matrix object
  // but this can be ignored for the theoretical tests
  
  //#### general parameters ####

  const int num_rows = 5;
  const int num_cols = 5;



  //#### init input and ouput matrices ####

  // generate random input matrix of integers 
  // output of convolutional layer (assuming batch norm after conv and before activation)
  // convolution -> batch normalization -> activation -> quantization

  MatrixXf float_target = MatrixXf::Random(num_rows,num_cols);
  float_target = (float_target + MatrixXf::Constant(num_rows,num_cols,1.0)) * 50;
  MatrixXi target = float_target.cast <int> ();
  cout << "target matrix =" << endl << target << endl;

  // init output matrix
  MatrixXi output = MatrixXi::Zero(num_rows,num_cols);
  MatrixXf float_output = MatrixXf::Zero(num_rows,num_cols);

  //#### Test float batch normalization (test normalization part) ####
  
  // set gamma to 1 for no scaling
  float gamma = 1;
  // set beta to 0 for no shifting
  float beta =  0;

  // init running mean as sample mean 
  float running_mean = target.mean();
  cout << "running mean =" << endl << running_mean << endl;

  // init variance as sample variance
  float running_var = get_variance_of_matrix (target,running_mean);
  cout << "running variance =" << endl << running_var << endl;

  // constant epsilon as zero to prevent addition of noise - division by zero might happen
  const float eps = 0;

  // call standard (float) batch norm
  batch_normalize_conv_inference (eps,float_output,float_target,gamma, beta,running_mean,running_var);



  //#### Test int batch normalization (test normalization part) ####

  // call int batch norm
  batch_normalize_conv_inference ((int)eps,output,target,(int)gamma, (int)beta,(int)running_mean,(int)running_var);



  //#### Test int batch normalization (test scaling and shifting part)####
  
  mt19937 rng;
  rng.seed(random_device()());
  uniform_int_distribution<mt19937::result_type> dist8(1,8);
  uniform_int_distribution<mt19937::result_type> dist_beta(1,128);

  // gamma has to be a power of 2 
  // in this case we'll just generate the power of two as an integer without computing it yet
  int gamma_rnd = dist8(rng);
  // can be any integer
  int beta_rnd =  dist_beta(rng);
  cout << "gamma =" << endl << gamma_rnd << endl;
  cout << "beta =" << endl << beta_rnd << endl;


  // call int batch norm with random shift and scale
  batch_normalize_conv_inference ((int)eps,output,target,gamma_rnd, beta_rnd,(int)running_mean,(int)running_var);
}
