#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <bitset>
#include <cmath>
// copy the Eigen folder to /usr/local/include/ or put symlink

// run this program as
// g++ main_test.cpp -std=c++11 -o test

// TODO: extend for multiple channel inputs -> would need Tensor instead of Matrix object
// but this can be ignored for the theoretical tests
using namespace Eigen;
using namespace std;

template<typename ParamT,typename Derived>
void batch_normalize_conv_inference (
	const ParamT eps,
        MatrixBase<Derived>& dest,
        const MatrixBase<Derived>& target,
        const ParamT gamma, 
        const ParamT beta,
        const ParamT running_mean,
        const ParamT running_variance
)
{
  
  const float invstd = 1.0f/std::sqrt(running_variance + eps);
  for (int i = 0; i < target.rows(); i++)
  	{
  	for (int j = 0; j < target.cols(); j++)
        	{
                dest(i,j) = gamma* (ParamT)( (target(i,j) - running_mean)*invstd) + beta;
                }
	}
  cout << "Standard BN result matrix =" << endl << dest << endl;
}

// computes approximate square root
unsigned int approximate_sqrt (
	const unsigned int value
)
{
  bitset<16> bit_value{value};
  //cout << "bit value: "<< bit_value << endl;
  // clz computes the number of running zeros regarding a 32-bit encoding
  unsigned int pos_highest_bit = 32 - __builtin_clz(value);
  //cout << "clz: "<< __builtin_clz(value) << endl;
  //cout << "position of highest bit: "<< pos_highest_bit << endl;
  // shift number of to the right by one bit (means dividing by 2)
  bitset<16> highest_bit{pos_highest_bit};
  highest_bit >>= 1;
  //cout << "highest bit: "<< highest_bit << endl;
  unsigned int sqrt_shift = (unsigned int)(highest_bit.to_ulong());
  // shift variance
  bitset<16> bit_sqrt;
  bit_sqrt = bit_value >> sqrt_shift;
    
  unsigned int approximate_solution = (unsigned int)(bit_sqrt.to_ulong());
  
  return approximate_solution;
}

// the function is using 16bit int as type but it could be an 8-bit type as well 
// int just worked better with the Eigen matrix type
void bitwise_batch_normalize_inference (
	const unsigned int eps,
        MatrixXi& dest,
        const MatrixXi& target,
        const unsigned int gamma, 
        const unsigned int beta,
        const unsigned int running_mean,
        const unsigned int running_variance
)
{
  // ### compute square root ###
  // find leading bit of variance
  //cout << "variance: "<< running_variance << endl;
  
  unsigned int sqrt_approx = approximate_sqrt(running_variance);
  //cout << "approximated standard deviation: "<< sqrt_approx << " exact: "<< sqrt(running_variance) << endl;
  // Adding 1 to the std - likely to have little effect on the result but prevents division by zero
  unsigned int pow_2_std = 32 - __builtin_clz(sqrt_approx+1)-1;
  //cout << "power of 2 standard deviation: "<< pow_2_std << endl;  

  for (int i = 0; i < target.rows(); i++)
  	{
  	for (int j = 0; j < target.cols(); j++)
        	{
  		// ### center inputs by subtracting the mean ###
		int centered_value = target(i,j) - (int) running_mean;
		// negative numbers caused some trouble here
		// Ugly fix: use absolute values for scaling return sign information later

		//cout << "centered value: "<< centered_value << endl;
		// Convert to bitstring, there is no sign bit so bit shifting is on absolute values
		bitset<16> bit_value{abs(centered_value)};
		//cout << "centered input bits: "<< bit_value << endl;
		
		// shift to the right to divide by the standard deviation with added const
		// equivalent in formula: (target(i,j) - running_mean)*invstd
		// probably some bug here how to treat the standard deviation!
		bit_value = bit_value >> pow_2_std;
   		//cout << "normalized bits: "<< bit_value << endl;

		// ### multiply with gamma ###

		// equivalent with formula
		//dest(i,j) = gamma*(target(i,j) - running_mean)*invstd 
		// substract one to account for shift of values 
		// example 2^0 = 1 shifting by 0 is the same as multiply by 1
		// 2^1 = 2 shifting by 1 is the same as multiply by 2
		unsigned int pow_2_gamma = 32 - __builtin_clz(gamma) -1;
		bit_value = bit_value << pow_2_gamma;
   		//cout << "scaled normalized bits: "<< bit_value << endl;

		// ### add beta ###
		unsigned int abs_scaled_value = (unsigned int)(bit_value.to_ulong());
		int scaled_value = (int) abs_scaled_value;
		if (centered_value<0){scaled_value = - scaled_value;}

		dest(i,j) = scaled_value + beta;
   		//cout << "shifted and scaled normalized integers: "<< dest(i,j) << endl;
		// equivalent with formula
                //dest(i,j) = gamma*(target(i,j) - running_mean)*invstd + beta;
                }
	}
  cout << "Bitshift BN result matrix =" << endl << dest << endl;
}

template<typename ParamT,typename Derived>
ParamT get_variance_of_matrix (
	const MatrixBase<Derived>& target,
        const ParamT mean
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

  //#### Test square root approximation ####
  /*
  const int num_samples = 1000;
  const int start_number = 0;
  VectorXf approximation_errors = VectorXf::Zero(num_samples);
  VectorXf approximation_errors_pow2 = VectorXf::Zero(num_samples);
  
  for (int i = 0; i < num_samples; i++)
  	{
	approximation_errors(i) = abs(sqrt(i+start_number) - approximate_sqrt(i+start_number));
  	approximation_errors_pow2(i) = abs(sqrt(i+start_number) - 32 + __builtin_clz(approximate_sqrt(i+start_number)+1));
  	}
  cout << "average approximation error: "<< approximation_errors.mean() << endl;
  cout << "maximum approximation error: "<< approximation_errors.maxCoeff() << endl;
  cout << "minimum approximation error: "<< approximation_errors.minCoeff() << endl;

  cout << "### Power of 2 approximation of approximated sqrt compared to float version ###" << endl;
  cout << "average approximation error: "<< approximation_errors_pow2.mean() << endl;
  cout << "maximum approximation error: "<< approximation_errors_pow2.maxCoeff() << endl;
  cout << "minimum approximation error: "<< approximation_errors_pow2.minCoeff() << endl;
  /**/

  //#### general parameters ####

  const int num_rows = 5;
  const int num_cols = 5;



  //#### init input and ouput matrices ####
  
  // generate random input matrix of integers 
  // output of convolutional layer (assuming batch norm after conv and before activation)
  // convolution -> batch normalization -> activation -> quantization
  
  MatrixXf float_target = MatrixXf::Random(num_rows,num_cols);
  float_target = (float_target + MatrixXf::Constant(num_rows,num_cols,1.0)) * 60;
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
  cout << "sample mean =" << endl << running_mean << endl;

  // init variance as sample variance
  float running_var = get_variance_of_matrix (target,running_mean);
  cout << "sample variance =" << endl << running_var << endl;

  // constant epsilon as zero to prevent addition of noise - division by zero might happen
  const float eps = 0;

  // call standard (float) batch norm
  //batch_normalize_conv_inference (eps,float_output,float_target,gamma, beta,running_mean,running_var);
  /**/


  //#### Test int batch normalization (test normalization part) ####

  // call int batch norm
  //batch_normalize_conv_inference ((int)eps,output,target,(int)gamma, (int)beta,(int)running_mean,(int)running_var);



  //#### Test int batch normalization (test scaling and shifting part)####
  
  mt19937 rng;
  rng.seed(random_device()());
  uniform_int_distribution<mt19937::result_type> dist8(1,118);
  uniform_int_distribution<mt19937::result_type> dist_beta(1,118);

  // gamma could be any value but is later reduced to a power of 2 
  int gamma_rnd = dist8(rng);
  // can be any integer
  int beta_rnd =  dist_beta(rng);
  cout << "gamma =" << endl << gamma_rnd << endl;
  cout << "beta =" << endl << beta_rnd << endl;
  
  // call int batch norm with random shift and scale
  batch_normalize_conv_inference (eps,output,target,(float)gamma_rnd, (float)beta_rnd,running_mean,running_var);
  /**/

  // call int batch norm with random shift and scale
  batch_normalize_conv_inference ((int)eps,output,target,gamma_rnd, beta_rnd,(int)running_mean,(int)running_var);
  /**/
 //#### Test bitwise batch normalization (with scaling and shifting)####
  
  // call bitwise batch norm with no shift and scale
  bitwise_batch_normalize_inference ((int)eps,output,target,(int)gamma_rnd, (int)beta_rnd,(int)running_mean,(int)running_var);



}
