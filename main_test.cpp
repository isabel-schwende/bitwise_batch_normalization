#include <iostream>
#include <Eigen/Dense>
#include <random>
// copied the Eigen folder to /usr/local/include/ or put symlink

// run this program as
// g++ main_test.cpp -std=c++11 -o test
using namespace Eigen;
using namespace std;
int main()
{
  // TODO: extend for multiple channel inputs -> would need Tensor instead of Matrix object
  // but this can be ignored for the theoretical tests
  
  const int num_rows = 5;
  const int num_cols = 5;
  // generate random input matrix of integers 
  // output of convolutional layer (assuming batch norm after conv and before activation)
  // convolution -> batch normalization -> activation -> quantization

  MatrixXd rand_floats = MatrixXd::Random(num_rows,num_cols);
  rand_floats = (rand_floats + MatrixXd::Constant(num_rows,num_cols,1.0)) * 50;
  MatrixXi target = rand_floats.cast <int> ();
  cout << "target matrix =" << endl << target << endl;
  
  mt19937 rng;
  rng.seed(random_device()());
  uniform_int_distribution<mt19937::result_type> dist8(1,8);
  uniform_int_distribution<mt19937::result_type> dist_beta(1,128);
  uniform_int_distribution<mt19937::result_type> dist_var(1,10);

  // gamma has to be a power of 2 
  // in this case we'll just generate the power of two as an integer without computing it yet
  int gamma = dist8(rng);
  // can be any integer
  int beta =  dist_beta(rng);
  cout << "gamma =" << endl << gamma << endl;
  cout << "beta =" << endl << beta << endl;

  // init running mean as mean of target but could be also randomly initalized
  float float_mean = target.mean();
  int running_mean = (int)float_mean;
  cout << "running mean =" << endl << running_mean << endl;

  // init variance randomly
  int running_var =  dist_var(rng);
  cout << "running variance =" << endl << running_var << endl;

  // constant epsilon to prevent division by zero
  const int eps = 1;

  // init output matrix
  MatrixXi output = MatrixXi::Zero(num_rows,num_cols);
  
}
