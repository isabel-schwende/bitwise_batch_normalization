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
  // generate random input matrix of integers 
  // output of convolutional layer (assuming batch norm after conv and before activation)
  MatrixXd m = MatrixXd::Random(3,3);
  m = (m + MatrixXd::Constant(3,3,1.0)) * 50;
  MatrixXi src = m.cast <int> ();
  cout << "src =" << endl << src << endl;
  
  mt19937 rng;
  rng.seed(random_device()());
  uniform_int_distribution<mt19937::result_type> dist8(1,8);
  uniform_int_distribution<mt19937::result_type> dist_beta(1,128);

  // gamma has to be a power of 2 (or could apply highest_power_of2)
  int gamma = dist8(rng);
  // can be any integer
  int beta =  dist_beta(rng);
  cout << "gamma =" << endl << gamma << endl;
  cout << "beta =" << endl << beta << endl;
}
