#include <iostream>
#include <Eigen/Dense>
// copied the Eigen folder to /usr/local/include/ or put symlink
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

  // gamma has to be a power of 2 (or could apply highest_power_of2)
  int gamma = 8;
  // can be any integer
  int beta =  5;
}
