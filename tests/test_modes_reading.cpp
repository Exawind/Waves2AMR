#include "Waves2AMR.h"

double L1_norm(std::vector<double> inp_vec1, std::vector<double> inp_vec2) {
  double sum{0.};
  for (int n = 0; n < inp_vec1.size(); ++n) {
    sum += std::abs(inp_vec1[n] - inp_vec2[n]);
  }
  return sum;
}

int main(int argc, char *argv[]) {
  // Set up AMReX
  amrex::Initialize(argc, argv, true, MPI_COMM_WORLD, []() {});

  // Name of modes file
  std::string fname = "/Users/mkuhn/testruns_data/HOS-NWT/"
                      "Matteo_case4_milestone/modes_HOS_SWENSE.dat";
  // Initialize mode reader and dimensionalize params
  ReadModes<double> rmodes(fname, false, false);
  int n0 = rmodes.get_first_fft_dimension();
  int n1 = rmodes.get_second_fft_dimension();
  int n0_sp = rmodes.get_first_spatial_dimension();
  int n1_sp = rmodes.get_second_spatial_dimension();
  int n2 = rmodes.get_third_dimension();
  double depth = rmodes.get_depth();
  double xlen = rmodes.get_xlen();
  double ylen = rmodes.get_ylen();
  double dimL = rmodes.get_L();
  double dimT = rmodes.get_T();
  // Print constants to screen
  std::cout << "HOS simulation constants\n";
  rmodes.print_file_constants();

  // Initialize variables to store modes
  int vsize = rmodes.get_vector_size();
  int vasize = rmodes.get_addl_vector_size();
  double initval = 0.0;
  std::vector<double> mX(vsize, initval);
  std::vector<double> mY(vsize, initval);
  std::vector<double> mZ(vsize, initval);
  std::vector<double> mFS(vsize, initval);
  std::vector<double> mAdd(vasize, initval);

  std::vector<double> mX_(vsize, initval);
  std::vector<double> mY_(vsize, initval);
  std::vector<double> mZ_(vsize, initval);
  std::vector<double> mFS_(vsize, initval);
  std::vector<double> mAdd_(vasize, initval);

  // Get modes at t = 300
  rmodes.get_data(300., mX, mY, mZ, mFS, mAdd);

  // Get modes at t = 350
  rmodes.get_data(350., mX_, mY_, mZ_, mFS_, mAdd_);

  // Calculate and print difference
  double sum = L1_norm(mX, mX_);
  std::cout << "First comparison, between t = 300 and 350: \n   |mX|  = " << sum << std::endl;
  sum = L1_norm(mY, mY_);
  std::cout << "   |mY|  = " << sum << std::endl;
  sum = L1_norm(mZ, mZ_);
  std::cout << "   |mZ|  = " << sum << std::endl;
  sum = L1_norm(mFS, mFS_);
  std::cout << "   |mFS| = " << sum << std::endl;
  sum = L1_norm(mAdd, mAdd_);
  std::cout << "   |mAdd|= " << sum << std::endl;

  // Get modes at t = 400
  rmodes.get_data(400., mX, mY, mZ, mFS, mAdd);

  // Calculate and print difference
  sum = L1_norm(mX, mX_);
  std::cout << "Second comparison, between t = 350 and 400: \n   |mX|  = " << sum << std::endl;
  sum = L1_norm(mY, mY_);
  std::cout << "   |mY|  = " << sum << std::endl;
  sum = L1_norm(mZ, mZ_);
  std::cout << "   |mZ|  = " << sum << std::endl;
  sum = L1_norm(mFS, mFS_);
  std::cout << "   |mFS| = " << sum << std::endl;
  sum = L1_norm(mAdd, mAdd_);
  std::cout << "   |mAdd|= " << sum << std::endl;

  // Get modes at t = 600
  rmodes.get_data(600., mX_, mY_, mZ_, mFS_, mAdd_);

  // Calculate and print difference
  sum = L1_norm(mX, mX_);
  std::cout << "Third comparison, between t = 400 and 600: \n   |mX|  = " << sum << std::endl;
  sum = L1_norm(mY, mY_);
  std::cout << "   |mY|  = " << sum << std::endl;
  sum = L1_norm(mZ, mZ_);
  std::cout << "   |mZ|  = " << sum << std::endl;
  sum = L1_norm(mFS, mFS_);
  std::cout << "   |mFS| = " << sum << std::endl;
  sum = L1_norm(mAdd, mAdd_);
  std::cout << "   |mAdd|= " << sum << std::endl;

  // Get modes at t = 650
  rmodes.get_data(650., mX, mY, mZ, mFS, mAdd);

  // Calculate and print difference
  sum = L1_norm(mX, mX_);
  std::cout << "Fifth comparison, between t = 600 and 650: \n   |mX|  = " << sum << std::endl;
  sum = L1_norm(mY, mY_);
  std::cout << "   |mY|  = " << sum << std::endl;
  sum = L1_norm(mZ, mZ_);
  std::cout << "   |mZ|  = " << sum << std::endl;
  sum = L1_norm(mFS, mFS_);
  std::cout << "   |mFS| = " << sum << std::endl;
  sum = L1_norm(mAdd, mAdd_);
  std::cout << "   |mAdd|= " << sum << std::endl;

  // Finalize AMReX
  amrex::Finalize();
}