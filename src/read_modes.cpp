#include "read_modes.h"

template class ReadModes<double>;
template class ReadModes<std::complex<double>>;

/*template <>
ReadModes::ReadModes(std::string filename, bool is_ocean, bool allmodes)
    : m_filename(filename), is_HOS_Ocean(is_ocean), is_init(true) {
  // Set time index value
  itime_now = 0;

  // Initialize
  bool file_exists = ascii_initialize(is_HOS_Ocean);
  if (!file_exists) {
    std::cout << "ABORT: ReadModes is initializing with a file, but the "
                 "specified file does not exist.\n";
    std::exit(1);
  }

  // Get working dimensions
  n1o2p1 = n1 / 2 + 1;

  // Calculate size of mode vectors
  vec_size = n2 * (is_HOS_Ocean ? n1o2p1 : n1);

  // Set size of mode vectors
  if (is_ocean) {
    // HOS-Ocean
    n1o2p1 = n1 / 2 + 1;
    vec_size = n2 * n1o2p1;
    c_modeX.resize(vec_size);
    c_modeY.resize(vec_size);
    c_modeZ.resize(vec_size);
    c_modeFS.resize(vec_size);
    // These modes are optional
    if (allmodes) {
      c_modeT.resize(vec_size);
      c_modeFST.resize(vec_size);
    }
  } else {
    // NWT
    vec_size = n2 * n1;
    r_modeX.resize(vec_size);
    r_modeY.resize(vec_size);
    r_modeZ.resize(vec_size);
    r_modeFS.resize(vec_size);
    // These modes are optional
    if (allmodes) {
      r_modeT.resize(vec_size);
      r_modeFST.resize(vec_size);
    }
  }

  // Dimensionalize all nondim scalar quantities
  dimensionalize();
}

ReadModes::ReadModes(double dt_out_, double T_stop_, double xlen_, double ylen_,
                     double depth_, double g_, double L_, double T_)
    : dt_out(dt_out_), T_stop(T_stop_), xlen(xlen_), ylen(ylen_), depth(depth_),
      g(g_), L(L_), T(T_), is_init(true) {
  // ^Manually set metadata for the sake of testing, do other expected steps
  // No treatment of integer dimensions needed at the moment

  // Initialize time index
  itime_now = 0;
  // Initialize output frequency
  f_out = 1.0 / dt_out;

  dimensionalize();
}

// Do-nothing constructor, initializer must be called later
ReadModes::ReadModes() : is_init(false) {}

bool ReadModes::initialize(std::string filename, bool is_ocean, bool allmodes) {
  // Check if already initialized
  if (is_init) {
    std::cout << "ABORT: ReadModes has already been initialized, but "
                 "initialize has been called again.\n";
    std::exit(1);
  }
  is_init = true;
  m_filename = filename;
  is_HOS_Ocean = is_ocean;

  // Set time index value
  itime_now = 0;
  // TODO: Determine filetype

  // Initialize (TODO: according to file type)
  bool file_exists = ascii_initialize(is_HOS_Ocean);
  if (!file_exists) {
    return file_exists;
  }

  // Set size of mode vectors
  if (is_ocean) {
    // HOS-Ocean
    n1o2p1 = n1 / 2 + 1;
    vec_size = n2 * n1o2p1;
    c_modeX.resize(vec_size);
    c_modeY.resize(vec_size);
    c_modeZ.resize(vec_size);
    c_modeFS.resize(vec_size);
    // These modes are optional
    if (allmodes) {
      c_modeT.resize(vec_size);
      c_modeFST.resize(vec_size);
    }
  } else {
    // NWT
    vec_size = n2 * n1;
    r_modeX.resize(vec_size);
    r_modeY.resize(vec_size);
    r_modeZ.resize(vec_size);
    r_modeFS.resize(vec_size);
    // These modes are optional
    if (allmodes) {
      r_modeT.resize(vec_size);
      r_modeFST.resize(vec_size);
    }
  }

  // Dimensionalize all nondim scalar quantities
  dimensionalize();

  return file_exists;
}

// Version that uses stored index as guess and increments it
int ReadModes::time2step(const double time) {
  itime_now = time2step(time, itime_now);
  return itime_now;
}

int ReadModes::time2step(const double time, const int itime_guess) {
  // Return -1 if time is negative
  if (time < 0.0)
    return -1;
  // Begin with guess
  int itime = itime_guess;
  // Look for same time or after
  bool done = false;
  while (!done) {
    // Use a tolerance to avoid skipping close matches
    if (itime * dt_out < (time - dt_out * 1e-8)) {
      ++itime;
    } else if ((itime - 1) * dt_out > time) {
      --itime;
    } else {
      done = true;
    }
  }
  return itime;
}

void ReadModes::dimensionalize() {
  // Dimensionalize read-in nondim quantities
  dt_out *= T;
  f_out /= T;
  T_stop *= T;
  xlen *= L;
  ylen *= L;
  depth *= L;
  g *= L / T / T;
}

bool ReadModes::read_data(double time) {
  int itime = time2step(time);
  return read_data(itime);
}

bool ReadModes::read_data(int itime) {
  // Read (TODO: according to file type)
  return ascii_read(itime);
}

void ReadModes::output_data(std::vector<std::complex<double>> &v1,
                            std::vector<std::complex<double>> &v2,
                            std::vector<std::complex<double>> &v3,
                            std::vector<std::complex<double>> &v4,
                            std::vector<std::complex<double>> &v5,
                            std::vector<std::complex<double>> &v6) {
  // Copy class variables to input/output variables
  std::copy(c_modeX.begin(), c_modeX.end(), v1.begin());
  std::copy(c_modeY.begin(), c_modeY.end(), v2.begin());
  std::copy(c_modeZ.begin(), c_modeZ.end(), v3.begin());
  std::copy(c_modeT.begin(), c_modeT.end(), v4.begin());
  std::copy(c_modeFS.begin(), c_modeFS.end(), v5.begin());
  std::copy(c_modeFST.begin(), c_modeFST.end(), v6.begin());
}

void ReadModes::output_data(std::vector<std::complex<double>> &v1,
                            std::vector<std::complex<double>> &v2,
                            std::vector<std::complex<double>> &v3,
                            std::vector<std::complex<double>> &v4) {
  // Copy class variables to input/output variables
  std::copy(c_modeX.begin(), c_modeX.end(), v1.begin());
  std::copy(c_modeY.begin(), c_modeY.end(), v2.begin());
  std::copy(c_modeZ.begin(), c_modeZ.end(), v3.begin());
  std::copy(c_modeFS.begin(), c_modeFS.end(), v4.begin());
}

bool ReadModes::get_data(double time, std::vector<std::complex<double>> &mX,
                         std::vector<std::complex<double>> &mY,
                         std::vector<std::complex<double>> &mZ,
                         std::vector<std::complex<double>> &mT,
                         std::vector<std::complex<double>> &mFS,
                         std::vector<std::complex<double>> &mFST) {
  // Read data
  auto flag = read_data(time);
  // Copy data to output
  output_data(mX, mY, mZ, mT, mFS, mFST);
  // Pass read flag (for detecting EOF)
  return flag;
}

bool ReadModes::get_data(int itime, std::vector<std::complex<double>> &mX,
                         std::vector<std::complex<double>> &mY,
                         std::vector<std::complex<double>> &mZ,
                         std::vector<std::complex<double>> &mT,
                         std::vector<std::complex<double>> &mFS,
                         std::vector<std::complex<double>> &mFST) {
  // Read data
  auto flag = read_data(itime);
  // Copy data to output
  output_data(mX, mY, mZ, mT, mFS, mFST);
  // Pass read flag (for detecting EOF)
  return flag;
}

bool ReadModes::get_data(double time, std::vector<std::complex<double>> &mX,
                         std::vector<std::complex<double>> &mY,
                         std::vector<std::complex<double>> &mZ,
                         std::vector<std::complex<double>> &mFS) {
  // Read data
  auto flag = read_data(time);
  // Copy data to output
  output_data(mX, mY, mZ, mFS);
  // Pass read flag (for detecting EOF)
  return flag;
}

bool ReadModes::get_data(int itime, std::vector<std::complex<double>> &mX,
                         std::vector<std::complex<double>> &mY,
                         std::vector<std::complex<double>> &mZ,
                         std::vector<std::complex<double>> &mFS) {
  // Read data
  auto flag = read_data(itime);
  // Copy data to output
  output_data(mX, mY, mZ, mFS);
  // Pass read flag (for detecting EOF)
  return flag;
}

void ReadModes::print_file_constants() {
  std::cout << "f_out " << f_out << " T " << T << " T_stop " << T_stop
            << std::endl;
  std::cout << "n1 " << n1 << " n2 " << n2 << std::endl;
  std::cout << "xlen " << xlen << " ylen " << ylen << std::endl;
  std::cout << "depth " << depth << " g " << g << " L " << L << std::endl;
}
  */