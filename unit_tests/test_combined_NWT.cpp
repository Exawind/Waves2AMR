#include "Waves2AMR.h"
#include "gtest/gtest.h"
#include <array>

namespace w2a_tests {

class CombinedTestNWT : public testing::Test {};

TEST_F(CombinedTestNWT, ReadFFTNonDim2D) {
  std::string fname = "../tests/nwt_2D_modes_HOS_SWENSE.dat";
  // Read
  ReadModes<double> rmodes(fname, false);
  // Initialize and size output variables
  int vsize = rmodes.get_vector_size();
  int vasize = rmodes.get_addl_vector_size();
  std::vector<double> mX(vsize, 0.0);
  std::vector<double> mY(vsize, 0.0);
  std::vector<double> mZ(vsize, 0.0);
  std::vector<double> mFS(vsize, 0.0);
  std::vector<double> mAdd(vsize, 0.0);

  EXPECT_EQ(vsize, 1 * 256);
  EXPECT_EQ(vasize, 1 * 64);

  // Populate mode data
  rmodes.get_data((int)6, mX, mY, mZ, mFS, mAdd);
  // Get dimensions
  int n0 = rmodes.get_first_fft_dimension();
  int n1 = rmodes.get_second_fft_dimension();
  int n2 = rmodes.get_third_dimension();
  double nd_xlen = rmodes.get_nondim_xlen();
  double nd_ylen = rmodes.get_nondim_ylen();
  double nd_depth = rmodes.get_nondim_depth();

  // Allocate complex pointers and get plan
  std::vector<fftw_plan> plan_vector;
  modes_hosgrid::plan_ifftw_nwt(n0, n1, plan_vector, &mFS[0],
                                modes_hosgrid::planner_flags::patient);
  auto u_modes = modes_hosgrid::allocate_real(n0, n1);
  auto v_modes = modes_hosgrid::allocate_real(n0, n1);
  auto w_modes = modes_hosgrid::allocate_real(n0, n1);
  auto au_modes = modes_hosgrid::allocate_real(n0, 1);
  auto av_modes = modes_hosgrid::allocate_real(n0, 1);
  auto aw_modes = modes_hosgrid::allocate_real(n0, 1);

  // Set up local working
  amrex::Vector<amrex::Real> u_work(n0 * n1, 0.0);
  amrex::Vector<amrex::Real> v_work(n0 * n1, 0.0);
  amrex::Vector<amrex::Real> w_work(n0 * n1, 0.0);

  // Set up output vectors
  amrex::Gpu::DeviceVector<amrex::Real> eta(n0 * n1, 0.0);
  amrex::Gpu::DeviceVector<amrex::Real> u(n0 * n1, 0.0);
  amrex::Gpu::DeviceVector<amrex::Real> v(n0 * n1, 0.0);
  amrex::Gpu::DeviceVector<amrex::Real> w(n0 * n1, 0.0);

  // Get spatial data for eta
  modes_hosgrid::populate_hos_nwt_eta_nondim(n0, n1, plan_vector, &mFS[0], eta);

  // Transfer to host
  std::vector<amrex::Real> etalocal;
  etalocal.resize(eta.size());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, eta.begin(), eta.end(),
                   &etalocal[0]);
  // Get max and min
  double max_eta = -100.0;
  double min_eta = 100.0;
  for (int i0 = 0; i0 < n0; ++i0) {
    for (int i1 = 0; i1 < n1; ++i1) {
      int idx = i0 * n1 + i1;
      max_eta = std::max(max_eta, etalocal[idx]);
      min_eta = std::min(min_eta, etalocal[idx]);
    }
  }

  // !! -- Reference values are from Grid2Grid library -- !! //

  // Check max and min
  EXPECT_NEAR(max_eta, 1.5059722949275829E-002, 1e-10);
  EXPECT_NEAR(min_eta, -1.9719905099789568E-002, 1e-10);

  // Store values to check velocity
  double ht[3]{-0.32280115456367126, 3.1918911957973251E-016,
               0.38888888888888928};
  double umaxref_[3]{9.1788477599439051E-003, 4.5390042349902321E-002,
                     0.13411654860836050};
  double uminref_[3]{-4.9171749226979318E-003, -5.7686227470213139E-002,
                     -0.17769950634856563};
  double wmaxref_[3]{7.0120317161333108E-003, 4.4400484210156757E-002,
                     0.13423685447894013};
  double wminref_[3]{-9.3915354314830553E-003, -4.6514951704988788E-002,
                     -0.13968892340760958};
  double umaxref[3]{4.9153695398747293E-003, 3.9587659476485480E-002,
                    0.13207261836925066};
  double uminref[3]{-7.9131256274804426E-003, -5.9558596641518795E-002,
                    -0.17895145574253227};
  double wmaxref[3]{5.9978774426146632E-003, 4.2625638369894081E-002,
                    0.13172677143217495};
  double wminref[3]{-6.9093491974525460E-003, -4.9123634147938001E-002,
                    -0.14160677461014401};
  // Get spatial data for velocity at different heights
  for (int n = 0; n < 3; ++n) {

    modes_hosgrid::populate_hos_nwt_vel_nondim(
        n0, n1, nd_xlen, nd_ylen, ht[n], mX, mY, mZ, plan_vector, u_modes,
        v_modes, w_modes, u_work, v_work, w_work);

    // Check velocities after base set of modes
    double max_u = -100.0;
    double min_u = 100.0;
    double max_v = -100.0;
    double min_v = 100.0;
    double max_w = -100.0;
    double min_w = 100.0;
    for (int i0 = 0; i0 < n0; ++i0) {
      for (int i1 = 0; i1 < n1; ++i1) {
        int idx = i0 * n1 + i1;
        max_u = std::max(max_u, u_work[idx]);
        min_u = std::min(min_u, u_work[idx]);
        max_v = std::max(max_v, v_work[idx]);
        min_v = std::min(min_v, v_work[idx]);
        max_w = std::max(max_w, w_work[idx]);
        min_w = std::min(min_w, w_work[idx]);
      }
    }

    // Check max and min
    EXPECT_NEAR(max_u, umaxref_[n], 1e-10);
    EXPECT_NEAR(min_u, uminref_[n], 1e-10);
    EXPECT_NEAR(max_v, 0.0, 1e-10);
    EXPECT_NEAR(min_v, 0.0, 1e-10);
    EXPECT_NEAR(max_w, wmaxref_[n], 1e-10);
    EXPECT_NEAR(min_w, wminref_[n], 1e-10);

    // Incorporate additional modes
    modes_hosgrid::populate_additional_hos_nwt_vel_nondim(
        n0, n1, n2, nd_xlen, nd_ylen, ht[n], mAdd, plan_vector, au_modes,
        av_modes, aw_modes, u_work, v_work, w_work);

    // Check velocities after additional modes incorporated
    max_u = -100.0;
    min_u = 100.0;
    max_v = -100.0;
    min_v = 100.0;
    max_w = -100.0;
    min_w = 100.0;
    for (int i0 = 0; i0 < n0; ++i0) {
      for (int i1 = 0; i1 < n1; ++i1) {
        int idx = i0 * n1 + i1;
        max_u = std::max(max_u, u_work[idx]);
        min_u = std::min(min_u, u_work[idx]);
        max_v = std::max(max_v, v_work[idx]);
        min_v = std::min(min_v, v_work[idx]);
        max_w = std::max(max_w, w_work[idx]);
        min_w = std::min(min_w, w_work[idx]);
      }
    }

    // Check max and min
    EXPECT_NEAR(max_u, umaxref[n], 1e-10);
    EXPECT_NEAR(min_u, uminref[n], 1e-10);
    EXPECT_NEAR(max_v, 0.0, 1e-10);
    EXPECT_NEAR(min_v, 0.0, 1e-10);
    EXPECT_NEAR(max_w, wmaxref[n], 1e-10);
    EXPECT_NEAR(min_w, wminref[n], 1e-10);

    modes_hosgrid::copy_vel_nondim_to_device(u_work, v_work, w_work, u, v, w,
                                             0);

    // Transfer to host
    std::vector<amrex::Real> ulocal, vlocal, wlocal;
    ulocal.resize(n0 * n1);
    vlocal.resize(n0 * n1);
    wlocal.resize(n0 * n1);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, u.begin(), u.end(), &ulocal[0]);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, v.begin(), v.end(), &vlocal[0]);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, w.begin(), w.end(), &wlocal[0]);
    max_u = -100.0;
    min_u = 100.0;
    max_v = -100.0;
    min_v = 100.0;
    max_w = -100.0;
    min_w = 100.0;
    for (int i0 = 0; i0 < n0; ++i0) {
      for (int i1 = 0; i1 < n1; ++i1) {
        int idx = i0 * n1 + i1;
        max_u = std::max(max_u, ulocal[idx]);
        min_u = std::min(min_u, ulocal[idx]);
        max_v = std::max(max_v, vlocal[idx]);
        min_v = std::min(min_v, vlocal[idx]);
        max_w = std::max(max_w, wlocal[idx]);
        min_w = std::min(min_w, wlocal[idx]);
      }
    }

    // Check max and min
    EXPECT_NEAR(max_u, umaxref[n], 1e-10);
    EXPECT_NEAR(min_u, uminref[n], 1e-10);
    EXPECT_NEAR(max_v, 0.0, 1e-10);
    EXPECT_NEAR(min_v, 0.0, 1e-10);
    EXPECT_NEAR(max_w, wmaxref[n], 1e-10);
    EXPECT_NEAR(min_w, wminref[n], 1e-10);
  }

  // Delete complex pointers to allocated data
  delete[] u_modes;
  delete[] v_modes;
  delete[] w_modes;
  delete[] au_modes;
  delete[] av_modes;
  delete[] aw_modes;
  // Delete plan
  for (int n = 0; n < plan_vector.size(); ++n) {
    fftw_destroy_plan(plan_vector[n]);
  }
  plan_vector.clear();
}

TEST_F(CombinedTestNWT, ReadFFTNonDim3D) {
  std::string fname = "../tests/nwt_3D_modes_HOS_SWENSE.dat";
  // Read
  ReadModes<double> rmodes(fname, false);
  // Initialize and size output variables
  int vsize = rmodes.get_vector_size();
  int vasize = rmodes.get_addl_vector_size();
  std::vector<double> mX(vsize, 0.0);
  std::vector<double> mY(vsize, 0.0);
  std::vector<double> mZ(vsize, 0.0);
  std::vector<double> mFS(vsize, 0.0);
  std::vector<double> mAdd(vasize, 0.0);

  // Populate mode data
  rmodes.get_data((int)6, mX, mY, mZ, mFS, mAdd);
  // Get dimensions
  int n0 = rmodes.get_first_fft_dimension();
  int n1 = rmodes.get_second_fft_dimension();
  int n2 = rmodes.get_third_dimension();
  double nd_xlen = rmodes.get_nondim_xlen();
  double nd_ylen = rmodes.get_nondim_ylen();
  double nd_depth = rmodes.get_nondim_depth();

  // Allocate complex pointers and get plan
  std::vector<fftw_plan> plan_vector;
  modes_hosgrid::plan_ifftw_nwt(n0, n1, plan_vector, &mFS[0],
                                modes_hosgrid::planner_flags::patient);
  auto u_modes = modes_hosgrid::allocate_real(n0, n1);
  auto v_modes = modes_hosgrid::allocate_real(n0, n1);
  auto w_modes = modes_hosgrid::allocate_real(n0, n1);
  auto au_modes = modes_hosgrid::allocate_real(n0, 1);
  auto av_modes = modes_hosgrid::allocate_real(n0, 1);
  auto aw_modes = modes_hosgrid::allocate_real(n0, 1);

  // Set up local working
  amrex::Vector<amrex::Real> u_work(n0 * n1, 0.0);
  amrex::Vector<amrex::Real> v_work(n0 * n1, 0.0);
  amrex::Vector<amrex::Real> w_work(n0 * n1, 0.0);

  // Set up output vectors
  amrex::Gpu::DeviceVector<amrex::Real> eta(n0 * n1, 0.0);
  amrex::Gpu::DeviceVector<amrex::Real> u(n0 * n1, 0.0);
  amrex::Gpu::DeviceVector<amrex::Real> v(n0 * n1, 0.0);
  amrex::Gpu::DeviceVector<amrex::Real> w(n0 * n1, 0.0);

  // Get spatial data for eta
  modes_hosgrid::populate_hos_nwt_eta_nondim(n0, n1, plan_vector, &mFS[0], eta);

  // Transfer to host
  std::vector<amrex::Real> etalocal;
  etalocal.resize(eta.size());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, eta.begin(), eta.end(),
                   &etalocal[0]);
  // Get max and min
  double max_eta = -100.0;
  double min_eta = 100.0;
  for (int i0 = 0; i0 < n0; ++i0) {
    for (int i1 = 0; i1 < n1; ++i1) {
      int idx = i0 * n1 + i1;
      max_eta = std::max(max_eta, etalocal[idx]);
      min_eta = std::min(min_eta, etalocal[idx]);
    }
  }

  // !! -- Reference values are from Grid2Grid library -- !! //

  // Check max and min
  EXPECT_NEAR(max_eta, 1.5787227348319759E-002, 1e-10);
  EXPECT_NEAR(min_eta, -1.6624551984605426E-002, 1e-10);

  // Store values to check velocity
  double ht[3]{-0.32280115456367126, 3.1918911957973251E-016,
               0.38888888888888928};
  double umaxref_[3]{1.0025164587775751E-002, 3.8424328117794920E-002,
                     0.11221213106061016};
  double uminref_[3]{-5.0827207851649216E-003, -4.1333660767169295E-002,
                     -0.12850450299308788};
  double wmaxref_[3]{7.5745784866818062E-003, 3.7740522310771546E-002,
                     0.11467590921752700};
  double wminref_[3]{-9.3906198628731577E-003, -3.2105602006349016E-002,
                     -9.6336423370944488E-002};
  double umaxref[3]{6.0751997221759049E-003, 3.4395234261114155E-002,
                    0.11012966187700549};
  double uminref[3]{-7.7382241342878206E-003, -4.3211673484922393E-002,
                    -0.12976012082538291};
  double wmaxref[3]{6.5893567121181182E-003, 3.5840604212930019E-002,
                    0.11192450388783164};
  double wminref[3]{-6.9523618382818439E-003, -3.3549751535864931E-002,
                    -9.8301929994299639E-002};
  // Get spatial data for velocity at different heights
  for (int n = 0; n < 3; ++n) {

    modes_hosgrid::populate_hos_nwt_vel_nondim(
        n0, n1, nd_xlen, nd_ylen, ht[n], mX, mY, mZ, plan_vector, u_modes,
        v_modes, w_modes, u_work, v_work, w_work);

    // Check velocities after base set of modes
    double max_u = -100.0;
    double min_u = 100.0;
    double max_v = -100.0;
    double min_v = 100.0;
    double max_w = -100.0;
    double min_w = 100.0;
    for (int i0 = 0; i0 < n0; ++i0) {
      for (int i1 = 0; i1 < n1; ++i1) {
        int idx = i0 * n1 + i1;
        max_u = std::max(max_u, u_work[idx]);
        min_u = std::min(min_u, u_work[idx]);
        max_v = std::max(max_v, v_work[idx]);
        min_v = std::min(min_v, v_work[idx]);
        max_w = std::max(max_w, w_work[idx]);
        min_w = std::min(min_w, w_work[idx]);
      }
    }

    // Check max and min
    EXPECT_NEAR(max_u, umaxref_[n], 1e-10);
    EXPECT_NEAR(min_u, uminref_[n], 1e-10);
    EXPECT_NEAR(max_v, 0.0, 1e-10);
    EXPECT_NEAR(min_v, 0.0, 1e-10);
    EXPECT_NEAR(max_w, wmaxref_[n], 1e-10);
    EXPECT_NEAR(min_w, wminref_[n], 1e-10);

    // Incorporate additional modes
    modes_hosgrid::populate_additional_hos_nwt_vel_nondim(
        n0, n1, n2, nd_xlen, nd_ylen, ht[n], mAdd, plan_vector, au_modes,
        av_modes, aw_modes, u_work, v_work, w_work);

    // Check velocities after additional modes incorporated
    max_u = -100.0;
    min_u = 100.0;
    max_v = -100.0;
    min_v = 100.0;
    max_w = -100.0;
    min_w = 100.0;
    for (int i0 = 0; i0 < n0; ++i0) {
      for (int i1 = 0; i1 < n1; ++i1) {
        int idx = i0 * n1 + i1;
        max_u = std::max(max_u, u_work[idx]);
        min_u = std::min(min_u, u_work[idx]);
        max_v = std::max(max_v, v_work[idx]);
        min_v = std::min(min_v, v_work[idx]);
        max_w = std::max(max_w, w_work[idx]);
        min_w = std::min(min_w, w_work[idx]);
      }
    }

    // Check max and min
    EXPECT_NEAR(max_u, umaxref[n], 1e-10);
    EXPECT_NEAR(min_u, uminref[n], 1e-10);
    EXPECT_NEAR(max_v, 0.0, 1e-10);
    EXPECT_NEAR(min_v, 0.0, 1e-10);
    EXPECT_NEAR(max_w, wmaxref[n], 1e-10);
    EXPECT_NEAR(min_w, wminref[n], 1e-10);

    modes_hosgrid::copy_vel_nondim_to_device(u_work, v_work, w_work, u, v, w,
                                             0);

    // Transfer to host
    std::vector<amrex::Real> ulocal, vlocal, wlocal;
    ulocal.resize(n0 * n1);
    vlocal.resize(n0 * n1);
    wlocal.resize(n0 * n1);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, u.begin(), u.end(), &ulocal[0]);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, v.begin(), v.end(), &vlocal[0]);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, w.begin(), w.end(), &wlocal[0]);
    max_u = -100.0;
    min_u = 100.0;
    max_v = -100.0;
    min_v = 100.0;
    max_w = -100.0;
    min_w = 100.0;
    for (int i0 = 0; i0 < n0; ++i0) {
      for (int i1 = 0; i1 < n1; ++i1) {
        int idx = i0 * n1 + i1;
        max_u = std::max(max_u, ulocal[idx]);
        min_u = std::min(min_u, ulocal[idx]);
        max_v = std::max(max_v, vlocal[idx]);
        min_v = std::min(min_v, vlocal[idx]);
        max_w = std::max(max_w, wlocal[idx]);
        min_w = std::min(min_w, wlocal[idx]);
      }
    }

    // Check max and min
    EXPECT_NEAR(max_u, umaxref[n], 1e-10);
    EXPECT_NEAR(min_u, uminref[n], 1e-10);
    EXPECT_NEAR(max_v, 0.0, 1e-10);
    EXPECT_NEAR(min_v, 0.0, 1e-10);
    EXPECT_NEAR(max_w, wmaxref[n], 1e-10);
    EXPECT_NEAR(min_w, wminref[n], 1e-10);
  }

  // Delete complex pointers to allocated data
  delete[] u_modes;
  delete[] v_modes;
  delete[] w_modes;
  delete[] au_modes;
  delete[] av_modes;
  delete[] aw_modes;
  // Delete plan
  for (int n = 0; n < plan_vector.size(); ++n) {
    fftw_destroy_plan(plan_vector[n]);
  }
  plan_vector.clear();
}

} // namespace w2a_tests