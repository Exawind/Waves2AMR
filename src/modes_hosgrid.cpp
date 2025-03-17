#include "modes_hosgrid.h"
#include "cmath"
#include <iostream>

void modes_hosgrid::copy_complex(
    const int n0, const int n1,
    std::vector<std::complex<double>> complex_vector, fftw_complex *ptr) {
  for (int i = 0; i < n0; ++i) {
    for (int j = 0; j < n1 / 2 + 1; ++j) {
      const int idx = i * (n1 / 2 + 1) + j;
      ptr[idx][0] = complex_vector[idx].real();
      ptr[idx][1] = complex_vector[idx].imag();
    }
  }
}

fftw_complex *modes_hosgrid::allocate_complex(const int n0, const int n1) {
  // Allocate data needed for modes and create pointer
  fftw_complex *a_ptr = new fftw_complex[n0 * (n1 / 2 + 1)];
  // Return pointer to fftw_complex data
  return a_ptr;
}

fftw_plan
modes_hosgrid::plan_ifftw(const int n0, const int n1, fftw_complex *in,
                          const planner_flags wisdom = planner_flags::patient) {
  unsigned int flag;
  switch (wisdom) {
  case planner_flags::estimate:
    flag = FFTW_ESTIMATE;
    break;
  case planner_flags::patient:
    flag = FFTW_PATIENT;
    break;
  case planner_flags::exhaustive:
    flag = FFTW_EXHAUSTIVE;
    break;
  case planner_flags::measure:
    flag = FFTW_MEASURE;
    break;
  default:
    std::cout << "ABORT: Planner flag supplied to modes_hosgrid::plan_ifftw is "
                 "invalid or unsupported.\n";
    std::exit(1);
  }
  // Output array is used for planning (except for FFTW_ESTIMATE)
  double out[n0][n1];
  // Make and return plan
  return fftw_plan_dft_c2r_2d(n0, n1, in, &out[0][0], flag);
}

void modes_hosgrid::plan_ifftw_nwt(
    const int n0, const int n1, std::vector<fftw_plan> plan_vector, double *in,
    double *in_sin, double *in_y,
    const planner_flags wisdom = planner_flags::patient) {
  unsigned int flag;
  switch (wisdom) {
  case planner_flags::estimate:
    flag = FFTW_ESTIMATE;
    break;
  case planner_flags::patient:
    flag = FFTW_PATIENT;
    break;
  case planner_flags::exhaustive:
    flag = FFTW_EXHAUSTIVE;
    break;
  case planner_flags::measure:
    flag = FFTW_MEASURE;
    break;
  default:
    std::cout << "ABORT: Planner flag supplied to modes_hosgrid::plan_ifftw is "
                 "invalid or unsupported.\n";
    std::exit(1);
  }
  // Output array is used for planning (except for FFTW_ESTIMATE)
  double out[n0][n1];
  double out_sin[n0 - 2][n1];
  double out_y[n1];
  // Very confused on the importance of indexing here??
  if (n1 == 1) {
    // CC
    plan_vector.emplace_back(
        fftw_plan_r2r_1d(n0, in, &out[0][0], FFTW_REDFT00, flag));
    // SC
    plan_vector.emplace_back(
        fftw_plan_r2r_1d(n0 - 2, in_sin, &out_sin[0][0], FFTW_RODFT00, flag));
    // None for CS, SS, Cy, Sy

  } else {
    // CC
    plan_vector.emplace_back(fftw_plan_r2r_2d(
        n0, n1, in, &out[0][0], FFTW_REDFT00, FFTW_REDFT00, flag));
    // SC
    plan_vector.emplace_back(fftw_plan_r2r_2d(
        n0 - 2, n1, in_sin, &out_sin[0][0], FFTW_RODFT00, FFTW_REDFT00, flag));
    // CS
    plan_vector.emplace_back(fftw_plan_r2r_2d(
        n0, n1 - 2, in + n0, &out[0][1], FFTW_REDFT00, FFTW_RODFT00, flag));
    // SS
    plan_vector.emplace_back(fftw_plan_r2r_2d(n0 - 2, n1 - 2, in_sin + n0 - 2,
                                              &out_sin[0][1], FFTW_RODFT00,
                                              FFTW_RODFT00, flag));
    // Cy
    plan_vector.emplace_back(
        fftw_plan_r2r_1d(n1, in_y, &out_y[0], FFTW_REDFT00, flag));
    // Sy
    plan_vector.emplace_back(
        fftw_plan_r2r_1d(n1 - 2, in_y + 1, &out_y[1], FFTW_RODFT00, flag));
  }
}

fftw_complex *modes_hosgrid::allocate_plan_copy(
    const int n0, const int n1, fftw_plan &p,
    std::vector<std::complex<double>> complex_vector) {
  // Allocate and get pointer
  auto a_ptr = allocate_complex(n0, n1);
  // Create plan before data is initialized
  p = plan_ifftw(n0, n1, a_ptr);
  // Copy mode data from input vector
  copy_complex(n0, n1, complex_vector, a_ptr);
  // Return pointer to fftw_complex data
  return a_ptr;
}

fftw_complex *
modes_hosgrid::allocate_copy(const int n0, const int n1,
                             std::vector<std::complex<double>> complex_vector) {
  // Allocate and get pointer
  auto a_ptr = allocate_complex(n0, n1);
  // Copy mode data from input vector
  copy_complex(n0, n1, complex_vector, a_ptr);
  // Return pointer to fftw_complex data
  return a_ptr;
}

void modes_hosgrid::populate_hos_eta(
    const int n0, const int n1, const double dimL,
    std::vector<fftw_plan> p_vector, fftw_complex *eta_modes,
    amrex::Gpu::DeviceVector<amrex::Real> &HOS_eta) {

  // Get nondimensional interface height (eta)
  if (p_vector.size() == 1) {
    populate_hos_ocean_eta_nondim(n0, n1, p_vector[0], eta_modes, HOS_eta);
  } else {
    populate_hos_nwt_eta_nondim(n0, n1, p_vector, eta_modes, HOS_eta);
  }

  // Dimensionalize the interface height
  dimensionalize_eta(dimL, HOS_eta);
}

void modes_hosgrid::populate_hos_ocean_eta_nondim(
    const int n0, const int n1, fftw_plan p, fftw_complex *eta_modes,
    amrex::Gpu::DeviceVector<amrex::Real> &HOS_eta) {
  // Local array for output data
  double out[n0 * n1];
  // Perform complex-to-real (inverse) FFT
  do_ifftw(n0, n1, p, eta_modes, &out[0]);

  // Copy data to output vector
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, &out[0], &out[0] + HOS_eta.size(),
                   HOS_eta.begin());

  // !! -- This function MODIFIES the modes -- !! //
  //   .. they are not intended to be reused ..   //
}

void modes_hosgrid::populate_hos_nwt_eta_nondim(
    const int n0, const int n1, std::vector<fftw_plan> p_vector,
    fftw_complex *eta_modes, amrex::Gpu::DeviceVector<amrex::Real> &HOS_eta) {
  // Local array for output data
  double out[n0 * n1];
  // Perform complex-to-real (inverse) FFT (cos, cos)
  do_ifftw(n0, n1, true, true, p_vector, eta_modes, &out[0]);

  // Copy data to output vector
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, &out[0], &out[0] + HOS_eta.size(),
                   HOS_eta.begin());

  // !! -- This function MODIFIES the modes -- !! //
  //   .. they are not intended to be reused ..   //
}

void modes_hosgrid::dimensionalize_eta(
    const double dimL, amrex::Gpu::DeviceVector<amrex::Real> &HOS_eta) {
  // Get pointers to eta because it is on device
  auto *eta_ptr = HOS_eta.data();
  // Get size of eta for loop
  const int n2D = HOS_eta.size();
  // Multiply each eta vector in given range of indices to dimensionalize eta
  amrex::ParallelFor(n2D, [=] AMREX_GPU_DEVICE(int n) { eta_ptr[n] *= dimL; });
}

void modes_hosgrid::populate_hos_vel(
    int n0, int n1, double xlen, double ylen, double depth, double z,
    const double dimL, const double dimT,
    std::vector<std::complex<double>> mX_vector,
    std::vector<std::complex<double>> mY_vector,
    std::vector<std::complex<double>> mZ_vector,
    std::vector<fftw_plan> p_vector, fftw_complex *x_modes,
    fftw_complex *y_modes, fftw_complex *z_modes,
    amrex::Gpu::DeviceVector<amrex::Real> &HOS_u,
    amrex::Gpu::DeviceVector<amrex::Real> &HOS_v,
    amrex::Gpu::DeviceVector<amrex::Real> &HOS_w, int indv_start) {

  // Nondimensionalize lengths from AMR domain
  const amrex::Real nd_xlen = xlen / dimL;
  const amrex::Real nd_ylen = ylen / dimL;
  const amrex::Real nd_depth = depth / dimL;
  const amrex::Real nd_z = z / dimL;

  // Get nondimensional velocities
  if (p_vector.size() == 1) {
    populate_hos_ocean_vel_nondim(n0, n1, nd_xlen, nd_ylen, nd_depth, nd_z,
                                  mX_vector, mY_vector, mZ_vector, p_vector[0],
                                  x_modes, y_modes, z_modes, HOS_u, HOS_v,
                                  HOS_w, indv_start);
  } else {
    populate_hos_nwt_vel_nondim(
        n0, n1, nd_xlen, nd_ylen, nd_z, mX_vector, mY_vector, mZ_vector,
        p_vector, x_modes, y_modes, z_modes, HOS_u, HOS_v, HOS_w, indv_start);
  }

  // Dimensionalize velocities
  dimensionalize_vel(n0, n1, dimL, dimT, HOS_u, HOS_v, HOS_w, indv_start);
}

void modes_hosgrid::populate_hos_ocean_vel_nondim(
    const int n0, const int n1, const double nd_xlen, const double nd_ylen,
    const double nd_depth, const double nd_z,
    std::vector<std::complex<double>> mX_vector,
    std::vector<std::complex<double>> mY_vector,
    std::vector<std::complex<double>> mZ_vector, fftw_plan p,
    fftw_complex *x_modes, fftw_complex *y_modes, fftw_complex *z_modes,
    amrex::Gpu::DeviceVector<amrex::Real> &HOS_u,
    amrex::Gpu::DeviceVector<amrex::Real> &HOS_v,
    amrex::Gpu::DeviceVector<amrex::Real> &HOS_w, int indv_start) {
  // Everything within this routine is nondimensionalized, including xlen, ylen,
  // depth, and z as inputs and HOS_u, HOS_v, and HOS_w as outputs

  // Reused constants
  const double twoPi_xlen = 2.0 * M_PI / nd_xlen;
  const double twoPi_ylen = 2.0 * M_PI / nd_ylen;
  // Loop modes to modify them
  for (int ix = 0; ix < n0; ++ix) {
    for (int iy = 0; iy < n1 / 2 + 1; ++iy) {

      // Get wavenumbers
      const double kxN2 = (double)(ix < n0 / 2 + 1 ? ix : ix - n0) * twoPi_xlen;
      const double ky = (double)iy * twoPi_ylen;
      const double k = sqrt(kxN2 * kxN2 + ky * ky);
      // Get depth-related quantities
      const double kZ = k * (nd_z + nd_depth);
      const double kD = k * nd_depth;
      // Get coefficients
      double coeff = 1.0;
      double coeff2 = 1.0;
      if (iy == 0) {
        // Do nothing for ix = 0, iy = 0
        if (ix != 0) {
          // Modified coeffs for iy = 0, ix > 0
          if ((kZ < 50.0) && (kD <= 50.0)) {
            coeff =
                exp(k * nd_z) * (1.0 + exp(-2.0 * kZ)) / (1.0 + exp(-2.0 * kD));
            coeff2 =
                exp(k * nd_z) * (1.0 - exp(-2.0 * kZ)) / (1.0 - exp(-2.0 * kD));
          } else {
            coeff = exp(k * nd_z);
            coeff2 = coeff;
          }
          if (coeff >= 3.0) {
            coeff = 3.0;
          }
          if (coeff2 >= 3.0) {
            coeff2 = 3.0;
          }
        }
      } else {
        // Ordinary coefficients for other cases
        if ((kZ < 50.0) && (kD <= 50.0)) {
          coeff = cosh(kZ) / cosh(kD);
          coeff2 = sinh(kZ) / sinh(kD);
        } else {
          coeff = exp(k * nd_z);
          coeff2 = coeff;
        }
        if (coeff >= 1000.0) {
          coeff = 1000.0;
        }
        if (coeff2 >= 1000.0) {
          coeff2 = 1000.0;
        }
      }
      // Multiply modes by coefficients
      // hosProcedure is velocity, I think
      const int idx = ix * (n1 / 2 + 1) + iy;
      (x_modes[idx])[0] = coeff * mX_vector[idx].real();
      (x_modes[idx])[1] = coeff * mX_vector[idx].imag();
      (y_modes[idx])[0] = coeff * mY_vector[idx].real();
      (y_modes[idx])[1] = coeff * mY_vector[idx].imag();
      (z_modes[idx])[0] = coeff2 * mZ_vector[idx].real();
      (z_modes[idx])[1] = coeff2 * mZ_vector[idx].imag();
    }
  }
  // Output pointer
  const int xy_size = n0 * n1;
  amrex::Vector<amrex::Real> out(xy_size, 0.0);
  // Perform inverse fft
  do_ifftw(n0, n1, p, x_modes, out.data());
  // Copy to output vectors
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, out.begin(), out.end(),
                   &HOS_u[indv_start]);
  // Repeat in other directions
  do_ifftw(n0, n1, p, y_modes, out.data());
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, out.begin(), out.end(),
                   &HOS_v[indv_start]);
  do_ifftw(n0, n1, p, z_modes, out.data());
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, out.begin(), out.end(),
                   &HOS_w[indv_start]);
}

void modes_hosgrid::populate_hos_nwt_vel_nondim(
    const int n0, const int n1, const double nd_xlen, const double nd_ylen,
    const double nd_z, std::vector<std::complex<double>> mX_vector,
    std::vector<std::complex<double>> mY_vector,
    std::vector<std::complex<double>> mZ_vector,
    std::vector<fftw_plan> p_vector, fftw_complex *x_modes,
    fftw_complex *y_modes, fftw_complex *z_modes,
    amrex::Gpu::DeviceVector<amrex::Real> &HOS_u,
    amrex::Gpu::DeviceVector<amrex::Real> &HOS_v,
    amrex::Gpu::DeviceVector<amrex::Real> &HOS_w, int indv_start) {
  // Everything within this routine is nondimensionalized, including xlen, ylen,
  // depth, and z as inputs and HOS_u, HOS_v, and HOS_w as outputs

  // Reused constants
  const double pi_xlen = M_PI / nd_xlen;
  const double pi_ylen = M_PI / nd_ylen;
  // Loop modes to modify them
  for (int ix = 0; ix < n0; ++ix) {
    const double kx = pi_xlen * (ix - 1.);
    for (int iy = 0; iy < n1; ++iy) {

      const double ky = pi_ylen * (iy - 1.);
      // Get wavenumber and depth-related quantities
      const double k = sqrt(kx * kx + ky * ky);
      const double kZ = k * (nd_z + 1.0);
      // Get coefficients !!
      double coeff = 1.0;
      double coeff2 = 1.0;
      if (iy == 0) {
        // Do nothing for ix = 0, iy = 0
        if (ix != 0) {
          // Modified coeffs for iy = 0, ix > 0
          if ((kZ < 50.0) && (k < 50.0)) {
            coeff =
                exp(k * nd_z) * (1.0 + exp(-2.0 * kZ)) / (1.0 + exp(-2.0 * k));
            coeff2 =
                exp(k * nd_z) * (1.0 - exp(-2.0 * kZ)) / (1.0 - exp(-2.0 * k));
          } else {
            coeff = exp(k * nd_z);
            coeff2 = coeff;
          }
          if (coeff >= 3.0) {
            coeff = 3.0;
          }
          if (coeff2 >= 3.0) {
            coeff2 = 3.0;
          }
        }
      } else {
        // Ordinary coefficients for other cases
        if ((kZ < 50.0) && (k < 50.0)) {
          coeff = cosh(kZ) / cosh(k);
          coeff2 = sinh(kZ) / sinh(k);
        } else {
          coeff = exp(k * nd_z);
          coeff2 = coeff;
        }
        if (coeff >= 3.0) {
          coeff = 3.0;
        }
        if (coeff2 >= 3.0) {
          coeff2 = 3.0;
        }
      }
      // Multiply modes by coefficients
      // hosProcedure is velocity, I think
      const int idx = ix * n1 + iy;
      (x_modes[idx])[0] = coeff * mX_vector[idx].real();
      (x_modes[idx])[1] = coeff * mX_vector[idx].imag();
      (y_modes[idx])[0] = coeff * mY_vector[idx].real();
      (y_modes[idx])[1] = coeff * mY_vector[idx].imag();
      (z_modes[idx])[0] = coeff2 * mZ_vector[idx].real();
      (z_modes[idx])[1] = coeff2 * mZ_vector[idx].imag();
    }
  }

  // Output vectors
  const int xy_size = n0 * n1;
  amrex::Vector<amrex::Real> out_u(xy_size, 0.0);
  amrex::Vector<amrex::Real> out_v(xy_size, 0.0);
  amrex::Vector<amrex::Real> out_w(xy_size, 0.0);
  // Perform inverse ffts
  do_ifftw(n0, n1, false, true, p_vector, x_modes, out_u.data()); // sin, cos
  do_ifftw(n0, n1, true, false, p_vector, y_modes, out_v.data()); // cos, sin
  do_ifftw(n0, n1, true, true, p_vector, z_modes, out_w.data());  // cos, cos

  /*// Include additional modes for wavemaker
  if (n_add > 0) {
    populate_additional_hos_nwt_vel_nondim(
        n0, n1, n_add, nd_xlen, nd_ylen, nd_z, add_modes_vec, p_vector,
        ax_modes, ay_modes, az_modes, out_u, out_v, out_w);
  }*/

  // Copy to output vectors
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, out_u.begin(), out_u.end(),
                   &HOS_u[indv_start]);
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, out_v.begin(), out_v.end(),
                   &HOS_v[indv_start]);
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, out_w.begin(), out_w.end(),
                   &HOS_w[indv_start]);

  // Exception for 1D scenario?
}

void modes_hosgrid::populate_additional_hos_nwt_vel_nondim(
    const int n0, const int n1, const int n_add, const double nd_xlen,
    const double nd_ylen, const double nd_z,
    std::vector<std::complex<double>> add_modes_vec,
    std::vector<fftw_plan> p_vector, fftw_complex *x_modes,
    fftw_complex *y_modes, fftw_complex *z_modes,
    amrex::Vector<amrex::Real> &nwt_u, amrex::Vector<amrex::Real> &nwt_v,
    amrex::Vector<amrex::Real> &nwt_w) {

  // Reused constants
  constexpr int l_add = 2;
  constexpr double k_add_x_max = 700.;
  const double xlen_add = 2.0 * (double)(l_add);
  const double pi_xlen_add = M_PI / xlen_add;
  const double pi_ylen = M_PI / nd_ylen;
  // Output pointer
  amrex::Vector<amrex::Real> out_u(n1, 0.0);
  amrex::Vector<amrex::Real> out_v(n1, 0.0);
  amrex::Vector<amrex::Real> out_w(n1, 0.0);
  // Loop modes to modify them
  for (int ix = 0; ix < n0; ++ix) {
    const double nd_x = ix * (nd_xlen / (n0 - 1.));
    for (int iy = 0; iy < n1; ++iy) {
      const double ky = pi_ylen * (iy - 1.);

      // Start modes at 0 to accumulate
      (x_modes[iy])[0] = 0.;
      (x_modes[iy])[1] = 0.;
      (y_modes[iy])[0] = 0.;
      (y_modes[iy])[1] = 0.;
      (z_modes[iy])[0] = 0.;
      (z_modes[iy])[1] = 0.;

      for (int ia = 0; ia < n_add; ++ia) {

        const double kx_add = (2. * ia - 1.) * pi_xlen_add;
        const double kxy_add = sqrt(kx_add * kx_add + ky * ky);
        const double coskx_add = cos(kx_add * (nd_z + 1.));
        const double sinkx_add = sin(kx_add * (nd_z + 1.));
        const double k_add_x1 = kxy_add * nd_x;
        const double k_add_x2 = (2. * kx_add) * (nd_xlen - nd_x);

        double expon3 = 0.;
        if (2. * kxy_add * nd_xlen <= k_add_x_max) {
          expon3 = exp(-2. * kxy_add * nd_xlen);
        }
        double expon1 = 0.;
        if (k_add_x1 <= k_add_x_max) {
          expon1 = exp(-k_add_x1) / (expon3 + 1.);
        }
        double expon2 = 0.;
        if (k_add_x2 <= k_add_x_max) {
          expon2 = exp(-k_add_x2);
        }
        double expon12 = 0.;
        if ((k_add_x1 + k_add_x2) <= k_add_x_max) {
          expon12 = expon1 * expon2;
        }

        const double csh_add = expon1 + expon12;
        const double sh_add = expon1 - expon12;

        const double kx_add_csh_add_x = csh_add * kx_add;
        const double k_add_sh_add_x = sh_add * kxy_add;
        const double kycsh_add_x = csh_add * ky;

        const double coeff2 = sinkx_add * kx_add_csh_add_x;
        const double coeff3 = coskx_add * k_add_sh_add_x;
        const double coeff4 = coskx_add * kycsh_add_x;

        const int idx = iy * n_add + ia;
        (x_modes[iy])[0] -= (add_modes_vec[idx]).real() * coeff3;
        (x_modes[iy])[1] -= (add_modes_vec[idx]).imag() * coeff3;
        (y_modes[iy])[0] -= (add_modes_vec[idx]).real() * coeff4;
        (y_modes[iy])[1] -= (add_modes_vec[idx]).imag() * coeff4;
        (z_modes[iy])[0] -= (add_modes_vec[idx]).real() * coeff2;
        (z_modes[iy])[1] -= (add_modes_vec[idx]).imag() * coeff2;
      }
    }

    // Perform inverse ffts ... exception for 1D?
    do_ifftw(n1, true, p_vector, x_modes, out_u.data());  // cos
    do_ifftw(n1, false, p_vector, y_modes, out_v.data()); // sin
    do_ifftw(n1, true, p_vector, z_modes, out_w.data());  // cos
    // Add to incoming vectors, copied elsewhere
    for (int iy = 0; iy < n1; ++iy) {
      nwt_u[ix * n1 + iy] += out_u[iy];
      nwt_v[ix * n1 + iy] += out_v[iy];
      nwt_w[ix * n1 + iy] += out_w[iy];
    }
  }
}

void modes_hosgrid::dimensionalize_vel(
    const int n0, const int n1, const double dimL, const double dimT,
    amrex::Gpu::DeviceVector<amrex::Real> &HOS_u,
    amrex::Gpu::DeviceVector<amrex::Real> &HOS_v,
    amrex::Gpu::DeviceVector<amrex::Real> &HOS_w, int indv_start) {
  // Get pointers to velocity because it is on device
  auto *u_ptr = HOS_u.data();
  auto *v_ptr = HOS_v.data();
  auto *w_ptr = HOS_w.data();
  // Multiply each component of velocity vectors in given range of indices to
  // dimensionalize the velocity
  amrex::ParallelFor(n0 * n1, [=] AMREX_GPU_DEVICE(int n) {
    u_ptr[indv_start + n] *= dimL / dimT;
    v_ptr[indv_start + n] *= dimL / dimT;
    w_ptr[indv_start + n] *= dimL / dimT;
  });
}

void modes_hosgrid::do_ifftw(const int n0, const int n1, fftw_plan p,
                             fftw_complex *f_in, double *sp_out) {
  // Modify modes with conversion coefficients
  for (int ix = 0; ix < n0; ++ix) {
    for (int iy = 0; iy < n1 / 2 + 1; ++iy) {
      const int idx = ix * (n1 / 2 + 1) + iy;
      const double f2s = (iy == 0 ? 1.0 : 0.5);
      (f_in[idx])[0] *= f2s;
      (f_in[idx])[1] *= f2s;
    }
  }
  // Perform fft
  fftw_execute_dft_c2r(p, f_in, sp_out);
}

void modes_hosgrid::do_ifftw(const int n0, const int n1, const bool cos_x,
                             const bool cos_y, std::vector<fftw_plan> p_vector,
                             fftw_complex *f_in, double *sp_out) {

  // Select plan
  int iplan = 0;
  if (n1 == 1) {
    if (cos_x) {
      iplan = 0; // CC
    } else {
      iplan = 1; // SC
    }
  } else {
    if (cos_x && cos_y) {
      iplan = 0; // CC
    } else if (!cos_x && cos_y) {
      iplan = 1; // SC
    } else if (cos_x && !cos_y) {
      iplan = 2; // CS
    } else {
      iplan = 3; // SS
    }
  }

  // Modify modes with conversion coefficients
  for (int ix = 0; ix < n0; ++ix) {
    for (int iy = 0; iy < n1; ++iy) {
      const int idx = ix * n1 + iy;
      const double f2s = (iy == 0 ? 1.0 : 0.5);
      (f_in[idx])[0] *= f2s;
      (f_in[idx])[1] *= f2s;
    }
    if (n1 != 1 && !cos_y) {
      (f_in[ix * n1 + 0])[0] = 0.;
      (f_in[ix * n1 + 0])[1] = 0.;
      (f_in[ix * n1 + n1 - 1])[0] = 0.;
      (f_in[ix * n1 + n1 - 1])[1] = 0.;
    }
  }
  // Perform fft
  fftw_execute_dft_c2r(p_vector[iplan], f_in, sp_out);

  if (!cos_x) {
    for (int iy = 0; iy < n1; ++iy) {
      sp_out[0 * n1 + iy] = 0.;
      sp_out[(n0 - 1) * n1 + iy] = 0.;
    }
  }

  if (n1 != 1 && !cos_y) {
    for (int ix = 0; ix < n0; ++ix) {
      sp_out[ix * n1 + 0] = 0.;
      sp_out[ix * n1 + n1 - 1] = 0.;
    }
  }
}

void modes_hosgrid::do_ifftw(const int n1, const bool cos_y,
                             std::vector<fftw_plan> p_vector,
                             fftw_complex *f_in, double *sp_out) {
  // Select plan
  int iplan = 0;
  if (cos_y) {
    iplan = 4; // Cy
  } else {
    iplan = 5; // Sy
  }

  // Modify modes with conversion coefficients
  for (int iy = 0; iy < n1; ++iy) {
    const double f2s = ((iy == 0 || iy == n1 - 1) ? 1.0 : 0.5);
    (f_in[iy])[0] *= f2s;
    (f_in[iy])[1] *= f2s;
  }
  if (!cos_y) {
    (f_in[0])[0] = 0.;
    (f_in[0])[1] = 0.;
    (f_in[n1 - 1])[0] = 0.;
    (f_in[n1 - 1])[1] = 0.;
  }

  // Perform fft
  fftw_execute_dft_c2r(p_vector[iplan], f_in, sp_out);

  if (!cos_y) {
    sp_out[0] = 0.;
    sp_out[n1 - 1] = 0.;
  }
}