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

// Uses ReadModes object directly instead of of separate variables
void modes_hosgrid::populate_hos_eta(
    ReadModes rm_obj, std::vector<fftw_plan> p_vector, fftw_complex *eta_modes,
    amrex::Gpu::DeviceVector<amrex::Real> &HOS_eta) {

  // Pass parameters to function via object calls
  populate_hos_eta(rm_obj.get_first_dimension(), rm_obj.get_second_dimension(),
                   rm_obj.get_L(), p_vector, eta_modes, HOS_eta);
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

// Uses ReadModes object directly instead of of separate variables
void modes_hosgrid::populate_hos_vel(
    ReadModes rm_obj, const double z,
    std::vector<std::complex<double>> mX_vector,
    std::vector<std::complex<double>> mY_vector,
    std::vector<std::complex<double>> mZ_vector,
    std::vector<fftw_plan> p_vector, fftw_complex *x_modes,
    fftw_complex *y_modes, fftw_complex *z_modes,
    amrex::Gpu::DeviceVector<amrex::Real> &HOS_u,
    amrex::Gpu::DeviceVector<amrex::Real> &HOS_v,
    amrex::Gpu::DeviceVector<amrex::Real> &HOS_w, int indv_start) {

  // Get nondimensional velocities
  populate_hos_vel(rm_obj.get_first_dimension(), rm_obj.get_second_dimension(),
                   rm_obj.get_xlen(), rm_obj.get_ylen(), rm_obj.get_depth(), z,
                   rm_obj.get_L(), rm_obj.get_T(), mX_vector, mY_vector,
                   mZ_vector, p_vector, x_modes, y_modes, z_modes, HOS_u, HOS_v,
                   HOS_w, indv_start);
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
  // Need to prep modes according to sin, cos rules

  // Output pointer
  const int xy_size = n0 * n1;
  amrex::Vector<amrex::Real> out(xy_size, 0.0);
  // Perform inverse fft
  do_ifftw(n0, n1, false, true, p_vector, x_modes, out.data()); // sin, cos
  // Copy to output vectors
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, out.begin(), out.end(),
                   &HOS_u[indv_start]);
  // Repeat in other directions (exception for 1D dara)
  do_ifftw(n0, n1, true, false, p_vector, y_modes, out.data()); // cos, sin
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, out.begin(), out.end(),
                   &HOS_v[indv_start]);
  do_ifftw(n0, n1, true, true, p_vector, z_modes, out.data()); // cos, cos
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, out.begin(), out.end(),
                   &HOS_w[indv_start]);
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
