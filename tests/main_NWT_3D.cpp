#include "Waves2AMR.h"

int main(int argc, char *argv[]) {
  // Set up AMReX
  amrex::Initialize(argc, argv, true, MPI_COMM_WORLD, []() {});

  // Name of modes file
  std::string fname = "../tests/nwt_3D_modes_HOS_SWENSE.dat";
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

  // Timestep stored: t = dt
  double dt_out = rmodes.get_dtout();
  rmodes.get_data(10.0 * dt_out, mX, mY, mZ, mFS, mAdd);

  // Set up fftw_complex ptr for eta and get plan
  std::vector<fftw_plan> plan_vector{};

  double *eta_modes =
      modes_hosgrid::allocate_plan_copy(n0, n1, plan_vector, mFS);

  // Allocate ptrs for velocity as well, copy is built-in later
  auto u_modes = modes_hosgrid::allocate_real(n0, n1);
  auto v_modes = modes_hosgrid::allocate_real(n0, n1);
  auto w_modes = modes_hosgrid::allocate_real(n0, n1);
  // Additional modes
  auto au_modes = modes_hosgrid::allocate_real(n0, 1);
  auto av_modes = modes_hosgrid::allocate_real(n0, 1);
  auto aw_modes = modes_hosgrid::allocate_real(n0, 1);

  // --- Workflow for AMR-Wind --- //
  // Create heights where velocity will be sampled
  auto nheights = 40;
  const amrex::Real dz0 = 0.05;
  amrex::Vector<amrex::Real> hvec;
  int flag =
      interp_to_mfab::create_height_vector(hvec, nheights, dz0, 0.0, -depth);
  // Fail if flag indicates it should
  if (flag > 0) {
    amrex::Abort("create_height_vector error, failure code " +
                 std::to_string(flag));
  }
  // Create representation of entire AMR-Wind mesh
  // Physical bounds of domain
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> problo{0., 0., -3.6};
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> probhi{28., 4., 1.4};
  amrex::RealBox rbox(problo.data(), probhi.data());
  // Domain boxes for each level
  int nxg0 = 64;
  int nyg0 = 16;
  int nzg0 = 16;
  amrex::Box domainbox0(amrex::IntVect{0, 0, 0},
                        amrex::IntVect{nxg0 - 1, nyg0 - 1, nzg0 - 1});
  amrex::Box domainbox1(
      amrex::IntVect{0, 0, 0},
      amrex::IntVect{2 * nxg0 - 1, 2 * nyg0 - 1, 2 * nzg0 - 1});
  // Geometry objects for each level
  amrex::Geometry geom0(domainbox0, &rbox);
  amrex::Geometry geom1(domainbox1, &rbox);
  // Geometry vector
  amrex::Vector<amrex::Geometry> geom_all{geom0, geom1};

  // Create vector of multifab to represent part of AMR-Wind mesh
  // This part of the mesh is what current processor has access to
  int nx_box = 8;
  amrex::Box localbox0(amrex::IntVect{0, 0, nx_box},
                       amrex::IntVect{nx_box - 1, nx_box - 1, 2 * nx_box - 1});
  amrex::BoxArray ba0(localbox0);
  amrex::DistributionMapping dm0{ba0};
  amrex::Box localbox1(
      amrex::IntVect{0, 0, 2 * nx_box},
      amrex::IntVect{2 * nx_box - 1, 2 * nx_box - 1, 2 * 2 * nx_box - 1});
  std::cout << "nz for boxes " << nx_box << " " << 2 * nx_box - 1 << " "
            << 2 * 2 * nx_box - 1 << std::endl;
  amrex::BoxArray ba1(localbox1);
  amrex::DistributionMapping dm1{ba1};
  const int ncomp = 3;
  const int nghost = 3;
  // Create multifabs and vector version of fields
  amrex::MultiFab mf_ls0(ba0, dm0, 1, nghost);
  amrex::MultiFab mf_ls1(ba1, dm1, 1, nghost);
  amrex::MultiFab mf_v0(ba0, dm0, ncomp, nghost);
  amrex::MultiFab mf_v1(ba1, dm1, ncomp, nghost);
  amrex::Vector<amrex::MultiFab *> phi_field{&mf_ls0, &mf_ls1};
  amrex::Vector<amrex::MultiFab *> velocity_field{&mf_v0, &mf_v1};

  // Get indices of heights that overlap
  amrex::Vector<int> indvec;
  flag = interp_to_mfab::get_local_height_indices(indvec, hvec, velocity_field,
                                                  geom_all);
  // Flag = 0 indicates that there are no overlapping points
  if (flag == 0) {
    amrex::Abort(
        "get_local_height_indices: no valid points between MF and hvec");
  }

  // Perform fftw for eta
  amrex::Gpu::DeviceVector<amrex::Real> hos_eta_vec(n0 * n1, 0.0);
  modes_hosgrid::populate_hos_eta(n0, n1, dimL, plan_vector, eta_modes,
                                  hos_eta_vec);

  // Loop through heights to check and print
  int n_hvec = 0;
  int n_mfab = 2 * nx_box + nghost - 1;
  int n_ivec = 0;
  std::cout << std::endl << "Heights, descending order: \n";
  for (int n = 0; n < 2 * nx_box + 2 * nghost + hvec.size(); ++n) {
    const auto ploz1 = geom1.ProbLo(2);
    const auto dz1 = geom1.CellSize(2);
    const amrex::Real h_mfab = ploz1 + dz1 * (2 * 3 * nx_box + n_mfab);
    if (hvec[n_hvec] > h_mfab || n_mfab < -nghost) {
      std::cout << "hvec " << n_hvec << " " << hvec[n_hvec];
      if (n_ivec < indvec.size() && indvec[n_ivec] == n_hvec) {
        std::cout << "          ivec " << n_ivec;
        ++n_ivec;
      }
      std::cout << std::endl;
      ++n_hvec;
    } else {
      std::cout << "mfab " << n_mfab << "            " << h_mfab << std::endl;
      --n_mfab;
    }
  }

  // Create vector of velocities to sample
  amrex::Gpu::DeviceVector<amrex::Real> hos_u_vec;
  amrex::Gpu::DeviceVector<amrex::Real> hos_v_vec;
  amrex::Gpu::DeviceVector<amrex::Real> hos_w_vec;

  // Resize 1D velocity vectors
  int nht = indvec.size();
  hos_u_vec.resize(n0 * n1 * nht);
  hos_v_vec.resize(n0 * n1 * nht);
  hos_w_vec.resize(n0 * n1 * nht);
  // Sample velocities
  int indv = 0;
  for (int iht = 0; iht < nht; ++iht) {
    // Get sample height
    amrex::Real ht = hvec[indvec[iht]];
    // Sample velocity
    modes_hosgrid::populate_hos_vel(n0, n1, n2, xlen, ylen, ht, dimL, dimT, mX,
                                    mY, mZ, mAdd, plan_vector, u_modes, v_modes,
                                    w_modes, au_modes, av_modes, aw_modes,
                                    hos_u_vec, hos_v_vec, hos_w_vec, indv);
    indv += n0 * n1;
  }

  // Interpolate to multifab
  const amrex::Real spd_dx = xlen / n0_sp;
  const amrex::Real spd_dy = ylen / n1_sp;
  const amrex::Real zero_sea_level = 0.0;
  interp_to_mfab::interp_eta_to_levelset_field(
      n0_sp, n1_sp, spd_dx, spd_dy, 0.0, 0.0, zero_sea_level, true, hos_eta_vec,
      phi_field, geom_all);
  interp_to_mfab::interp_velocity_to_field(
      n0_sp, n1_sp, spd_dx, spd_dy, 0.0, 0.0, true, indvec, hvec, hos_u_vec,
      hos_v_vec, hos_w_vec, velocity_field, geom_all);

  // Try to get next timestep (not available)
  bool readflag = rmodes.get_data(20.0 * dt_out, mX, mY, mZ, mFS, mAdd);
  if (!readflag) {
    amrex::Print() << "Data at time " << 2.0 * dt_out
                   << " not available, looping back to initial time " << 0
                   << ".\n";
    readflag = rmodes.get_data(0.0, mX, mY, mZ, mFS, mAdd);
    if (readflag) {
      amrex::Print() << "Data read at time " << 0 << " successful.\n";
    }
  }

  // Delete ptrs and plan
  delete[] eta_modes;
  delete[] u_modes;
  delete[] v_modes;
  delete[] w_modes;
  for (int n = 0; n < plan_vector.size(); ++n) {
    fftw_destroy_plan(plan_vector[n]);
  }
  plan_vector.clear();

  // Finalize AMReX
  amrex::Finalize();
}