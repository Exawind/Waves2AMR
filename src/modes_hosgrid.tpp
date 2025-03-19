// Uses ReadModes object directly instead of of separate variables
template <typename VT>
void modes_hosgrid::populate_hos_vel(
    ReadModes<VT> rm_obj, const double z, std::vector<VT> mX_vector,
    std::vector<VT> mY_vector, std::vector<VT> mZ_vector,
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

template <typename VT>
void modes_hosgrid::populate_hos_vel(
    ReadModes<VT> rm_obj, const double z, std::vector<VT> mX_vector,
    std::vector<VT> mY_vector, std::vector<VT> mZ_vector,
    std::vector<VT> mAdd_vector, std::vector<fftw_plan> p_vector,
    double *x_modes, double *y_modes, double *z_modes, double *add_x_modes,
    double *add_y_modes, double *add_z_modes,
    amrex::Gpu::DeviceVector<amrex::Real> &HOS_u,
    amrex::Gpu::DeviceVector<amrex::Real> &HOS_v,
    amrex::Gpu::DeviceVector<amrex::Real> &HOS_w, int indv_start) {

  // Get nondimensional velocities
  populate_hos_vel(rm_obj.get_first_dimension(), rm_obj.get_second_dimension(),
                   rm_obj.get_third_dimension(), rm_obj.get_xlen(),
                   rm_obj.get_ylen(), z, rm_obj.get_L(), rm_obj.get_T(),
                   mX_vector, mY_vector, mZ_vector, mAdd_vector, p_vector,
                   x_modes, y_modes, z_modes, add_x_modes, add_y_modes,
                   add_z_modes, HOS_u, HOS_v, HOS_w, indv_start);
}

// Uses ReadModes object directly instead of of separate variables
template <typename VT>
void modes_hosgrid::populate_hos_eta(
    ReadModes<VT> rm_obj, std::vector<fftw_plan> p_vector,
    fftw_complex *eta_modes, amrex::Gpu::DeviceVector<amrex::Real> &HOS_eta) {

  // Pass parameters to function via object calls
  populate_hos_eta(rm_obj.get_first_dimension(), rm_obj.get_second_dimension(),
                   rm_obj.get_L(), p_vector, eta_modes, HOS_eta);
}