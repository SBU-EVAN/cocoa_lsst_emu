#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cmath>
#include <stdexcept>
#include <array>
#include <random>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/cfg/env.h>

#include <boost/algorithm/string.hpp>

// Python Binding
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include "cosmolike/basics.h"
#include "cosmolike/bias.h"
#include "cosmolike/baryons.h"
#include "cosmolike/cosmo2D.h"
#include "cosmolike/cosmo3D.h"
#include "cosmolike/halo.h"
#include "cosmolike/radial_weights.h"
#include "cosmolike/recompute.h"
#include "cosmolike/pt_cfastpt.h"
#include "cosmolike/redshift_spline.h"
#include "cosmolike/structs.h"

#include "interface.hpp"

namespace ima = interface_mpp_aux;

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// init functions
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

void cpp_initial_setup()
{
  spdlog::cfg::load_env_levels();
  spdlog::debug("\x1b[90m{}\x1b[0m: Begins", "initial_setup");

  // restart variables to 0 so error check can flag bad initialization
  tomo.shear_Nbin = 0;
  tomo.clustering_Nbin = 0;

  like.shear_shear = 0;
  like.shear_pos = 0;
  like.pos_pos = 0;

  // bias
  gbias.b1_function = &b1_per_bin;

  // no priors
  like.clusterN = 0;
  like.clusterWL = 0;
  like.clusterCG = 0;
  like.clusterCC = 0;

  // reset bias
  for (int i = 0; i < MAX_SIZE_ARRAYS; i++)
  {
    gbias.b[i] = 0.0;
    gbias.b2[i] = 0.0;
    gbias.b_mag[i] = 0.0;
  }

  // reset IA
  for (int i = 0; i < MAX_SIZE_ARRAYS; i++)
  {
    nuisance.A_z[i] = 0.0;
    nuisance.A2_z[i] = 0.0;
    nuisance.b_ta_z[i] = 0.0;
  }

  like.high_def_integration = 1;

  spdlog::debug("\x1b[90m{}\x1b[0m: Ends", "initial_setup");
}

void cpp_init_accuracy_boost(const double accuracy_boost, const double sampling_boost,
const int integration_accuracy)
{
  const double from_desy3_to_lsst_acc = 2;

  Ntable.N_a = static_cast<int>(ceil(Ntable.N_a*accuracy_boost));
  Ntable.N_ell_TATT = static_cast<int>(ceil(Ntable.N_ell_TATT*accuracy_boost));
  Ntable.N_ell_TATT = static_cast<int>(ceil(Ntable.N_ell_TATT*from_desy3_to_lsst_acc));

  Ntable.N_k_lin = static_cast<int>(ceil(Ntable.N_k_lin*sampling_boost));
  Ntable.N_k_nlin = static_cast<int>(ceil(Ntable.N_k_nlin*sampling_boost));
  Ntable.N_ell = static_cast<int>(ceil(Ntable.N_ell*sampling_boost));
  
  Ntable.N_theta  = static_cast<int>(ceil(Ntable.N_theta*sampling_boost));

  Ntable.N_M = static_cast<int>(ceil(Ntable.N_M*sampling_boost));

  precision.low /= accuracy_boost;
  precision.medium /= accuracy_boost;
  precision.high /= accuracy_boost;
  precision.insane /= accuracy_boost; 
  
  like.high_def_integration = integration_accuracy;
}


void cpp_init_probes(std::string possible_probes)
{
  spdlog::debug("\x1b[90m{}\x1b[0m: Begins", "init_probes");

  if (possible_probes.compare("xi") == 0)
  { // cosmolike c interface
    like.shear_shear = 1;

    spdlog::debug("\x1b[90m{}\x1b[0m: {} = {} selected", "init_probes",
      "possible_probes", "xi");
  }
  else if (possible_probes.compare("wtheta") == 0)
  {
    like.pos_pos = 1;

    spdlog::debug("\x1b[90m{}\x1b[0m: {} = {} selected", "init_probes",
      "possible_probes", "wtheta");
  }
  else if (possible_probes.compare("gammat") == 0)
  {
    like.shear_pos = 1;

    spdlog::debug("\x1b[90m{}\x1b[0m: {} = {} selected", "init_probes",
      "possible_probes", "gammat");
  }
  else if (possible_probes.compare("2x2pt") == 0)
  {
    like.shear_pos = 1;
    like.pos_pos = 1;

    spdlog::debug("\x1b[90m{}\x1b[0m: {} = {} selected", "init_probes",
      "possible_probes", "2x2pt");
  }
  else if (possible_probes.compare("3x2pt") == 0)
  {
    like.shear_shear = 1;
    like.shear_pos = 1;
    like.pos_pos = 1;

    spdlog::debug("\x1b[90m{}\x1b[0m: {} = {} selected", "init_probes",
      "possible_probes", "3x2pt");
  }
  else if (possible_probes.compare("xi_ggl") == 0)
  {
    like.shear_shear = 1;
    like.shear_pos = 1;

    spdlog::debug("\x1b[90m{}\x1b[0m: {} = {} selected", "init_probes",
      "possible_probes", "xi + ggl (2x2pt)");
  }
  else
  {
    spdlog::critical("\x1b[90m{}\x1b[0m: {} = {} probe not supported",
      "init_probes", "possible_probes", possible_probes);
    exit(1);
  }

  spdlog::debug("\x1b[90m{}\x1b[0m: Ends", "init_probes");
}

void cpp_init_survey(std::string surveyname, double area, double sigma_e)
{
  spdlog::debug("\x1b[90m{}\x1b[0m: Begins", "init_survey");

  if (surveyname.size() > CHAR_MAX_SIZE - 1)
  {
    exit(1);
  }
  if (!(surveyname.size()>0))
  {
    spdlog::critical("{}: incompatible input", "init_survey");
    exit(1);
  }

  memcpy(survey.name, surveyname.c_str(), surveyname.size() + 1);
  survey.area = area;
  survey.sigma_e = sigma_e;

  spdlog::debug("\x1b[90m{}\x1b[0m: Ends", "init_survey");
}

void cpp_init_cosmo_runmode(const bool is_linear)
{
  spdlog::debug("\x1b[90m{}\x1b[0m: Begins", "init_cosmo_runmode");

  std::string mode = is_linear ? "linear" : "Halofit";
  const size_t size = mode.size();
  memcpy(pdeltaparams.runmode, mode.c_str(), size + 1);

  spdlog::debug("\x1b[90m{}\x1b[0m: {} = {} selected",
    "init_cosmo_runmode", "runmode", mode);

  spdlog::debug("\x1b[90m{}\x1b[0m: Ends", "init_cosmo_runmode");
}

void cpp_init_IA(int N)
{
  spdlog::debug("\x1b[90m{}\x1b[0m: Begins", "init_IA");

  spdlog::debug("\x1b[90m{}\x1b[0m: {} = {} selected.", "init_IA", "IA", N);

  if (N == 3 || N == 4 || N == 5 || N == 6)
  {
    like.IA = N;
  }
  else
  {
    spdlog::critical("{}: {} = {} not supported", "init_IA", "like.IA", N);
    exit(1);
  }

  spdlog::debug("\x1b[90m{}\x1b[0m: Ends", "init_IA");
}

void cpp_init_baryons_contamination(
const bool use_baryonic_simulations_contamination,
const std::string which_baryonic_simulations_contamination)
{
  spdlog::debug("\x1b[90m{}\x1b[0m: Begins", "init_baryons_contamination");

  spdlog::info("\x1b[90m{}\x1b[0m: {} = {} selected",
    "init_baryons_contamination", "use_baryonic_simulations",
    use_baryonic_simulations_contamination);

  if (use_baryonic_simulations_contamination)
  {
    init_baryons(which_baryonic_simulations_contamination.c_str());

    spdlog::info("\x1b[90m{}\x1b[0m: {} = {} selected",
      "init_baryons_contamination", "which_baryonic_simulations_contamination",
      which_baryonic_simulations_contamination);
  }
  else
  {
    reset_bary_struct();
  }

  spdlog::debug("\x1b[90m{}\x1b[0m: Ends", "init_baryons_contamination");
}

void cpp_init_binning(const int Ntheta, const double theta_min_arcmin,
const double theta_max_arcmin)
{
  spdlog::debug("\x1b[90m{}\x1b[0m: Begins", "init_binning");

  if (!(Ntheta > 0))
  {
    spdlog::critical("\x1b[90m{}\x1b[0m: {} = {} not supported", "init_binning",
      "like.Ntheta", Ntheta);
    exit(1);
  }

  spdlog::debug("\x1b[90m{}\x1b[0m: {} = {} selected.",
    "init_binning", "Ntheta", Ntheta);

  spdlog::debug("\x1b[90m{}\x1b[0m: {} = {} selected.",
    "init_binning", "theta_min_arcmin", theta_min_arcmin);

  spdlog::debug("\x1b[90m{}\x1b[0m: {} = {} selected.",
    "init_binning", "theta_max_arcmin", theta_max_arcmin);

  like.Ntheta = Ntheta;
  like.vtmin = theta_min_arcmin * 2.90888208665721580e-4;
  like.vtmax = theta_max_arcmin * 2.90888208665721580e-4;
  const double logdt = (std::log(like.vtmax)-std::log(like.vtmin))/like.Ntheta;
  like.theta = (double*) calloc(like.Ntheta, sizeof(double));

  constexpr double x = 2./ 3.;

  for (int i = 0; i < like.Ntheta; i++)
  {
    const double thetamin = std::exp(log(like.vtmin) + (i + 0.0) * logdt);
    const double thetamax = std::exp(log(like.vtmin) + (i + 1.0) * logdt);
    like.theta[i] = x * (std::pow(thetamax, 3) - std::pow(thetamin, 3)) /
      (thetamax*thetamax - thetamin*thetamin);

    spdlog::debug(
      "\x1b[90m{}\x1b[0m: Bin {:d} - {} = {:.4e}, {} = {:.4e} and {} = {:.4e}",
      "init_binning", i, "theta_min [rad]", thetamin, "theta [rad]",
      like.theta[i], "theta_max [rad]", thetamax);
  }

  spdlog::debug("\x1b[90m{}\x1b[0m: Ends", "init_binning");
}

void cpp_init_lens_sample(std::string multihisto_file, const int Ntomo, const double ggl_cut)
{
  spdlog::debug("\x1b[90m{}\x1b[0m: Begins", "init_lens_sample");

  if (tomo.shear_Nbin == 0)
  {
    spdlog::critical("{}: {} not set prior to this function call",
      "init_lens_sample", "tomo.shear_Nbin");
    exit(1);
  }
  if (multihisto_file.size()>CHAR_MAX_SIZE-1)
  {
    spdlog::critical(
      "\x1b[90m{}\x1b[0m: insufficient pre-allocated char memory (max = {}) for"
      "the string: {}", "init_lens_sample", CHAR_MAX_SIZE-1, multihisto_file);
    exit(1);
  }
  if (!(multihisto_file.size() > 0))
  {
    spdlog::critical("\x1b[90m{}\x1b[0m: empty {} string not supported",
      "init_lens_sample", "multihisto_file");
    exit(1);
  }
  if (!(Ntomo > 0) || Ntomo > MAX_SIZE_ARRAYS)
  {
    spdlog::critical("\x1b[90m{}\x1b[0m: {} = {} not supported (max = {})",
      "init_lens_sample", "Ntomo", Ntomo, MAX_SIZE_ARRAYS);
    exit(1);
  }

  memcpy(redshift.clustering_REDSHIFT_FILE, multihisto_file.c_str(), multihisto_file.size()+1);

  redshift.clustering_photoz = 4;
  tomo.clustering_Nbin = Ntomo;
  tomo.clustering_Npowerspectra = tomo.clustering_Nbin;

  spdlog::debug("\x1b[90m{}\x1b[0m: {} = {} selected.", "init_lens_sample",
    "clustering_REDSHIFT_FILE", multihisto_file);

  spdlog::debug("\x1b[90m{}\x1b[0m: {} = {} selected.", "init_lens_sample",
    "clustering_Nbin", Ntomo);

  if (ggl_cut > 0)
  {
    survey.ggl_overlap_cut = ggl_cut;
  }
  else
  {
    survey.ggl_overlap_cut = 0.0;
  }

  spdlog::debug("\x1b[90m{}\x1b[0m: {} = {} selected.", "init_lens_sample",
    "survey.ggl_overlap_cut", survey.ggl_overlap_cut);

  pf_photoz(0.1, 0);
  {
    int n = 0;
    for (int i = 0; i < tomo.clustering_Nbin; i++)
    {
      for (int j = 0; j < tomo.shear_Nbin; j++)
      {
        n += test_zoverlap(i, j);
      }
    }
    tomo.ggl_Npowerspectra = n;

    spdlog::debug("\x1b[90m{}\x1b[0m: tomo.ggl_Npowerspectra = {}",
      "init_lens_sample", tomo.ggl_Npowerspectra);
  }
  spdlog::debug("\x1b[90m{}\x1b[0m: Ends", "init_lens_sample");
}

void cpp_init_source_sample(std::string multihisto_file, const int Ntomo)
{
  spdlog::debug("\x1b[90m{}\x1b[0m: Begins", "init_source_sample");

  if (multihisto_file.size() > CHAR_MAX_SIZE - 1)
  {
    spdlog::critical(
      "\x1b[90m{}\x1b[0m: insufficient pre-allocated char memory (max = {}) for"
      "the string: {}", "init_source_sample", CHAR_MAX_SIZE-1, multihisto_file);
    exit(1);
  }
  if (!(multihisto_file.size() > 0))
  {
    spdlog::critical("\x1b[90m{}\x1b[0m: empty {} string not supported",
      "init_source_sample", "multihisto_file");
    exit(1);
  }
  if (!(Ntomo > 0) || Ntomo > MAX_SIZE_ARRAYS)
  {
    spdlog::critical("\x1b[90m{}\x1b[0m: {} = {} not supported (max = {})",
      "init_source_sample", "Ntomo", Ntomo, MAX_SIZE_ARRAYS);
    exit(1);
  }

  // convert std::string to char*
  memcpy(redshift.shear_REDSHIFT_FILE, multihisto_file.c_str(), multihisto_file.size() + 1);

  redshift.shear_photoz = 4;
  tomo.shear_Nbin = Ntomo;
  tomo.shear_Npowerspectra = tomo.shear_Nbin * (tomo.shear_Nbin + 1) / 2;

  spdlog::debug("\x1b[90m{}\x1b[0m: tomo.shear_Npowerspectra = {}", 
    "init_source_sample", tomo.shear_Npowerspectra);

  for (int i=0; i<tomo.shear_Nbin; i++)
  {
    nuisance.bias_zphot_shear[i] = 0.0;

    spdlog::info("\x1b[90m{}\x1b[0m: bin {} - {} = {}.",
      "init_source_sample", i, "<z_s>", zmean_source(i));
  }

  spdlog::debug("\x1b[90m{}\x1b[0m: {} = {} selected.", "init_source_sample",
    "shear_REDSHIFT_FILE", multihisto_file);

  spdlog::debug("\x1b[90m{}\x1b[0m: {} = {} selected.", "init_source_sample",
    "shear_Nbin", Ntomo);

  spdlog::debug("\x1b[90m{}\x1b[0m: Ends", "init_source_sample");
}

void cpp_init_size_data_vector()
{
  spdlog::debug("\x1b[90m{}\x1b[0m: Begins", "init_size_data_vector");

  if (tomo.shear_Nbin == 0)
  {
    spdlog::critical("{}: {} not set prior to this function call",
      "init_size_data_vector", "tomo.shear_Nbin");
    exit(1);
  }
  if (tomo.clustering_Nbin == 0)
  {
    spdlog::critical("{}: {} not set prior to this function call",
      "init_size_data_vector", "tomo.clustering_Nbin");
    exit(1);
  }
  if (like.Ntheta == 0)
  {
    spdlog::critical("{}: {} not set prior to this function call",
      "init_size_data_vector", "like.Ntheta");
    exit(1);
  }

  like.Ndata = like.Ntheta*(2*tomo.shear_Npowerspectra +
                        tomo.ggl_Npowerspectra + tomo.clustering_Npowerspectra);

  spdlog::debug("\x1b[90m{}\x1b[0m: {} = {} selected.",
    "init_size_data_vector", "Ndata", like.Ndata);

  spdlog::debug("\x1b[90m{}\x1b[0m: Ends", "init_size_data_vector");
}

void cpp_init_linear_power_spectrum(std::vector<double> io_log10k,
std::vector<double> io_z, std::vector<double> io_lnP)
{
  spdlog::debug("\x1b[90m{}\x1b[0m: Begins", "init_linear_power_spectrum");

  {
    bool debug_fail = false;
    if (io_z.size()*io_log10k.size() != io_lnP.size())
    {
      debug_fail = true;
    }
    else
    {
      if (io_z.size() == 0 || io_log10k.size() == 0)
      {
        debug_fail = true;
      }
    }
    if (debug_fail)
    {
      spdlog::critical(
        "\x1b[90m{}\x1b[0m: incompatible input w/ k.size = {}, z.size = {}, "
        "and lnP.size = {}", "init_linear_power_spectrum", io_log10k.size(),
        io_z.size(), io_lnP.size());
      exit(1);
    }

    if(io_z.size() < 5 || io_log10k.size() < 5)
    {
      spdlog::critical(
        "\x1b[90m{}\x1b[0m: bad input w/ k.size = {}, z.size = {}, "
        "and lnP.size = {}", "init_linear_power_spectrum", io_log10k.size(),
        io_z.size(), io_lnP.size());
      exit(1);
    }
  }

  int nlog10k = static_cast<int>(io_log10k.size());
  int nz = static_cast<int>(io_z.size());
  double* log10k = io_log10k.data();
  double* z = io_z.data();
  double* lnP = io_lnP.data();
  setup_p_lin(&nlog10k, &nz, &log10k, &z, &lnP, 1);

  // force initialization - imp to avoid seg fault when openmp is on
  const double io_a = 1.0;
  const double io_k = 0.1*cosmology.coverH0;
  p_lin(io_k, io_a);

  spdlog::debug("\x1b[90m{}\x1b[0m: Ends", "init_linear_power_spectrum");

  return;
}

void cpp_init_non_linear_power_spectrum(std::vector<double> io_log10k,
std::vector<double> io_z, std::vector<double> io_lnP)
{
  spdlog::debug("\x1b[90m{}\x1b[0m: Begins", "init_non_linear_power_spectrum");

  {
    bool debug_fail = false;
    if (io_z.size()*io_log10k.size() != io_lnP.size())
    {
      debug_fail = true;
    }
    else
    {
      if (io_z.size() == 0)
      {
        debug_fail = true;
      }
    }
    if (debug_fail)
    {
      spdlog::critical(
        "\x1b[90m{}\x1b[0m: incompatible input w/ k.size = {}, z.size = {}, "
        "and lnP.size = {}", "init_non_linear_power_spectrum", io_log10k.size(),
        io_z.size(), io_lnP.size());
      exit(1);
    }

    if(io_z.size() < 5 || io_log10k.size() < 5)
    {
      spdlog::critical(
        "\x1b[90m{}\x1b[0m: bad input w/ k.size = {}, z.size = {}, "
        "and lnP.size = {}", "init_non_linear_power_spectrum", io_log10k.size(),
        io_z.size(), io_lnP.size());
      exit(1);
    }
  }

  int nlog10k = static_cast<int>(io_log10k.size());
  int nz = static_cast<int>(io_z.size());
  double* log10k = io_log10k.data();
  double* z = io_z.data();
  double* lnP = io_lnP.data();
  setup_p_nonlin(&nlog10k, &nz, &log10k, &z, &lnP, 1);

  // force initialization - imp to avoid seg fault when openmp is on
  const double io_a = 1.0;
  const double io_k = 0.1*cosmology.coverH0;
  p_nonlin(io_k, io_a);

  spdlog::debug("\x1b[90m{}\x1b[0m: Ends", "init_non_linear_power_spectrum");

  return;
}

// Growth: D = G * a
void cpp_init_growth(std::vector<double> io_z, std::vector<double> io_G)
{
  spdlog::debug("\x1b[90m{}\x1b[0m: Begins", "init_growth");

  {
    bool debug_fail = false;
    if (io_z.size() != io_G.size())
    {
      debug_fail = true;
    }
    else
    {
      if (io_z.size() == 0)
      {
        debug_fail = true;
      }
    }
    if (debug_fail)
    {
      spdlog::critical("\x1b[90m{}\x1b[0m: incompatible input w/ z.size = {} and G.size = {}",
        "init_growth", io_z.size(), io_G.size());
      exit(1);
    }
  }

  int nz = static_cast<int>(io_z.size());
  double* z = io_z.data();
  double* G = io_G.data();
  setup_growth(&nz, &z, &G, 1);

  // force initialization - imp to avoid seg fault when openmp is on
  const double io_a = 1.0;
  const double zz = 0.0;
  f_growth(zz);
  growfac_all(io_a);
  growfac(io_a);

  spdlog::debug("\x1b[90m{}\x1b[0m: Ends", "init_growth");

  return;
}

void cpp_init_distances(std::vector<double> io_z, std::vector<double> io_chi)
{
  spdlog::debug("\x1b[90m{}\x1b[0m: Begins", "init_distances");

  {
    bool debug_fail = false;
    if (io_z.size() != io_chi.size())
    {
      debug_fail = true;
    }
    else
    {
      if (io_z.size() == 0)
      {
        debug_fail = true;
      }
    }
    if (debug_fail)
    {
      spdlog::critical(
        "\x1b[90m{}\x1b[0m: incompatible input w/ z.size = {} and G.size = {}",
        "init_distances",
        io_z.size(),
        io_chi.size()
      );
      exit(1);
    }
  }

  int nz = static_cast<int>(io_z.size());
  double* vz = io_z.data();
  double* vchi = io_chi.data();
  setup_chi(&nz, &vz, &vchi, 1);

  // force initialization - imp to avoid seg fault when openmp is on
  const double io_a = 1.0;
  chi(io_a);

  spdlog::debug("\x1b[90m{}\x1b[0m: Ends", "init_distances");

  return;
}

void cpp_init_data_real(std::string COV, std::string MASK, std::string DATA)
{
  spdlog::debug("\x1b[90m{}\x1b[0m: Begins", "init_data_real");

  ima::RealData& instance = ima::RealData::get_instance();

  instance.set_mask(MASK); // set_mask must be called first
  instance.set_data(DATA);
  instance.set_inv_cov(COV);

  spdlog::debug("\x1b[90m{}\x1b[0m: Ends", "init_data_real");

  return;
}

void cpp_init_baryon_pca_scenarios(std::string scenarios)
{
  spdlog::debug("\x1b[90m{}\x1b[0m: Begins", "init_baryon_pca_scenarios");

  ima::BaryonScenario& instance = ima::BaryonScenario::get_instance();

  instance.set_scenarios(scenarios);

  spdlog::debug("\x1b[90m{}\x1b[0m: Ends", "init_baryon_pca_scenarios");

  return;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// SET PARAM FUNCTIONS
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

void cpp_set_cosmological_parameters(const double omega_matter,
const double hubble, const bool is_cached_cosmology)
{
  spdlog::debug("\x1b[90m{}\x1b[0m: Begins", "set_cosmological_parameters");

  if(!is_cached_cosmology)
  {
    // Cosmolike should not need parameters from inflation or dark energy.
    // because Cobaya provides P(k,z), H(z), D(z), Chi(z)...
    // It may require H0 to set scales and \Omega_M to set the halo model

    // cosmolike c interface
    cosmology.Omega_m = omega_matter;
    cosmology.Omega_v = 1.0-omega_matter;
    // Cosmolike only needs to know that there are massive neutrinos (>0)
    cosmology.Omega_nu = 0.1;
    cosmology.h0 = hubble/100.0; // assuming H0 in km/s/Mpc
    cosmology.MGSigma = 0.0;
    cosmology.MGmu = 0.0;

    // Technical Problem: we want Cosmolike to calculate the data vector when
    // Cobaya request (no cache). To avoid cache in Cosmolike, we use a
    // random number generators to set cosmology.random
    cosmology.random = ima::RandomNumber::get_instance().get();
    cosmology.is_cached = 0;
  }
  else
  {
    cosmology.is_cached = 1;
  }

  spdlog::debug("\x1b[90m{}\x1b[0m: Ends", "set_cosmological_parameters");
}

void cpp_set_nuisance_shear_calib(std::vector<double> M)
{
  spdlog::debug("\x1b[90m{}\x1b[0m: Begins", "set_nuisance_shear_calib");

  if (tomo.shear_Nbin == 0)
  {
    spdlog::critical(
      "\x1b[90m{}\x1b[0m: {} = 0 is invalid", "set_nuisance_shear_calib",
      "shear_Nbin");
    exit(1);
  }
  if (tomo.shear_Nbin != static_cast<int>(M.size()))
  {
    spdlog::critical(
      "\x1b[90m{}\x1b[0m: incompatible input w/ size = {} (!= {})",
      "set_nuisance_shear_calib", M.size(), tomo.shear_Nbin);
    exit(1);
  }

  for (int i=0; i<tomo.shear_Nbin; i++)
  {
    nuisance.shear_calibration_m[i] = M[i];
  }

   spdlog::debug("\x1b[90m{}\x1b[0m: Ends", "set_nuisance_shear_calib");
}

void cpp_set_nuisance_shear_photoz(std::vector<double> SP)
{
  spdlog::debug("\x1b[90m{}\x1b[0m: Begins", "set_nuisance_shear_photoz");

  if (tomo.shear_Nbin == 0)
  {
    spdlog::critical(
      "\x1b[90m{}\x1b[0m: {} = 0 is invalid",
      "set_nuisance_shear_photoz",
      "shear_Nbin"
    );
    exit(1);
  }
  if (tomo.shear_Nbin != static_cast<int>(SP.size()))
  {
    spdlog::critical(
      "\x1b[90m{}\x1b[0m: incompatible input w/ size = {} (!= {})",
      "set_nuisance_shear_photoz",
      SP.size(),
      tomo.shear_Nbin
    );
    exit(1);
  }

  for (int i=0; i<tomo.shear_Nbin; i++)
  {
    nuisance.bias_zphot_shear[i] = SP[i];
  }

  spdlog::debug("\x1b[90m{}\x1b[0m: Ends", "set_nuisance_shear_photoz");
}

void cpp_set_nuisance_clustering_photoz(std::vector<double> CP)
{
  spdlog::debug("\x1b[90m{}\x1b[0m: Begins", "set_nuisance_clustering_photoz");

  if (tomo.clustering_Nbin == 0)
  {
    spdlog::critical(
      "\x1b[90m{}\x1b[0m: {} = 0 is invalid",
      "set_nuisance_clustering_photoz",
      "clustering_Nbin"
    );
    exit(1);
  }
  if (tomo.clustering_Nbin != static_cast<int>(CP.size()))
  {
    spdlog::critical(
      "\x1b[90m{}\x1b[0m: incompatible input w/ size = {} (!= {})",
      "set_nuisance_clustering_photoz",
      CP.size(),
      tomo.clustering_Nbin
    );
    exit(1);
  }

  for (int i=0; i<tomo.clustering_Nbin; i++)
  {
    nuisance.bias_zphot_clustering[i] = CP[i];
  }

  spdlog::debug("\x1b[90m{}\x1b[0m: Ends", "set_nuisance_clustering_photoz");
}

void cpp_set_nuisance_linear_bias(std::vector<double> B1)
{
  spdlog::debug("\x1b[90m{}\x1b[0m: Begins", "set_nuisance_linear_bias");

  if (tomo.clustering_Nbin == 0)
  {
    spdlog::critical("\x1b[90m{}\x1b[0m: {} = 0 is invalid",
      "set_nuisance_linear_bias", "clustering_Nbin");
    exit(1);
  }
  if (tomo.clustering_Nbin != static_cast<int>(B1.size()))
  {
    spdlog::critical(
      "\x1b[90m{}\x1b[0m: incompatible input w/ size = {} (!= {})",
      "set_nuisance_linear_bias", B1.size(), tomo.clustering_Nbin);
    exit(1);
  }

  for (int i=0; i<tomo.clustering_Nbin; i++)
  {
    gbias.b[i] = B1[i];
  }

  spdlog::debug("\x1b[90m{}\x1b[0m: Ends", "set_nuisance_linear_bias");
}

void cpp_set_nuisance_nonlinear_bias(std::vector<double> B1,
std::vector<double> B2)
{
  spdlog::debug("\x1b[90m{}\x1b[0m: Begins", "set_nuisance_nonlinear_bias");

  if (tomo.clustering_Nbin == 0)
  {
    spdlog::critical("\x1b[90m{}\x1b[0m: {} = 0 is invalid",
      "set_nuisance_nonlinear_bias", "clustering_Nbin"
    );
    exit(1);
  }
  if (tomo.clustering_Nbin != static_cast<int>(B1.size()) ||
      tomo.clustering_Nbin != static_cast<int>(B2.size()))
  {
    spdlog::critical(
      "\x1b[90m{}\x1b[0m: incompatible input w/ sizes = {} and {} (!= {})",
      "set_nuisance_nonlinear_bias", B1.size(), B2.size(), tomo.clustering_Nbin
    );
    exit(1);
  }

  constexpr double tmp = -4./7.;
  for (int i=0; i<tomo.clustering_Nbin; i++)
  {
    gbias.b2[i] = B2[i];
    gbias.bs2[i] = ima::almost_equal(B2[i], 0.) ? 0 : tmp*(B1[i]-1.0);
  }

  spdlog::debug("\x1b[90m{}\x1b[0m: Ends", "set_nuisance_nonlinear_bias");
}

void cpp_set_nuisance_magnification_bias(std::vector<double> B_MAG)
{
  spdlog::debug("\x1b[90m{}\x1b[0m: Begins", "set_nuisance_magnification_bias");

  if (tomo.clustering_Nbin == 0)
  {
    spdlog::critical("\x1b[90m{}\x1b[0m: {} = 0 is invalid",
      "set_nuisance_magnification_bias",
      "clustering_Nbin");
    exit(1);
  }
  if (tomo.clustering_Nbin != static_cast<int>(B_MAG.size()))
  {
    spdlog::critical(
      "\x1b[90m{}\x1b[0m: incompatible input w/ size = {} (!= {})",
      "set_nuisance_magnification_bias", B_MAG.size(), tomo.clustering_Nbin);
    exit(1);
  }

  for (int i=0; i<tomo.clustering_Nbin; i++)
  {
    gbias.b_mag[i] = B_MAG[i];
  }

  spdlog::debug("\x1b[90m{}\x1b[0m: Ends", "set_nuisance_magnification_bias");
}

void cpp_set_nuisance_bias(std::vector<double> B1, std::vector<double> B2,
std::vector<double> B_MAG)
{
  cpp_set_nuisance_linear_bias(B1);
  cpp_set_nuisance_nonlinear_bias(B1, B2);
  cpp_set_nuisance_magnification_bias(B_MAG);
}

void cpp_set_nuisance_ia(std::vector<double> A1, std::vector<double> A2,
std::vector<double> B_TA)
{
  spdlog::debug("\x1b[90m{}\x1b[0m: Begins", "set_nuisance_ia");

  if (tomo.shear_Nbin == 0)
  {
    spdlog::critical("\x1b[90m{}\x1b[0m: {} = 0 is invalid",
      "set_nuisance_ia", "shear_Nbin");
    exit(1);
  }
  if (tomo.shear_Nbin != static_cast<int>(A1.size()) ||
      tomo.shear_Nbin != static_cast<int>(A2.size()) ||
  		tomo.shear_Nbin != static_cast<int>(B_TA.size()))
  {
    spdlog::critical(
      "\x1b[90m{}\x1b[0m: incompatible input w/ sizes = {}, {} and {} (!= {})",
      "set_nuisance_ia", A1.size(), A2.size(), B_TA.size(), tomo.shear_Nbin
    );
    exit(1);
  }

  nuisance.c1rhocrit_ia = 0.01389;
  if (like.IA == 3 || like.IA == 5)
  {
    for (int i=0; i<tomo.shear_Nbin; i++)
    {
      nuisance.A_z[i] = A1[i];
      nuisance.A2_z[i] = A2[i];
      nuisance.b_ta_z[i] = B_TA[i];
    }
  }
  else if (like.IA == 4 || like.IA == 6)
  {
    nuisance.A_ia = A1[0];
    nuisance.eta_ia = A1[1];
    nuisance.oneplusz0_ia = 1.62;

    nuisance.A2_ia = A2[0];
    nuisance.eta_ia_tt = A2[1];
    nuisance.b_ta_z[0] = B_TA[0];

    for (int i=2; i<tomo.shear_Nbin; i++)
    {
      if ( !(ima::almost_equal(A1[i], 0.)) || !(ima::almost_equal(A2[i], 0.)) ||
           !(ima::almost_equal(B_TA[i], 0.)))
      {
        spdlog::critical(
        	"set_nuisance_ia: one of nuisance.A_z[{}]={}, nuisance.A2_z[{}]="
          "{}, nuisance.b_ta[{}]={} was specified w/ power-law evolution\n",
          i, nuisance.A_z[i], i, nuisance.A2_z[i], i, nuisance.b_ta_z[i]);
        exit(1);
      }
    }
  }

  spdlog::debug("\x1b[90m{}\x1b[0m: Ends", "set_nuisance_ia");
}

void cpp_set_pm(std::vector<double> pm)
{
  ima::PointMass& instance = ima::PointMass::get_instance();
  instance.set_pm_vector(pm);
  return;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// GET FUNCTIONS
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

arma::Mat<double> cpp_get_covariance_masked()
{
  ima::RealData& instance = ima::RealData::get_instance();
  return instance.get_covariance_masked();
}

arma::Mat<double> cpp_get_covariance_masked_reduced_dim()
{
  ima::RealData& instance = ima::RealData::get_instance();
  return instance.get_covariance_masked_reduced_dim();
}

int cpp_get_mask(const int i)
{
  ima::RealData& instance = ima::RealData::get_instance();
  return instance.get_mask(i);
}

int cpp_get_ndim()
{
  ima::RealData& instance = ima::RealData::get_instance();
  return instance.get_ndim();
}

int cpp_get_nreduced_dim()
{
  ima::RealData& instance = ima::RealData::get_instance();
  return instance.get_nreduced_dim();
}

int cpp_get_index_reduced_dim(const int i)
{
  ima::RealData& instance = ima::RealData::get_instance();
  return instance.get_index_reduced_dim(i);
}

// The conversion between STL vector and python np array is cleaner
// arma:Col is cast to 2D np array with 1 column (not as nice!)
std::vector<double> cpp_get_expand_dim_from_masked_reduced_dim(
std::vector<double> reduced_dim_vector)
{
  ima::RealData& instance = ima::RealData::get_instance();

  arma::Col<double> tmp = instance.get_expand_dim_from_masked_reduced_dim(
    arma::Col<double>(reduced_dim_vector));

  std::vector<double> result(tmp.n_elem, 0.0);
  for(int i=0; i<static_cast<int>(tmp.n_elem); i++)
  {
    result[i] = tmp(i);
  }

  return result;
}

int cpp_get_baryon_pca_nscenarios()
{
  ima::BaryonScenario& instance = ima::BaryonScenario::get_instance();
  return instance.nscenarios();
}

std::string cpp_get_baryon_pca_scenario_name(const int i)
{
  ima::BaryonScenario& instance = ima::BaryonScenario::get_instance();
  return instance.get_scenario(i);
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// COMPUTE FUNCTIONS
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

double cpp_compute_chi2(std::vector<double> datavector)
{
  ima::RealData& instance = ima::RealData::get_instance();
  return instance.get_chi2(datavector);
}

double cpp_compute_pm(const int zl, const int zs,
const double theta)
{
  ima::PointMass& instance = ima::PointMass::get_instance();
  return instance.get_pm(zl,zs,theta);
}

std::vector<double> cpp_compute_data_vector_masked()
{
  spdlog::debug("\x1b[90m{}\x1b[0m: Begins", "compute_data_vector_masked");

  if (tomo.shear_Nbin == 0)
  {
    spdlog::critical("\x1b[90m{}\x1b[0m: {} = 0 is invalid",
      "compute_data_vector_masked", "shear_Nbin");
    exit(1);
  }
  if (tomo.clustering_Nbin == 0)
  {
    spdlog::critical("\x1b[90m{}\x1b[0m: {} = 0 is invalid",
      "compute_data_vector_masked", "clustering_Nbin");
    exit(1);
  }
  if (like.Ntheta == 0)
  {
    spdlog::critical("\x1b[90m{}\x1b[0m: {} = 0 is invalid",
      "compute_data_vector_masked", "Ntheta");
    exit(1);
  }
  if (!ima::RealData::get_instance().is_mask_set())
  {
    spdlog::critical(
      "\x1b[90m{}\x1b[0m: {} not set prior to this function call",
      "compute_data_vector_masked", "mask");
    exit(1);
  }
  if (!ima::RealData::get_instance().is_data_set())
  {
    spdlog::critical(
      "\x1b[90m{}\x1b[0m: {} not set prior to this function call",
      "compute_data_vector_masked", "data_vector");
    exit(1);
  }
  if (!ima::RealData::get_instance().is_inv_cov_set())
  {
    spdlog::critical(
      "\x1b[90m{}\x1b[0m: {} not set prior to this function call",
      "compute_data_vector_masked", "inv_cov");
    exit(1);
  }

  std::vector<double> data_vector(like.Ndata, 0.0);

  int start = 0;
  if (like.shear_shear == 1)
  {
    for (int nz=0; nz<tomo.shear_Npowerspectra; nz++)
    {
      const int z1 = Z1(nz);
      const int z2 = Z2(nz);
      for (int i = 0; i<like.Ntheta; i++)
      {
        if (cpp_get_mask(like.Ntheta*nz+i))
        {
          data_vector[like.Ntheta*nz+i] =
            xi_pm_tomo(1, i, z1, z2, 1 /* limber option = 1 -> limber */)*
            (1.0 + nuisance.shear_calibration_m[z1])*
            (1.0 + nuisance.shear_calibration_m[z2]);
        }
        if (cpp_get_mask(like.Ntheta*(tomo.shear_Npowerspectra+nz)+i))
        {
          data_vector[like.Ntheta*(tomo.shear_Npowerspectra+nz)+i] =
            xi_pm_tomo(-1, i, z1, z2, 1 /*limber*/)*
            (1. + nuisance.shear_calibration_m[z1])*
            (1. + nuisance.shear_calibration_m[z2]);
        }
      }
    }
  }

  start = start + 2*like.Ntheta*tomo.shear_Npowerspectra;
  if (like.shear_pos == 1)
  {
    for (int nz=0; nz<tomo.ggl_Npowerspectra; nz++)
    {
      const int zl = ZL(nz);
      const int zs = ZS(nz);
      for (int i=0; i<like.Ntheta; i++)
      {
        if (cpp_get_mask(start+(like.Ntheta*nz)+i))
        {
          const double theta = like.theta[i];
          data_vector[start+(like.Ntheta*nz)+i] = (
            w_gammat_tomo(i, zl, zs, 1 /* limber option=1 -> limber */) +
            cpp_compute_pm(zl, zs, theta))*(1.0+nuisance.shear_calibration_m[zs]);
        }
      }
    }
  }

  start = start + like.Ntheta*tomo.ggl_Npowerspectra;
  if (like.pos_pos == 1)
  {
    for (int nz=0; nz<tomo.clustering_Npowerspectra; nz++)
    {
      for (int i=0; i<like.Ntheta; i++)
      {
        if (cpp_get_mask(start+(like.Ntheta*nz)+i))
        {
          data_vector[start+(like.Ntheta*nz)+i] =
            w_gg_tomo(i, nz, nz, 0 /* limber option = 0 -> nonlimber */);
        }
      }
    }
  }

  spdlog::debug("\x1b[90m{}\x1b[0m: Ends", "compute_data_vector_masked");

  return data_vector;
}

std::vector<double> cpp_compute_data_vector_masked_reduced_dim()
{
  std::vector<double> data_vector_masked = cpp_compute_data_vector_masked();

  const int ndim = data_vector_masked.size();
  const int ndim_reduced = cpp_get_nreduced_dim();

  std::vector<double> data_vector_masked_reduced_dim(ndim_reduced, 0.0);

  for(int i=0; i<ndim; i++)
  {
    if(cpp_get_mask(i)>0.99)
    {
      if(cpp_get_index_reduced_dim(i) < 0)
      {
        spdlog::critical("\x1b[90m{}\x1b[0m: logical error, internal"
          " inconsistent mask operation",
          "cpp_compute_data_vector_masked_reduced_dim");
        exit(1);
      }

      data_vector_masked_reduced_dim[cpp_get_index_reduced_dim(i)] =
        data_vector_masked[i];
    }
  }

  return data_vector_masked_reduced_dim;
}

double cpp_compute_baryon_ratio(double log10k, double a)
{
  const double KNL = pow(10.0,log10k)*cosmology.coverH0;
  return PkRatio_baryons(KNL, a);
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// RESET FUNCTIONS
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

void cpp_reset_baryionic_struct()
{
  reset_bary_struct();
  return;
}

// ----------------------------------------------------------------------------
// ------------------------ INTERNAL C++ FUNCTIONS ----------------------------
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// ------------------------ INTERNAL C++ FUNCTIONS ----------------------------
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// ------------------------ INTERNAL C++ FUNCTIONS ----------------------------
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// ------------------------ INTERNAL C++ FUNCTIONS ----------------------------
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// ------------------------ INTERNAL C++ FUNCTIONS ----------------------------
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// ------------------------ INTERNAL C++ FUNCTIONS ----------------------------
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// CLASS RealData MEMBER FUNCTIONS (& RELATED) - READ MASK, COV..
// THERE ARE "C" WRAPS FOR MOST OF THESE FUNCTIONS
// ----------------------------------------------------------------------------

arma::Mat<double> ima::read_table(const std::string file_name)
{
  std::ifstream input_file(file_name);

  if (!input_file.is_open())
  {
    spdlog::critical(
      "\x1b[90m{}\x1b[0m: file {} cannot be opened",
      "read_table",
      file_name
    );
    exit(1);
  }

  // Read the entire file into memory
  std::string tmp;
  input_file.seekg(0,std::ios::end);
  tmp.resize(static_cast<size_t>(input_file.tellg()));
  input_file.seekg(0,std::ios::beg);
  input_file.read(&tmp[0],tmp.size());
  input_file.close();
  if(tmp.empty())
  {
    spdlog::critical(
      "\x1b[90m{}\x1b[0m: file {} is empty",
      "read_table",
      file_name
    );
    exit(1);
  }
  std::vector<std::string> lines;
  lines.reserve(50000);
  // Second: Split file into lines
  boost::trim_if(tmp,boost::is_any_of("\t "));
  boost::trim_if(tmp,boost::is_any_of("\n"));
  boost::split(lines, tmp,boost::is_any_of("\n"), boost::token_compress_on);
  // Erase comment/blank lines
  auto check = [](std::string mystr) -> bool
  {
    return boost::starts_with(mystr, "#");
  };
  lines.erase(std::remove_if(lines.begin(), lines.end(), check), lines.end());
  // Third: Split line into words
  arma::Mat<double> result;
  size_t ncols = 0;
  { // first line
    std::vector<std::string> words;
    words.reserve(100);
    boost::split(words,lines[0], boost::is_any_of(" \t"),
      boost::token_compress_on);
    ncols = words.size();
    result.set_size(lines.size(), ncols);
    for (size_t j=0; j<ncols; j++)
    {
      result(0,j) = std::stod(words[j]);
    }
  }
  #pragma omp parallel for
  for (size_t i=1; i<lines.size(); i++)
  {
    std::vector<std::string> words;
    boost::split(words, lines[i], boost::is_any_of(" \t"),
      boost::token_compress_on);
    if (words.size() != ncols)
    {
      spdlog::critical("\x1b[90m{}\x1b[0m: file {} is not well formatted"
      " (regular table required)", "read_table", file_name);
      exit(1);
    }
    for (size_t j=0; j<ncols; j++)
    {
      result(i,j) = std::stod(words[j]);
    }
  };
  return result;
}

std::vector<double> ima::convert_arma_col_to_stl_vector(arma::Col<double> in)
{
  std::vector<double> out(in.n_elem, 0.0);

  for(int i=0; i<static_cast<int>(in.n_elem); i++)
  {
    out[i] = in(i);
  }

  return out;
}

void ima::RealData::set_mask(std::string MASK)
{
  if (!(like.Ndata>0))
  {
    spdlog::critical("\x1b[90m{}\x1b[0m: {} not set prior to this function call",
      "set_mask", "like.Ndata");
    exit(1);
  }

  this->ndata_ = like.Ndata;
  this->mask_.set_size(this->ndata_);

  arma::Mat<double> table = ima::read_table(MASK);
  for (int i=0; i<this->ndata_; i++)
  {
    this->mask_(i) = static_cast<int>(table(i,1)+1e-13);
    if(!(this->mask_(i) == 0 || this->mask_(i) == 1))
    {
      spdlog::critical("\x1b[90m{}\x1b[0m: inconsistent mask", "set_mask");
      exit(1);
    }
  }

  // overwriting mask if some part of the observable is not wanted
  if (like.shear_shear == 0)
  {
    const int M = like.Ntheta*2*tomo.shear_Npowerspectra;
    for (int i=0; i<M; i++)
    {
      this->mask_(i) = 0;
    }
  }
  if (like.shear_pos == 0)
  {
    const int N = 2*like.Ntheta*tomo.shear_Npowerspectra;
    const int M = N + like.Ntheta*tomo.ggl_Npowerspectra;
    for (int i=N; i<M; i++)
    {
      this->mask_(i) = 0;
    }
  }
  if (like.pos_pos == 0)
  {
    const int N = like.Ntheta*(2*tomo.shear_Npowerspectra + tomo.ggl_Npowerspectra);
    const int M = N + like.Ntheta*tomo.clustering_Npowerspectra;
    for (int i=N; i<M; i++)
    {
      this->mask_(i) = 0;
    }
  }

  this->mask_filename_ = MASK;
  this->ndata_masked_ = arma::accu(this->mask_);

  if(!(this->ndata_masked_>0))
  {
    spdlog::critical(
      "\x1b[90m{}\x1b[0m: mask file {} left no data points after masking",
      "set_mask", MASK);
    exit(1);
  }
  spdlog::info(
    "\x1b[90m{}\x1b[0m: mask file {} left {} non-masked elements after masking",
    "set_mask", MASK, this->ndata_masked_);

  this->index_reduced_dim_.set_size(this->ndata_);
  {
    double j=0;
    for(int i=0; i<this->ndata_; i++)
    {
      if(this->get_mask(i) > 0)
      {
        this->index_reduced_dim_(i) = j;
        j++;
      }
      else
      {
        this->index_reduced_dim_(i) = -1;
      }
    }
    if(j != this->ndata_masked_)
    {
      spdlog::critical(
       "\x1b[90m{}\x1b[0m: logical error, internal inconsistent mask operation",
       "set_mask");
      exit(1);
    }
  }

  this->is_mask_set_ = true;
}

void ima::RealData::set_data(std::string DATA)
{
  if (!(this->is_mask_set_))
  {
    spdlog::critical(
      "\x1b[90m{}\x1b[0m: {} not set prior to this function call", "set_data",
      "mask");
    exit(1);
  }

  this->data_masked_.set_size(this->ndata_);
  this->data_filename_ = DATA;
  this->data_masked_reduced_dim_.set_size(this->ndata_masked_);

  arma::Mat<double> table = ima::read_table(DATA);

  for(int i=0; i<like.Ndata; i++)
  {
    this->data_masked_(i) = table(i,1);
    this->data_masked_(i) *= this->get_mask(i);

    if(this->get_mask(i) == 1)
    {
      if(this->get_index_reduced_dim(i) < 0)
      {
        spdlog::critical("\x1b[90m{}\x1b[0m: logical error, internal"
          " inconsistent mask operation", "set_data");
        exit(1);
      }

      this->data_masked_reduced_dim_(this->get_index_reduced_dim(i)) =
        this->data_masked_(i);
    }
  }

  this->is_data_set_ = true;
}

void ima::RealData::set_inv_cov(std::string COV)
{
  if (!(this->is_mask_set_))
  {
    spdlog::critical(
      "\x1b[90m{}\x1b[0m: {} not set prior to this function call",
      "set_inv_cov",
      "mask"
    );
    exit(1);
  }

  arma::Mat<double> table = ima::read_table(COV); // this reads cov!

  this->cov_masked_.set_size(this->ndata_, this->ndata_);
  this->cov_masked_.zeros();

  this->inv_cov_masked_.set_size(this->ndata_, this->ndata_);
  this->inv_cov_masked_.zeros();

  switch (table.n_cols)
  {
    case 3:
    {
      for (int i=0; i<static_cast<int>(table.n_rows); i++)
      {
        const int j = static_cast<int>(table(i,0));
        const int k = static_cast<int>(table(i,1));

        this->cov_masked_(j,k) = table(i,2);
        this->inv_cov_masked_(j,k) = table(i,2);

        if (j!=k)
        {
          // apply mask to off-diagonal covariance elements
          this->cov_masked_(j,k) *= this->get_mask(j);
          this->cov_masked_(j,k) *= this->get_mask(k);

          this->inv_cov_masked_(j,k) *= this->get_mask(j);
          this->inv_cov_masked_(j,k) *= this->get_mask(k);

          // m(i,j) = m(j,i)
          this->cov_masked_(k,j) = this->cov_masked_(j,k);
          this->inv_cov_masked_(k,j) = this->inv_cov_masked_(j,k);
        }
      };
      break;
    }
    case 4:
    {
      for (int i=0; i<static_cast<int>(table.n_rows); i++)
      {
        const int j = static_cast<int>(table(i,0));
        const int k = static_cast<int>(table(i,1));

        this->cov_masked_(j,k) = table(i,2) + table(i,3);
        this->inv_cov_masked_(j,k) = table(i,2) + table(i,3);

        if (j!=k)
        {
          // apply mask to off-diagonal covariance elements
          this->cov_masked_(j,k) *= this->get_mask(j);
          this->cov_masked_(j,k) *= this->get_mask(k);

          this->inv_cov_masked_(j,k) *= this->get_mask(j);
          this->inv_cov_masked_(j,k) *= this->get_mask(k);

          // m(i,j) = m(j,i)
          this->cov_masked_(k,j) = this->cov_masked_(j,k);
          this->inv_cov_masked_(k,j) = this->inv_cov_masked_(j,k);
        }
      };
      break;
    }
    case 10:
    {
      for (int i=0; i<static_cast<int>(table.n_rows); i++)
      {
        const int j = static_cast<int>(table(i,0));
        const int k = static_cast<int>(table(i,1));

        this->cov_masked_(j,k) = table(i,8) + table(i,9);
        this->inv_cov_masked_(j,k) = table(i,8) + table(i,9);

        if (j!=k)
        {
          // apply mask to off-diagonal covariance elements
          this->cov_masked_(j,k) *= this->get_mask(j);
          this->cov_masked_(j,k) *= this->get_mask(k);

          this->inv_cov_masked_(j,k) *= this->get_mask(j);
          this->inv_cov_masked_(j,k) *= this->get_mask(k);

          // m(i,j) = m(j,i)
          this->cov_masked_(k,j) = this->cov_masked_(j,k);
          this->inv_cov_masked_(k,j) = this->inv_cov_masked_(j,k);
        }
      }
      break;
    }
    default:
      spdlog::critical("{}: data format for covariance file = {} is invalid",
        "set_inv_cov", COV);
      exit(1);
  }

  this->inv_cov_masked_ = arma::inv(this->inv_cov_masked_);

  // apply mask again, to make sure numerical errors in matrix
  // inversion don't cause problems...
  // also, set diagonal elements corresponding to datavector elements
  // outside mask to zero, so that these elements don't contribute to chi2
  for (int i=0; i<this->ndata_; i++)
  {
    this->inv_cov_masked_(i,i) *= this->get_mask(i)*this->get_mask(i);
    for (int j=0; j<i; j++)
    {
      this->inv_cov_masked_(i,j) *= this->get_mask(i)*this->get_mask(j);
      this->inv_cov_masked_(j,i) = this->inv_cov_masked_(i,j);
    }
  };
  this->cov_filename_ = COV;
  this->is_inv_cov_set_ = true;

  this->cov_masked_reduced_dim_.set_size(this->ndata_masked_,
    this->ndata_masked_);

  this->inv_cov_masked_reduced_dim_.set_size(this->ndata_masked_,
    this->ndata_masked_);

  for(int i=0; i<this->ndata_; i++)
  {
    for(int j=0; j<this->ndata_; j++)
    {
      if((this->mask_(i)>0.99) && (this->mask_(j)>0.99))
      {
        if(this->get_index_reduced_dim(i) < 0)
        {
          spdlog::critical("\x1b[90m{}\x1b[0m: logical error, internal"
            " inconsistent mask operation", "set_inv_cov");
          exit(1);
        }
        if(this->get_index_reduced_dim(j) < 0)
        {
          spdlog::critical("\x1b[90m{}\x1b[0m: logical error, internal"
            " inconsistent mask operation", "set_inv_cov");
          exit(1);
        }

        this->cov_masked_reduced_dim_(this->get_index_reduced_dim(i),
          this->get_index_reduced_dim(j)) = this->cov_masked_(i,j);

        this->inv_cov_masked_reduced_dim_(this->get_index_reduced_dim(i),
          this->get_index_reduced_dim(j)) = this->inv_cov_masked_(i,j);
      }
    }
  }
}

arma::Col<int> ima::RealData::get_mask() const
{
  return this->mask_;
}

int ima::RealData::get_mask(const int ci) const
{
  if (ci > like.Ndata || ci < 0)
  {
    spdlog::critical(
      "\x1b[90m{}\x1b[0m: index i = {} is not valid (min = {}, max = {})",
      "get_mask", ci, 0.0, like.Ndata);
    exit(1);
  }

  return this->mask_(ci);
}

int ima::RealData::get_ndim() const
{
  return this->ndata_;
}

int ima::RealData::get_nreduced_dim() const
{
  return this->ndata_masked_;
}

int ima::RealData::get_index_reduced_dim(const int ci) const
{
  if (ci > like.Ndata || ci < 0)
  {
    spdlog::critical("\x1b[90m{}\x1b[0m: index i = {} is not valid"
      " (min = {}, max = {})", "get_index_reduced_dim", ci, 0.0, like.Ndata);
    exit(1);
  }

  return this->index_reduced_dim_(ci);
}

arma::Col<double> ima::RealData::get_data_masked() const
{
  return this->data_masked_;
}

double ima::RealData::get_data_masked(const int ci) const
{
  if (ci > like.Ndata || ci < 0)
  {
    spdlog::critical(
      "\x1b[90m{}\x1b[0m: index i = {} is not valid (min = {}, max = {})",
      "get_data_masked", ci, 0, like.Ndata);
    exit(1);
  }

  return this->data_masked_(ci);
}

arma::Col<double> ima::RealData::get_data_masked_reduced_dim() const
{
  return this->data_masked_reduced_dim_;
}

double ima::RealData::get_data_masked_reduced_dim(const int ci) const
{
  if (ci > like.Ndata || ci < 0)
  {
    spdlog::critical(
      "\x1b[90m{}\x1b[0m: index i = {} is not valid (min = {}, max = {})",
      "get_data_masked_reduced_dim", ci, 0, like.Ndata);
    exit(1);
  }

  return this->data_masked_reduced_dim_(ci);
}

arma::Mat<double> ima::RealData::get_inverse_covariance_masked() const
{
  return this->inv_cov_masked_;
}

double ima::RealData::get_inverse_covariance_masked(const int ci,
const int cj) const
{
  if (ci > like.Ndata || ci < 0)
  {
    spdlog::critical(
      "\x1b[90m{}\x1b[0m: index i = {} is not valid (min = {}, max = {})",
      "get_inverse_covariance_masked", ci, 0.0, like.Ndata);
    exit(1);
  }
  if (cj > like.Ndata || cj < 0)
  {
    spdlog::critical(
      "\x1b[90m{}\x1b[0m: index j = {} is not valid (min = {}, max = {})",
      "get_inverse_covariance_masked", cj, 0.0, like.Ndata);
    exit(1);
  }

  return this->inv_cov_masked_(ci, cj);
}

arma::Mat<double> ima::RealData::get_inverse_covariance_masked_reduced_dim() const
{
  return this->inv_cov_masked_reduced_dim_;
}

double ima::RealData::get_inverse_covariance_masked_reduced_dim(const int ci,
const int cj) const
{
  if (ci > like.Ndata || ci < 0)
  {
    spdlog::critical(
      "\x1b[90m{}\x1b[0m: index i = {} is not valid (min = {}, max = {})",
      "get_inverse_covariance_masked_reduced_dim", ci, 0.0, like.Ndata);
    exit(1);
  }
  if (cj > like.Ndata || cj < 0)
  {
    spdlog::critical(
      "\x1b[90m{}\x1b[0m: index j = {} is not valid (min = {}, max = {})",
      "get_inverse_covariance_masked_reduced_dim", cj, 0.0, like.Ndata);
    exit(1);
  }

  return this->inv_cov_masked_reduced_dim_(ci, cj);
}

arma::Mat<double> ima::RealData::get_covariance_masked() const
{
  return this->cov_masked_;
}

arma::Mat<double> ima::RealData::get_covariance_masked_reduced_dim() const
{
  return this->cov_masked_reduced_dim_;
}

double ima::RealData::get_chi2(std::vector<double> datavector) const
{
  if (!(this->is_data_set_))
  {
    spdlog::critical(
      "\x1b[90m{}\x1b[0m: {} not set prior to this function call",
      "get_chi2",
      "data_vector"
    );
    exit(1);
  }
  if (!(this->is_mask_set_))
  {
    spdlog::critical(
      "\x1b[90m{}\x1b[0m: {} not set prior to this function call",
      "get_chi2",
      "mask"
    );
    exit(1);
  }
  if (!(this->is_inv_cov_set_))
  {
    spdlog::critical(
      "\x1b[90m{}\x1b[0m: {} not set prior to this function call",
      "get_chi2",
      "inv_cov"
    );
    exit(1);
  }

  double chi2 = 0.0;
  for (int i=0; i<like.Ndata; i++)
  {
    if (this->get_mask(i))
    {
      const double x = datavector[i] - this->get_data_masked(i);
      for (int j=0; j<like.Ndata; j++)
      {
        if (this->get_mask(j))
        {
          const double y = datavector[j] - this->get_data_masked(j);
          chi2 += x*this->get_inverse_covariance_masked(i,j)*y;
        }
      }
    }
  }
  if (chi2 < 0.0)
  {
    spdlog::critical("\x1b[90m{}\x1b[0m: chi2 = {} (invalid)", "get_chi2", chi2);
    exit(1);
  }
  return chi2;
}

bool ima::RealData::is_mask_set() const
{
  return this->is_mask_set_;
}

bool ima::RealData::is_data_set() const
{
  return this->is_data_set_;
}

bool ima::RealData::is_inv_cov_set() const
{
  return this->is_inv_cov_set_;
}

arma::Col<double> ima::RealData::get_expand_dim_from_masked_reduced_dim(
arma::Col<double> reduced_dim_vector) const
{
  if (this->ndata_masked_ != static_cast<int>(reduced_dim_vector.n_elem))
  {
    spdlog::critical("\x1b[90m{}\x1b[0m: {} invalid input vector",
      "get_expand_dim_from_masked_reduced_dim"
    );
    exit(1);
  }

  arma::Col<double> vector;
  vector.set_size(this->ndata_);
  vector.zeros();

  for(int i=0; i<this->ndata_; i++)
  {
    if(this->mask_(i) > 0.99)
    {
      if(this->get_index_reduced_dim(i) < 0)
      {
        spdlog::critical("\x1b[90m{}\x1b[0m: logical error, internal "
        "inconsistent mask operation",
        "get_expand_dim_from_masked_reduced_dim");
        exit(1);
      }

      vector(i) = reduced_dim_vector(this->get_index_reduced_dim(i));
    }
  }

  return vector;
}

// ----------------------------------------------------------------------------
// CLASS PointMass MEMBER FUNCTIONS
// THERE ARE "C" WRAPS FOR MOST OF THESE FUNCTIONS
// ----------------------------------------------------------------------------

void ima::PointMass::set_pm_vector(std::vector<double> pm)
{
  this->pm_ = pm;
  return;
}

std::vector<double> ima::PointMass::get_pm_vector() const
{
  return this->pm_;
}

double ima::PointMass::get_pm(const int zl, const int zs,
const double theta) const
{
  constexpr double G_over_c2 = 1.6e-23;
  const double a_lens = 1.0/(1.0 + zmean(zl));
  const double chi_lens = chi(a_lens);

  return 4*G_over_c2*this->pm_[zl]*1.e+13*g_tomo(a_lens, zs)/(theta*theta)/
    (chi_lens*a_lens);
}

// ----------------------------------------------------------------------------
// CLASS BaryonScenario MEMBER FUNCTIONS
// THERE ARE "C" WRAPS FOR MOST OF THESE FUNCTIONS
// ----------------------------------------------------------------------------

int ima::BaryonScenario::nscenarios() const
{
  return this->nscenarios_;
}

void ima::BaryonScenario::set_scenarios(std::string scenarios)
{
  std::vector<std::string> lines;
  lines.reserve(50);

  // Second: Split file into lines
  boost::trim_if(scenarios, boost::is_any_of("\t "));
  boost::trim_if(scenarios, boost::is_any_of("\n"));

  if (scenarios.empty())
  {
    spdlog::critical("\x1b[90m{}\x1b[0m: invalid string input (empty)",
      "init_baryon_pca_scenarios");
    exit(1);
  }

  boost::split(lines, scenarios, boost::is_any_of("/"),
    boost::token_compress_on);

  this->nscenarios_ = lines.size();

  for(int i=0; i<this->nscenarios_; i++)
  {
    this->scenarios_[i] = lines[i];
  }

  return;
}

std::string ima::BaryonScenario::get_scenario(const int i) const
{
  return this->scenarios_.at(i);
}

// ----------------------------------------------------------------------------
// ---------------------------- PYTHON WRAPPER --------------------------------
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// ---------------------------- PYTHON WRAPPER --------------------------------
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// ---------------------------- PYTHON WRAPPER --------------------------------
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// ---------------------------- PYTHON WRAPPER --------------------------------
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// ---------------------------- PYTHON WRAPPER --------------------------------
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// ---------------------------- PYTHON WRAPPER --------------------------------
// ----------------------------------------------------------------------------


PYBIND11_MODULE(cosmolike_lsst_y1_interface, m)
{
  m.doc() = "CosmoLike Interface for LSST-Y1 3x2 Module";

  // --------------------------------------------------------------------
  // --------------------------------------------------------------------
  // INIT FUNCTIONS
  // --------------------------------------------------------------------
  // --------------------------------------------------------------------

  m.def("initial_setup",
    &cpp_initial_setup,
    "Initialize Many Cosmolike Variables to their Default Values"
  );

  m.def("init_probes",
    &cpp_init_probes,
    "Init Probes (cosmic shear or 2x2pt or 3x2pt...)",
    py::arg("possible_probes")
  );

  m.def("init_survey_parameters",
    &cpp_init_survey,
    "Init Survey Parameters",
    py::arg("surveyname"),
    py::arg("area"),
    py::arg("sigma_e")
  );

  m.def("init_cosmo_runmode",
    &cpp_init_cosmo_runmode,
    "Init Run Mode",
    py::arg("is_linear")
  );

  m.def("init_baryons_contamination",
    &cpp_init_baryons_contamination,
    "Add baryonic simulations to power spectrum as contamination",
    py::arg("use_baryonic_simulations_contamination"),
    py::arg("which_baryonic_simulations_contamination")
  );

  m.def("init_IA",
    &cpp_init_IA,
    "Init IA related options",
    py::arg("ia_model")
  );

  m.def("init_binning",
    &cpp_init_binning,
    "Init Bining related variables",
    py::arg("Ntheta"),
    py::arg("theta_min_arcmin"),
    py::arg("theta_max_arcmin")
  );

  m.def("init_data_real",
    &cpp_init_data_real,
    "Init covariance, mask and data vector by providing the file names that"
    "hold their values",
    py::arg("COV"),
    py::arg("MASK"),
    py::arg("DATA")
  );

  m.def("init_lens_sample",
    &cpp_init_lens_sample,
    "Init Lens Sample",
    py::arg("multihisto_file"),
    py::arg("Ntomo"),
    py::arg("ggl_cut")
  );

  m.def("init_source_sample",
    &cpp_init_source_sample,
    "Init Source Sample",
    py::arg("multihisto_file"),
    py::arg("Ntomo")
  );

  m.def("init_size_data_vector",
    &cpp_init_size_data_vector,
    "Init Size Data Vector"
  );

  m.def("init_linear_power_spectrum",
    &cpp_init_linear_power_spectrum,
    "Load Linear Matter Power Spectrum from Cobaya to Cosmolike",
    py::arg("log10k"),
    py::arg("z"),
    py::arg("lnP")
  );

  m.def("init_non_linear_power_spectrum",
    &cpp_init_non_linear_power_spectrum,
    "Load Matter Power Spectrum from Cobaya to Cosmolike",
    py::arg("log10k"),
    py::arg("z"),
    py::arg("lnP")
  );

  m.def("init_growth",
    &cpp_init_growth,
    "Load Growth Factor from Cobaya to Cosmolike",
    py::arg("z"),
    py::arg("G")
  );

  m.def("init_distances",
    &cpp_init_distances,
    "Load chi(z) from Cobaya to Cosmolike",
    py::arg("z"),
    py::arg("chi")
  );

  m.def("init_baryon_pca_scenarios",
    &cpp_init_baryon_pca_scenarios,
    "Init scenario selection to generate baryonic PCA",
    py::arg("scenarios")
  );

  m.def("init_accuracy_boost",
    &cpp_init_accuracy_boost,
    "Init Accuracy and Sampling Boost (can slow down Cosmolike a lot)",
    py::arg("accuracy_boost"),
    py::arg("sampling_boost"),
    py::arg("integration_accuracy")
  );

  // --------------------------------------------------------------------
  // --------------------------------------------------------------------
  // SET FUNCTIONS
  // --------------------------------------------------------------------
  // --------------------------------------------------------------------

  m.def("set_nuisance_ia",
    &cpp_set_nuisance_ia,
    "Set Nuisance IA Parameters",
    py::arg("A1"),
    py::arg("A2"),
    py::arg("B_TA")
  );

  m.def("set_nuisance_bias",
    &cpp_set_nuisance_bias,
    "Set Nuisance Bias Parameters",
    py::arg("B1"),
    py::arg("B2"),
    py::arg("B_MAG")
  );

  m.def("set_nuisance_shear_calib",
    &cpp_set_nuisance_shear_calib,
    "Set Shear Calibration Parameters",
    py::arg("M")
  );

  m.def("set_nuisance_clustering_photoz",
    &cpp_set_nuisance_clustering_photoz,
    "Set Clustering Shear Photo-Z Parameters",
    py::arg("bias")
  );

  m.def("set_nuisance_shear_photoz",
    &cpp_set_nuisance_shear_photoz,
    "Set Shear Photo-z Parameters",
    py::arg("bias")
  );

  m.def("set_cosmological_parameters",
    &cpp_set_cosmological_parameters,
    "Set Cosmological Parameters",
    py::arg("omega_matter"),
    py::arg("hubble"),
    py::arg("is_cached")
  );

  m.def("set_point_mass",
    &cpp_set_pm,
    py::arg("PMV")
  );

  // --------------------------------------------------------------------
  // --------------------------------------------------------------------
  // GET FUNCTIONS
  // --------------------------------------------------------------------
  // --------------------------------------------------------------------

  m.def("get_covariance_masked",
    &cpp_get_covariance_masked,
    "Get Masked Covariance Matrix - masked dimensions are filled w/ zeros"
  );

  m.def("get_covariance_masked_reduced_dim",
    &cpp_get_covariance_masked_reduced_dim,
    "Get Masked Covariance Matrix - it does not to contain masked dimensions"
  );

  m.def("get_ndim", &cpp_get_ndim, "Get number of data points");

  m.def("get_nreduced_dim", &cpp_get_nreduced_dim,
    "Get number of non-masked points"
  );

  m.def("get_baryon_pca_scenario_name", &cpp_get_baryon_pca_scenario_name,
    "Get jth scenario name selected to generate baryonic PCA", py::arg("i"));

  m.def("get_baryon_pca_nscenarios",
    &cpp_get_baryon_pca_nscenarios,
    "Get number of scenarios selected to generate baryonic PCA"
  );

  m.def("get_expand_dim_from_masked_reduced_dim",
    &cpp_get_expand_dim_from_masked_reduced_dim,
    "Get expanded vector (w/ zeros on masked dim) from masked reduced dim vector"
  );

  // --------------------------------------------------------------------
  // --------------------------------------------------------------------
  // COMPUTE FUNCTIONS
  // --------------------------------------------------------------------
  // --------------------------------------------------------------------

  m.def("compute_data_vector_masked",
    &cpp_compute_data_vector_masked,
    "Get theoretical data vector - masked dimensions are filled w/ zeros"
  );

  m.def("compute_data_vector_masked_reduced_dim",
    &cpp_compute_data_vector_masked_reduced_dim,
    "Get theoretical data vector - it does not to contain masked dimensions"
  );

  m.def("compute_chi2",
    &cpp_compute_chi2,
    "Get chi^2",
    py::arg("datavector")
  );

  m.def("compute_baryon_ratio",
    &cpp_compute_baryon_ratio,
    "Get Baryon Ratio"
  );

  m.def("reset_baryionic_struct",
    &cpp_reset_baryionic_struct,
    "reset baryionic struct to original values"
  );
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

int main()
{
  cpp_initial_setup();
  std::cout << "GOODBYE" << std::endl;
  exit(1);
}
