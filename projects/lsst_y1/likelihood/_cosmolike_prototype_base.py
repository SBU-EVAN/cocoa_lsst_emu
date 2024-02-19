# Python 2/3 compatibility - must be first line
from __future__ import absolute_import, division, print_function
import os
import numpy as np
import scipy
from scipy.interpolate import UnivariateSpline
import sys
import time

# Local
from cobaya.likelihoods.base_classes import DataSetLikelihood
from cobaya.log import LoggedError
from getdist import IniFile

import cosmolike_lsst_y1_interface as ci

# default is best fit LCDM - just need to be an ok Cosmology
default_omega_matter = 0.315
default_hubble = 74.03
default_omega_nu_h2 = 0.0

default_z = np.array([
  0.,          0.11101010,  0.22202020,  0.33303030,  0.44404040,  0.55505051,
  0.66606061,  0.77707071,  0.88808081,  0.99909091,  1.11010101,  1.22111111,
  1.33212121,  1.44313131,  1.55414141,  1.66515152,  1.77616162,  1.88717172,
  1.99818182,  2.10919192,  2.22020202,  2.33121212,  2.44222222,  2.55323232,
  2.66424242,  2.77525253,  2.88626263,  2.99727273,  3.10828283,  3.21929293,
  3.33030303,  3.44131313,  3.55232323,  3.66333333,  3.77434343,  3.88535354,
  3.99636364,  4.10737374,  4.21838384,  4.32939394,  4.44040404,  4.55141414,
  4.66242424,  4.77343434,  4.88444444,  4.99545455,  5.10646465,  5.21747475,
  5.32848485,  5.43949495,  5.55050505,  5.66151515,  5.77252525,  5.88353535,
  5.99454545,  6.10555556,  6.21656566,  6.32757576,  6.43858586,  6.54959596,
  6.66060606,  6.77161616,  6.88262626,  6.99363636,  7.10464646,  7.21565657,
  7.32666667,  7.43767677,  7.54868687,  7.65969697,  7.77070707,  7.88171717,
  7.99272727,  8.10373737,  8.21474747,  8.32575758,  8.43676768,  8.54777778,
  8.65878788,  8.76979798,  8.88080808,  8.99181818,  9.10282828,  9.21383838,
  9.32484848,  9.43585859,  9.54686869,  9.65787879,  9.76888889,  9.87989899,
  9.99090909, 10.10191919, 10.21292929, 10.32393939, 10.43494949, 10.54595960,
  10.6569697, 10.76797980, 10.87898990, 10.99000000
  ])

default_chi = np.array([
   0.0,439.4981441901221,858.0416237527756,1254.8408009609223,1629.709749743714,
   1982.9621087888709,2315.2896285308143,2627.6450104613505,2921.140613926014,
   3196.968260667916,3456.339782258587,3700.4456271142944,3930.4279102286982,
   4147.3643438650115,4352.259985452185,4546.044385041486,4729.5721949378385,
   4903.626171493769,5068.921287405417,5226.109582520614,5375.785225614292,
   5518.489536109791,5654.715801869407,5784.913800415938,5909.493978237531,
   6028.8312839788305,6143.26859641929,6253.119923208842,6358.673177750374,
   6460.192721741891,6557.921633986121,6652.083741920545,6742.885440028272,
   6830.517317562262,6915.1556161061435,6996.963542799694,7076.092412212595,
   7152.682741675389,7226.865166025485,7298.761294688355,7368.4844768438925,
   7436.140491426934,7501.828170160967,7565.63996211106,7627.6624396908255,
   7687.976770284773,7746.659112922598,7803.781032248162,7859.409823733494,
   7913.608835484643,7966.437756844108,8017.952882055976,8068.207351447877,
   8117.251372316519,8165.132421467811,8211.895435317594,8257.582964035817,
   8302.23535405978,8345.890875994064,8388.585862443648,8430.354831318795,
   8471.230599781875,8511.244389653995,8550.425925019172,8588.803522692133,
   8626.404179504872,8663.253636772872,8699.376473063612,8734.796157982593,
   8769.535118356289,8803.614796722586,8837.055705886985,8869.877479851164,
   8902.098921393474,8933.738046556602,8964.812129049888,8995.337728085331,
   9025.330741701371,9054.80642977566,9083.779449000327,9112.263883075733,
   9140.273271085562,9167.820634180674,9194.918500688977,9221.578929759315,
   9247.813535983449,9273.63350098463,9299.04960740926,9324.072248139235,
   9348.711446338273,9372.976872144722,9396.877858455327,9420.423415857691,
   9443.622246764959,9466.482758802447,9489.013079507331,9511.221060266003,
   9533.114299897283,9554.700147330219,9575.985713926271
 ])

class _cosmolike_prototype_base(DataSetLikelihood):
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def initialize(self, probe):

    # ------------------------------------------------------------------------
    ini = IniFile(os.path.normpath(os.path.join(self.path, self.data_file)))

    self.probe = probe

    self.IA_type = ini.int('IA_model')

    self.data_vector_file = ini.relativeFileName('data_file')

    self.cov_file = ini.relativeFileName('cov_file')

    self.mask_file = ini.relativeFileName('mask_file')

    self.lens_file = ini.relativeFileName('nz_lens_file')

    self.source_file = ini.relativeFileName('nz_source_file')

    self.ggl_olap_cut = ini.float("lensing_overlap_cut")

    self.lens_ntomo = ini.int("lens_ntomo") #5

    self.source_ntomo = ini.int("source_ntomo") #4

    self.ntheta = ini.int("n_theta")

    self.theta_min_arcmin = ini.float("theta_min_arcmin")

    self.theta_max_arcmin = ini.float("theta_max_arcmin")

    self.force_cache_false = False

    # ------------------------------------------------------------------------
    self.z_interp_1D = np.linspace(0,2.0,1000)
    self.z_interp_1D = np.concatenate((self.z_interp_1D,
      np.linspace(2.0,10.1,200)),axis=0)
    self.z_interp_1D = np.concatenate((self.z_interp_1D,
      np.linspace(1080,2000,20)),axis=0) #CMB 6x2pt g_CMB (possible in the future)
    self.z_interp_1D[0] = 0

    self.z_interp_2D = np.linspace(0,2.0,100)
    self.z_interp_2D = np.concatenate((self.z_interp_2D, np.linspace(2.0,10.1,50)),axis=0)
    self.z_interp_2D[0] = 0

    self.len_z_interp_2D = len(self.z_interp_2D)
    self.len_log10k_interp_2D = 1200
    self.log10k_interp_2D = np.linspace(-4.2,2.0,self.len_log10k_interp_2D)

    # Cobaya wants k in 1/Mpc
    self.k_interp_2D = np.power(10.0,self.log10k_interp_2D)
    self.len_k_interp_2D = len(self.k_interp_2D)
    self.len_pkz_interp_2D = self.len_log10k_interp_2D*self.len_z_interp_2D
    self.extrap_kmax = 2.5e2 * self.accuracyboost

    # ------------------------------------------------------------------------

    ci.initial_setup()
    ci.init_accuracy_boost(self.accuracyboost, self.samplingboost, self.integration_accuracy)

    ci.init_probes(possible_probes=self.probe)

    ci.init_binning(self.ntheta, self.theta_min_arcmin, self.theta_max_arcmin)

    ci.init_cosmo_runmode(is_linear=False)

    # to set lens tomo bins, we need a default \chi(z)
    ci.set_cosmological_parameters(omega_matter = default_omega_matter,
      hubble = default_hubble, is_cached = False)

    # convert chi to Mpc/h
    ci.init_distances(default_z, default_chi*default_hubble/100.0)

    ci.init_IA(self.IA_type)

    ci.init_source_sample(self.source_file, self.source_ntomo)

    ci.init_lens_sample(self.lens_file, self.lens_ntomo, self.ggl_olap_cut)

    ci.init_size_data_vector()

    ci.init_data_real(self.cov_file, self.mask_file, self.data_vector_file)

    # FOR ALLOWED OPTIONS FOR `which_baryonic_simulations`, SEE BARYONS.C
    # FUNCTION `void init_baryons(char* scenario)`. SIMS INCLUDE
    # TNG100, HzAGN, mb2, illustris, eagle, owls_AGN_T80, owls_AGN_T85,
    # owls_AGN_T87, BAHAMAS_T76, BAHAMAS_T78, BAHAMAS_T80
    ci.init_baryons_contamination(
      self.use_baryonic_simulations_for_dv_contamination,
      self.which_baryonic_simulations_for_dv_contamination)

    if self.create_baryon_pca:
      ci.init_baryon_pca_scenarios(self.baryon_pca_select_simulations)
      self.use_baryon_pca = False
    else:
      if ini.string('baryon_pca_file', default=''):
        baryon_pca_file = ini.relativeFileName('baryon_pca_file')
        self.baryon_pcs = np.loadtxt(baryon_pca_file)
        self.log.info('use_baryon_pca = True')
        self.log.info('baryon_pca_file = %s loaded', baryon_pca_file)
        self.use_baryon_pca = True
      else:
        self.log.info('use_baryon_pca = False')
        self.use_baryon_pca = False

    self.baryon_pcs_qs = np.zeros(4)
    # ------------------------------------------------------------------------

    self.do_cache_lnPL = np.zeros(
      self.len_log10k_interp_2D*self.len_z_interp_2D)

    self.do_cache_lnPNL = np.zeros(
      self.len_log10k_interp_2D*self.len_z_interp_2D)

    self.do_cache_chi = np.zeros(len(self.z_interp_1D))

    self.do_cache_cosmo = np.zeros(2)

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def get_requirements(self):
    return {
      "H0": None,
      "omegam": None,
      "Pk_interpolator": {
        "z": self.z_interp_2D,
        "k_max": self.kmax_boltzmann * self.accuracyboost,
        "nonlinear": (True,False),
        "vars_pairs": ([("delta_tot", "delta_tot")])
      },
      "comoving_radial_distance": {
        "z": self.z_interp_1D
      # Get comoving radial distance from us to redshift z in Mpc.
      },
      "Cl": { # DONT REMOVE THIS - SOME WEIRD BEHAVIOR IN CAMB WITHOUT WANTS_CL
        'tt': 0
      }
    }

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def compute_logp(self, datavector):
    return -0.5 * ci.compute_chi2(datavector)

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def set_cache_alert(self, chi, lnPL, lnPNL):
    cache_alert_1 = np.array_equal(self.do_cache_chi, chi)

    cache_alert_2 = np.array_equal(self.do_cache_lnPL, lnPL)

    cache_alert_3 = np.array_equal(self.do_cache_lnPNL, lnPNL)

    cache_alert_4 = np.array_equal(
      self.do_cache_cosmo,
      np.array([
        self.provider.get_param("omegam"),
        self.provider.get_param("H0")
      ])
    )

    return cache_alert_1 and cache_alert_2 and cache_alert_3 and cache_alert_4 and not self.force_cache_false

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def set_cosmo_related(self):
    h = self.provider.get_param("H0")/100.0

    # Compute linear matter power spectrum
    PKL = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"),
      nonlinear=False, extrap_kmax = self.extrap_kmax)

    # Compute non-linear matter power spectrum
    PKNL = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"),
      nonlinear=True, extrap_kmax = self.extrap_kmax)

    lnPL = np.empty(self.len_pkz_interp_2D)
    lnPNL = np.empty(self.len_pkz_interp_2D)
    #for i in range(self.len_k_interp_2D) :
    #  lnPNL[i*self.len_z_interp_2D:(i+1)*self.len_z_interp_2D] = PKNL.logP(self.z_interp_2D, self.k_interp_2D[i])[0:self.len_z_interp_2D]
    #  lnPL[i*self.len_z_interp_2D:(i+1)*self.len_z_interp_2D] = PKL.logP(self.z_interp_2D, self.k_interp_2D[i])[0:self.len_z_interp_2D]
    tmp1 = PKNL.logP(self.z_interp_2D, self.k_interp_2D).flatten()
    tmp2 = PKL.logP(self.z_interp_2D, self.k_interp_2D).flatten()
    for i in range(self.len_z_interp_2D):
      lnPNL[i::self.len_z_interp_2D] = tmp1[i*self.len_k_interp_2D:(i+1)*self.len_k_interp_2D]
      lnPL[i::self.len_z_interp_2D] = tmp2[i*self.len_k_interp_2D:(i+1)*self.len_k_interp_2D]

    # Cosmolike wants k in h/Mpc
    log10k_interp_2D = self.log10k_interp_2D - np.log10(h)
    lnPNL += np.log((h**3))
    lnPL += np.log((h**3))

    # Compute chi(z) - convert to Mpc/h
    chi = self.provider.get_comoving_radial_distance(self.z_interp_1D) * h

    cache_alert = self.set_cache_alert(chi, lnPL, lnPNL)

    ci.set_cosmological_parameters(
      omega_matter = self.provider.get_param("omegam"),
      hubble = self.provider.get_param("H0"),
      is_cached = cache_alert
    )

    if cache_alert == False :
      self.do_cache_chi = np.copy(chi)

      self.do_cache_lnPL = np.copy(lnPL)

      self.do_cache_lnPNL = np.copy(lnPNL)

      self.do_cache_cosmo = np.array([
        self.provider.get_param("omegam"),
        self.provider.get_param("H0")
      ])

      ci.init_linear_power_spectrum(log10k = log10k_interp_2D,
        z = self.z_interp_2D, lnP = lnPL)

      ci.init_non_linear_power_spectrum(log10k = log10k_interp_2D,
        z = self.z_interp_2D, lnP = lnPNL)

      G_growth = np.sqrt(PKL.P(self.z_interp_2D,0.0005)/PKL.P(0,0.0005))
      G_growth = G_growth*(1 + self.z_interp_2D)/G_growth[len(G_growth)-1]

      ci.init_growth(z = self.z_interp_2D, G = G_growth)

      ci.init_distances(z = self.z_interp_1D, chi = chi)

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def set_source_related(self, **params_values):
    ci.set_nuisance_shear_calib(
      M = [
        params_values.get(p, None) for p in [
          "LSST_M"+str(i+1) for i in range(self.source_ntomo)
        ]
      ]
    )

    ci.set_nuisance_shear_photoz(
      bias = [
        params_values.get(p, None) for p in [
          "LSST_DZ_S"+str(i+1) for i in range(self.source_ntomo)
        ]
      ]
    )

    ci.set_nuisance_ia(
      A1 = [
        params_values.get(p, None) for p in [
          "LSST_A1_"+str(i+1) for i in range(self.source_ntomo)
        ]
      ],
      A2 = [
        params_values.get(p, None) for p in [
          "LSST_A2_"+str(i+1) for i in range(self.source_ntomo)
        ]
      ],
      B_TA = [
        params_values.get(p, None) for p in [
          "LSST_BTA_"+str(i+1) for i in range(self.source_ntomo)
        ]
      ],
    )

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def set_lens_related(self, **params_values):
    ci.set_nuisance_bias(
      B1 = [
        params_values.get(p, None) for p in [
          "LSST_B1_"+str(i+1) for i in range(self.lens_ntomo)
        ]
      ],
      B2 = [
        params_values.get(p, None) for p in [
          "LSST_B2_"+str(i+1) for i in range(self.lens_ntomo)
        ]
      ],
      B_MAG = [
        params_values.get(p, None) for p in [
          "LSST_BMAG_"+str(i+1) for i in range(self.lens_ntomo)
        ]
      ]
    )
    ci.set_nuisance_clustering_photoz(
      bias = [
        params_values.get(p, None) for p in [
          "LSST_DZ_L"+str(i+1) for i in range(self.lens_ntomo)
        ]
      ]
    )
    ci.set_point_mass(
      PMV = [
        params_values.get(p, None) for p in [
          "LSST_PM"+str(i+1) for i in range(self.lens_ntomo)
        ]
      ]
    )

  # ------------------------------------------------------------------------
  # --------------------------- baryonic PCAs ------------------------------
  # ------------------------------------------------------------------------

  def set_baryon_related(self, **params_values):
    self.baryon_pcs_qs[0] = params_values.get("LSST_BARYON_Q1", 0.0)
    self.baryon_pcs_qs[1] = params_values.get("LSST_BARYON_Q2", 0.0)
    self.baryon_pcs_qs[2] = params_values.get("LSST_BARYON_Q3", 0.0)
    self.baryon_pcs_qs[3] = params_values.get("LSST_BARYON_Q4", 0.0)
    
  def add_baryon_pcs_to_datavector(self, datavector):    
    return datavector[:] + self.baryon_pcs_qs[0]*self.baryon_pcs[:,0] \
      + self.baryon_pcs_qs[1]*self.baryon_pcs[:,1] \
      + self.baryon_pcs_qs[2]*self.baryon_pcs[:,2] \
      + self.baryon_pcs_qs[3]*self.baryon_pcs[:,3]

  def compute_dm_datavector_masked_reduced_dim(self, **params_values):

    self.force_cache_false = True

    ci.init_baryons_contamination(False, "")

    self.set_cosmo_related()

    self.force_cache_false = False

    if (self.probe != "xi"):
      self.set_lens_related(**params_values)

    if (self.probe != "wtheta"):
      self.set_source_related(**params_values)

    # datavector C++ returns a list (not numpy array)
    return np.array(ci.compute_data_vector_masked_reduced_dim())

  # Hack to create baryonic PCAs
  def compute_barion_datavector_masked_reduced_dim(self, sim, **params_values):

    self.force_cache_false = True

    ci.init_baryons_contamination(True, sim)

    self.set_cosmo_related()

    self.force_cache_false = False

    if (self.probe != "xi"):
      self.set_lens_related(**params_values)

    if (self.probe != "wtheta"):
      self.set_source_related(**params_values)

    # datavector C++ returns a list (not numpy array)
    return np.array(ci.compute_data_vector_masked_reduced_dim())

  def generate_baryonic_PCA(self, **params_values):

    cov_L_cholesky = np.linalg.cholesky(
      ci.get_covariance_masked_reduced_dim())

    inv_cov_L_cholesky = np.linalg.inv(cov_L_cholesky)

    ndata_reduced = ci.get_nreduced_dim()

    nbaryons_scenario = ci.get_baryon_pca_nscenarios()

    modelv_dm = self.compute_dm_datavector_masked_reduced_dim(**params_values)

    baryon_diff = np.zeros(shape=(ndata_reduced, nbaryons_scenario))

    for i in range(nbaryons_scenario):
      modelv_baryon = self.compute_barion_datavector_masked_reduced_dim(
        ci.get_baryon_pca_scenario_name(i), **params_values)

      baryon_diff[:,i] = (modelv_baryon-modelv_dm)

    baryon_weighted_diff = np.dot(inv_cov_L_cholesky, baryon_diff)

    U, Sdig, VT = np.linalg.svd(baryon_weighted_diff, full_matrices=True)

    # MAKE SURE WHATEVER VERSION OF NP HAVE U IN THE RIGHT ORDER
    if(np.all(np.diff(Sdig) <= 0) != True):
      raise LoggedError(self.log, "LOGICAL ERROR WITH NUMPY FUNCTION GEN PCA")

    PCs = np.empty(shape=(ndata_reduced, nbaryons_scenario))

    for i in range(nbaryons_scenario):
      PCs[:,i] = U[:,i]

    PCs = np.dot(cov_L_cholesky, PCs)

    # Now we need to expand the number of dimensions
    ndata = ci.get_ndim()
    PCS_FINAL = np.empty(shape=(ndata, nbaryons_scenario))

    for i in range(nbaryons_scenario):
      PCS_FINAL[:,i] = ci.get_expand_dim_from_masked_reduced_dim(PCs[:,i])

    np.savetxt(self.filename_baryon_pca, PCS_FINAL)

    ci.init_baryons_contamination(False,"")
