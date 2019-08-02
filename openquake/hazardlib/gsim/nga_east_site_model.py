# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2013-2019 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.
"""
Module exports :class:`NGAEastSite`
"""
from copy import deepcopy
import numpy as np
from openquake.hazardlib.gsim.base import registry, GMPE, CoeffsTable
from openquake.hazardlib.imt import PGA, SA
from openquake.hazardlib import const


uppernames = '''
DEFINED_FOR_INTENSITY_MEASURE_TYPES
DEFINED_FOR_STANDARD_DEVIATION_TYPES
REQUIRES_SITES_PARAMETERS
REQUIRES_RUPTURE_PARAMETERS
REQUIRES_DISTANCES
'''.split()


class NGAEastSite(GMPE):
    """
    Implements a meta GMPE object to apply the NGA East site amplification
    model to a given GMPE. The primary objective is the application to the
    current suite of NGA East Models, which are already defined for a
    reference very hard rock condition of Vs30 = 3000 m/s. Nonetheless, the
    model can be applied to any GMPE in the OpenQuake library, albeit that
    in practice it would be preferable to adjust other GMPEs to the very hard
    rock condition rather than rely entirely on extrapolating any existing
    Vs30 amplification model to this extreme reference condition.

    The model contains a linear and a non-linear component of amplification.

    The linear model is described in Stewart et al., (2017):

    Stewart, J. P., Parker, G. A., Harmon, J. A., Atkinson, G. A., Boore, D.
    M., Darragh, R. B., Silva, W. J. and Hashash, Y. M. A. (2017) "Expert Panel
    Recommendations for Ergodic Site Amplification in Central and Eastern
    North America", PEER Report No. 2017/04, Pacific Earthquake Engineering
    Research Center, University of California, Berkeley.

    The nonlinear model is described in Hashash et al. (2017):

    Hashash, Y. M. A., Harmon, J. A., Ilhan, O., Parker, G. and Stewart, J. P.
    (2017), "Recommendation for Ergonic Nonlinear Site Amplification in
    Central and Eastern North America", PEER Report No. 2017/05, Pacific
    Earthquake Engineering Research Center, University of California, Berkeley.

    Note that the uncertainty provided in this model is treated as an
    epistemic rather than aleatory. As such there is no modification of the
    standard deviation model used for the bedrock case. The epistemic
    uncertainty can be added to the model by the user input site_epsilon term,
    which describes the number of standard deviations by which to multiply
    the epistemic uncertainty model, to then be added or subtracted from the
    median amplification model

    :param gmpe:
        Input GMPE for calculation of ground motion on rock

    :param float site_epsilon:
        Number of standard deviations above or below the median to apply
        the epistemic uncertainty model
    """

    #: Supported tectonic region type is 'active shallow crust'
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST

    #: Set of :mod:`intensity measure types <openquake.hazardlib.imt>`
    #: this GSIM can calculate. A set should contain classes from module
    #: :mod:`openquake.hazardlib.imt`.
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set()

    #: Supported intensity measure component is the geometric mean of two
    #: horizontal components
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.RotD50

    #: Supported standard deviation types are inter-event, intra-event
    #: and total
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = set([
        const.StdDev.TOTAL,
        const.StdDev.INTER_EVENT,
        const.StdDev.INTRA_EVENT
    ])

    #: Required site parameter is vs30
    REQUIRES_SITES_PARAMETERS = set(("vs30", ))

    #: Required rupture parameters are magnitude, others will be taken from
    #: the GMPE
    REQUIRES_RUPTURE_PARAMETERS = set(('mag',))

    #: Required distance measure will be set by the GMPE
    REQUIRES_DISTANCES = set(())

    def __init__(self, gmpe_name, site_epsilon=0.0, tau_model="cena",
                 phi_model="cena", phi_s2ss_model="cena", tau_quantile=None,
                 phi_ss_quantile=None, phi_s2ss_quantile=None):
        super().__init__(gmpe_name=gmpe_name, tau_model=tau_model,
                         phi_model=phi_model, phi_s2ss_model=phi_s2ss_model,
                         tau_quantile=tau_quantile,
                         phi_ss_quantile=phi_ss_quantile,
                         phi_s2ss_quantile=phi_s2ss_quantile)
        self.gmpe = registry[gmpe_name]()
        for name in uppernames:
            setattr(self, name,
                    frozenset(getattr(self, name) | getattr(self.gmpe, name)))
        self.site_epsilon = site_epsilon

    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        Returns the amplified median ground motion and standard deviations
        """
        # Get the coefficients for the IMT
        C_LIN = self.LINEAR_COEFFS[imt]
        f760 = self.F760[imt]
        C_NL = self.NONLINEAR_COEFFS[imt]

        # Get PGA on rock (default to Vs30 3000 m/s in NGA Tables
        sites_r = deepcopy(sites)
        sites_r = 3000.0 * np.ones(sites.vs30.shape)
        # In the case that z1pt0 is needed then for the reference rock this
        # must default to 0.0 m
        if hasattr(sites_r, "z1pt0"):
            sites_r.z1pt0 = np.zeros(sites.vs30.shape)
        # In the case that z2pt5 is needed then for the reference rock this
        # must default to 0.0 km
        if hasattr(sites_r, "z2pt5"):
            sites_r.z2pt5 = np.zeros(sites.vs30.shape)

        # Get the PGA on the reference rock condition
        if PGA in self.DEFINED_FOR_INTENSITY_MEASURE_TYPES:
            pga, stddevs = self.gmpe.get_mean_and_stddevs(sites_r, rup, dists,
                                                          PGA(), stddev_types)
        else:
            # If the GMPE is not defined for PGA (as is the case for some
            # NGA East models) then take Sa (0.01 s)
            pga, stddevs = self.gmpe.get_mean_and_stddevs(sites_r, rup, dists,
                                                          SA(0.01),
                                                          stddev_types)

        if not str(imt) == "PGA":
            # Calculate the ground motion at required spectral period for
            # the reference rock
            mean, stddevs = self.gmpe.get_mean_and_stddevs(sites_r, rup, dists,
                                                           imt, stddev_types)
        else:
            # Avoid re-calculating PGA if that was already done!
            mean = np.copy(pga)
        # Get the linear amplification factor
        f_v = self._get_fv(C_LIN, sites)
        # Linear amplification is the sum of the 3000 m/s to 760 m/s adjustment
        # and the linear Vs30-dependent model
        f_lin = f760["F760"] + f_v
        # Get the nonlinear amplification from Hashash et al., (2017)
        f_nl, f_rk = self.get_fnl(C_NL, np.exp(pga), sites.vs30)
        mean += (f_lin + f_nl)
        # If an epistemic uncertainty is required then retrieve the epistemic
        # sigma of both models and multiply by the input epsilon
        if self.site_epsilon:
            # In the case of the linear model sigma_f760 and sigma_fv are
            # assumed independent and the resulting sigma_flin is the root
            # sum of squares (SRSS)
            f_lin_stddev = np.sqrt(
                f760["sigma_F760"] ** 2. +
                self.get_linear_stddev(C_LIN, sites.vs30) ** 2.)
            # Likewise, the epistemic uncertainty on the linear and nonlinear
            # model are assumed independent and the SRSS is taken
            f_nl_stddev = self.get_nonlinear_stddev(C_NL, sites.vs30) * f_rk
            site_epistemic = np.sqrt(f_lin_stddev ** 2. + f_nl_stddev ** 2.)
            mean += (self.site_epsilon * site_epistemic)
        return mean, stddevs

    def _get_fv(self, C_LIN, sites):
        """
        Returns the Vs30-dependent component of the mean linear amplification
        model, as defined in equation 2.3 of Stewart et al. (2017)
        """
        f_v = C_LIN["c"] * np.log(sites.vs30 / self.CONSTANTS["vref"])
        idx = sites.vs30 <= C_LIN["v1"]
        if np.any(idx):
            f_v[idx] = C_LIN["c"] * np.log(C_LIN["v1"] /
                                           self.CONSTANTS["vref"])
        idx = sites.vs30 > C_LIN["v2"]
        if np.any(idx):
            f_v[idx] = C_LIN["c"] *\
                np.log(C_LIN["v2"] / self.CONSTANTS["vref"]) +\
                (C_LIN["c"] / 2.) * np.log(sites.vs30[idx] / C_LIN["v2"])
        return f_v

    def get_fnl(self, C_NL, pga_rock, vs30):
        """
        Returns the nonlinear mean amplification according to equation 2.2
        of Hashash et al. (2017)
        """
        f_nl = np.zeros(vs30.shape)
        f_rk = np.log((pga_rock + C_NL["f3"]) / C_NL["f3"])
        idx = vs30 < C_NL["Vc"]
        if np.any(idx):
            f_2 = self._get_f2(C_NL, vs30[idx])
            f_nl[idx] = f_2 * f_rk[idx]
        return f_nl, f_rk

    def _get_f2(self, C_NL, vs30):
        """
        Returns the f2 term of the mean nonlinear amplification model
        according to equation 2.3 of Hashash et al., (2017)
        """
        c_vs = np.copy(vs30)
        c_vs[c_vs > 3000.] = 3000.
        return C_NL["f4"] * (np.exp(C_NL["f5"] * (c_vs - 360.)) -
                             np.exp(C_NL["f5"] * (3000.0 - 360.)))

    def get_linear_stddev(self, C_LIN, vs30):
        """
        Returns the standard deviation of the linear amplification function,
        as defined in equation 2.4 of Stewart et al., (2017)
        """
        sigma_v = C_LIN["sigma_vc"] + np.zeros(vs30.shape)
        idx = vs30 < C_LIN["vf"]
        if np.any(idx):
            dsig = C_LIN["sigma_L"] - C_LIN["sigma_vc"]
            d_v = (vs30[idx] - self.CONSTANTS["vL"]) /\
                (C_LIN["vf"] - self.CONSTANTS["vL"])
            sigma_v[idx] = C_LIN["sigma_L"] - (2. * dsig * d_v) +\
                dsig * (d_v ** 2.)
        idx = np.logical_and(vs30 > C_LIN["v2"], vs30 <= self.CONSTANTS["vU"])
        if np.any(idx):
            d_v = (vs30[idx] - C_LIN["v2"]) /\
                (self.CONSTANTS["vU"] - C_LIN["v2"])
            sigma_v[idx] = C_LIN["sigma_vc"] + \
                (C_LIN["sigma_U"] - C_LIN["sigma_vc"]) * (d_v ** 2.)
        idx = vs30 > self.CONSTANTS["vU"]
        if np.any(idx):
            sigma_v[idx] = C_LIN["sigma_U"] *\
                (1. - (np.log(vs30[idx] / self.CONSTANTS["vU"]) /
                       np.log(3000. / self.CONSTANTS["vU"])))
        return sigma_v

    def get_nonlinear_stddev(self, C_NL, vs30):
        """
        Returns the standard deviation of the nonlinear amplification function,
        as defined in equation 2.5 of Hashash et al. (2017)
        """
        sigma_f2 = np.zeros(vs30.shape)
        sigma_f2[vs30 < 300.] = C_NL["sigma_c"]
        idx = np.logical_and(vs30 >= 300, vs30 < 1000)
        if np.any(idx):
            sigma_f2[idx] = (-C_NL["sigma_c"] / np.log(1000. / 300.)) *\
                np.log(vs30[idx] / 300.) + C_NL["sigma_c"]
        return sigma_f2

    # Three constants: vref, vL and vU
    CONSTANTS = {"vref": 760., "vL": 200., "vU": 2000.0}

    # Coefficients for the linear model, taken from the electronic supplement
    # to Stewart et al., (2017)
    LINEAR_COEFFS = CoeffsTable(sa_damping=5, table="""\
    imt         c      v1       v2      vf   sigma_vc   sigma_L   sigma_U
    pga    -0.359   333.0    760.0   333.0      0.268     0.281     0.472
    0.01   -0.359   333.0    760.0   333.0      0.268     0.281     0.472
    0.08   -0.359   333.0    760.0   333.0      0.268     0.281     0.472
    0.10   -0.344   319.0    760.0   319.0      0.270     0.263     0.470
    0.20   -0.408   314.0    760.0   314.0      0.251     0.306     0.334
    0.30   -0.416   200.0    760.0   250.0      0.225     0.276     0.381
    0.40   -0.418   200.0    760.0   250.0      0.225     0.275     0.381
    0.50   -0.446   200.0    764.0   280.0      0.225     0.311     0.323
    0.80   -0.527   230.0    942.0   280.0      0.225     0.334     0.308
    1.00   -0.554   278.0   1103.0   300.0      0.225     0.377     0.361
    2.00   -0.580   286.0   1201.0   300.0      0.259     0.548     0.388
    3.00   -0.593   313.0    876.0   313.0      0.306     0.538     0.551
    4.00   -0.597   322.0    881.0   322.0      0.340     0.435     0.585
    5.00   -0.582   325.0    855.0   325.0      0.340     0.400     0.587
    """)

    # Coefficients for the nonlinear model, taken from Table 2.1 of
    # Hashash et al., (2017)
    NONLINEAR_COEFFS = CoeffsTable(sa_damping=5, table="""\
    imt         f3         f4         f5       Vc    sigma_c   Cratio
    pga    0.16249   -0.50667   -0.00273   2990.0       0.12    2.275
    0.01   0.16249   -0.50667   -0.00273   2990.0       0.12    2.275
    0.08   0.16249   -0.50667   -0.00273   2990.0       0.12    2.275
    0.10   0.15083   -0.44661   -0.00335   2990.0       0.12    2.275
    0.20   0.12815   -0.30481   -0.00488   1533.0       0.12    2.275
    0.30   0.13070   -0.22825   -0.00655   1152.0       0.15    2.275
    0.40   0.09414   -0.11591   -0.00872   1018.0       0.15    2.275
    0.50   0.09888   -0.07793   -0.01028    938.0       0.15    2.275
    0.80   0.07357   -0.01592   -0.01515    832.0       0.10    2.275
    1.00   0.04367   -0.00478   -0.01823    951.0       0.06    2.275
    2.00   0.00164   -0.00236   -0.01296    879.0       0.04    2.275
    3.00   0.00746   -0.00626   -0.01043    894.0       0.04    2.275
    4.00   0.00269   -0.00331   -0.01215    875.0       0.03    2.275
    5.00   0.00242   -0.00256   -0.01325    856.0       0.02    2.275
    10.0   0.05329   -0.00631   -0.01403    837.0       0.02    2.275
    """)

    # Amplification factors for reference hard rock (Vs30 = 3000 m/s) to
    # reference rock (Vs30 = 700 m/s), taken from the electronic supplement to
    # Stewart et al., (2017)
    F760 = CoeffsTable(sa_damping=5, table="""\
    imt     F760  sigma_F760
    pga    0.185       0.434
    0.010  0.185       0.434
    0.011  0.185       0.434
    0.011  0.185       0.434
    0.012  0.185       0.434
    0.013  0.185       0.434
    0.014  0.185       0.434
    0.015  0.185       0.434
    0.016  0.185       0.434
    0.017  0.185       0.434
    0.019  0.185       0.434
    0.020  0.185       0.434
    0.022  0.185       0.434
    0.023  0.189       0.432
    0.025  0.195       0.429
    0.027  0.203       0.422
    0.028  0.212       0.414
    0.031  0.224       0.404
    0.033  0.238       0.393
    0.035  0.252       0.387
    0.038  0.267       0.387
    0.040  0.283       0.390
    0.043  0.300       0.390
    0.046  0.318       0.381
    0.050  0.337       0.363
    0.053  0.356       0.340
    0.057  0.377       0.320
    0.061  0.400       0.308
    0.066  0.425       0.306
    0.071  0.454       0.312
    0.076  0.475       0.322
    0.081  0.512       0.335
    0.087  0.558       0.346
    0.093  0.613       0.357
    0.100  0.674       0.366
    0.107  0.730       0.372
    0.115  0.760       0.370
    0.123  0.759       0.351
    0.132  0.714       0.323
    0.142  0.647       0.284
    0.152  0.586       0.253
    0.163  0.534       0.234
    0.175  0.488       0.222
    0.187  0.449       0.214
    0.201  0.419       0.214
    0.215  0.390       0.207
    0.231  0.362       0.195
    0.248  0.332       0.177
    0.266  0.301       0.156
    0.285  0.278       0.141
    0.305  0.270       0.131
    0.327  0.262       0.124
    0.351  0.242       0.117
    0.376  0.224       0.115
    0.404  0.209       0.112
    0.433  0.197       0.113
    0.464  0.186       0.111
    0.498  0.175       0.105
    0.534  0.166       0.104
    0.572  0.157       0.113
    0.614  0.150       0.123
    0.658  0.142       0.132
    0.705  0.135       0.139
    0.756  0.127       0.138
    0.811  0.120       0.133
    0.870  0.111       0.130
    0.933  0.103       0.128
    1.000  0.095       0.124
    1.072  0.088       0.118
    1.150  0.083       0.112
    1.233  0.080       0.108
    1.322  0.078       0.110
    1.417  0.078       0.114
    1.520  0.077       0.120
    1.630  0.077       0.122
    1.748  0.078       0.124
    1.874  0.079       0.124
    2.009  0.079       0.118
    2.154  0.078       0.113
    2.310  0.076       0.112
    2.477  0.075       0.111
    2.656  0.074       0.109
    2.848  0.073       0.108
    3.054  0.073       0.111
    3.275  0.072       0.116
    3.511  0.070       0.120
    3.765  0.068       0.122
    4.037  0.066       0.120
    4.329  0.065       0.116
    4.642  0.065       0.112
    4.977  0.064       0.108
    5.337  0.063       0.104
    5.722  0.061       0.100
    6.136  0.060       0.096
    6.579  0.058       0.091
    7.055  0.057       0.087
    7.565  0.056       0.082
    8.111  0.056       0.078
    8.697  0.055       0.074
    9.326  0.055       0.070
    10.00  0.053       0.069
    """)
