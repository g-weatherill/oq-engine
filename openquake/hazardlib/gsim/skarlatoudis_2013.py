# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2015-2017 GEM Foundation
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
Module exports :class:`SkarlatoudisEtAl2015SSlab`
               :class:`SkarlatoudisEtAl2015SInter`
"""
from __future__ import division
import numpy as np
# standard acceleration of gravity in m/s**2
from scipy.constants import g


from openquake.hazardlib.gsim.base import GMPE, CoeffsTable
from openquake.hazardlib import const
from openquake.hazardlib.imt import PGA, PGV, SA

class SkarlatoudisEtAl2013SSlab(GMPE):
    """

    """
        #: The GMPE is derived for non-subduction deep earthquakes in Vrancea
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.SUBDUCTION_INTRASLAB

    #: Supported intensity measure types are peak ground acceleration,
    #:and spectral acceleration
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set([
        PGA,
        SA
    ])

    #: Supported intensity measure component is assumed to be geometric mean
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.AVERAGE_HORIZONTAL

    #: Supported standard deviation types is total.
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = set([
        const.StdDev.INTER_EVENT,
        const.StdDev.INTRA_EVENT,
        const.StdDev.TOTAL,
    ])

    #: The GMPE provides a Vs30-dependent site scaling term and a forearc/
    #: backarc attenuation term
    REQUIRES_SITES_PARAMETERS = set(('vs30', 'backarc'))

    #: Required rupture parameters are magnitude and focal depth
    REQUIRES_RUPTURE_PARAMETERS = set(('mag', 'hypo_depth'))

    #: Required distance measure is hypocentral distance
    REQUIRES_DISTANCES = set(('rhypo',))

    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.get_mean_and_stddevs>`
        for spec of input and result values.
        """
        C = self.COEFFS[imt]
        mean = (self.get_magnitude_term(C, rup.mag) +
                self.get_distance_term(C, dists, sites, rup.hypo_depth) +
                self.get_site_amplification(C, sites.vs30))
        if isinstance(imt, PGV):
            # Returns the PGV in cm/s, but need to conver to natural log
            mean = np.log(10.0 ** mean)
        else:
            # Returns the PGA, SA in cm/s/s, convert to fractiol g and natural
            # logarithm
            mean = np.log((10.0 ** mean) / (100.0 * g))
        stddevs = self.get_stddevs(C, dists.rhypo.shape, stddev_types)
        return mean, stddevs


    def get_magnitude_term(self, C, mag):
        """
        Returns the linear magnitude scaling term
        """
        return C["c1"] + C["c2"] * (mag - 5.5)

    def get_distance_term(self, C, dists, sites, hypo_depth):
        """
        """
        f_r = self.CONSTANTS["c31"] * np.log10(dists.rhypo) + \
            C["c32"] * (dists.rhypo - self.CONSTANTS["rref"])
        # In this formulation the ARC term is 0 for backarc, and 1 for forearc
        arc = np.ones(sites.backarc.shape, dtype=float)
        fhr = self._get_fhr(hypo_depth, dists.rhypo)
        arc[sites.backarc] = 0.0
        if hypo_depth > 100.0:
            f_r += (C["c41"] * (1.0 - arc) + C["c51"] * arc)
        else:
            f_r += (C["c42"] * (1.0 - arc) * fhr + C["c52"] * arc * fhr)
        return f_r


    def _get_fhr(self, hypo_depth, rhypo):
        """

        """
        fhr = np.zeros_like(rhypo)
        if hypo_depth < 80.0:
            idx = np.logical_and(rhypo > 205.0, rhypo <= 355.0)
            fhr[idx] = (205.0 - rhypo[idx]) / 150.0
            idx = rhypo > 355.0
            fhr[idx] = 1.0
        else:
            idx = np.logical_and(rhypo > 140.0, rhypo <= 240.0)
            fhr[idx] = (140.0 - rhypo[idx]) / 100.0
            idx = rhypo > 240.0
            fhr[idx] = 1.0
        return fhr
            

    def get_site_amplification(self, C, vs30):
        """
        Returns the linear amplification terms for NEHRP site classes C and D
        """
        s_c = np.zeros_like(vs30)
        s_d = np.zeros_like(vs30)
        s_c[np.logical_and(vs30 < 760.0, vs30 >= 360.0)] = C["c61"]
        s_d[vs30 < 360.0] = C["c62"]
        return s_c + s_d

    def get_stddevs(self, C, num_sites, stddev_types):
        """
        """
        # Convert standard deviations from log10 to natural log
        tau = np.log(10.0 ** C["tau"]) * np.ones(num_sites)
        phi = np.log(10.0 ** C["phi"]) * np.ones(num_sites)
        stddevs = []
        for stddev_type in stddev_types:
            assert stddev_type in self.DEFINED_FOR_STANDARD_DEVIATION_TYPES
            if stddev_type == const.StdDev.TOTAL:
                stddevs.append(np.sqrt(tau ** 2. + phi ** 2.))
            elif stddev_type == const.StdDev.INTER_EVENT:
                stddevs.append(tau)
            elif stddev_type == const.StdDev.INTRA_EVENT:
                stddevs.append(phi)
        return stddevs

    
    COEFFS = CoeffsTable(sa_damping=5, table="""\
    imt        c1     c2       c32     c41     c42    c51    c52    c61    c62    phi     tau  sigma
    pgv     2.965  1.069  -0.00178  -0.264   0.018  0.390  0.333  0.408  0.599  0.315   0.144  0.346
    pga     4.229  0.877  -0.00206  -0.481  -0.152  0.425  0.303  0.267  0.491  0.352   0.112  0.369
    0.010   4.235  0.876  -0.00206  -0.482  -0.153  0.425  0.304  0.265  0.488  0.353   0.111  0.370
    0.025   4.119  0.877  -0.00202  -0.490  -0.140  0.415  0.326  0.301  0.511  0.352   0.103  0.367
    0.050   4.320  0.863  -0.00212  -0.483  -0.178  0.410  0.286  0.245  0.475  0.376   0.095  0.388
    0.100   4.565  0.867  -0.00244  -0.515  -0.185  0.452  0.371  0.234  0.442  0.404   0.066  0.410
    0.200   4.613  0.842  -0.00199  -0.596  -0.221  0.396  0.291  0.289  0.469  0.379   0.154  0.409
    0.400   4.463  0.926  -0.00190  -0.427  -0.110  0.459  0.295  0.298  0.516  0.322   0.141  0.351
    1.000   3.952  1.102  -0.00178  -0.199   0.112  0.316  0.442  0.371  0.512  0.305   0.201  0.365
    2.000   3.281  1.260  -0.00106  -0.136   0.055  0.196  0.352  0.408  0.578  0.277   0.203  0.343
    4.000   2.588  1.384  -0.00039  -0.179  -0.046  0.113  0.189  0.264  0.475  0.278   0.176  0.329
    """)

    CONSTANTS = {"c31": -1.7, "rref": 1.0}


class SkarlatoudisEtAl2013SInter(SkarlatoudisEtAl2013SSlab):
    """
    Implements subduction interface model of Skarlatoudis et al. 2013
    """
    def get_distance_term(self, C, dists, sites, hypo_depth):
        """
        """
        # In this formulation the ARC term is 0 for backarc, and 1 for forearc
        arc = np.ones(sites.backarc.shape, dtype=float)
        return (self.CONSTANTS["c31"] * np.log10(dists.rhypo) +
            C["c41"] * (1.0 - arc) * (dists.rhypo - self.CONSTANTS["rref"]) +
            C["c42"] * arc * (dists.rhypo - self.CONSTANTS["rref"]))

    COEFFS = CoeffsTable(sa_damping=5, table="""\
    imt        c1    c2      c41      c42   c61   c62 sigma   tau   phi
    pgv     2.783 1.186 -0.00122 -0.00064 0.232 0.428 0.261 0.095 0.277
    pga     3.945 0.974 -0.00172 -0.00099 0.189 0.707 0.330 0.257 0.418
    0.010   3.950 0.972 -0.00172 -0.00099 0.187 0.708 0.331 0.261 0.421
    0.025   3.842 0.951 -0.00169 -0.00096 0.193 0.792 0.326 0.261 0.418
    0.050   4.005 0.938 -0.00167 -0.00100 0.167 0.694 0.347 0.288 0.451
    0.100   4.112 0.910 -0.00163 -0.00091 0.163 0.731 0.377 0.364 0.524
    0.200   4.296 0.907 -0.00174 -0.00099 0.182 0.725 0.354 0.299 0.463
    0.400   4.244 0.985 -0.00177 -0.00089 0.251 0.736 0.338 0.149 0.369
    1.000   3.900 1.171 -0.00162 -0.00094 0.329 0.521 0.259 0.110 0.282
    """)
