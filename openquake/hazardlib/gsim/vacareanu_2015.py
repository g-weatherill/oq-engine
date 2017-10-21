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
Module exports :class:`VacareanuEtAl2015`
               :class:`VacareanuEtAl2015AverageSoil`
"""
from __future__ import division
import numpy as np
# standard acceleration of gravity in m/s**2
from scipy.constants import g


from openquake.hazardlib.gsim.base import GMPE, CoeffsTable
from openquake.hazardlib import const
from openquake.hazardlib.imt import PGA, SA

class VacareanuEtAl2015(GMPE):
    """
    Implements the GMPE of Vacareanu et al. (2015) for application to the
    Vrancea region:

    Vacareanu, R, Radulian M, Iancovici, M, Pavel, F and Neagu C. (2015)
    "Fore-Arc and Back-Arc Ground Motion Prediction Model for Vrancea
    Intermediate Depth Seismic Source", Journal of Earthquake Engineering,
    19, 535-562

    In this version the site response term is separated into rock/stiff soil
    and soft soil conditions.
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
                self.get_distance_term(C, dists.rhypo, sites.backarc) +
                self.get_focal_depth_term(C, rup.hypo_depth) +
                self.get_site_term(C, sites))
        # Mean is returned in terms of ln Y in cm/s/s, convert to g
        mean -= np.log(100.0 * g)
        stddevs = self.get_stddevs(C, dists.rhypo.shape, stddev_types)
        return mean, stddevs

    def get_magnitude_term(self, C, mag):
        """
        Returns the quadratic magnitude scaling term from equation 1
        """
        return C["c1"] + C["c2"] * (mag - 6.0) + C["c3"] * ((mag - 6.0) ** 2.0)

    def get_distance_term(self, C, rhypo, backarc):
        """
        Returns the forearc/backarc adjusted distance term from equation 1
        """
        # ARC term is 0 for backarc sites, 1 for forearc
        arc = np.ones_like(backarc, dtype="float")
        if np.any(backarc):
            arc[backarc] = 0.0
        return C["c4"] * np.log(rhypo) + C["c5"] * (1.0 - arc) * rhypo +\
            C["c6"] * arc * rhypo

    def get_focal_depth_term(self, C, hypo_depth):
        """
        Returns the focal depth calcing term
        """
        return C["c7"] * hypo_depth

    def get_site_term(self, C, sites):
        """
        Returns the site scaling term for soil classes EC8 A - C
        """
        # Soil class B is EC8 A and B combined
        amp = C["c8"] * np.ones_like(sites.vs30)
        # Site classes C or softer are treated as soft soil
        amp[sites.vs30 < 360.0] = C["c9"]
        return amp
        
        
    def get_stddevs(self, C, num_sites, stddev_types):
        """
        Returns the standard deviations
        """
        tau = C["tau"] + np.zeros(num_sites)
        phi = C["phi"] + np.zeros(num_sites)
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

    # Coefficients taken from Table 4
    COEFFS = CoeffsTable(sa_damping=5, table="""\
    imt       c1      c2       c3       c4       c5       c6       c7       c8      c9      c10  sigma    tau    phi
    pga   9.6231  1.4232  -0.1555  -1.1316  -0.0114  -0.0024  -0.0007  -0.0835  0.1589   0.0488  0.698  0.406  0.568
    0.01  9.6231  1.4232  -0.1555  -1.1316  -0.0114  -0.0024  -0.0007  -0.0835  0.1589   0.0488  0.698  0.406  0.568
    0.1   9.6981  1.3679  -0.1423  -0.9889  -0.0135  -0.0026  -0.0017  -0.1965  0.1670   0.0020  0.806  0.468  0.656
    0.2  10.0090  1.3620  -0.1138  -1.0371  -0.0127  -0.0032  -0.0004  -0.1547  0.2861   0.0860  0.792  0.469  0.638
    0.3  10.7033  1.4580  -0.1187  -1.2340  -0.0106  -0.0026   0.0000  -0.1014  0.2659   0.0991  0.783  0.480  0.619
    0.4  10.7701  1.5748  -0.1439  -1.3207  -0.0093  -0.0022   0.0005  -0.1076  0.3062   0.1183  0.810  0.519  0.622
    0.5   9.2327  1.6739  -0.1664  -1.0022  -0.0100  -0.0041   0.0007  -0.0259  0.2576   0.0722  0.767  0.461  0.613
    0.6   8.6445  1.7672  -0.1925  -0.8938  -0.0099  -0.0045  -0.0004  -0.1038  0.2181   0.0179  0.740  0.429  0.603
    0.7   8.7134  1.8500  -0.1990  -0.9780  -0.0088  -0.0039   0.0002  -0.1867  0.1564   0.0006  0.735  0.426  0.599
    0.8   9.0835  1.9066  -0.2022  -1.1044  -0.0078  -0.0031   0.0005  -0.2901  0.0546  -0.1019  0.726  0.417  0.594
    0.9   9.1274  1.9662  -0.2465  -1.1437  -0.0074  -0.0031   0.0001  -0.2804  0.0884  -0.0790  0.719  0.403  0.596
    1.0   8.9987  1.9964  -0.2658  -1.1226  -0.0071  -0.0031  -0.0009  -0.2992  0.0739  -0.0955  0.715  0.400  0.592
    1.2   8.0465  2.0432  -0.2241  -0.9654  -0.0072  -0.0041  -0.0013  -0.2681  0.1476  -0.0412  0.713  0.392  0.595
    1.4   7.0585  2.1148  -0.2167  -0.8011  -0.0078  -0.0049  -0.0013  -0.2566  0.2009  -0.0068  0.714  0.392  0.597
    1.6   6.8329  2.1668  -0.2418  -0.8036  -0.0075  -0.0047  -0.0018  -0.2268  0.2272   0.0211  0.732  0.418  0.601
    1.8   6.4292  2.1988  -0.2468  -0.7625  -0.0073  -0.0047  -0.0020  -0.2464  0.2200   0.0082  0.745  0.427  0.611
    2.0   6.3876  2.2151  -0.2289  -0.8004  -0.0066  -0.0043  -0.0024  -0.2767  0.2134  -0.0091  0.744  0.425  0.611
    2.5   4.4248  2.2541  -0.2144  -0.4280  -0.0079  -0.0061  -0.0031  -0.2924  0.2108  -0.0177  0.750  0.420  0.622
    3.0   4.5395  2.2812  -0.2256  -0.5340  -0.0072  -0.0054  -0.0034  -0.3066  0.1840  -0.0387  0.765  0.436  0.629
    3.5   4.7407  2.2803  -0.2456  -0.6250  -0.0065  -0.0045  -0.0041  -0.3728  0.0918  -0.1192  0.778  0.436  0.645
    4.0   4.4928  2.2796  -0.2580  -0.6215  -0.0062  -0.0041  -0.0048  -0.3763  0.0512  -0.1428  0.792  0.443  0.657
    """)


class VacareanuEtAl2015AverageSoil(VacareanuEtAl2015):
    """
    Implements the GMPE of Vacareanu et al (2015) for strong motion records
    from Vrancea earthquakes.

    Average soil term is used; hence, soil term is switched off and single
    coefficient used for site term
    """
    #: The GMPE requires only the forearc/backarc attenuation term
    REQUIRES_SITES_PARAMETERS = set(('backarc',))

    def get_site_term(self, C, sites):
        """
        Returns the C10 coefficient, representng "average soil" conditions
        """
        return C["c10"] * np.ones_like(sites.backarc)

