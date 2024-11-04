# The Hazard Library
# Copyright (C) 2012-2023 GEM Foundation
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Module :mod:`openquake.hazardlib.mgmpe.m9_basin_term` implements
:class:`~openquake.hazardlib.mgmpe.M9BasinTerm`
"""
import numpy as np

from openquake.hazardlib import const
from openquake.hazardlib.gsim.base import GMPE, registry


def _get_m9_basin_term(ctx, imt, mean):
    if imt.period > 1.9: # Only apply to long-period SA
        fb_m9 = np.log(2.0)
        idx = ctx.z2pt5 > 6.0 # Apply only if z2pt5 > 6
        mean[idx] += fb_m9

    return mean


class M9BasinTerm(GMPE):
    """
    Implements a modified GMPE class that can be used to account for basin
    amplification of long period ground-motions within the Seattle Basin
    through the use of the M9 basin amplification model (Frankel et al. 2018,
    Wirth et al. 2018).

    The amplification of the mean ground-motion is uniformly modelled across
    the Seattle Basin as an additive factor of log(2.0) for long period
    ground-motions with a z2pt5 greater than 6.0 km.

    :param gmpe_name:
        The name of a GMPE class
    """
    # Req Params
    REQUIRES_SITES_PARAMETERS = {}

    # Others are set from underlying GMM
    REQUIRES_DISTANCES = set() 
    REQUIRES_RUPTURE_PARAMETERS = set()
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = ""
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set()
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = {const.StdDev.TOTAL}
    DEFINED_FOR_TECTONIC_REGION_TYPE = ""
    DEFINED_FOR_REFERENCE_VELOCITY = None

    def __init__(self, gmpe_name, **kwargs):
        self.gmpe = registry[gmpe_name]()
        self.set_parameters()    

    def compute(self, ctx: np.recarray, imts, mean, sig, tau, phi):      
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.compute>`
        for spec of input and result values.
        """
        self.gmpe.compute(ctx, imts, mean, sig, tau, phi)
        for m, imt in enumerate(imts):
             mean[m] += _get_m9_basin_term(ctx, imt, mean[m])