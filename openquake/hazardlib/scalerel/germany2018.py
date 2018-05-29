# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2012-2018 GEM Foundation
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
Module :mod:`openquake.hazardlib.scalerel.wc1994` implements :class:`WC1994`.
"""
from math import log10
from openquake.hazardlib.scalerel.base import BaseMSRSigma, BaseASRSigma


class GermanyMSR(BaseMSRSigma):
    """
    Source files contain an MSR for Germany
    Log10 A = -2.44 + 0.59 * mag (sigma = 0.16)
    Implements both magnitude-area scaling relationships.
    """
    def get_median_area(self, mag, rake):
        """
        The values are a function of both magnitude and rake.

        Setting the rake to ``None`` causes their "All" rupture-types
        to be applied.
        """
        return 10.0 ** (-2.44 + 0.59 * mag)

    def get_std_dev_area(self, mag, rake):
        """
        Standard deviation for WC1994. Magnitude is ignored.
        """
        return 0.16
