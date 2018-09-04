# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2015-2018 GEM Foundation
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

import numpy as np
from openquake.hazardlib.gsim.abrahamson_2015 import (
    AbrahamsonEtAl2015SInter, AbrahamsonEtAl2015SInterLow,
    AbrahamsonEtAl2015SInterHigh, AbrahamsonEtAl2015SSlab,
    AbrahamsonEtAl2015SSlabLow, AbrahamsonEtAl2015SSlabHigh)


EPI_FACT = 0.3033 * 1.581

# Attenuation Adjustment
ALA = 0.5
AUA = 1.5


class BCHSInterMidAttCentralMidEpi(AbrahamsonEtAl2015SInter):
    pass


class BCHSInterMidAttLowerMidEpi(AbrahamsonEtAl2015SInterLow):
    pass


class BCHSInterMidAttUpperMidEpi(AbrahamsonEtAl2015SInterHigh):
    pass


class BCHSSlabMidAttCentralMidEpi(AbrahamsonEtAl2015SSlab):
    pass


class BCHSSlabMidAttLowerMidEpi(AbrahamsonEtAl2015SSlabLow):
    pass


class BCHSSlabMidAttUpperMidEpi(AbrahamsonEtAl2015SSlabHigh):
    pass


# =============================================================================
# BCHydro GMPEs for the "Mid" Magnitude Scaling Case
# =============================================================================
class BCHSInterMidAttCentralLowEpi(AbrahamsonEtAl2015SInter):
    """
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super().get_mean_and_stddevs(sites, rup, dists, imt,
                                                     stddev_types)
        return mean - EPI_FACT, stddevs


class BCHSInterMidAttCentralHighEpi(AbrahamsonEtAl2015SInter):
    """
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super().get_mean_and_stddevs(sites, rup, dists, imt,
                                                     stddev_types)
        return mean + EPI_FACT, stddevs


class BCHSInterHighAttCentralMidEpi(AbrahamsonEtAl2015SInter):
    """
    Variant of the AbrahamsonEtAl2015SInter model with reduced attenuation
    (0.5 * theta6)
    """
    def _compute_distance_term(self, C, mag, dists):
        """
        Computes the distance scaling term, as contained within equation (1)
        """
        return (C['theta2'] + self.CONSTS['theta3'] * (mag - 7.8)) *\
            np.log(dists.rrup + self.CONSTS['c4'] * np.exp((mag - 6.) *
                   self.CONSTS['theta9'])) + ((ALA * C['theta6']) * dists.rrup)


class BCHSInterHighAttCentralLowEpi(BCHSInterHighAttCentralMidEpi):
    """
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super().get_mean_and_stddevs(sites, rup, dists, imt,
                                                     stddev_types)
        return mean - EPI_FACT, stddevs


class BCHSInterHighAttCentralHighEpi(BCHSInterHighAttCentralMidEpi):
    """
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super().get_mean_and_stddevs(sites, rup, dists, imt,
                                                     stddev_types)
        return mean + EPI_FACT, stddevs


class BCHSInterLowAttCentralMidEpi(AbrahamsonEtAl2015SInter):
    """
    Variant of the AbrahamsonEtAl2015SInter model with increased attenuation
    (1.5 * theta6)
    """
    def _compute_distance_term(self, C, mag, dists):
        """
        Computes the distance scaling term, as contained within equation (1)
        """
        return (C['theta2'] + self.CONSTS['theta3'] * (mag - 7.8)) *\
            np.log(dists.rrup + self.CONSTS['c4'] * np.exp((mag - 6.) *
                   self.CONSTS['theta9'])) + ((AUA * C['theta6']) * dists.rrup)


class BCHSInterLowAttCentralLowEpi(BCHSInterLowAttCentralMidEpi):
    """
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super().get_mean_and_stddevs(sites, rup, dists, imt,
                                                     stddev_types)
        return mean - EPI_FACT, stddevs


class BCHSInterLowAttCentralHighEpi(BCHSInterLowAttCentralMidEpi):
    """
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super().get_mean_and_stddevs(sites, rup, dists, imt,
                                                     stddev_types)
        return mean + EPI_FACT, stddevs



# =============================================================================
# BCHydro GMPEs for the "Low" Magnitude Scaling Case
# =============================================================================
class BCHSInterMidAttLowerLowEpi(AbrahamsonEtAl2015SInterLow):
    """
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super().get_mean_and_stddevs(sites, rup, dists, imt,
                                                     stddev_types)
        return mean - EPI_FACT, stddevs


class BCHSInterMidAttLowerHighEpi(AbrahamsonEtAl2015SInterLow):
    """
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super().get_mean_and_stddevs(sites, rup, dists, imt,
                                                     stddev_types)
        return mean + EPI_FACT, stddevs


class BCHSInterHighAttLowerMidEpi(AbrahamsonEtAl2015SInterLow):
    """
    Variant of the AbrahamsonEtAl2015SInter model with reduced attenuation
    (0.5 * theta6)
    """
    def _compute_distance_term(self, C, mag, dists):
        """
        Computes the distance scaling term, as contained within equation (1)
        """
        return (C['theta2'] + self.CONSTS['theta3'] * (mag - 7.8)) *\
            np.log(dists.rrup + self.CONSTS['c4'] * np.exp((mag - 6.) *
                   self.CONSTS['theta9'])) + ((ALA * C['theta6']) * dists.rrup)


class BCHSInterHighAttLowerLowEpi(BCHSInterHighAttLowerMidEpi):
    """
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super().get_mean_and_stddevs(sites, rup, dists, imt,
                                                     stddev_types)
        return mean - EPI_FACT, stddevs


class BCHSInterHighAttLowerHighEpi(BCHSInterHighAttLowerMidEpi):
    """
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super().get_mean_and_stddevs(sites, rup, dists, imt,
                                                     stddev_types)
        return mean + EPI_FACT, stddevs


class BCHSInterLowAttLowerMidEpi(AbrahamsonEtAl2015SInterLow):
    """
    Variant of the AbrahamsonEtAl2015SInter model with increased attenuation
    (1.5 * theta6)
    """
    def _compute_distance_term(self, C, mag, dists):
        """
        Computes the distance scaling term, as contained within equation (1)
        """
        return (C['theta2'] + self.CONSTS['theta3'] * (mag - 7.8)) *\
            np.log(dists.rrup + self.CONSTS['c4'] * np.exp((mag - 6.) *
                   self.CONSTS['theta9'])) + ((AUA * C['theta6']) * dists.rrup)


class BCHSInterLowAttLowerLowEpi(BCHSInterLowAttLowerMidEpi):
    """
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super().get_mean_and_stddevs(sites, rup, dists, imt,
                                                     stddev_types)
        return mean - EPI_FACT, stddevs


class BCHSInterLowAttLowerHighEpi(BCHSInterLowAttLowerMidEpi):
    """
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super().get_mean_and_stddevs(sites, rup, dists, imt,
                                                     stddev_types)
        return mean + EPI_FACT, stddevs


# =============================================================================
# BCHydro GMPEs for the "High" Magnitude Scaling Case
# =============================================================================
class BCHSInterMidAttUpperLowEpi(AbrahamsonEtAl2015SInterHigh):
    """
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super().get_mean_and_stddevs(sites, rup, dists, imt,
                                                     stddev_types)
        return mean - EPI_FACT, stddevs


class BCHSInterMidAttUpperHighEpi(AbrahamsonEtAl2015SInterHigh):
    """
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super().get_mean_and_stddevs(sites, rup, dists, imt,
                                                     stddev_types)
        return mean + EPI_FACT, stddevs


class BCHSInterHighAttUpperMidEpi(AbrahamsonEtAl2015SInterHigh):
    """
    Variant of the AbrahamsonEtAl2015SInter model with reduced attenuation
    (0.5 * theta6)
    """
    def _compute_distance_term(self, C, mag, dists):
        """
        Computes the distance scaling term, as contained within equation (1)
        """
        return (C['theta2'] + self.CONSTS['theta3'] * (mag - 7.8)) *\
            np.log(dists.rrup + self.CONSTS['c4'] * np.exp((mag - 6.) *
                   self.CONSTS['theta9'])) + ((ALA * C['theta6']) * dists.rrup)


class BCHSInterHighAttUpperLowEpi(BCHSInterHighAttUpperMidEpi):
    """
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super().get_mean_and_stddevs(sites, rup, dists, imt,
                                                     stddev_types)
        return mean - EPI_FACT, stddevs


class BCHSInterHighAttUpperHighEpi(BCHSInterHighAttUpperMidEpi):
    """
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super().get_mean_and_stddevs(sites, rup, dists, imt,
                                                     stddev_types)
        return mean + EPI_FACT, stddevs


class BCHSInterLowAttUpperMidEpi(AbrahamsonEtAl2015SInterHigh):
    """
    Variant of the AbrahamsonEtAl2015SInter model with increased attenuation
    (1.5 * theta6)
    """
    def _compute_distance_term(self, C, mag, dists):
        """
        Computes the distance scaling term, as contained within equation (1)
        """
        return (C['theta2'] + self.CONSTS['theta3'] * (mag - 7.8)) *\
            np.log(dists.rrup + self.CONSTS['c4'] * np.exp((mag - 6.) *
                   self.CONSTS['theta9'])) + ((AUA * C['theta6']) * dists.rrup)


class BCHSInterLowAttUpperLowEpi(BCHSInterLowAttUpperMidEpi):
    """
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super().get_mean_and_stddevs(sites, rup, dists, imt,
                                                     stddev_types)
        return mean - EPI_FACT, stddevs


class BCHSInterLowAttUpperHighEpi(BCHSInterLowAttUpperMidEpi):
    """
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super().get_mean_and_stddevs(sites, rup, dists, imt,
                                                     stddev_types)
        return mean + EPI_FACT, stddevs


# =============================================================================
# BCHydro GMPEs for the "Mid" Magnitude Scaling Case
# =============================================================================
class BCHSSlabMidAttCentralLowEpi(AbrahamsonEtAl2015SSlab):
    """
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super().get_mean_and_stddevs(sites, rup, dists, imt,
                                                     stddev_types)
        return mean - EPI_FACT, stddevs


class BCHSSlabMidAttCentralHighEpi(AbrahamsonEtAl2015SSlab):
    """
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super().get_mean_and_stddevs(sites, rup, dists, imt,
                                                     stddev_types)
        return mean + EPI_FACT, stddevs


class BCHSSlabHighAttCentralMidEpi(AbrahamsonEtAl2015SSlab):
    """
    Variant of the AbrahamsonEtAl2015SSlab model with reduced attenuation
    (0.5 * theta6)
    """
    def _compute_distance_term(self, C, mag, dists):
        """
        Computes the distance scaling term, as contained within equation (1)
        """
        return ((C['theta2'] + C['theta14'] + self.CONSTS['theta3'] *
                (mag - 7.8)) * np.log(dists.rhypo + self.CONSTS['c4'] *
                np.exp((mag - 6.) * self.CONSTS['theta9'])) +
                ((ALA * C['theta6']) * dists.rhypo)) + C["theta10"]


class BCHSSlabHighAttCentralLowEpi(BCHSSlabHighAttCentralMidEpi):
    """
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super().get_mean_and_stddevs(sites, rup, dists, imt,
                                                     stddev_types)
        return mean - EPI_FACT, stddevs


class BCHSSlabHighAttCentralHighEpi(BCHSSlabHighAttCentralMidEpi):
    """
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super().get_mean_and_stddevs(sites, rup, dists, imt,
                                                     stddev_types)
        return mean + EPI_FACT, stddevs


class BCHSSlabLowAttCentralMidEpi(AbrahamsonEtAl2015SSlab):
    """
    Variant of the AbrahamsonEtAl2015SSlab model with increased attenuation
    (1.5 * theta6)
    """
    def _compute_distance_term(self, C, mag, dists):
        """
        Computes the distance scaling term, as contained within equation (1)
        """
        return ((C['theta2'] + C['theta14'] + self.CONSTS['theta3'] *
                (mag - 7.8)) * np.log(dists.rhypo + self.CONSTS['c4'] *
                np.exp((mag - 6.) * self.CONSTS['theta9'])) +
                ((AUA * C['theta6']) * dists.rhypo)) + C["theta10"]


class BCHSSlabLowAttCentralLowEpi(BCHSSlabLowAttCentralMidEpi):
    """
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super().get_mean_and_stddevs(sites, rup, dists, imt,
                                                     stddev_types)
        return mean - EPI_FACT, stddevs


class BCHSSlabLowAttCentralHighEpi(BCHSSlabLowAttCentralMidEpi):
    """
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super().get_mean_and_stddevs(sites, rup, dists, imt,
                                                     stddev_types)
        return mean + EPI_FACT, stddevs


# =============================================================================
# BCHydro GMPEs for the "Low" Magnitude Scaling Case
# =============================================================================
class BCHSSlabMidAttLowerLowEpi(AbrahamsonEtAl2015SSlabLow):
    """
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super().get_mean_and_stddevs(sites, rup, dists, imt,
                                                     stddev_types)
        return mean - EPI_FACT, stddevs


class BCHSSlabMidAttLowerHighEpi(AbrahamsonEtAl2015SSlabLow):
    """
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super().get_mean_and_stddevs(sites, rup, dists, imt,
                                                     stddev_types)
        return mean + EPI_FACT, stddevs


class BCHSSlabHighAttLowerMidEpi(AbrahamsonEtAl2015SSlabLow):
    """
    Variant of the AbrahamsonEtAl2015SSlab model with reduced attenuation
    (0.5 * theta6)
    """
    def _compute_distance_term(self, C, mag, dists):
        """
        Computes the distance scaling term, as contained within equation (1b)
        """
        return ((C['theta2'] + C['theta14'] + self.CONSTS['theta3'] *
                (mag - 7.8)) * np.log(dists.rhypo + self.CONSTS['c4'] *
                np.exp((mag - 6.) * self.CONSTS['theta9'])) +
                ((ALA * C['theta6']) * dists.rhypo)) + C["theta10"]


class BCHSSlabHighAttLowerLowEpi(BCHSSlabHighAttLowerMidEpi):
    """
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super().get_mean_and_stddevs(sites, rup, dists, imt,
                                                     stddev_types)
        return mean - EPI_FACT, stddevs


class BCHSSlabHighAttLowerHighEpi(BCHSSlabHighAttLowerMidEpi):
    """
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super().get_mean_and_stddevs(sites, rup, dists, imt,
                                                     stddev_types)
        return mean + EPI_FACT, stddevs


class BCHSSlabLowAttLowerMidEpi(AbrahamsonEtAl2015SSlabLow):
    """
    Variant of the AbrahamsonEtAl2015SSlab model with increased attenuation
    (1.5 * theta6)
    """
    def _compute_distance_term(self, C, mag, dists):
        """
        Computes the distance scaling term, as contained within equation (1)
        """
        return ((C['theta2'] + C['theta14'] + self.CONSTS['theta3'] *
                (mag - 7.8)) * np.log(dists.rhypo + self.CONSTS['c4'] *
                np.exp((mag - 6.) * self.CONSTS['theta9'])) +
                ((AUA * C['theta6']) * dists.rhypo)) + C["theta10"]


class BCHSSlabLowAttLowerLowEpi(BCHSSlabLowAttLowerMidEpi):
    """
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super().get_mean_and_stddevs(sites, rup, dists, imt,
                                                     stddev_types)
        return mean - EPI_FACT, stddevs


class BCHSSlabLowAttLowerHighEpi(BCHSSlabLowAttLowerMidEpi):
    """
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super().get_mean_and_stddevs(sites, rup, dists, imt,
                                                     stddev_types)
        return mean + EPI_FACT, stddevs


# =============================================================================
# BCHydro GMPEs for the "High" Magnitude Scaling Case
# =============================================================================
class BCHSSlabMidAttUpperLowEpi(AbrahamsonEtAl2015SSlabHigh):
    """
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super().get_mean_and_stddevs(sites, rup, dists, imt,
                                                     stddev_types)
        return mean - EPI_FACT, stddevs


class BCHSSlabMidAttUpperHighEpi(AbrahamsonEtAl2015SSlabHigh):
    """
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super().get_mean_and_stddevs(sites, rup, dists, imt,
                                                     stddev_types)
        return mean + EPI_FACT, stddevs


class BCHSSlabHighAttUpperMidEpi(AbrahamsonEtAl2015SSlabHigh):
    """
    Variant of the AbrahamsonEtAl2015SSlab model with reduced attenuation
    (0.5 * theta6)
    """
    def _compute_distance_term(self, C, mag, dists):
        """
        Computes the distance scaling term, as contained within equation (1)
        """
        return ((C['theta2'] + C['theta14'] + self.CONSTS['theta3'] *
                (mag - 7.8)) * np.log(dists.rhypo + self.CONSTS['c4'] *
                np.exp((mag - 6.) * self.CONSTS['theta9'])) +
                ((ALA * C['theta6']) * dists.rhypo)) + C["theta10"]


class BCHSSlabHighAttUpperLowEpi(BCHSSlabHighAttUpperMidEpi):
    """
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super().get_mean_and_stddevs(sites, rup, dists, imt,
                                                     stddev_types)
        return mean - EPI_FACT, stddevs


class BCHSSlabHighAttUpperHighEpi(BCHSSlabHighAttUpperMidEpi):
    """
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super().get_mean_and_stddevs(sites, rup, dists, imt,
                                                     stddev_types)
        return mean + EPI_FACT, stddevs


class BCHSSlabLowAttUpperMidEpi(AbrahamsonEtAl2015SSlabHigh):
    """
    Variant of the AbrahamsonEtAl2015SSlab model with increased attenuation
    (1.5 * theta6)
    """
    def _compute_distance_term(self, C, mag, dists):
        """
        Computes the distance scaling term, as contained within equation (1)
        """
        return ((C['theta2'] + C['theta14'] + self.CONSTS['theta3'] *
                (mag - 7.8)) * np.log(dists.rhypo + self.CONSTS['c4'] *
                np.exp((mag - 6.) * self.CONSTS['theta9'])) +
                ((AUA * C['theta6']) * dists.rhypo)) + C["theta10"]


class BCHSSlabLowAttUpperLowEpi(BCHSSlabLowAttUpperMidEpi):
    """
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super().get_mean_and_stddevs(sites, rup, dists, imt,
                                                     stddev_types)
        return mean - EPI_FACT, stddevs


class BCHSSlabLowAttUpperHighEpi(BCHSSlabLowAttUpperMidEpi):
    """
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super().get_mean_and_stddevs(sites, rup, dists, imt,
                                                     stddev_types)
        return mean + EPI_FACT, stddevs
