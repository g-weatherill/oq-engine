"""
Germany GMPEs for National PSHA
"""
import numpy as np
from openquake.hazardlib.gsim.akkar_2014 import AkkarEtAlRhyp2014
from openquake.hazardlib.gsim.bindi_2014 import BindiEtAl2014Rhyp
from openquake.hazardlib.gsim.cauzzi_2014 import CauzziEtAl2014
from openquake.hazardlib.gsim.derras_2014 import DerrasEtAl2014
from openquake.hazardlib.gsim.bindi_2017 import BindiEtAl2017Rhypo


STRESS_DROP_ADJUST = {
    "L": 0.75,
    "M": 1.25,
    "H": 1.5}

def adjustment_fact(median, branch):
    """
    Applies stress drop adjustment factors
    """
    return median + np.log(STRESS_DROP_ADJUST[branch])


class AkkarEtAlRhyp2014LSD(AkkarEtAlRhyp2014):
    """
    Akkar et al. + "Low" Stress Drop Adjustment
    """
    SD_BRANCH = "L"
    
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super(AkkarEtAlRhyp2014LSD, self).get_mean_and_stddevs(
            sites, rup, dists, imt, stddev_types)
        return adjustment_fact(mean, self.SD_BRANCH), stddevs


class AkkarEtAlRhyp2014MSD(AkkarEtAlRhyp2014LSD):
    """
    Akkar et al + "Middle Stress Drop Adjustment
    """
    SD_BRANCH = "M"


class AkkarEtAlRhyp2014HSD(AkkarEtAlRhyp2014LSD):
    """
    Akkar et al "High Stress Drop Adjustment"
    """
    SD_BRANCH = "H"


# Bindi et al 2014


class BindiEtAl2014RhypLSD(BindiEtAl2014Rhyp):
    """
    Bindi et al. + "Low" Stress Drop Adjustment
    """
    SD_BRANCH = "L"
    
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super(BindiEtAl2014RhypLSD, self).get_mean_and_stddevs(
            sites, rup, dists, imt, stddev_types)
        return adjustment_fact(mean, self.SD_BRANCH), stddevs

class BindiEtAl2014RhypMSD(BindiEtAl2014RhypLSD):
    """
    Bindi et al + "Middle Stress Drop Adjustment
    """
    SD_BRANCH = "M"
    

class BindiEtAl2014RhypHSD(BindiEtAl2014RhypLSD):
    """
    Bindi et al "High Stress Drop Adjustment"
    """
    SD_BRANCH = "H"



def rhypo_to_rrup(rhypo, mag):
    """
    """
    rrup = rhypo - (0.7108 + 2.496E-6 * (mag ** 7.982))
    rrup[rrup < 3.0] = 3.0    

def rhypo_to_rjb(rhypo, mag):
    """
    """
    epsilon = rhypo - (4.853 + 1.347E-6 * (mag ** 8.163))
    rjb = np.zeros_like(repi)
    idx = epsilon >= 3.
    rjb[idx] = sqrt((epsilon[idx] ** 2.) - 9.0)
    rjb[rjb < 0.0] = 0.0
    return rjb


# Cauzzi et al. 2014 - Converted from Rhypo
class CauzziEtAl2014Rhypo(CauzziEtAl2014):
    """
    """
    REQUIRES_DISTANCES = set(("rhypo", ))

    def _compute_mean(self, C, rup, dists, sites, imt):
        """
        Returns the mean ground motion acceleration and velocity
        """
        # Convert rhypo to rrup
        rrup = rhypo_to_rrup(dists.rhypo, mag)
        mean = (self._get_magnitude_scaling_term(C, rup.mag) +
                self._get_distance_scaling_term(C, rup.mag, rrup) +
                self._get_style_of_faulting_term(C, rup.rake) +
                self._get_site_amplification_term(C, sites.vs30))
        # convert from cm/s**2 to g for SA and from cm/s**2 to g for PGA (PGV
        # is already in cm/s) and also convert from base 10 to base e.
        if isinstance(imt, PGA):
            mean = np.log((10 ** mean) * ((2 * np.pi / 0.01) ** 2) *
                          1e-2 / g)
        elif isinstance(imt, SA):
            mean = np.log((10 ** mean) * ((2 * np.pi / imt.period) ** 2) *
                          1e-2 / g)
        else:
            mean = np.log(10 ** mean)

        return mean


class CauzziEtAl2014LSD(CauzziEtAl2014Rhypo):
    """
    Cauzzi et al. + "Low" Stress Drop Adjustment
    """
    SD_BRANCH = "L"
    
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super(CauzziEtAl2014LSD, self).get_mean_and_stddevs(
            sites, rup, dists, imt, stddev_types)
        return adjustment_fact(mean, self.SD_BRANCH), stddevs


class CauzziEtAl2014MSD(CauzziEtAl2014LSD):
    """
    Cauzzi et al + "Middle Stress Drop Adjustment
    """
    SD_BRANCH = "M"
    

class CauzziEtAl2014HSD(CauzziEtAl2014LSD):
    """
    Bindi et al "High Stress Drop Adjustment"
    """
    SD_BRANCH = "H"


# Derras et al 2014
class DerrasEtAl2014Rhypo(DerrasEtAl2014):
    """
    Re-calibration of the Derras et al. (2014) GMPE taking hypocentral
    distance as an input and converting to Rjb
    """
    #: The required distance parameter is hypocentral distance
    REQUIRES_DISTANCES = set(('rhypo', ))

    def get_mean(self, C, rup, sites, dists):
        """
        Returns the mean ground motion in terms of log10 m/s/s, implementing
        equation 2 (page 502)
        """
        # W2 needs to be a 1 by 5 matrix (not a vector
        w_2 = np.array([
            [C["W_21"], C["W_22"], C["W_23"], C["W_24"], C["W_25"]]
            ])
        # Gets the style of faulting dummy variable
        sof = self._get_sof_dummy_variable(rup.rake)
        # Get the normalised coefficients
        p_n = self.get_pn(rup, sites, dists, sof)
        mean = np.zeros_like(dists.rhypo)
        # Need to loop over sites - maybe this can be improved in future?
        # ndenumerate is used to allow for application to 2-D arrays
        for idx, rval in np.ndenumerate(p_n[0]):
            # Place normalised coefficients into a single array
            p_n_i = np.array([rval, p_n[1], p_n[2][idx], p_n[3], p_n[4]])
            # Executes the main ANN model
            mean_i = np.dot(w_2, np.tanh(np.dot(self.W_1, p_n_i) + self.B_1))
            mean[idx] = (0.5 * (mean_i + C["B_2"] + 1.0) *
                         (C["tmax"] - C["tmin"])) + C["tmin"]
        return mean

    def get_pn(self, rup, sites, dists, sof):
        """
        Normalise the input parameters within their upper and lower
        defined range
        """
        # List must be in following order
        p_n = []
        # Rjb
        # Note that Rjb must be clipped at 0.1 km
        rjb = rhypo_to_rjb(dists.rhypo, rup.mag)
        rjb[rjb < 0.1] = 0.1
        p_n.append(self._get_normalised_term(np.log10(rjb),
                                             self.CONSTANTS["logMaxR"],
                                             self.CONSTANTS["logMinR"]))
        # Magnitude
        p_n.append(self._get_normalised_term(rup.mag,
                                             self.CONSTANTS["maxMw"],
                                             self.CONSTANTS["minMw"]))
        # Vs30
        p_n.append(self._get_normalised_term(np.log10(sites.vs30),
                                             self.CONSTANTS["logMaxVs30"],
                                             self.CONSTANTS["logMinVs30"]))
        # Depth
        p_n.append(self._get_normalised_term(rup.hypo_depth,
                                             self.CONSTANTS["maxD"],
                                             self.CONSTANTS["minD"]))
        # Style of Faulting
        p_n.append(self._get_normalised_term(sof,
                                             self.CONSTANTS["maxFM"],
                                             self.CONSTANTS["minFM"]))
        return p_n


class DerrasEtAl2014LSD(DerrasEtAl2014Rhypo):
    """
    Derras et al. + "Low" Stress Drop Adjustment
    """
    SD_BRANCH = "L"
    
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super(DerrasEtAl2014LSD, self).get_mean_and_stddevs(
            sites, rup, dists, imt, stddev_types)
        return adjustment_fact(mean, self.SD_BRANCH), stddevs


class DerrasEtAl2014MSD(DerrasEtAl2014LSD):
    """
    Derras et al + "Middle Stress Drop Adjustment
    """
    SD_BRANCH = "M"
    

class DerrasEtAl2014HSD(DerrasEtAl2014LSD):
    """
    Derras et al "High Stress Drop Adjustment"
    """
    SD_BRANCH = "H"


# Bindi et al. (2017)

class BindiEtAl2017RhypoLSD(BindiEtAl2017Rhypo):
    """
    Bindi et al. 2017 + "Low" Stress Drop Adjustment
    """
    SD_BRANCH = "L"
    
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        mean, stddevs = super(BindiEtAl2017RhypoLSD, self).get_mean_and_stddevs(
            sites, rup, dists, imt, stddev_types)
        return adjustment_fact(mean, self.SD_BRANCH), stddevs


class BindiEtAl2017RhypoMSD(BindiEtAl2017RhypoLSD):
    """
    Bindi et al 2017 + "Middle Stress Drop Adjustment
    """
    SD_BRANCH = "M"
    

class BindiEtAl2017RhypoHSD(BindiEtAl2017RhypoLSD):
    """
    Bindi et al 2017 "High Stress Drop Adjustment"
    """
    SD_BRANCH = "H"
