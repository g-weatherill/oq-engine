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


# Cauzzi et al. 2014

class CauzziEtAl2014LSD(CauzziEtAl2014):
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


class DerrasEtAl2014LSD(DerrasEtAl2014):
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
