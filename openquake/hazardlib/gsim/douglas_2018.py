"""
Implementation of the Douglas (2018) proposed logic tree
"""
import numpy as np
from openquake.hazardlib.imt import PGA, SA
from openquake.hazardlib.gsim.base import GMPE
from openquake.hazardlib.gsim.kotha_2016 import (KothaEtAl2016Italy,
                                                 KothaEtAl2016Turkey,
                                                 KothaEtAl2016Other,
                                                 KothaEtAl2016)

def sigma_mu(mag, rake, imt):
    """
    Implements the epistemic adjustment factor of Al-Atik and Youngs (2014)
    """
    if isinstance(imt, PGA):
        period = 0.0
    elif isinstance(imt, SA):
        period = imt.period
    else:
        raise ValueError("Al Atik & Youngs Correction not applicable for %s"
                         % imt)
    if mag < 7.0:
        sigma_mu = 0.083
    else:
        sigma_mu = 0.056 * (mag - 7.0) + 0.083
    if period >= 0.5:
        sigma_mu += (0.0171 * np.log(period))
    if (rake < 120.0) and (rake > 60.0):
        # In original DB normal faulting case increases uncertainty,
        # here is it reverse faulting
        sigma_mu += 0.038
    return 1.645 * sigma_mu


STRESS_DROP_FACTORS_SP = {
    "low": -0.5,
    "middle": 0.2,
    "high": 0.9}

STRESS_DROP_FACTORS_LP = {
    "low": -0.4,
    "middle": 0.1,
    "high": 0.6}


def stress_drop_factor(imt, branch="middle"):
    """
    """
    if isinstance(imt, PGA):
        return STRESS_DROP_FACTORS_SP[branch]
    elif isinstance(imt, SA):
        if imt.period < 1.0:
            return STRESS_DROP_FACTORS_SP[branch]
        else:
            return STRESS_DROP_FACTORS_LP[branch]
    else:
        raise ValueError("Stress drop factor not supported for IMT %s" % imt)

# ////////////////////////////////////////////////////////////////////////////
# Kotha Italy
# ////////////////////////////////////////////////////////////////////////////
class KothaItalySDMiddleSigmaMuCentral(KothaEtAl2016):
    """
    """
    BRANCH = "middle"
    EPSILON_MU = 0.0
    GSIM = KothaEtAl2016Italy()
    REQUIRES_RUPTURE_PARAMETERS = set(("mag", "rake"))

    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        """
        imean, stddevs = self.GSIM.get_mean_and_stddevs(sites, rup, dists, imt,
                                                       stddev_types)
        mean = imean +\
            (self.EPSILON_MU * sigma_mu(rup.mag, rup.rake, imt)) +\
            stress_drop_factor(imt, self.BRANCH)
        return mean, stddevs


class KothaItalySDLowSigmaMuCentral(KothaItalySDMiddleSigmaMuCentral):
    BRANCH = "low"
    EPSILON_MU = 0.0
    GSIM = KothaEtAl2016Italy()


class KothaItalySDHighSigmaMuCentral(KothaItalySDMiddleSigmaMuCentral):
    BRANCH = "high"
    EPSILON_MU = 0.0
    GSIM = KothaEtAl2016Italy()
               

class KothaItalySDMiddleSigmaMuLow(KothaItalySDMiddleSigmaMuCentral):
    """
    """
    BRANCH = "middle"
    EPSILON_MU = -1.0
    GSIM = KothaEtAl2016Italy()


class KothaItalySDLowSigmaMuLow(KothaItalySDMiddleSigmaMuCentral):
    BRANCH = "low"
    EPSILON_MU = -1.0
    GSIM = KothaEtAl2016Italy()


class KothaItalySDHighSigmaMuLow(KothaItalySDMiddleSigmaMuCentral):
    BRANCH = "high"
    EPSILON_MU = -1.0
    GSIM = KothaEtAl2016Italy()


class KothaItalySDMiddleSigmaMuHigh(KothaItalySDMiddleSigmaMuCentral):
    """
    """
    BRANCH = "middle"
    EPSILON_MU = 1.0
    GSIM = KothaEtAl2016Italy()


class KothaItalySDLowSigmaMuHigh(KothaItalySDMiddleSigmaMuCentral):
    BRANCH = "low"
    EPSILON_MU = 1.0
    GSIM = KothaEtAl2016Italy()


class KothaItalySDHighSigmaMuHigh(KothaItalySDMiddleSigmaMuCentral):
    BRANCH = "high"
    EPSILON_MU = 1.0
    GSIM = KothaEtAl2016Italy()

# ////////////////////////////////////////////////////////////////////////////
# Kotha Turkey
# ////////////////////////////////////////////////////////////////////////////

class KothaTurkeySDMiddleSigmaMuCentral(KothaItalySDMiddleSigmaMuCentral):
    BRANCH = "middle"
    EPSILON_MU = 0.0
    GSIM = KothaEtAl2016Turkey()


class KothaTurkeySDLowSigmaMuCentral(KothaItalySDMiddleSigmaMuCentral):
    BRANCH = "low"
    EPSILON_MU = 0.0
    GSIM = KothaEtAl2016Turkey()


class KothaTurkeySDHighSigmaMuCentral(KothaItalySDMiddleSigmaMuCentral):
    BRANCH = "high"
    EPSILON_MU = 0.0
    GSIM = KothaEtAl2016Turkey()
               

class KothaTurkeySDMiddleSigmaMuLow(KothaItalySDMiddleSigmaMuCentral):
    """
    """
    BRANCH = "middle"
    EPSILON_MU = -1.0
    GSIM = KothaEtAl2016Turkey()


class KothaTurkeySDLowSigmaMuLow(KothaItalySDMiddleSigmaMuCentral):
    BRANCH = "low"
    EPSILON_MU = -1.0
    GSIM = KothaEtAl2016Turkey()


class KothaTurkeySDHighSigmaMuLow(KothaItalySDMiddleSigmaMuCentral):
    BRANCH = "high"
    EPSILON_MU = -1.0
    GSIM = KothaEtAl2016Turkey()


class KothaTurkeySDMiddleSigmaMuHigh(KothaItalySDMiddleSigmaMuCentral):
    """
    """
    BRANCH = "middle"
    EPSILON_MU = 1.0
    GSIM = KothaEtAl2016Turkey()


class KothaTurkeySDLowSigmaMuHigh(KothaItalySDMiddleSigmaMuCentral):
    BRANCH = "low"
    EPSILON_MU = 1.0
    GSIM = KothaEtAl2016Turkey()


class KothaTurkeySDHighSigmaMuHigh(KothaItalySDMiddleSigmaMuCentral):
    BRANCH = "high"
    EPSILON_MU = 1.0
    GSIM = KothaEtAl2016Turkey()


# ////////////////////////////////////////////////////////////////////////////
# Kotha Other
# ////////////////////////////////////////////////////////////////////////////

class KothaOtherSDMiddleSigmaMuCentral(KothaItalySDMiddleSigmaMuCentral):
    BRANCH = "middle"
    EPSILON_MU = 0.0
    GSIM = KothaEtAl2016Other()


class KothaOtherSDLowSigmaMuCentral(KothaItalySDMiddleSigmaMuCentral):
    BRANCH = "low"
    EPSILON_MU = 0.0
    GSIM = KothaEtAl2016Other()


class KothaOtherSDHighSigmaMuCentral(KothaItalySDMiddleSigmaMuCentral):
    BRANCH = "high"
    EPSILON_MU = 0.0
    GSIM = KothaEtAl2016Other()
               

class KothaOtherSDMiddleSigmaMuLow(KothaItalySDMiddleSigmaMuCentral):
    """
    """
    BRANCH = "middle"
    EPSILON_MU = -1.0
    GSIM = KothaEtAl2016Other()


class KothaOtherSDLowSigmaMuLow(KothaItalySDMiddleSigmaMuCentral):
    BRANCH = "low"
    EPSILON_MU = -1.0
    GSIM = KothaEtAl2016Other()


class KothaOtherSDHighSigmaMuLow(KothaItalySDMiddleSigmaMuCentral):
    BRANCH = "high"
    EPSILON_MU = -1.0
    GSIM = KothaEtAl2016Other()


class KothaOtherSDMiddleSigmaMuHigh(KothaItalySDMiddleSigmaMuCentral):
    """
    """
    BRANCH = "middle"
    EPSILON_MU = 1.0
    GSIM = KothaEtAl2016Other()


class KothaOtherSDLowSigmaMuHigh(KothaItalySDMiddleSigmaMuCentral):
    BRANCH = "low"
    EPSILON_MU = 1.0
    GSIM = KothaEtAl2016Other()


class KothaOtherSDHighSigmaMuHigh(KothaItalySDMiddleSigmaMuCentral):
    BRANCH = "high"
    EPSILON_MU = 1.0
    GSIM = KothaEtAl2016Other()
