# The Hazard Library
# Copyright (C) 2013-2019 GEM Foundation
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
from openquake.hazardlib.tests.gsim.utils import BaseGSIMTestCase
from openquake.hazardlib.gsim.nga_east_site_model import NGAEastSite


MAX_DISCREP = 0.1


class NGAEastSiteTestCase(BaseGSIMTestCase):
    GSIM_CLASS = NGAEastSite

    def test_mean(self):
        self.check("nga_east_site/NGAEAST_SITE_AMPLIFICATION_MEAN.csv",
                   max_discrep_percentage=MAX_DISCREP,
                   gmpe_name='DarraghEtAl2015NGAEast1CVSP',
                   site_epsilon=0.0)


class NGAEastSitePlus1SigmaTestCase(BaseGSIMTestCase):
    GSIM_CLASS = NGAEastSite

    def test_mean(self):
        self.check("nga_east_site/NGAEAST_SITE_AMPLIFICATION_MEAN_PLUS_1_SIGMA.csv",
                   max_discrep_percentage=MAX_DISCREP,
                   gmpe_name='DarraghEtAl2015NGAEast1CVSP',
                   site_epsilon=1.0)


class NGAEastSiteMinus1SigmaTestCase(BaseGSIMTestCase):
    GSIM_CLASS = NGAEastSite

    def test_mean(self):
        self.check("nga_east_site/NGAEAST_SITE_AMPLIFICATION_MEAN_MINUS_1_SIGMA.csv",
                   max_discrep_percentage=MAX_DISCREP,
                   gmpe_name='DarraghEtAl2015NGAEast1CVSP',
                   site_epsilon=-1.0)

