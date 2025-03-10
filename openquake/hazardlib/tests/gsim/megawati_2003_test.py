# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2025 GEM Foundation
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

from openquake.hazardlib.gsim.megawati_2003 import MegawatiEtAl2003
from openquake.hazardlib.tests.gsim.utils import BaseGSIMTestCase


class Megawati2003TestCase(BaseGSIMTestCase):
    GSIM_CLASS = MegawatiEtAl2003

    def test_all(self):
        # built with utils/build_vtable MegawatiEtAl2003
        self.check('MegawatiEtAl2003.csv', max_discrep_percentage=1.0)
