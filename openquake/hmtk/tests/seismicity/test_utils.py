# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4

#
# LICENSE
#
# Copyright (C) 2010-2025 GEM Foundation, G. Weatherill, M. Pagani,
# D. Monelli.
#
# The Hazard Modeller's Toolkit is free software: you can redistribute
# it and/or modify it under the terms of the GNU Affero General Public
# License as published by the Free Software Foundation, either version
# 3 of the License, or (at your option) any later version.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>
#
# DISCLAIMER
#
# The software Hazard Modeller's Toolkit (hmtk) provided herein
# is released as a prototype implementation on behalf of
# scientists and engineers working within the GEM Foundation (Global
# Earthquake Model).
#
# It is distributed for the purpose of open collaboration and in the
# hope that it will be useful to the scientific, engineering, disaster
# risk and software design communities.
#
# The software is NOT distributed as part of GEM’s OpenQuake suite
# (https://www.globalquakemodel.org/tools-products) and must be considered as a
# separate entity. The software provided herein is designed and implemented
# by scientific staff. It is not developed to the design standards, nor
# subject to same level of critical review by professional software
# developers, as GEM’s OpenQuake software suite.
#
# Feedback and contribution to the software is welcome, and can be
# directed to the hazard scientific staff of the GEM Model Facility
# (hazard@globalquakemodel.org).
#
# The Hazard Modeller's Toolkit (hmtk) is therefore distributed WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details.
#
# The GEM Foundation, and the authors of the software, assume no
# liability for use of the software.

"""
Module :mod:`openquake.hmtk.tests.seismicity.test_utils`
implements :class:`TestDecimaltime`.
"""

import os
import numpy
import unittest
from openquake.hmtk.parsers.catalogue import CsvCatalogueParser

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class TestDecimaltime(unittest.TestCase):
    """
    Tests calculation of decimal time
    """

    def test_convert_to_decimal_1(self):
        fname = os.path.join(DATA_DIR, "test_cat_01.csv")
        parser = CsvCatalogueParser(fname)
        cat = parser.read_file()
        with self.assertRaises(ValueError):
            _ = cat.get_decimal_time()

    def test_convert_to_decimal_2(self):
        fname = os.path.join(DATA_DIR, "test_cat_02.csv")
        parser = CsvCatalogueParser(fname)
        cat = parser.read_file()

        for lab in ["day", "hour", "minute", "second"]:
            idx = numpy.isnan(cat.data[lab])
            if lab == "day":
                cat.data[lab][idx] = 1
            elif lab == "second":
                cat.data[lab][idx] = 0.0
            else:
                cat.data[lab][idx] = 0
        computed = cat.get_decimal_time()
        expected = numpy.array(
            [
                2015.0,
                1963.65205479,
                1963.65217088,
                1963.58082192,
                1999.62793753,
            ]
        )
        numpy.testing.assert_almost_equal(computed, expected)
