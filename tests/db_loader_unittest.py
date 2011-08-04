# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4

# Copyright (c) 2010-2011, GEM Foundation.
#
# OpenQuake is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License version 3
# only, as published by the Free Software Foundation.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License version 3 for more details
# (a copy is included in the LICENSE file that accompanied this code).
#
# You should have received a copy of the GNU Lesser General Public License
# version 3 along with OpenQuake.  If not, see
# <http://www.gnu.org/licenses/lgpl-3.0.txt> for a copy of the LGPLv3 License.

"""
    Unittests for NRML/CSV input files loaders to the database
"""

import geoalchemy
import unittest

from openquake import java
from openquake import xml
from openquake.utils import db
from openquake.utils.db import loader as db_loader
from tests.utils import helpers

TEST_SRC_FILE = helpers.get_data_path('example-source-model.xml')

SIMPLE_FAULT_OUTLINE_WKT = \
'''SRID=4326;POLYGON((
-121.7467204676017 37.7997646326561 8.0,
 -121.75532031089188 37.805642887351446 8.0,
 -121.76392151966904 37.81152051630329 8.0,
 -121.77252409429458 37.817397519279645 8.0,
 -121.78112803513002 37.82327389604836 8.0,
 -121.78973334253678 37.82914964637729 8.0,
 -121.79834001687624 37.83502477003418 8.0,
 -121.80694805850989 37.840899266786685 8.0,
 -121.81555746779918 37.84677313640241 8.0,
 -121.82416824510553 37.85264637864887 8.0,
 -121.8327803907904 37.85851899329353 8.0,
 -121.8413939052153 37.86439098010374 8.0,
 -121.85000878874159 37.87026233884681 8.0,
 -121.85862504173078 37.87613306928992 8.0,
 -121.86724266454432 37.88200317120026 8.0,
 -121.8758616575437 37.88787264434489 8.0,
 -121.88448202109029 37.893741488490775 8.0,
 -121.89310375554568 37.89960970340485 8.0,
 -121.90172686127129 37.90547728885395 8.0,
 -121.91035133862856 37.91134424460486 8.0,
 -121.91897718797895 37.91721057042425 8.0,
 -121.92760440968398 37.92307626607874 8.0,
 -121.93623300410509 37.92894133133485 8.0,
 -121.94486297160377 37.93480576595908 8.0,
 -121.95349431254148 37.94066956971779 8.0,
 -121.96212702727968 37.9465327423773 8.0,
 -121.96212702727968 37.9465327423773 8.0,
 -121.95625230605336 37.9518957106558 8.615661475325659,
 -121.95037672704326 37.95725838678357 9.231322950651316,
 -121.94450029000527 37.96262077066159 9.846984425976975,
 -121.93862299469524 37.967982862190716 10.462645901302633,
 -121.932744840869 37.97334466127188 11.078307376628292,
 -121.92686582828229 37.978706167805875 11.69396885195395,
 -121.92098595669077 37.98406738169352 12.309630327279608,
 -121.91510522585013 37.98942830283558 12.925291802605265,
 -121.91510522585013 37.98942830283558 12.925291802605265,
 -121.90647626854076 37.98356513215041 12.925291802605265,
 -121.8978486843434 37.97770133036593 12.925291802605265,
 -121.88922247289645 37.97183689771583 12.925291802605265,
 -121.88059763383829 37.96597183443374 12.925291802605265,
 -121.87197416680722 37.96010614075318 12.925291802605265,
 -121.8633520714417 37.95423981690762 12.925291802605265,
 -121.85473134738001 37.94837286313044 12.925291802605265,
 -121.84611199426057 37.94250527965494 12.925291802605265,
 -121.83749401172173 37.936637066714376 12.925291802605265,
 -121.82887739940193 37.93076822454189 12.925291802605265,
 -121.82026215693944 37.92489875337057 12.925291802605265,
 -121.8116482839727 37.919028653433436 12.925291802605265,
 -121.80303578014009 37.913157924963414 12.925291802605265,
 -121.79442464507997 37.90728656819335 12.925291802605265,
 -121.7858148784307 37.90141458335603 12.925291802605265,
 -121.77720647983072 37.89554197068414 12.925291802605265,
 -121.76859944891841 37.88966873041036 12.925291802605265,
 -121.75999378533214 37.88379486276721 12.925291802605265,
 -121.7513894887103 37.87792036798718 12.925291802605265,
 -121.74278655869134 37.87204524630267 12.925291802605265,
 -121.73418499491355 37.86616949794599 12.925291802605265,
 -121.7255847970154 37.86029312314944 12.925291802605265,
 -121.71698596463537 37.854416122145146 12.925291802605265,
 -121.70838849741173 37.84853849516526 12.925291802605265,
 -121.69979239498296 37.84266024244175 12.925291802605265,
 -121.69979239498296 37.84266024244175 12.925291802605265,
 -121.70566138792265 37.83729930973308 12.309630327279608,
 -121.71152952784557 37.831938085823225 11.69396885195395,
 -121.71739681499416 37.82657657081076 11.078307376628292,
 -121.72326324961074 37.82121476479418 10.462645901302633,
 -121.72912883193759 37.815852667871994 9.846984425976975,
 -121.73499356221691 37.81049028014264 9.231322950651316,
 -121.7408574406909 37.805127601704555 8.615661475325659,
 -121.7467204676017 37.7997646326561 8.0))'''.replace('\n', '')

SIMPLE_FAULT_EDGE_WKT = \
    'SRID=4326;LINESTRING(-121.8229 37.7301 0.0, -122.0388 37.8771 0.0)'


class NrmlModelLoaderTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        """
        One-time setup stuff for this entire test case class.
        """
        super(NrmlModelLoaderTestCase, self).__init__(*args, **kwargs)

        self.src_reader = java.jclass('SourceModelReader')(
            TEST_SRC_FILE, db_loader.SourceModelLoader.DEFAULT_MFD_BIN_WIDTH)
        java.jvm().java.lang.System.setProperty("openquake.nrml.schema",
                                                xml.nrml_schema_file())
        self.sources = self.src_reader.read()
        self.simple, self.complex, self.area, self.point = self.sources

    def test_get_simple_fault_surface(self):
        surface = db_loader.get_fault_surface(self.simple)
        surface_type = surface.__javaclass__.getName()
        self.assertEqual('org.opensha.sha.faultSurface.StirlingGriddedSurface',
            surface_type)

        # these surfaces are complex objects
        # there is a lot here we can test
        # for now, we only test the overall surface area
        # the test value is derived from our test data
        self.assertEqual(200.0, surface.getSurfaceArea())

    def test_get_complex_fault_surface(self):
        surface = db_loader.get_fault_surface(self.complex)
        surface_type = surface.__javaclass__.getName()
        self.assertEqual(
            'org.opensha.sha.faultSurface.ApproxEvenlyGriddedSurface',
            surface_type)

        # as with the simple fault surface, we're only going to test
        # the surface area (for now)
        # the test value is derived from our test data
        self.assertEqual(126615.0, surface.getSurfaceArea())

    def test_get_fault_surface_raises(self):
        """
        Test that the
        :py:function:`openquake.utils.db.loader.get_fault_surface` function
        raises when passed an inappropriate source type.
        """
        self.assertRaises(ValueError, db_loader.get_fault_surface, self.area)
        self.assertRaises(ValueError, db_loader.get_fault_surface, self.point)

    def test_parse_mfd_simple_fault(self):

        expected = {
            'table': 'hzrdi.mfd_evd',
            'data': {
                'max_val': 6.9500000000000002,
                'total_cumulative_rate': 1.8988435199999998e-05,
                'min_val': 6.5499999999999998,
                'bin_size': 0.10000000000000009,
                'mfd_values': [
                    0.0010614989,
                    0.00088291626999999998,
                    0.00073437776999999999,
                    0.00061082879999999995,
                    0.00050806530000000003],
                'total_moment_rate': 281889786038447.25,
                'owner_id': None}}

        mfd = self.simple.getMfd()

        # Assume that this is an 'evenly discretized' MFD
        # We want to do this check so we know right away if our test data
        # has been altered.
        mfd_type = mfd.__javaclass__.getName()
        self.assertEqual(
            '%s.IncrementalMagFreqDist' % db_loader.MFD_PACKAGE, mfd_type)

        # this is the dict we'll be passing to sqlalchemy to do the db insert
        mfd_insert = db_loader.parse_mfd(self.simple, mfd)

        helpers.assertDictAlmostEqual(self, expected, mfd_insert)

    def test_parse_mfd_complex_fault(self):
        expected = {
            'table': 'hzrdi.mfd_tgr',
            'data': {
                'b_val': 0.80000000000000004,
                'total_cumulative_rate': 4.933442096397671e-10,
                'min_val': 6.5499999999999998,
                'max_val': 8.9499999999999993,
                'total_moment_rate': 198544639016.43918,
                'a_val': 1.0,
                'owner_id': None}}

        mfd = self.complex.getMfd()

        mfd_type = mfd.__javaclass__.getName()
        self.assertEqual(
            '%s.GutenbergRichterMagFreqDist' % db_loader.MFD_PACKAGE, mfd_type)

        mfd_insert = db_loader.parse_mfd(self.complex, mfd)

        helpers.assertDictAlmostEqual(self, expected, mfd_insert)

    def test_parse_simple_fault_src(self):

        expected = (
            {'table': 'hzrdi.mfd_evd', 'data': {
                'max_val': 6.9500000000000002,
                'total_cumulative_rate': 1.8988435199999998e-05,
                'min_val': 6.5499999999999998,
                'bin_size': 0.10000000000000009,
                'mfd_values': [
                    0.0010614989, 0.00088291626999999998,
                    0.00073437776999999999, 0.00061082879999999995,
                    0.00050806530000000003],
                'total_moment_rate': 281889786038447.25,
                'owner_id': None}},
            {'table': 'hzrdi.simple_fault', 'data': {
                'name': u'Mount Diablo Thrust',
                'upper_depth': 8.0,
                'mgf_evd_id': None,
                'mfd_tgr_id': None,
                'outline': \
                    geoalchemy.WKTSpatialElement(SIMPLE_FAULT_OUTLINE_WKT),
                'edge': geoalchemy.WKTSpatialElement(SIMPLE_FAULT_EDGE_WKT),
                'lower_depth': 13.0,
                'gid': u'src01',
                'owner_id': None,
                'dip': 38.0,
                'description': None}},
            {'table': 'hzrdi.source', 'data': {
                'r_depth_distr_id': None,
                'name': u'Mount Diablo Thrust',
                'tectonic_region': 'active',
                'rake': 90.0,
                'si_type': 'simple',
                'gid': u'src01',
                'simple_fault_id': None,
                'owner_id': None,
                'hypocentral_depth': None,
                'description': None,
                'input_id': None}})

        simple_data = db_loader.parse_simple_fault_src(self.simple)

        # The WKTSpatialElement objects do not make for nice comparisons.
        # So instead, we'll just test the text element of each object
        # to make sure the coords and spatial reference system match.
        # To do that, we need to remove the WKTSpatialElements so
        # we can compare the rest of the dicts for equality.
        exp_outline = expected[1]['data'].pop('outline')
        actual_outline = simple_data[1]['data'].pop('outline')

        self.assertEqual(exp_outline.geom_wkt, actual_outline.geom_wkt)

        exp_edge = expected[1]['data'].pop('edge')
        actual_edge = simple_data[1]['data'].pop('edge')

        self.assertEqual(exp_edge.geom_wkt, actual_edge.geom_wkt)

        # Now we can test the rest of the data.
        for idx, exp in enumerate(expected):
            helpers.assertDictAlmostEqual(self, exp, simple_data[idx])


class CsvLoaderTestCase(unittest.TestCase):
    """
        Main class to execute tests about CSV
    """

    def setUp(self):
        csv_file = "ISC_snippet.csv"
        self.csv_path = helpers.get_data_path(csv_file)
        self.db_loader = db_loader.CsvModelLoader(self.csv_path, None, 'eqcat')
        self.db_loader._read_model()
        self.csv_reader = self.db_loader.csv_reader

    def test_input_csv_is_of_the_right_len(self):
        # without the header line is 100
        expected_len = 100

        self.assertEqual(len(list(self.csv_reader)), expected_len)

    def test_csv_headers_are_correct(self):
        expected_headers = ['eventid', 'agency', 'identifier', 'year',
            'month', 'day', 'hour', 'minute', 'second', 'time_error',
            'longitude', 'latitude', 'semi_major', 'semi_minor', 'strike',
            'depth', 'depth_error', 'mw_val', 'mw_val_error',
            'ms_val', 'ms_val_error', 'mb_val', 'mb_val_error', 'ml_val',
            'ml_val_error']

        # it's not important that the headers of the csv are in the right or
        # wrong order, by using the DictReader it is sufficient to compare the
        # headers
        expected_headers = sorted(expected_headers)
        csv_headers = sorted(self.csv_reader.next().keys())
        self.assertEqual(csv_headers, expected_headers)
