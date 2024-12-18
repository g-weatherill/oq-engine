#!/bin/env python
# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2024, GEM Foundation
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
# along with OpenQuake.  If not, see <http://www.gnu.org/licenses/>.

import os
import logging
import numpy
from openquake.baselib import performance, sap, hdf5
from openquake.hazardlib import valid, gsim_lt
from openquake.commonlib import readinput, datastore
from openquake.calculators import base
from openquake.engine import engine

INPUTS = dict(
    calculation_mode='event_based',
    number_of_logic_tree_samples='2000',
    ses_per_logic_tree_path='50',
    investigation_time='1',
    ground_motion_fields='false',
    minimum_magnitude='5',
    minimum_intensity='.1')
MODELS = sorted('''
ALS AUS CEA EUR HAW KOR NEA PHL ARB IDN MEX NWA PNG SAM TWN
CAN CHN IND MIE NZL SEA USA ZAF CCA JPN NAF PAC SSA WAF
'''.split())  # GLD is missing
# MODELS = 'EUR MIE'.split()

dt = [('model', '<S3'), ('trt', '<S61'), ('gsim', hdf5.vstr), ('weight', float)]

def imts(dic):
    imtls = valid.dictionary(dic['intensity_measure_types_and_levels'])
    return ' '.join(imt for imt in imtls)


def check_imts(dicts, models):
    imts0 = imts(dicts[0])
    for model, imts1 in zip(models[1:], map(imts, dicts[1:])):
        if imts1 != imts0:
            raise ValueError(f'{imts1} != {imts0} for {model}')


def read_job_inis(mosaic_dir, models):
    out = []
    rows = []
    for model in models:
        fname = os.path.join(mosaic_dir, model, 'in', 'job_vs30.ini')
        dic = readinput.get_params(fname)
        dic.update(INPUTS)
        if 'truncation_level' not in dic:  # CAN
            dic['truncation_level'] = '5'
            dic['intensity_measure_types_and_levels'] = '''\
            {"PGA": logscale(0.005, 3.00, 25),
            "SA(0.1)": logscale(0.005, 8.00, 25),
            "SA(0.2)": logscale(0.005, 9.00, 25),
            "SA(0.3)": logscale(0.005, 8.00, 25),
            "SA(0.6)": logscale(0.005, 5.50, 25),
            "SA(1.0)": logscale(0.005, 3.60, 25),
            "SA(2.0)": logscale(0.005, 2.10, 25)}'''
        if model in ("KOR", "JPN"):
            dic['investigation_time'] = '50'
            dic['ses_per_logic_tree_path'] = '1'
        dic['mosaic_model'] = model
        gslt = gsim_lt.GsimLogicTree(dic['inputs']['gsim_logic_tree'])
        for trt, gsims in gslt.values.items():
            for gsim in gsims:
                q = (model, trt, gsim._toml, gsim.weight['default'])
                rows.append(q)
        out.append(dic)
    check_imts(out, models)
    return out, rows


def main(mosaic_dir, out):
    """
    Storing global SES
    """
    job_inis, rows = read_job_inis(mosaic_dir, MODELS)
    with performance.Monitor(measuremem=True) as mon:
        with hdf5.File(out, 'w') as h5:
            h5['models'] = MODELS
            h5['model_trt_gsim_weight'] = numpy.array(rows, dt)
        jobs = engine.run_jobs(
            engine.create_jobs(job_inis, log_level=logging.WARN),
            concurrent_jobs=4)
        fnames = [datastore.read(job.calc_id).filename for job in jobs]
        logging.warning(f'Saving {out}')
        with hdf5.File(out, 'a') as h5:
            base.import_sites_hdf5(h5, fnames)
            base.import_ruptures_hdf5(h5, fnames)
            h5['/'].attrs.update(INPUTS)
    print(mon)
    return fnames

main.mosaic_dir = 'Directory containing the hazard mosaic'
main.out = 'Output file'

if __name__ == '__main__':
    sap.run(main)
