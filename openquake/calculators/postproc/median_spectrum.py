# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2023, GEM Foundation
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
"""
Median spectrum post-processor
"""
import logging
import numpy as np
from openquake.baselib import sap, parallel, general
from openquake.hazardlib import contexts


def set_imls(cmaker, uhs):
    """
    Replace the imtls with the values in the uniform hazard spectrum (P
    levels per IMT).

    :param cmaker: a ContextMaker
    :param uhs: an array of shape (M, P)
    """
    imtls = {}
    loglevs = {}
    for imt, imls in zip(cmaker.imts, uhs):
        imtls[imt] = imls
        loglevs[imt] = np.log(imls)
    cmaker.imtls = general.DictArray(imtls)
    cmaker.loglevels = general.DictArray(loglevs)
    return cmaker


def compute_median_spectrum(cmaker, ctx, monitor):
    """
    For a given group, computes the median hazard spectrum using a weighted
    mean based on the poes.

    :param cmaker: ContextMaker for a group of sources
    :param ctx: context array generated by the group of sources
    """
    site_id = ctx[0].sids
    weights = []
    M = len(cmaker.imts)
    P = len(cmaker.poes)
    for poes, ctxt, _inv in cmaker.gen_poes(ctx):
        C, _, G = poes.shape  # L = M * P
        if np.isfinite(ctxt[0].occurrence_rate):
            ocr = ctxt.occurrence_rate
        else:
            probs = [rup.probs_occur[0] for rup in ctxt]
            ocr = -np.log(probs) / cmaker.investigation_time
        ws = np.empty((C, M, G, P), np.float32)
        for g, w in enumerate(cmaker.wei):
            poes_g = poes[:, :, g].reshape(C, M, P)
            for m in range(M):
                for p, poe in enumerate(cmaker.poes):
                    ws[:, m, g, p] = ocr * poes_g[:, m, p] / poe * w
        weights.append(ws)

    mea, _, _, _ = cmaker.get_mean_stds([ctx])  # shape (G, M, N)
    wei = np.concatenate(weights)  # shape (N, M, G, P)
    median_spectrum = np.einsum('nmgp,gmn->mp', wei, mea)

    return {(cmaker.grp_id, site_id): median_spectrum}


def main(dstore, csm):
    """
    Compute the median hazard spectrum for the reference poe,
    starting from the already stored mean hazard spectrum.

    :param dstore: DataStore of the parent calculation
    :param csm: CompositeRiskModel
    """
    # consistency checks
    oqp = dstore['oqparam']
    N = len(dstore['sitecol'])
    M = len(oqp.imtls)
    assert oqp.investigation_time == 1, oqp.investigation_time
    assert 'PGV' not in oqp.imtls
    assert N <= oqp.max_sites_disagg, N
    logging.warning('Median spectrum calculations are still '
                    'experimental')

    # read the precomputed mean hazard spectrum
    uhs = dstore.sel('hmaps-stats', stat='mean')
    ref_uhs = uhs[0, 0]  # shape SNMP -> MP
    cmakers = contexts.read_cmakers(dstore)
    ctx_by_grp = contexts.read_ctx_by_grp(dstore)
    
    smap = parallel.Starmap(compute_median_spectrum, h5=dstore)
    for grp_id, ctx in ctx_by_grp.items():
        # reduce the levels to 1 level per IMT
        cmaker = set_imls(cmakers[grp_id], ref_uhs)
        for sid in range(N):
            smap.submit((cmaker, ctx[ctx.sids==sid]))
    res = smap.reduce()

    # save the median_spectrum
    Gr = len(csm.src_groups)  # number of groups
    P = len(oqp.poes)
    median_spectrum = np.zeros((Gr, N, M, P), np.float32)
    for (grp_id, site_id), mhs in res.items():
        median_spectrum[grp_id, site_id] = np.exp(mhs)
    dstore.create_dset('median_spectrum', median_spectrum)
    dstore.set_shape_descr('median_spectrum',
                           grp_id=Gr, site_id=N,
                           period=[imt.period for imt in oqp.imt_periods()],
                           poe=oqp.poes)

if __name__ == '__main__':
    sap.run(main)
