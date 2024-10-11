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
from openquake.baselib import sap, parallel, general, performance
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


def get_mea_sig_wei(cmaker, ctx, uhs):
    """
    :param cmaker: a ContextMaker instance with G gsims and M imts
    :param ctx: a context array of size C
    :param uhs: an array of shape (M, P)
    :returns: mean[G, M, C], sigma[G, M, C], weights[G, M, C, P]
    """
    M = len(cmaker.imts)
    P = len(cmaker.poes)
    G = len(cmaker.gsims)
    C = len(ctx)
    # reduce the levels to P levels per IMT
    cmaker = set_imls(cmaker, uhs)
    wei = np.empty((G, M, C, P), np.float32)
    mean = np.empty((G, M, C), np.float32)
    sigma = np.empty((G, M, C), np.float32)
    start = 0
    for poes, mea, sig, ctxt in cmaker.gen_poes(ctx):
        c, _, _ = poes.shape  # L = M * P
        slc = slice(start, start + c)
        mean[:, :, slc] = mea
        sigma[:, :, slc] = sig
        start += c
        ocr = cmaker.get_occ_rates(ctxt)
        for g, w in enumerate(cmaker.wei):
            poes_g = poes[:, :, g].reshape(c, M, P)
            # NB: vectorizing the loops on M, P improves nothing;
            # the important loop is the one on C (up to 6000 elements
            # for Canada) which is vectorized
            for p, poe in enumerate(cmaker.poes):
                for m, imt in enumerate(cmaker.imtls):
                    wei[g, m, slc, p] = ocr * poes_g[:, m, p] / poe * w
    return mean, sigma, wei


# NB: we are ignoring IMT-dependent weight
    
def compute_median_spectrum(cmaker, context, uhs, monitor=performance.Monitor()):
    """
    For a given group, computes the median hazard spectrum using a weighted
    mean based on the poes.

    :param cmaker: ContextMaker for a group of sources
    :param context: context array generated by the group of sources
    :param uhs: array of Uniform Hazard Spectra of shape (N, M, P)
    """
    _N, M, P = uhs.shape
    for site_id in np.unique(context.sids):
        ctx = context[context.sids == site_id]
        mea, sig, wei = get_mea_sig_wei(cmaker, ctx, uhs[site_id])
        out = np.empty((3, M, P))  # <mea>, <sig>, tot_w
        out[0] = np.einsum("gmup,gmu->mp", wei, mea)
        out[1] = np.einsum("gmup,gmu->mp", wei, sig)
        out[2] = wei.sum(axis=(0, 2))
        yield {(cmaker.grp_id, site_id): out}


# NB: we are ignoring IMT-dependent weights
def main(dstore, csm):
    """
    Compute the median hazard spectrum for the reference poe,
    starting from the already stored mean hazard spectrum.

    :param dstore: DataStore of the parent calculation
    :param csm: CompositeRiskModel
    """
    # consistency checks
    oq = dstore["oqparam"]
    periods = [imt.period for imt in oq.imt_periods()]
    N = len(dstore["sitecol"])
    M = len(oq.imtls)
    assert oq.investigation_time == 1, oq.investigation_time
    assert len(periods) == M, 'IMTs different from PGA, SA'
    assert N <= oq.max_sites_disagg, N
    logging.warning("Median spectrum calculations are still " "experimental")

    # read the precomputed mean hazard spectrum
    ref_uhs = dstore.sel("hmaps-stats", stat="mean")[:, 0]  # shape NSMP -> NMP
    cmakers = contexts.read_cmakers(dstore)
    G = {cm.grp_id: len(cm.gsims) for cm in cmakers}
    ctx_by_grp = contexts.read_ctx_by_grp(dstore)
    totsize = sum(len(ctx) * G[grp_id] for grp_id, ctx in ctx_by_grp.items())
    blocksize = totsize / (oq.concurrent_tasks or 1)
    smap = parallel.Starmap(compute_median_spectrum, h5=dstore)
    for grp_id, ctx in ctx_by_grp.items():
        # reduce the levels to 1 level per IMT
        cmaker = cmakers[grp_id]
        splits = np.ceil(len(ctx) * G[cmaker.grp_id] / blocksize)
        for ctxt in np.array_split(ctx, splits):
            smap.submit((cmaker, ctxt, ref_uhs))
    res = smap.reduce()

    # save the median_spectrum
    Gr = len(csm.src_groups)  # number of groups
    P = len(oq.poes)
    log_median_spectra = np.zeros((Gr, N, 3, M, P), np.float32)
    tot_w = np.zeros((N, M, P))
    for (grp_id, site_id), out in res.items():
        log_median_spectra[grp_id, site_id] = out
        tot_w[site_id] += out[2]
    dstore.create_dset("log_median_spectra", log_median_spectra)
    dstore.set_shape_descr("log_median_spectra", grp_id=Gr,
                           site_id=N, kind=['mea', 'sig', 'wei'],
                           period=periods, poe=oq.poes)
    # sanity check on the weights
    # np.testing.assert_allclose(tot_w, 1, rtol=.01)


if __name__ == "__main__":
    sap.run(main)
