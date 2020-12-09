# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Class for reading MWA correlator FITS files."""
import warnings
import itertools
import numpy as np

from astropy.io import fits
from astropy.time import Time
from astropy import constants as const
from pyuvdata.data import DATA_PATH
from scipy.special import erf
from scipy.integrate import simps

from .. import _corr_fits
from . import UVData
from .. import utils as uvutils

__all__ = ["input_output_mapping", "MWACorrFITS"]


def input_output_mapping():
    """Build a mapping dictionary from pfb input to output numbers."""
    # the polyphase filter bank maps inputs to outputs, which the MWA
    # correlator then records as the antenna indices.
    # the following is taken from mwa_build_lfiles/mwac_utils.c
    # inputs are mapped to outputs via pfb_mapper as follows
    # (from mwa_build_lfiles/antenna_mapping.h):
    # floor(index/4) + index%4 * 16 = input
    # for the first 64 outputs, pfb_mapper[output] = input
    return _corr_fits.input_output_mapping()


def sighat_vector(x):
    """
    Generate quantized sigma using Van Vleck relation.

    For an explanation of the Van Vleck relations used and their implementation
    in this code, see the memos at
    https://github.com/EoRImaging/Memos/blob/master/PDFs/007_Van_Vleck_A.pdf and
    https://github.com/EoRImaging/Memos/blob/master/PDFs/008_Van_Vleck_B.pdf

    Parameters
    ----------
    x : numpy array
        Array of sigma inputs.
    """
    yy = np.arange(7)[:, np.newaxis]
    z = (2 * yy + 1) * erf((yy + 0.5) / (x * np.sqrt(2)))
    z = z.sum(axis=0)
    sighat = np.sqrt(7 ** 2 - z)
    return sighat


def sighat_vector_prime(x):
    """
    Calculate the derivative of sighat_vector.

    Parameters
    ----------
    x : numpy array
        Array of sigma inputs.
    """
    yy = np.arange(7)[:, np.newaxis]
    z = (
        (2 * yy + 1)
        * (yy + 0.5)
        * np.exp(-((yy + 0.5) ** 2) / (2 * (x ** 2)))
        / (np.sqrt(2 * np.pi) * (x ** 2))
    )
    sighat_prime = z.sum(axis=0)
    sighat_prime = sighat_prime / sighat_vector(x)
    return sighat_prime


def corrcorrect_simps(rho, sig1, sig2):
    """
    Generate quantized kappa using the Van Vleck relation.

    For an explanation of the Van Vleck relations used and their implementation
    in this code, see the memos at
    https://github.com/EoRImaging/Memos/blob/master/PDFs/007_Van_Vleck_A.pdf and
    https://github.com/EoRImaging/Memos/blob/master/PDFs/008_Van_Vleck_B.pdf

    Parameters
    ----------
    rho : numpy array
        Array of rho inputs.
    sig1 : numpy array
        Array of sigma inputs corresponding to antenna 1.
    sig2: numpy array
        Array of sigma inputs corresponding to antenna 2.
    """
    x = np.linspace(0, rho, 11, dtype=np.float64)
    khat = np.zeros((11, rho.size), dtype=np.float64)
    khat = _corr_fits.get_khat(x, sig1, sig2)
    integrated_khat = simps(khat, x, axis=0)
    return integrated_khat


def corrcorrect_vect_prime(rho, sig1, sig2):
    """
    Calculate the derivative of corrcorrect_simps.

    Parameters
    ----------
    rho : numpy array
        Array of rho inputs.
    sig1 : numpy array
        Array of sigma inputs corresponding to antenna 1.
    sig2: numpy array
        Array of sigma inputs corresponding to antenna 2.
    """
    return _corr_fits.get_khat(rho, sig1, sig2)


def van_vleck_autos(sighat_arr):
    """
    Use Newton's method to solve the inverse of sighat_vector.

    For an explanation of the Van Vleck corrections used and their implementation
    in this code, see the memos at
    https://github.com/EoRImaging/Memos/blob/master/PDFs/007_Van_Vleck_A.pdf and
    https://github.com/EoRImaging/Memos/blob/master/PDFs/008_Van_Vleck_B.pdf

    Parameters
    ----------
    sighat_arr : numpy array
        Array of quantized sigma to be corrected.
    """
    # cut off small sigmas that will not converge
    cutoff_inds = np.where(sighat_arr > 0.5)[0]
    sighat = sighat_arr[cutoff_inds]
    if len(sighat) > 0:
        guess = np.copy(sighat)
        inds = np.where(np.abs(sighat_vector(guess) - sighat) > 1e-10)[0]
        while len(inds) != 0:
            print(len(inds))
            guess[inds] = guess[inds] - (
                (sighat_vector(guess[inds]) - sighat[inds])
                / sighat_vector_prime(guess[inds])
            )
            inds = np.where(np.abs(sighat_vector(guess) - sighat) > 1e-10)[0]
        sighat_arr[cutoff_inds] = guess

    return sighat_arr


def van_vleck_crosses_int(k_arr, sig1_arr, sig2_arr, cheby_approx):
    """
    Use Newton's method to solve the inverse of corrcorrect_simps.

    For an explanation of the Van Vleck corrections used and their implementation
    in this code, see the memos at
    https://github.com/EoRImaging/Memos/blob/master/PDFs/007_Van_Vleck_A.pdf and
    https://github.com/EoRImaging/Memos/blob/master/PDFs/008_Van_Vleck_B.pdf

    Parameters
    ----------
    k_arr : numpy array
        Array of quantized kappa to be corrected.
    sig1_arr : numpy array
        Array of sigma inputs corresponding to antenna 1.
    sig2_arr: numpy array
        Array of sigma inputs corresponding to antenna 2.
    cheby_approx : bool
        Flag to warn if chebyshev approximation is being used.
    """
    nonzero_inds = np.where((k_arr != 0) & (sig1_arr != 0) & (sig2_arr != 0))[0]
    if len(nonzero_inds) > 0.0:
        if cheby_approx:
            warnings.warn(
                str(len(nonzero_inds))
                + " values are being corrected with the van vleck integral"
            )
        neg_inds = np.where(k_arr < 0.0)[0]
        khat = np.abs(k_arr[nonzero_inds])
        sig1 = sig1_arr[nonzero_inds]
        sig2 = sig2_arr[nonzero_inds]
        x0 = khat / (sig1 * sig2)
        corr = corrcorrect_simps(x0, sig1, sig2) - khat
        x0 = x0 - (corr / corrcorrect_vect_prime(x0, sig1, sig2))
        inds = np.where(np.abs(corr) > 1e-8)[0]
        while len(inds) != 0:
            corr = corrcorrect_simps(x0[inds], sig1[inds], sig2[inds]) - khat[inds]
            x0[inds] = x0[inds] - (
                corr / corrcorrect_vect_prime(x0[inds], sig1[inds], sig2[inds])
            )
            inds2 = np.where(np.abs(corr) > 1e-8)[0]
            inds = inds[inds2]
        k_arr[nonzero_inds] = x0 * sig1 * sig2
        k_arr[neg_inds] = np.negative(k_arr[neg_inds])

    return k_arr


def van_vleck_crosses_cheby(
    khat,
    sig1,
    sig2,
    broad_inds,
    rho_coeff,
    sv_inds_right1,
    sv_inds_right2,
    ds1,
    ds2,
    cheby_approx,
):
    """
    Compute a chebyshev approximation of corrcorrect_simps.

    Uses a bilinear interpolation to find chebyshev coefficients. Assumes distance
    between points of interpolation grid is 0.01. If sig1 or sig2 falls outside
    the interpolation grid, the corresponding values are corrected using
    van_vleck_crosses_int.

    For an explanation of the Van Vleck corrections used and their implementation
    in this code, see the memos at
    https://github.com/EoRImaging/Memos/blob/master/PDFs/007_Van_Vleck_A.pdf and
    https://github.com/EoRImaging/Memos/blob/master/PDFs/008_Van_Vleck_B.pdf

    Parameters
    ----------
    khat : numpy array
        Array of quantized kappa to be corrected.
    sig1 : numpy array
        Array of sigma inputs corresponding to antenna 1.
    sig2: numpy array
        Array of sigma inputs corresponding to antenna 2.
    broad_inds : numpy array
        Array indexing sigmas within the chebyshev approximation range.
    rho_coeff : numpy array
        Array of chebyshev polynomial coefficients.
    sv_inds_right1 : numpy array
        Array of right indices for sig1 for bilinear interpolation.
    sv_inds_right2 : numpy array
        Array of right indices for sig2 for bilinear interpolation.
    ds1 : numpy array
        Distance between sig1 and right-indexed value for bilinear interpolation.
    ds2 : numpy array
        Distance between sig2 and right-indexed value for bilinear interpolation.
    cheby_approx : bool
        Flag to warn if chebyshev approximation is being used.
    """
    kap = np.empty(broad_inds.shape, dtype=np.complex128)
    kap[broad_inds] = (
        _corr_fits.van_vleck_cheby(
            khat[broad_inds], rho_coeff, sv_inds_right1, sv_inds_right2, ds1, ds2
        )
        * sig1[broad_inds]
        * sig2[broad_inds]
    )
    kap[~broad_inds] = van_vleck_crosses_int(
        khat.real[~broad_inds], sig1[~broad_inds], sig2[~broad_inds], cheby_approx
    ) + 1j * van_vleck_crosses_int(
        khat.imag[~broad_inds], sig1[~broad_inds], sig2[~broad_inds], cheby_approx
    )

    return kap


class MWACorrFITS(UVData):
    """
    UVData subclass for reading MWA correlator fits files.

    This class should not be interacted with directly; instead use the
    read_mwa_corr_fits method on the UVData class.
    """

    def correct_cable_length(self, cable_lens):
        """
        Apply a cable length correction to the data array.

        Parameters
        ----------
        cable_lens : list of strings
        A list of strings containing the cable lengths for each antenna.
        """
        # from MWA_Tools/CONV2UVFITS/convutils.h
        cable_len_diffs = _corr_fits.get_cable_len_diffs(
            self.Nblts, self.ant_1_array, self.ant_2_array, cable_lens
        )
        self.data_array *= np.exp(
            -1j
            * 2
            * np.pi
            * cable_len_diffs.reshape(self.Nblts, 1)
            / const.c.to("m/s").value
            * self.freq_array.reshape(1, self.Nfreqs)
        )[:, :, None]
        history_add_string = " Applied cable length correction."
        self.history += history_add_string

    def flag_init(
        self,
        num_fine_chan,
        edge_width=80e3,
        start_flag=2.0,
        end_flag=0.0,
        flag_dc_offset=True,
    ):
        """
        Apply routine flagging to the MWA Correlator FITS file data.

        Includes options to flag the coarse channel edges, beginning and end
        of obs, as well as the center fine channel of each coarse channel.

        Parameters
        ----------
        edge_width: float
            The width to flag on the edge of each coarse channel, in hz. Set to
            0 for no edge flagging.
        start_flag: float
            The number of seconds to flag at the beginning of the observation.
            Set to 0 for no flagging.
        end_flag: floats
            The number of seconds to flag at the end of the observation. Set to
            0 for no flagging.
        flag_dc_offset: bool
            Set to True to flag the center fine channel of each coarse channel.

        Raises
        ------
        ValueError
            If edge_width is not an integer multiple of the channel_width of
            the data (0 also acceptable).
            If start_flag is not an integer multiple of the integration time
            (0 also acceptable).
            If end_flag is not an integer multiple of the integration time
            (0 also acceptable).

        """
        if (edge_width % self.channel_width) > 0:
            raise ValueError(
                "The edge_width must be an integer multiple of the "
                "channel_width of the data or zero."
            )
        if (start_flag % self.integration_time[0]) > 0:
            raise ValueError(
                "The start_flag must be an integer multiple of the "
                "integration_time of the data or zero."
            )
        if (end_flag % self.integration_time[0]) > 0:
            raise ValueError(
                "The end_flag must be an integer multiple of the "
                "integration_time of the data or zero."
            )

        num_ch_flag = int(edge_width / self.channel_width)
        num_start_flag = int(start_flag / self.integration_time[0])
        num_end_flag = int(end_flag / self.integration_time[0])

        if num_ch_flag > 0:
            edge_inds = []
            for ch_count in range(num_ch_flag):
                # count up from the left
                left_chans = list(range(ch_count, self.Nfreqs, num_fine_chan))
                # count down from the right
                right_chans = list(range(self.Nfreqs - 1 - ch_count, 0, -num_fine_chan))
                edge_inds = edge_inds + left_chans + right_chans

            self.flag_array[:, :, edge_inds, :] = True

        if flag_dc_offset:
            center_inds = list(range(num_fine_chan // 2, self.Nfreqs, num_fine_chan))

            self.flag_array[:, :, center_inds, :] = True

        if (num_start_flag > 0) or (num_end_flag > 0):
            shape = self.flag_array.shape
            # TODO: Spw axis to be collapsed in future release
            # Asserting this here because this is effectively a stripped down UVFITS
            # reader, and thus assuming that this should only support simple tables
            assert shape[1] == 1
            reshape = [self.Ntimes, self.Nbls, 1, self.Nfreqs, self.Npols]
            self.flag_array = np.reshape(self.flag_array, reshape)
            if num_start_flag > 0:
                self.flag_array[:num_start_flag, :, :, :, :] = True
            if num_end_flag > 0:
                self.flag_array[-num_end_flag:, :, :, :, :] = True
            self.flag_array = np.reshape(self.flag_array, shape)

    def _read_fits_file(
        self, filename, time_array, file_nums_to_index, num_fine_chans, int_time,
    ):
        """
        Read the fits file and populate into memory.

        This is an internal function and should not regularly be called except
        by read_mwa_corr_fits function.

        It is designed to close the fits files, headers, and all associated pointers.
        Without this read in a function, reading files has a large memory footprint.

        Parameters
        ----------
        filename : str
            The mwa gpubox fits file to read
        time_array : array of floats
            The time_array object constructed during read_mwa_corr_fits call
        file_nums_to_index : dict
            Mappings of file name to index in coarse channel
        num_fine_chans : int
            Number of fine channels in a coarse channel
        int_time : float
            The integration time of each observation.

        """
        # get the file number from the file name
        file_num = int(filename.split("_")[-2][-2:])
        # map file number to frequency index
        freq_ind = file_nums_to_index[file_num] * num_fine_chans
        with fits.open(filename, memmap=True, mode="denywrite") as hdu_list:
            for hdu in hdu_list:
                # entry 0 is a header, so we skip it.
                if hdu.data is None:
                    continue
                time = (
                    hdu.header["TIME"]
                    + hdu.header["MILLITIM"] / 1000.0
                    + int_time / 2.0
                )
                time_ind = np.where(time_array == time)[0][0]
                # dump data into matrix
                # and take data from real to complex numbers
                indices = np.index_exp[
                    time_ind, freq_ind : freq_ind + num_fine_chans, :
                ]
                self.data_array[indices] = hdu.data[:, 0::2] + 1j * hdu.data[:, 1::2]
                self.nsample_array[
                    time_ind, :, freq_ind : freq_ind + num_fine_chans, :
                ] = 1.0
                self.flag_array[time_ind, :, file_nums_to_index[file_num], :] = False
        return

    def van_vleck_correction(
        self,
        flagged_ants,
        cheby_approx,
        data_array_dtype,
        remove_flagged_ants,
        flag_small_sig_ants,
    ):
        """
        Apply a van vleck correction to the data array.

        For an explanation of the Van Vleck corrections used and their implementation
        in this code, see the memos at
        https://github.com/EoRImaging/Memos/blob/master/PDFs/007_Van_Vleck_A.pdf and
        https://github.com/EoRImaging/Memos/blob/master/PDFs/008_Van_Vleck_B.pdf

        Parameters
        ----------
        cheby_approx : bool
            Option to implement the van vleck correction with a chebyshev polynomial
            approximation.
        data_array_dtype : numpy dtype
            Datatype to store the output data_array as.
        """
        history_add_string = " Applied Van Vleck correction."
        # need data array to have 64 bit precision
        if self.data_array.dtype != np.complex128:
            self.data_array = self.data_array.astype(np.complex128)
        # scale the data
        # number of samples per fine channel is equal to channel width (Hz)
        # multiplied be the integration time (s)
        nsamples = self.channel_width * self.integration_time[0]
        # cast data to ints
        self.data_array = np.rint(self.data_array / self.extra_keywords["SCALEFAC"])
        # take advantage of circular symmetry! divide by two
        self.data_array = self.data_array / (nsamples * 2.0)
        # reshape to (nbls, ntimes, nfreqs, npols)
        self.data_array = np.swapaxes(self.data_array, 0, 1)
        # get indices for autos
        autos = np.where(
            self.ant_1_array[0 : self.Nbls] == self.ant_2_array[0 : self.Nbls]
        )[0]
        # get indices for crosses
        crosses = np.where(
            self.ant_1_array[0 : self.Nbls] != self.ant_2_array[0 : self.Nbls]
        )[0]
        # find polarizations
        xx = np.where(self.polarization_array == -5)[0][0]
        yy = np.where(self.polarization_array == -6)[0][0]
        xy = np.where(self.polarization_array == -7)[0][0]
        yx = np.where(self.polarization_array == -8)[0][0]
        pols = np.array([yy, xx])
        # pols = np.array([0, 3])
        # combine axes
        self.data_array = self.data_array.reshape(
            (self.Nbls, self.Nfreqs * self.Ntimes, self.Npols)
        )
        # square root autos
        auto_inds = autos[:, np.newaxis]
        self.data_array.real[auto_inds, :, pols] = np.sqrt(
            self.data_array.real[auto_inds, :, pols]
        )
        # look for small sigmas that will not converge
        small_sig_flags = np.logical_and(
            self.data_array.real[auto_inds, :, pols] != 0,
            self.data_array.real[auto_inds, :, pols] <= 0.5,
        )
        if flag_small_sig_ants:
            # find antenna indices for small sig ants and add to flagged_ants
            ant_inds = np.unique(np.nonzero(small_sig_flags)[0])
            ant_inds = ant_inds[~np.in1d(ant_inds, flagged_ants)]
            if len(ant_inds) != 0:
                history_add_string += (
                    " The following antennas were flagged by the Van Vleck \
                    correction and removed from the data: "
                    + str(ant_inds)
                    + "."
                )
                flagged_ants = np.concatenate((flagged_ants, ant_inds))
        else:
            # get flags for small sig ants and add to flag array
            small_sig_flags = np.logical_or(
                small_sig_flags[:, 0, :], small_sig_flags[:, 1, :]
            )
            small_sig_flags = np.logical_or(
                small_sig_flags[self.ant_1_array, :],
                small_sig_flags[self.ant_2_array, :],
            )
            small_sig_flags = small_sig_flags.reshape(
                self.Nbls, self.Ntimes, self.Nfreqs
            )
            small_sig_flags = np.swapaxes(small_sig_flags, 0, 1)
            small_sig_flags = small_sig_flags[:, :, :, np.newaxis]
            self.flag_array = np.logical_or(self.flag_array, small_sig_flags)
        # get unflagged autos
        good_autos = np.delete(autos, flagged_ants)
        sighat = self.data_array.real[good_autos[:, np.newaxis], :, pols].flatten()
        # correct autos
        sigma = van_vleck_autos(sighat)
        self.data_array.real[good_autos[:, np.newaxis], :, pols] = sigma.reshape(
            len(good_autos), len(pols), self.Ntimes * self.Nfreqs
        )
        # get good crosses
        bad_ant_inds = np.nonzero(
            np.logical_or(
                np.in1d(self.ant_1_array[0 : self.Nbls], flagged_ants),
                np.in1d(self.ant_2_array[0 : self.Nbls], flagged_ants),
            )
        )[0]
        crosses = np.delete(crosses, np.nonzero(np.in1d(crosses, bad_ant_inds))[0])
        # correct crosses
        if cheby_approx:
            history_add_string += " Used Van Vleck Chebychev approximation."
            # load in interpolation files
            with open(DATA_PATH + "/mwa_config_data/Chebychev_coeff.npy", "rb") as f:
                rho_coeff = np.load(f)
            with open(DATA_PATH + "/mwa_config_data/sigma1.npy", "rb") as f:
                sig_vec = np.load(f)
            rho_coeff = rho_coeff[:, :, np.array([1, 3, 5])]
            sigs = self.data_array.real[autos[:, np.newaxis], :, pols]
            smol_inds = np.nonzero(
                np.logical_and(sigs.flatten() != 0, sigs.flatten() <= 0.9)
            )[0]
            smol_ants = np.floor(smol_inds / (len(pols) * self.Ntimes * self.Nfreqs))
            print("num smol autos: " + str(len(smol_ants)))
            print("smol auto ants: " + str(np.unique(smol_ants)))
            # look for big sigmas
            big_inds = np.nonzero(sigs > 4.5)[0]
            big_ants = np.floor(big_inds / (len(pols) * self.Ntimes * self.Nfreqs))
            print("num big autos: " + str(len(big_ants)))
            print("smol big ants: " + str(np.unique(big_ants)))
            # find sigmas within interpolation range
            in_inds = np.logical_and(sigs > 0.9, sigs <= 4.5)
            # get indices and distances for bilinear interpolation
            sv_inds_right = np.zeros(in_inds.shape, dtype=np.int64)
            ds = np.zeros(in_inds.shape)
            sv_inds_right[in_inds] = np.searchsorted(sig_vec, sigs[in_inds])
            ds[in_inds] = sig_vec[sv_inds_right[in_inds]] - sigs[in_inds]
            # get indices for sigmas corresponding to crosses
            sig1_inds = self.ant_1_array[crosses]
            sig2_inds = self.ant_2_array[crosses]
            # iterate over polarization
            pol_dict = {
                yy: [(yy, yy), (0, 0)],
                yx: [(yy, xx), (0, 1)],
                xy: [(xx, yy), (1, 0)],
                xx: [(xx, xx), (1, 1)],
            }
            for i in [xx, yy, xy, yx]:
                (pol1, pol2) = pol_dict[i][1]
                (sig1_pol, sig2_pol) = pol_dict[i][0]
                # broadcast in_inds
                broad_inds = np.logical_and(
                    in_inds[sig1_inds, pol1, :], in_inds[sig2_inds, pol2, :],
                )
                # broadcast indices and distances for bilinear interpolation
                sv_inds_right1 = sv_inds_right[sig1_inds, pol1, :][broad_inds]
                sv_inds_right2 = sv_inds_right[sig2_inds, pol2, :][broad_inds]
                ds1 = ds[sig1_inds, pol1, :][broad_inds]
                ds2 = ds[sig2_inds, pol2, :][broad_inds]
                self.data_array[crosses, :, i] = van_vleck_crosses_cheby(
                    self.data_array[crosses, :, i],
                    self.data_array.real[autos[sig1_inds], :, sig1_pol],
                    self.data_array.real[autos[sig2_inds], :, sig2_pol],
                    broad_inds,
                    rho_coeff,
                    sv_inds_right1,
                    sv_inds_right2,
                    ds1,
                    ds2,
                    cheby_approx,
                )
            # correct yx autos
            sig_inds = self.ant_1_array[good_autos]
            broad_inds = np.logical_and(
                in_inds[sig_inds, 0, :], in_inds[sig_inds, 1, :]
            )
            sv_inds_right1 = sv_inds_right[sig_inds, 0, :][broad_inds]
            sv_inds_right2 = sv_inds_right[sig_inds, 1, :][broad_inds]
            ds1 = ds[sig_inds, 0, :][broad_inds]
            ds2 = ds[sig_inds, 1, :][broad_inds]
            self.data_array[good_autos, :, yx] = van_vleck_crosses_cheby(
                self.data_array[good_autos, :, yx],
                self.data_array.real[good_autos, :, yy],
                self.data_array.real[good_autos, :, xx],
                broad_inds,
                rho_coeff,
                sv_inds_right1,
                sv_inds_right2,
                ds1,
                ds2,
                cheby_approx,
            )
            # add back in frequency axis
            self.data_array = self.data_array.reshape(
                (self.Nbls, self.Ntimes, self.Nfreqs, self.Npols)
            )
        # solve integral directly
        else:
            # add back in frequency axis
            self.data_array = self.data_array.reshape(
                (self.Nbls, self.Ntimes, self.Nfreqs, self.Npols)
            )
            for k in crosses:
                auto1 = autos[self.ant_1_array[k]]
                auto2 = autos[self.ant_2_array[k]]
                for j in range(self.Nfreqs):
                    # get data
                    sig1 = self.data_array.real[
                        auto1, :, j, np.array([yy, yy, xx, xx])
                    ].flatten()
                    sig2 = self.data_array.real[
                        auto2, :, j, np.array([yy, xx, yy, xx])
                    ].flatten()
                    khat = self.data_array[
                        k, :, j, np.array([yy, yx, xy, xx])
                    ].flatten()
                    # correct real
                    kap = van_vleck_crosses_int(khat.real, sig1, sig2, cheby_approx)
                    self.data_array.real[k, :, j, :] = kap.reshape(
                        self.Ntimes, self.Npols
                    )
                    # correct imaginary
                    kap = van_vleck_crosses_int(khat.imag, sig1, sig2, cheby_approx)
                    self.data_array.imag[k, :, j, :] = kap.reshape(
                        self.Ntimes, self.Npols
                    )
            # correct yx autos
            for k in good_autos:
                for j in range(self.Nfreqs):
                    # get data
                    sig1 = self.data_array.real[k, :, j, yy]
                    sig2 = self.data_array.real[k, :, j, xx]
                    khat = self.data_array[k, :, j, yx]
                    # correct real
                    kap = van_vleck_crosses_int(khat.real, sig1, sig2, cheby_approx)
                    self.data_array.real[k, :, j, yx] = kap
                    # correct imaginary
                    kap = van_vleck_crosses_int(khat.imag, sig1, sig2, cheby_approx)
                    self.data_array.imag[k, :, j, yx] = kap
        # correct xy autos
        self.data_array[good_autos, :, :, xy] = np.conj(
            self.data_array[good_autos, :, :, yx]
        )
        # square autos
        self.data_array.real[auto_inds, :, :, pols] = (
            self.data_array.real[auto_inds, :, :, pols] ** 2
        )
        # reshape to (ntimes, nbls, nfreqs, npols)
        self.data_array = np.swapaxes(self.data_array, 0, 1)
        # rescale the data
        self.data_array = self.data_array * (
            self.extra_keywords["SCALEFAC"] * nsamples * 2
        )
        # return data array to desired precision
        if self.data_array.dtype != data_array_dtype:
            self.data_array = self.data_array.astype(data_array_dtype)
        self.history += history_add_string

        return flagged_ants

    def read_mwa_corr_fits(
        self,
        filelist,
        use_cotter_flags=None,
        remove_dig_gains=True,
        remove_coarse_band=True,
        correct_cable_len=False,
        correct_van_vleck=False,
        cheby_approx=True,
        flag_small_sig_ants=True,
        phase_to_pointing_center=False,
        propagate_coarse_flags=True,
        flag_init=True,
        edge_width=80e3,
        start_flag="goodtime",
        end_flag=0.0,
        flag_dc_offset=True,
        remove_flagged_ants=True,
        background_lsts=True,
        read_data=True,
        data_array_dtype=np.complex64,
        nsample_array_dtype=np.float32,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
    ):
        """
        Read in MWA correlator gpu box files.

        The default settings remove some of the instrumental effects in the bandpass
        by dividing out the digital gains and the coarse band shape.
        If the desired output is raw correlator data, set remove_dig_gains=False,
        remove_coarse_band=False, correct_cable_len=False, and
        phase_to_pointing_center=False.

        Parameters
        ----------
        filelist : list of str
            The list of MWA correlator files to read from. Must include at
            least one fits file and only one metafits file per data set.
            Can also be a list of lists to read multiple data sets.
        axis : str
            Axis to concatenate files along. This enables fast concatenation
            along the specified axis without the normal checking that all other
            metadata agrees. This method does not guarantee correct resulting
            objects. Please see the docstring for fast_concat for details.
            Allowed values are: 'blt', 'freq', 'polarization'. Only used if
            multiple files are passed.
        use_cotter_flags : bool
            Option to use cotter output mwaf flag files. Defaults to true if
            cotter flag files are submitted.
        remove_dig_gains : bool
            Option to divide out digital gains.
        remove_coarse_band : bool
            Option to divide out coarse band shape.
        correct_cable_len : bool
            Option to apply a cable delay correction.
        correct_van_vleck : bool
            Option to apply a van vleck correction.
        cheby_approx : bool
            Only used if correct_van_vleck is True. Option to implement the van
            vleck correction with a chebyshev polynomial approximation.
        flag_small_sig_ants : bool
            Only used if correct_van_vleck is True. Option to completely flag any
            antenna that has a sigma < 0.5. If set to False, only the times and
            frequencies at which sigma < 0.5 will be flagged for the antenna.
        phase_to_pointing_center : bool
            Option to phase to the observation pointing center.
        propagate_coarse_flags : bool
            Option to propagate flags for missing coarse channel integrations
            across frequency.
        flag_init: bool
            Set to True in order to do routine flagging of coarse channel edges,
            start or end integrations, or the center fine channel of each coarse
            channel. See associated keywords.
        edge_width: float
            Only used if flag_init is True. The width to flag on the edge of
            each coarse channel, in hz. Errors if not equal to integer multiple
            of channel_width. Set to 0 for no edge flagging.
        start_flag: float or str
            Only used if flag_init is True. The number of seconds to flag at the
            beginning of the observation. Set to 0 for no flagging. Default is
            'goodtime', which uses information in the metafits file to determine
            the length of time that should be flagged. Errors if input is not a
            float or 'goodtime'. Errors if float input is not equal to an
            integer multiple of the integration time.
        end_flag: floats
            Only used if flag_init is True. The number of seconds to flag at the
            end of the observation. Set to 0 for no flagging. Errors if not
            equal to an integer multiple of the integration time.
        flag_dc_offset: bool
            Only used if flag_init is True. Set to True to flag the center fine
            channel of each coarse channel.
        remove_flagged_ants : bool
            Option to perform a select to remove antennas flagged in the metafits
            file. If flag_small_sig_ants is True then antennas flagged by the Van
            Vleck correction are also removed.
        background_lsts : bool
            When set to True, the lst_array is calculated in a background thread.
        read_data : bool
            Read in the visibility, nsample and flag data. If set to False, only
            the metadata will be read in. Setting read_data to False results in
            a metadata only object.
        data_array_dtype : numpy dtype
            Datatype to store the output data_array as. Must be either
            np.complex64 (single-precision real and imaginary) or np.complex128
            (double-precision real and imaginary).
        nsample_array_dtype : numpy dtype
            Datatype to store the output nsample_array as. Must be either
            np.float64 (double-precision), np.float32 (single-precision), or
            np.float16 (half-precision). Half-precision is only recommended for
            cases where no sampling or averaging of baselines will occur,
            because round-off errors can be quite large (~1e-3).
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after after reading in the file (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            reading in the file (the default is True, meaning the acceptable
            range check will be done).
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.


        Raises
        ------
        ValueError
            If required files are missing or multiple files metafits files are
            included in filelist.
            If files from different observations are included in filelist.
            If files in fileslist have different fine channel widths
            If file types other than fits, metafits, and mwaf files are included
            in filelist.

        """
        metafits_file = None
        ppds_file = None
        obs_id = None
        file_dict = {}
        start_time = 0.0
        end_time = 0.0
        included_file_nums = []
        included_flag_nums = []
        cotter_warning = False
        num_fine_chans = 0

        # do datatype checks
        if data_array_dtype not in (np.complex64, np.complex128):
            raise ValueError("data_array_dtype must be np.complex64 or np.complex128")
        if nsample_array_dtype not in (np.float64, np.float32, np.float16):
            raise ValueError(
                "nsample_array_dtype must be one of: np.float64, np.float32, np.float16"
            )
        # do start_flag check
        if not isinstance(start_flag, (int, float)):
            if start_flag != "goodtime":
                raise ValueError("start_flag must be int or float or 'goodtime'")

        # iterate through files and organize
        # create a list of included coarse channels
        # find the first and last times that have data
        for file in filelist:
            if file.lower().endswith(".metafits"):
                # force only one metafits file
                if metafits_file is not None:
                    raise ValueError("multiple metafits files in filelist")
                metafits_file = file
            elif file.lower().endswith(".fits"):
                # check if ppds file
                try:
                    fits.getheader(file, extname="ppds")
                    ppds_file = file
                except Exception:
                    # check obsid
                    head0 = fits.getheader(file, 0)
                    if obs_id is None:
                        obs_id = head0["OBSID"]
                    else:
                        if head0["OBSID"] != obs_id:
                            raise ValueError(
                                "files from different observations submitted "
                                "in same list"
                            )
                    # check headers for first and last times containing data
                    headstart = fits.getheader(file, 1)
                    headfin = fits.getheader(file, -1)
                    first_time = headstart["TIME"] + headstart["MILLITIM"] / 1000.0
                    last_time = headfin["TIME"] + headfin["MILLITIM"] / 1000.0
                    if start_time == 0.0:
                        start_time = first_time
                    # check that files with a timing offset can be aligned
                    elif np.abs(start_time - first_time) % headstart["INTTIME"] != 0.0:
                        raise ValueError(
                            "coarse channel start times are misaligned by an amount that is not \
                            an integer multiple of the integration time"
                        )
                    elif start_time > first_time:
                        start_time = first_time
                    if end_time < last_time:
                        end_time = last_time
                    # get number of fine channels
                    if num_fine_chans == 0:
                        num_fine_chans = headstart["NAXIS2"]
                    elif num_fine_chans != headstart["NAXIS2"]:
                        raise ValueError(
                            "files submitted have different fine channel widths"
                        )

                    # get the file number from the file name;
                    # this will later be mapped to a coarse channel
                    file_num = int(file.split("_")[-2][-2:])
                    if file_num not in included_file_nums:
                        included_file_nums.append(file_num)
                    # organize files
                    if "data" not in file_dict.keys():
                        file_dict["data"] = [file]
                    else:
                        file_dict["data"].append(file)

                    # save bscale keyword
                    if "SCALEFAC" not in self.extra_keywords.keys():
                        if "BSCALE" in head0.keys():
                            self.extra_keywords["SCALEFAC"] = head0["BSCALE"]
                        else:
                            # correlator did a divide by 4 before october 2014
                            self.extra_keywords["SCALEFAC"] = 0.25

            # look for flag files
            elif file.lower().endswith(".mwaf"):
                if use_cotter_flags is None:
                    use_cotter_flags = True
                flag_num = int(file.split("_")[-1][0:2])
                included_flag_nums.append(flag_num)
                if use_cotter_flags is False and cotter_warning is False:
                    warnings.warn("mwaf files submitted with use_cotter_flags=False")
                    cotter_warning = True
                elif "flags" not in file_dict.keys():
                    file_dict["flags"] = [file]
                else:
                    file_dict["flags"].append(file)
            else:
                raise ValueError("only fits, metafits, and mwaf files supported")

        # checks:
        if metafits_file is None and ppds_file is None:
            raise ValueError("no metafits file submitted")
        elif metafits_file is None:
            metafits_file = ppds_file
        elif ppds_file is not None:
            ppds = fits.getheader(ppds_file, 0)
            meta = fits.getheader(metafits_file, 0)
            for key in ppds.keys():
                if key not in meta.keys():
                    self.extra_keywords[key] = ppds[key]
        if "data" not in file_dict.keys():
            raise ValueError("no data files submitted")
        if "flags" not in file_dict.keys() and use_cotter_flags:
            raise ValueError(
                "no flag files submitted. Rerun with flag files \
                             or use_cotter_flags=False"
            )

        # first set parameters that are always true
        self.Nspws = 1
        self.spw_array = np.array([0])
        self.phase_type = "drift"
        self.vis_units = "uncalib"
        self.Npols = 4
        self.xorientation = "east"

        # get information from metafits file
        with fits.open(metafits_file, memmap=True) as meta:
            meta_hdr = meta[0].header

            # get a list of coarse channels
            coarse_chans = meta_hdr["CHANNELS"].split(",")
            coarse_chans = np.array(sorted(int(i) for i in coarse_chans))

            # integration time in seconds
            int_time = meta_hdr["INTTIME"]

            # pointing center in degrees
            ra_deg = meta_hdr["RA"]
            dec_deg = meta_hdr["DEC"]
            ra_rad = np.pi * ra_deg / 180
            dec_rad = np.pi * dec_deg / 180

            # set start_flag with goodtime
            if flag_init and start_flag == "goodtime":
                # ppds file does not contain this key
                try:
                    if meta_hdr["GOODTIME"] > start_time:
                        start_flag = meta_hdr["GOODTIME"] - start_time
                    else:
                        start_flag = 0.0
                except KeyError:
                    raise ValueError(
                        "To use start_flag='goodtime', a .metafits file must \
                            be submitted"
                    )

            # get parameters from header
            # this assumes no averaging by this code so will need to be updated
            self.channel_width = float(meta_hdr.pop("FINECHAN") * 1000)
            if "HISTORY" in meta_hdr:
                self.history = str(meta_hdr["HISTORY"])
                meta_hdr.remove("HISTORY", remove_all=True)
            else:
                self.history = ""
            if not uvutils._check_history_version(
                self.history, self.pyuvdata_version_str
            ):
                self.history += self.pyuvdata_version_str
            self.instrument = meta_hdr["TELESCOP"]
            self.telescope_name = meta_hdr.pop("TELESCOP")
            self.object_name = meta_hdr.pop("FILENAME")

            # get rid of the instrument keyword so it doesn't get put back in
            meta_hdr.remove("INSTRUME")
            # get rid of keywords that uvfits.py gets rid of
            bad_keys = ["SIMPLE", "EXTEND", "BITPIX", "NAXIS", "DATE-OBS"]
            for key in bad_keys:
                meta_hdr.remove(key, remove_all=True)
            # store remaining keys in extra keywords
            for key in meta_hdr:
                if key == "COMMENT":
                    self.extra_keywords[key] = str(meta_hdr.get(key))
                elif key != "":
                    self.extra_keywords[key] = meta_hdr.get(key)
            # get antenna data from metafits file table
            meta_tbl = meta[1].data

            # because of polarization, each antenna # is listed twice
            antenna_numbers = meta_tbl["Antenna"][1::2]
            antenna_names = meta_tbl["TileName"][1::2]
            antenna_flags = meta_tbl["Flag"][1::2]
            cable_lens = np.asarray(meta_tbl["Length"][1::2]).astype(np.str_)
            dig_gains = meta_tbl["Gains"][1::2, :].astype(np.float64)

            # get antenna postions in enu coordinates
            antenna_positions = np.zeros((len(antenna_numbers), 3))
            antenna_positions[:, 0] = meta_tbl["East"][1::2]
            antenna_positions[:, 1] = meta_tbl["North"][1::2]
            antenna_positions[:, 2] = meta_tbl["Height"][1::2]

        # reorder antenna parameters from metafits ordering
        reordered_inds = antenna_numbers.argsort()
        self.antenna_numbers = antenna_numbers[reordered_inds]
        self.antenna_names = list(antenna_names[reordered_inds])
        antenna_positions = antenna_positions[reordered_inds, :]
        antenna_flags = antenna_flags[reordered_inds]
        cable_lens = cable_lens[reordered_inds]
        dig_gains = dig_gains[reordered_inds, :]

        # find flagged antenna
        flagged_ants = self.antenna_numbers[np.where(antenna_flags == 1)]

        # set parameters from other parameters
        self.Nants_data = len(self.antenna_numbers)
        self.Nants_telescope = len(self.antenna_numbers)
        self.Nbls = int(
            len(self.antenna_numbers) * (len(self.antenna_numbers) + 1) / 2.0
        )

        # get telescope parameters
        self.set_telescope_params()

        # build time array of centers
        time_array = np.arange(
            start_time + int_time / 2.0, end_time + int_time / 2.0 + int_time, int_time
        )

        # convert to time to jd floats
        float_time_array = Time(time_array, format="unix", scale="utc").jd.astype(float)
        # build into time array
        self.time_array = np.repeat(float_time_array, self.Nbls)

        self.Ntimes = len(time_array)

        self.Nblts = int(self.Nbls * self.Ntimes)

        # convert times to lst
        proc = self.set_lsts_from_time_array(background=background_lsts)

        self.integration_time = np.full((self.Nblts), int_time)

        # convert antenna positions from enu to ecef
        # antenna positions are "relative to
        # the centre of the array in local topocentric \"east\", \"north\",
        # \"height\". Units are meters."
        antenna_positions_ecef = uvutils.ECEF_from_ENU(
            antenna_positions, *self.telescope_location_lat_lon_alt
        )
        # make antenna positions relative to telescope location
        self.antenna_positions = antenna_positions_ecef - self.telescope_location

        # make initial antenna arrays, where ant_1 <= ant_2
        # itertools.combinations_with_replacement returns
        # all pairs in the range 0...Nants_telescope
        # including pairs with the same number (e.g. (0,0) auto-correlation).
        # this is a little faster than having nested for-loops moving over the
        # upper triangle of antenna-pair combinations matrix.
        ant_1_array, ant_2_array = np.transpose(
            list(
                itertools.combinations_with_replacement(
                    np.arange(self.Nants_telescope), 2
                )
            )
        )

        self.ant_1_array = np.tile(np.array(ant_1_array), self.Ntimes)
        self.ant_2_array = np.tile(np.array(ant_2_array), self.Ntimes)

        self.baseline_array = self.antnums_to_baseline(
            self.ant_1_array, self.ant_2_array
        )

        # create self.uvw_array
        self.set_uvws_from_antenna_positions(allow_phasing=False)

        # coarse channel mapping:
        # channels in group 0-128 go in order; channels in group 129-155 go in
        # reverse order
        # that is, if the lowest channel is 127, it will be assigned to the
        # first file
        # channel 128 will be assigned to the second file
        # then the highest channel will be assigned to the third file
        # and the next hightest channel assigned to the fourth file, and so on
        count = np.count_nonzero(coarse_chans <= 128)
        # map all file numbers to coarse channel numbers
        file_nums_to_coarse = {
            i + 1: coarse_chans[i]
            if i < count
            else coarse_chans[(len(coarse_chans) + count - i - 1)]
            for i in range(len(coarse_chans))
        }
        # map included coarse channels to file numbers
        coarse_to_incl_files = {}
        for i in included_file_nums:
            coarse_to_incl_files[file_nums_to_coarse[i]] = i
        # sort included coarse channels
        included_coarse_chans = sorted(coarse_to_incl_files.keys())
        # map included file numbers to an index that orders them
        file_nums_to_index = {}
        for i in included_coarse_chans:
            file_nums_to_index[coarse_to_incl_files[i]] = included_coarse_chans.index(i)
        # check that coarse channels are contiguous.
        chans = np.array(included_coarse_chans)
        for i in np.diff(chans):
            if i != 1:
                warnings.warn("coarse channels are not contiguous for this observation")
                break

        # warn user if not all coarse channels are included
        if len(included_coarse_chans) != len(coarse_chans):
            warnings.warn("some coarse channel files were not submitted")

        # build frequency array
        self.Nfreqs = len(included_coarse_chans) * num_fine_chans
        # TODO: Spw axis to be collapsed in future release
        self.freq_array = np.zeros((1, self.Nfreqs))

        # each coarse channel is split into 128 fine channels of width 10 kHz.
        # The first fine channel for each coarse channel is centered on the
        # lower bound frequency of that channel and its center frequency is
        # computed as fine_center = coarse_channel_number * 1280-640 (kHz).
        # If the fine channels have been averaged (added) by some factor, the
        # center of the resulting channel is found by averaging the centers of
        # the first and last fine channels it is made up of.
        # That is, avg_fine_center=(lowest_fine_center+highest_fine_center)/2
        # where highest_fine_center=lowest_fine_center+(avg_factor-1)*10 kHz
        # so avg_fine_center=(lowest_fine_center+lowest_fine_center+(avg_factor-1)*10)/2
        #                   =lowest_fine_center+((avg_factor-1)*10)/2
        #                   =lowest_fine_center+offset
        # Calculate offset=((avg_factor-1)*10)/2 to build the frequency array
        avg_factor = self.channel_width / 10000
        width = self.channel_width / 1000
        offset = (avg_factor - 1) * 10 / 2.0

        for i in range(len(included_coarse_chans)):
            # get the lowest fine freq of the coarse channel (kHz)
            lower_fine_freq = included_coarse_chans[i] * 1280 - 640
            # find the center of the lowest averaged channel
            first_center = lower_fine_freq + offset
            # add the channel centers for this coarse channel into
            # the frequency array (converting from kHz to Hz)
            self.freq_array[
                0, int(i * num_fine_chans) : int((i + 1) * num_fine_chans)
            ] = (
                np.arange(first_center, first_center + num_fine_chans * width, width)
                * 1000
            )
        # polarizations are ordered yy, yx, xy, xx
        self.polarization_array = np.array([-6, -8, -7, -5])

        if read_data:
            # read data into an array with dimensions (time, uv, baselines*pols)
            self.data_array = np.zeros(
                (self.Ntimes, self.Nfreqs, self.Nbls * self.Npols),
                dtype=data_array_dtype,
            )
            self.nsample_array = np.zeros(
                (self.Ntimes, self.Nbls, self.Nfreqs, self.Npols),
                dtype=nsample_array_dtype,
            )
            self.flag_array = np.full(
                (self.Ntimes, self.Nbls, len(included_coarse_chans), self.Npols), True
            )
            # read data files
            for filename in file_dict["data"]:
                self._read_fits_file(
                    filename, time_array, file_nums_to_index, num_fine_chans, int_time
                )
            # build mapper from antenna numbers and polarizations to pfb inputs
            corr_ants_to_pfb_inputs = {}
            for i in range(len(antenna_numbers)):
                for p in range(2):
                    corr_ants_to_pfb_inputs[(antenna_numbers[i], p)] = 2 * i + p

            # for mapping, start with a pair of antennas/polarizations
            # this is the pair we want to find the data for
            # map the pair to the corresponding pfb input indices
            # map the pfb input indices to the pfb output indices
            # these are the indices for the data corresponding to the initial
            # antenna/pol pair

            # These two 1D arrays will be both C and F contiguous
            # but we are explicitly declaring C to be consistent with the rest
            # of the python which interacts with the C/Cython code.
            # generate a mapping index array
            map_inds = np.zeros((self.Nbls * self.Npols), dtype=np.int32, order="C",)
            # generate a conjugation array
            conj = np.full((self.Nbls * self.Npols), False, dtype=np.bool_, order="C",)

            _corr_fits.generate_map(corr_ants_to_pfb_inputs, map_inds, conj)

            # propagate coarse flags
            if propagate_coarse_flags:
                self.flag_array = np.any(self.flag_array, axis=2)
                self.flag_array = self.flag_array[:, :, np.newaxis, :]
                self.flag_array = np.repeat(self.flag_array, self.Nfreqs, axis=2)
            else:
                self.flag_array = np.repeat(self.flag_array, num_fine_chans, axis=2)

            # reorder data
            self.data_array = np.take(self.data_array, map_inds, axis=2)

            # conjugate data
            self.data_array[:, :, conj] = np.conj(self.data_array[:, :, conj])

            # reshape data
            self.data_array = self.data_array.reshape(
                (self.Ntimes, self.Nfreqs, self.Nbls, self.Npols)
            )

            self.data_array = np.swapaxes(self.data_array, 1, 2)

            # van vleck correction
            if correct_van_vleck:
                flagged_ants = self.van_vleck_correction(
                    flagged_ants,
                    cheby_approx=cheby_approx,
                    data_array_dtype=data_array_dtype,
                    remove_flagged_ants=remove_flagged_ants,
                    flag_small_sig_ants=flag_small_sig_ants,
                )
            else:
                # when MWA data is cast to float for the correlator, the division
                # by 127 introduces small errors that are mitigated when the data
                # is cast back into integer
                self.data_array = np.rint(
                    self.data_array / self.extra_keywords["SCALEFAC"]
                )
                self.data_array = self.data_array * self.extra_keywords["SCALEFAC"]
            # combine baseline and time axes
            self.data_array = self.data_array.reshape(
                (self.Nblts, self.Nfreqs, self.Npols)
            )
            self.flag_array = self.flag_array.reshape(
                (self.Nblts, self.Nfreqs, self.Npols)
            )
            self.nsample_array = self.nsample_array.reshape(
                (self.Nblts, self.Nfreqs, self.Npols)
            )
            # divide out digital gains
            if remove_dig_gains:
                self.history += " Divided out digital gains."
                # get gains for included coarse channels
                coarse_inds = np.where(np.isin(coarse_chans, included_coarse_chans))[0]
                # during commissioning a shift in the bit selection in the digital
                # receiver was implemented which effectively multiplies the data by
                # a factor of 64. To account for this, the digital gains are divided
                # by a factor of 64 here. For a more detailed explanation, see PR #908.
                dig_gains = dig_gains[:, coarse_inds] / 64
                dig_gains = np.repeat(dig_gains, num_fine_chans, axis=1)

                self.data_array /= (
                    dig_gains[self.ant_1_array, :, np.newaxis]
                    * dig_gains[self.ant_2_array, :, np.newaxis]
                )

            # divide out coarse band shape
            if remove_coarse_band:
                self.history += " Divided out coarse channel bandpass."
                # get coarse band shape
                with open(
                    DATA_PATH + "/mwa_config_data/MWA_rev_cb_10khz_doubles.txt", "r"
                ) as f:
                    cb = f.read().splitlines()
                cb_array = np.array(cb).astype(np.float64)
                cb_array = cb_array.reshape(int(128 / avg_factor), int(avg_factor))
                cb_array = np.average(cb_array, axis=1)
                cb_array = cb_array[0:num_fine_chans]
                cb_array = np.tile(cb_array, len(included_coarse_chans))

                self.data_array /= cb_array[:, np.newaxis]

            # cable delay corrections
            if correct_cable_len:
                self.correct_cable_length(cable_lens)

            # add spectral window index
            # TODO: Spw axis to be collapsed in future release
            self.data_array = self.data_array[:, np.newaxis, :, :]
            self.flag_array = self.flag_array[:, np.newaxis, :, :]
            self.nsample_array = self.nsample_array[:, np.newaxis, :, :]

            if use_cotter_flags:
                # throw an error if matching files not submitted
                if included_file_nums != included_flag_nums:
                    raise ValueError(
                        "flag file coarse bands do not match data file coarse bands"
                    )
                warnings.warn(
                    "coarse channel, start time, and end time flagging will default \
                        to the more aggressive of flag_init and AOFlagger"
                )
                for file in file_dict["flags"]:
                    flag_num = int(file.split("_")[-1][0:2])
                    # map file number to frequency index
                    freq_ind = file_nums_to_index[flag_num] * num_fine_chans
                    with fits.open(file) as aoflags:
                        flags = aoflags[1].data.field("FLAGS")
                    # some flag files are longer than data; crop the ends
                    flags = flags[: self.Nblts, :]
                    # some flag files are shorter than data; assume same end time
                    blt_ind = self.Nblts - len(flags)
                    flags = flags[:, np.newaxis, :, np.newaxis]
                    self.flag_array[
                        blt_ind:, :, freq_ind : freq_ind + num_fine_chans, :
                    ] = np.logical_or(
                        self.flag_array[
                            blt_ind:, :, freq_ind : freq_ind + num_fine_chans, :
                        ],
                        flags,
                    )
            # remove bad antennas or flag bad ants
            if remove_flagged_ants:
                good_ants = np.delete(self.antenna_numbers, flagged_ants)
                self.select(antenna_nums=good_ants)
            else:
                # generage baseline flags for flagged ants
                bad_ant_inds = np.nonzero(
                    np.logical_or(
                        np.in1d(self.ant_1_array, flagged_ants),
                        np.in1d(self.ant_2_array, flagged_ants),
                    )
                )[0]
                # TODO: Spw axis to be collapsed in future release
                self.flag_array[bad_ant_inds, :, :, :] = True

            if flag_init:
                self.flag_init(
                    num_fine_chans,
                    edge_width=edge_width,
                    start_flag=start_flag,
                    end_flag=end_flag,
                    flag_dc_offset=flag_dc_offset,
                )

            # to account for discrepancies between file conventions, in order
            # to be consistent with the uvw vector direction, all the data must
            # be conjugated
            self.data_array = np.conj(self.data_array)

        # wait for LSTs if set in background
        if proc is not None:
            proc.join()

        if not self.metadata_only:
            # reorder polarizations
            # reorder pols calls check so must come after
            # lst thread is re-joined.
            self.reorder_pols()

        # phasing
        if phase_to_pointing_center:
            self.phase(ra_rad, dec_rad)

        if not self.metadata_only:

            # check if object is self-consistent
            # uvws are calcuated using pyuvdata, so turn off the check for speed.
            if run_check:
                self.check(
                    check_extra=check_extra,
                    run_check_acceptability=run_check_acceptability,
                    strict_uvw_antpos_check=strict_uvw_antpos_check,
                )
