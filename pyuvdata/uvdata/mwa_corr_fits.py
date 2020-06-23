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

# from scipy.optimize import root

from scipy.special import erf
from scipy.integrate import simps, quad

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


# @profile
def sighat_vector(x):
    """
    Generate quantized sigma from a given sigma input.

    Parameters
    ----------
    x : numpy array
        Array of sigma inputs into the inverse correction function.
    bits : int
        Number of quantization bits.
    """
    # assign the upper level of the quantization
    # m = 2 ** (bits - 1) - 1
    m = 7
    # create an array
    y = np.arange(m)
    # create a sparse array to perform the function accross
    yy = np.reshape(y, (len(y), 1))
    # compute terms of summation
    z = (2 * yy + 1) * erf((yy + 0.5) / (x * np.sqrt(2)))
    # sum terms
    z = z.sum(axis=0)
    # create a new array that is the standard deviation of the quantized signal
    sighat = np.sqrt(m ** 2 - z)
    return sighat


def sighat_vector_prime(x):
    # this is not actually the full derivative; have to divide by sighat_vector
    # assign the upper level of the quantization
    # m = 2 ** (nbits - 1) - 1
    m = 7
    # create an array
    y = np.arange(m)
    # create a sparse array to perform the function across
    yy = np.reshape(y, (len(y), 1))
    # compute terms of summation
    z = (
        (2 * yy + 1)
        * (yy + 0.5)
        * np.exp(-((yy + 0.5) ** 2) / (2 * (x ** 2)))
        / (np.sqrt(2 * np.pi) * (x ** 2))
    )
    # print(z.shape)
    # sum terms
    sighat_prime = z.sum(axis=0)
    return sighat_prime


# @profile
def autos_opt_func(x, sighat):
    return sighat_vector(x) - sighat


def autos_opt_func_prime(x):
    return sighat_vector_prime(x) / sighat_vector(x)


def autos_root_jac(x, sighat):
    return np.diag(sighat_vector_prime(x) / sighat_vector(x))


# =============================================================================
# @profile
# def corrcorrect_vect(rho, bits, xsig, ysig):
#     """Generate quantized covariance from correlation and sigma inputs."""
#     # create an integration grid for midpoint summation
#     x = np.array([np.arange(i / (2 * 10000), i, i / 10000) for i in rho])
#     x = np.transpose(x)
#     # assign the upper level of the quantization
#     m = 2 ** (bits - 1) - 1
#     # create variable for summation
#     i = np.arange(0, m, 1)
#     ii = np.reshape(i, (1, len(i), 1, 1))
#     kk = np.reshape(i, (len(i), 1, 1, 1))
#     xx = np.reshape(x, (1, 1, x.shape[0], x.shape[1]))
#     # set up summation in integrand
#     z = (1 / (np.pi * np.sqrt(1 - xx ** 2))) * (
#         np.exp(
#             -(1 / (2 * (1 - xx ** 2)))
#             * (
#                 ((ii + 0.5) ** 2 / xsig ** 2)
#                 + ((kk + 0.5) ** 2 / ysig ** 2)
#                 - 2 * xx * (ii + 0.5) * (kk + 0.5) / (xsig * ysig)
#             )
#         )
#         + np.exp(
#             -(1 / (2 * (1 - xx ** 2)))
#             * (
#                 ((ii + 0.5) ** 2 / xsig ** 2)
#                 + ((kk + 0.5) ** 2 / ysig ** 2)
#                 + 2 * xx * (ii + 0.5) * (kk + 0.5) / (xsig * ysig)
#             )
#         )
#     )
#     # sum over integration variable
#     z = z.sum(2)
#     # sum over i
#     z = z.sum(0)
#     # sum over k
#     z = z.sum(0)
#     # multiply by width for midpoint Riemann sum
#     result = z * rho / 10000
#     return result
# =============================================================================

# @profile
def corrcorrect_vect_prime(rho, sig1, sig2):
    i = np.arange(0.5, 7.5, 1)
    j = np.reshape(i, (1, len(i), 1)) / sig1
    k = np.reshape(i, (len(i), 1, 1)) / sig2
    xx = rho
    # set up summation
    khat = (1 / (np.pi * np.sqrt(1 - xx ** 2))) * (
        np.exp(-(1 / (2 * (1 - xx ** 2))) * ((j ** 2) + (k ** 2) - 2 * xx * j * k))
        + np.exp(-(1 / (2 * (1 - xx ** 2))) * ((j ** 2) + (k ** 2) + 2 * xx * j * k))
    )
    # sum over i
    khat = khat.sum(0)
    # sum over k
    khat = khat.sum(0)

    return khat


# =============================================================================
# @profile
# def corrcorrect_vect_prime(rho, xsig, ysig):
#     # assign the upper level of the quantization
#     # m = 2 ** (bits - 1) - 1
#     # create variables for summation
#     i = np.arange(0.5, 7.5, 1)
#     # ii, kk, xx = np.meshgrid(i, i, rho, sparse=True)
#     ii = np.reshape(i, (1, len(i), 1))
#     kk = np.reshape(i, (len(i), 1, 1))
#     xx = rho
#     # set up summation
#     z = (1 / (np.pi * np.sqrt(1 - xx ** 2))) * (
#         np.exp(
#             -(1 / (2 * (1 - xx ** 2)))
#             * (
#                 (ii ** 2 / xsig ** 2)
#                 + (kk ** 2 / ysig ** 2)
#                 - 2 * xx * ii * kk / (xsig * ysig)
#             )
#         )
#         + np.exp(
#             -(1 / (2 * (1 - xx ** 2)))
#             * (
#                 (ii ** 2 / xsig ** 2)
#                 + (kk ** 2 / ysig ** 2)
#                 + 2 * xx * ii * kk / (xsig * ysig)
#             )
#         )
#     )
#     # sum over i
#     z = z.sum(0)
#     # sum over k
#     z = z.sum(0)
#     return z
# =============================================================================


# @profile
def corrcorrect_quad(rho, bits, xsig, ysig):
    result = [
        quad(
            corrcorrect_vect_prime, 0, rho[i], args=(4, xsig[i], ysig[i],), epsabs=1e-10
        )[0]
        for i in range(len(rho))
    ]
    return np.array(result)


# @profile
def corr_root_func(x, kaphat, sig1, sig2):
    return corrcorrect_quad(x, 4, sig1, sig2) - kaphat


# @profile
def corr_root_jac(x, kaphat, xsig, ysig):
    # assign the upper level of the quantization
    m = 2 ** (4 - 1) - 1
    # create variables for summation
    i = np.arange(0, m, 1)
    ii = np.reshape(i, (1, len(i), 1))
    kk = np.reshape(i, (len(i), 1, 1))
    xsig = np.reshape(xsig, (1, 1, len(xsig)))
    z = (
        2
        * (np.exp(-(((ii + 0.5) ** 2 / xsig ** 2) + ((kk + 0.5) ** 2 / ysig ** 2)) / 2))
        / (np.pi)
    )
    z = z.sum(0)
    z = z.sum(0)
    return np.diag(z)


# @profile
def corrcorrect_simps(rho, sig1, sig2):
    i = np.arange(0.5, 7.5, 1)
    x = np.linspace(0, rho, 11)
    j = np.reshape(i, (1, len(i), 1, 1)) / sig1
    k = np.reshape(i, (len(i), 1, 1, 1)) / sig2
    xx = np.reshape(x, (1, 1, 11, len(rho)))
    # set up summation
    khat = (1 / (np.pi * np.sqrt(1 - xx ** 2))) * (
        np.exp(-(1 / (2 * (1 - xx ** 2))) * ((j ** 2) + (k ** 2) - 2 * xx * j * k))
        + np.exp(-(1 / (2 * (1 - xx ** 2))) * ((j ** 2) + (k ** 2) + 2 * xx * j * k))
    )
    # sum over i
    khat = khat.sum(0)
    # sum over k
    khat = khat.sum(0)
    # integrate with simps
    khat = simps(khat, x, axis=0)
    return khat


# =============================================================================
# @profile
# def corrcorrect_simps(rho, xsig, ysig):
#     # assign the upper level of the quantization
#     # m = 2 ** (bits - 1) - 1
#     # create variables for summation
#     i = np.arange(0.5, 7.5, 1)
#     x = np.linspace(0, rho, 101)
#     ii = np.reshape(i, (1, len(i), 1, 1))
#     kk = np.reshape(i, (len(i), 1, 1, 1))
#     xx = np.reshape(x, (1, 1, 101, len(rho)))
#     # set up summation
#     z = (1 / (np.pi * np.sqrt(1 - xx ** 2))) * (
#         np.exp(
#             -(1 / (2 * (1 - xx ** 2)))
#             * (
#                 (ii ** 2 / xsig ** 2)
#                 + (kk ** 2 / ysig ** 2)
#                 - 2 * xx * ii * kk / (xsig * ysig)
#             )
#         )
#         + np.exp(
#             -(1 / (2 * (1 - xx ** 2)))
#             * (
#                 (ii ** 2 / xsig ** 2)
#                 + (kk ** 2 / ysig ** 2)
#                 + 2 * xx * ii * kk / (xsig * ysig)
#             )
#         )
#     )
#     # sum over i
#     z = z.sum(0)
#     # sum over k
#     z = z.sum(0)
#     z = simps(z, x, axis=0)
#     return z
# =============================================================================

# @profile
def corr_root_func_simps(x, kaphat, sig1, sig2):
    return corrcorrect_simps(x, sig1, sig2) - kaphat


# @profile
def corrcorrect_approx_taylor(rho, sig1, sig2):
    a = np.arange(-6.5, 7.5, 1.0)
    # reshape
    aa = np.reshape(a, (len(a), 1, 1)) / sig1
    # print(aa.shape)
    bb = np.reshape(a, (1, len(a), 1)) / sig2
    # print(bb.shape)
    xx = np.reshape(rho, (1, 1, len(rho)))
    khat = (
        (1 / (2 * np.pi)) * np.exp(-(aa ** 2 + bb ** 2) / 2) * (xx + aa * bb * xx ** 2)
    )
    khat = khat.sum(0)
    khat = khat.sum(0)
    return khat


# @profile
def corr_root_func_approx_taylor(x, kaphat, sig1, sig2):
    return corrcorrect_approx_taylor(x, sig1, sig2) - kaphat


# @profile
def corr_root_jac_taylor(x, kaphat, sig1, sig2):
    a = np.arange(-6.5, 7.5, 1.0)
    # reshape
    aa = np.reshape(a, (len(a), 1, 1)) / sig1
    # print(aa.shape)
    bb = np.reshape(a, (1, len(a), 1)) / sig2
    # print(bb.shape)
    xx = np.reshape(x, (1, 1, len(x)))
    khat = (1 / (2 * np.pi)) * np.exp(-(aa ** 2 + bb ** 2) / 2) * (1 + aa * bb * xx * 2)
    khat = khat.sum(0)
    khat = khat.sum(0)
    return np.diag(khat)


# @profile
def corrcorrect_approx_integrand(rho, sig1, sig2):
    a = np.arange(-6.5, 7.5, 1.0)
    # reshape
    aa = np.reshape(a, (len(a), 1, 1)) / sig1
    # print(aa.shape)
    bb = np.reshape(a, (1, len(a), 1)) / sig2
    # print(bb.shape)
    xx = np.reshape(rho, (1, 1, len(rho)))
    # print(xx.shape)
    khat = (
        (1 / (4 * np.pi * (aa ** 2 + bb ** 2) ** (5 / 2)))
        * np.exp((-(aa ** 2 + bb ** 2) / 2))
        * (
            np.sqrt(aa ** 2 + bb ** 2) * aa * bb
            - np.sqrt(aa ** 2 + bb ** 2)
            * (aa * bb + xx * (aa ** 2 + bb ** 2))
            * np.exp(-(aa ** 2 + bb ** 2) * (xx ** 2) / 2 + aa * bb * xx)
            + np.sqrt(np.pi / 2)
            * (2 * (aa ** 4 + bb ** 4) + aa ** 2 + bb ** 2 + 5 * aa ** 2 * bb ** 2)
            * np.exp(aa ** 2 * bb ** 2 / (2 * (aa ** 2 + bb ** 2)))
            * (
                erf(
                    (-aa * bb + xx * (aa ** 2 + bb ** 2))
                    / np.sqrt(2 * (aa ** 2 + bb ** 2))
                )
                - erf(-aa * bb / np.sqrt(2 * (aa ** 2 + bb ** 2)))
            )
        )
    )
    khat = khat.sum(0)
    khat = khat.sum(0)
    return khat


# @profile
def corr_root_func_approx_integrand(x, kaphat, sig1, sig2):
    return corrcorrect_approx_integrand(x, sig1, sig2) - kaphat


# @profile
# this one requires more integral evaluations
def corr_root_jac2(x, kaphat, sig1, sig2):
    return np.diag(sig1 * sig2)


class MWACorrFITS(UVData):
    """
    UVData subclass for reading MWA correlator fits files.

    This class should not be interacted with directly; instead use the
    read_mwa_corr_fits method on the UVData class.
    """

    # @profile
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

    def van_vleck_correction(self):
        """Apply a van vleck correction to the data array."""
        # get indices for autos
        autos = np.where(
            self.ant_1_array[0 : self.Nbls] == self.ant_2_array[0 : self.Nbls]
        )[0]
        # get indices for crosses
        crosses = np.where(
            self.ant_1_array[0 : self.Nbls] != self.ant_2_array[0 : self.Nbls]
        )[0]
        # generate dict for getting auto pols
        # polarizations are ordered yy, yx, xy, xx
        # TODO: generalize this for any polarization ordering
        # pol_dict = {0: (0, 0), 1: (0, 3), 2: (3, 0), 3: (3, 3)}

        # correct xx and yy autos
        pols = np.array([0, 3])
        # combine axes for speed-up
        self.data_array = self.data_array.reshape(
            (self.Nbls, self.Nfreqs * self.Ntimes, self.Npols)
        )
        # print('before square root')
        # print(self.data_array.real[0, 0, 0])
        # print(self.data_array.real[0, 0, 3])
        auto_inds = autos[:, np.newaxis, np.newaxis]
        self.data_array.real[auto_inds, :, pols] = np.sqrt(
            self.data_array.real[auto_inds, :, pols]
        )
        # print('after square root')
        # print(self.data_array.real[0, 0, 0])
        # print(self.data_array.real[0, 0, 3])

        for k in autos:
            print("correcting antenna " + str(k))
            # TODO: think about correcting zeros
            # so one weird thing is at low sigma things get rounded up to 0.06
            # don't correct zeros?

            flat_array = self.data_array.real[k, :, pols].flatten()
            # print(flat_array.shape)
            zero_inds = np.where(flat_array != 0)[0]
            sighat_array = flat_array[zero_inds]

            if len(sighat_array) > 0:
                guess = np.copy(sighat_array)
                inds = np.where(np.abs(autos_opt_func(guess, sighat_array)) > 1e-8)[0]
                # print(len(inds))
                while len(inds) != 0:
                    # print(len(inds))
                    # print(guess[inds][0:10])
                    # print(autos_opt_func(guess[inds],sighat_array[inds])[0:10])
                    # print(autos_opt_func_prime(guess[inds])[0:10])
                    guess[inds] = guess[inds] - (
                        (sighat_vector(guess[inds]) - sighat_array[inds])
                        * sighat_vector(guess[inds])
                        / sighat_vector_prime(guess[inds])
                    )
                    inds = np.where(np.abs(sighat_vector(guess) - sighat_array) > 1e-8)[
                        0
                    ]
                flat_array[zero_inds] = guess
                self.data_array.real[k, :, pols] = flat_array.reshape(
                    2, self.Ntimes * self.Nfreqs
                )

        # add back in frequency axis
        self.data_array = self.data_array.reshape(
            (self.Nbls, self.Ntimes, self.Nfreqs, self.Npols)
        )
        sig1_inds = np.array([0, 0, 3, 3])
        sig2_inds = np.array([0, 3, 0, 3])
        for k in crosses:
            print("correcting baseline " + str(k))
            auto1 = autos[self.ant_1_array[k]]
            auto2 = autos[self.ant_2_array[k]]
            for j in range(self.Nfreqs):
                # get sigmas
                sig_1 = self.data_array.real[auto1, :, j, sig1_inds].swapaxes(0, 1)
                sig_2 = self.data_array.real[auto2, :, j, sig2_inds].swapaxes(0, 1)
                # print(flat_array.shape)
                flat_array = np.abs(self.data_array.real[k, :, j, :].flatten())
                zero_inds = np.where(flat_array != 0)[0]
                kaphat_array = flat_array[zero_inds]
                if len(kaphat_array) > 0:
                    sig_array1 = sig_1.flatten()[zero_inds]
                    sig_array2 = sig_2.flatten()[zero_inds]
                    x0 = kaphat_array / (sig_array1 * sig_array2)
                    x0 = x0 - (
                        corrcorrect_simps(x0, sig_array1, sig_array2) - kaphat_array
                    ) / corrcorrect_vect_prime(x0, sig_array1, sig_array2)
                    inds = np.where(
                        np.abs(
                            corrcorrect_simps(x0, sig_array1, sig_array2) - kaphat_array
                        )
                        > 1e-8
                    )[0]
                    while len(inds) != 0:
                        # print(len(inds))
                        x0[inds] = x0[inds] - (
                            corrcorrect_simps(
                                x0[inds], sig_array1[inds], sig_array2[inds]
                            )
                            - kaphat_array[inds]
                        ) / corrcorrect_vect_prime(
                            x0[inds], sig_array1[inds], sig_array2[inds]
                        )
                        inds2 = np.where(
                            np.abs(
                                corrcorrect_simps(
                                    x0[inds], sig_array1[inds], sig_array2[inds]
                                )
                                - kaphat_array[inds]
                            )
                            > 1e-8
                        )[0]
                        inds = inds[inds2]

                    flat_array[zero_inds] = x0 * sig_array1 * sig_array2
                    self.data_array.real[k, :, j, :] = flat_array.reshape(
                        self.Ntimes, self.Npols
                    )

                flat_array = np.abs(self.data_array.imag[k, :, j, :].flatten())
                zero_inds = np.where(flat_array != 0)[0]
                kaphat_array = flat_array[zero_inds]
                if len(kaphat_array) > 0:
                    sig_array1 = sig_1.flatten()[zero_inds]
                    sig_array2 = sig_2.flatten()[zero_inds]
                    x0 = kaphat_array / (sig_array1 * sig_array2)
                    x0 = x0 - (
                        corrcorrect_simps(x0, sig_array1, sig_array2) - kaphat_array
                    ) / corrcorrect_vect_prime(x0, sig_array1, sig_array2)
                    inds = np.where(
                        np.abs(
                            corrcorrect_simps(x0, sig_array1, sig_array2) - kaphat_array
                        )
                        > 1e-8
                    )[0]
                    while len(inds) != 0:
                        # print(len(inds))
                        x0[inds] = x0[inds] - (
                            corrcorrect_simps(
                                x0[inds], sig_array1[inds], sig_array2[inds]
                            )
                            - kaphat_array[inds]
                        ) / corrcorrect_vect_prime(
                            x0[inds], sig_array1[inds], sig_array2[inds]
                        )
                        inds2 = np.where(
                            np.abs(
                                corrcorrect_simps(
                                    x0[inds], sig_array1[inds], sig_array2[inds]
                                )
                                - kaphat_array[inds]
                            )
                            > 1e-8
                        )[0]
                        inds = inds[inds2]

                    flat_array[zero_inds] = x0 * sig_array1 * sig_array2
                    self.data_array.imag[k, :, j, :] = flat_array.reshape(
                        self.Ntimes, self.Npols
                    )
        # correct xy autos
        for k in autos:
            print("correcting xy auto " + str(k))
            for j in range(self.Nfreqs):
                zero_inds = np.where(self.data_array.real[k, :, j, 1] != 0)[0]
                kaphat_array = np.abs(self.data_array.real[k, zero_inds, j, 1])
                sig_array1 = self.data_array.real[k, zero_inds, j, 0]
                sig_array2 = self.data_array.real[k, zero_inds, j, 3]
                if len(kaphat_array) > 0:
                    x0 = kaphat_array / (sig_array1 * sig_array2)
                    x0 = x0 - (
                        corrcorrect_simps(x0, sig_array1, sig_array2) - kaphat_array
                    ) / corrcorrect_vect_prime(x0, sig_array1, sig_array2)
                    inds = np.where(
                        np.abs(
                            corrcorrect_simps(x0, sig_array1, sig_array2) - kaphat_array
                        )
                        > 1e-8
                    )[0]
                    while len(inds) != 0:
                        # print(len(inds))
                        x0[inds] = x0[inds] - (
                            corrcorrect_simps(
                                x0[inds], sig_array1[inds], sig_array2[inds]
                            )
                            - kaphat_array[inds]
                        ) / corrcorrect_vect_prime(
                            x0[inds], sig_array1[inds], sig_array2[inds]
                        )
                        inds2 = np.where(
                            np.abs(
                                corrcorrect_simps(
                                    x0[inds], sig_array1[inds], sig_array2[inds]
                                )
                                - kaphat_array[inds]
                            )
                            > 1e-8
                        )[0]
                        inds = inds[inds2]
                    self.data_array.real[k, zero_inds, j, 1] = (
                        x0 * sig_array1 * sig_array2
                    )
                    self.data_array.real[k, zero_inds, j, 2] = (
                        x0 * sig_array1 * sig_array2
                    )

                zero_inds = np.where(self.data_array.imag[k, :, j, 1] != 0)[0]
                kaphat_array = np.abs(self.data_array.real[k, zero_inds, j, 1])
                sig_array1 = self.data_array.real[k, zero_inds, j, 0]
                sig_array2 = self.data_array.real[k, zero_inds, j, 3]
                if len(kaphat_array) > 0:
                    x0 = kaphat_array / (sig_array1 * sig_array2)
                    x0 = x0 - (
                        corrcorrect_simps(x0, sig_array1, sig_array2) - kaphat_array
                    ) / corrcorrect_vect_prime(x0, sig_array1, sig_array2)
                    inds = np.where(
                        np.abs(
                            corrcorrect_simps(x0, sig_array1, sig_array2) - kaphat_array
                        )
                        > 1e-8
                    )[0]
                    while len(inds) != 0:
                        # print(len(inds))
                        x0[inds] = x0[inds] - (
                            corrcorrect_simps(
                                x0[inds], sig_array1[inds], sig_array2[inds]
                            )
                            - kaphat_array[inds]
                        ) / corrcorrect_vect_prime(
                            x0[inds], sig_array1[inds], sig_array2[inds]
                        )
                        inds2 = np.where(
                            np.abs(
                                corrcorrect_simps(
                                    x0[inds], sig_array1[inds], sig_array2[inds]
                                )
                                - kaphat_array[inds]
                            )
                            > 1e-8
                        )[0]
                        inds = inds[inds2]
                    self.data_array.imag[k, zero_inds, j, 1] = (
                        x0 * sig_array1 * sig_array2
                    )
                    self.data_array.imag[k, zero_inds, j, 2] = -(
                        x0 * sig_array1 * sig_array2
                    )

        self.data_array.real[auto_inds, :, :, pols] = (
            self.data_array.real[auto_inds, :, :, pols] ** 2
        )

    # @profile
    def read_mwa_corr_fits(
        self,
        filelist,
        use_cotter_flags=None,
        remove_dig_gains=True,
        remove_coarse_band=True,
        correct_cable_len=False,
        correct_van_vleck=False,
        phase_to_pointing_center=False,
        propagate_coarse_flags=True,
        flag_init=True,
        edge_width=80e3,
        start_flag="goodtime",
        end_flag=0.0,
        flag_dc_offset=True,
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
        bscale = None
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

                    # get scaling info
                    if bscale is None:
                        if "BSCALE" in head0.keys():
                            bscale = head0["BSCALE"]
                            self.extra_keywords["SCALEFAC"] = head0["BSCALE"]
                        else:
                            # correlator did a divide by 4 before october 2014
                            bscale = 0.25
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

            # generage baseline flags for flagged ants
            bad_ant_inds = np.nonzero(
                np.logical_or(
                    np.in1d(ant_1_array, flagged_ants),
                    np.in1d(ant_2_array, flagged_ants),
                )
            )[0]
            self.flag_array[:, bad_ant_inds, :, :] = True

            # van vleck correction
            if correct_van_vleck:
                # need data array to have 64 bit precision
                if self.data_array.dtype != np.complex128:
                    self.data_array = self.data_array.astype(np.complex128)
                # scale the data
                # number of samples per fine channel is equal to channel width (Hz)
                nsamples = self.channel_width * self.integration_time[0]
                # cast data to ints
                self.data_array = np.rint(self.data_array / bscale)
                # take advantage of cicular polarization! divide by two
                self.data_array = self.data_array / nsamples
                self.data_array = self.data_array / 2.0
                # reshape to (nbls, ntimes, nfreqs, npols)
                self.data_array = np.swapaxes(self.data_array, 0, 1)
                # self.data_array = np.around(self.data_array, round_factor)
                self.van_vleck_correction()
                # reshape to (ntimes, nbls, nfreqs, npols)
                self.data_array = np.swapaxes(self.data_array, 0, 1)
                # rescale the data
                self.data_array = self.data_array * (bscale * nsamples * 2)
                # return data array to desired precision
                if self.data_array.dtype != data_array_dtype:
                    self.data_array = self.data_array.astype(data_array_dtype)

            else:
                # when MWA data is cast to float for the correlator, the division
                # by 127 introduces small errors that are mitigated when the data
                # is cast back into integer
                self.data_array = np.rint(self.data_array / bscale)
                self.data_array = self.data_array * bscale

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
            self.data_array = self.data_array[:, np.newaxis, :, :]
            self.flag_array = self.flag_array[:, np.newaxis, :, :]
            self.nsample_array = self.nsample_array[:, np.newaxis, :, :]

            # because of an annoying discrepancy between file conventions, in order
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
            if flag_init:
                self.flag_init(
                    num_fine_chans,
                    edge_width=edge_width,
                    start_flag=start_flag,
                    end_flag=end_flag,
                    flag_dc_offset=flag_dc_offset,
                )

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

            # check if object is self-consistent
            # uvws are calcuated using pyuvdata, so turn off the check for speed.
            if run_check:
                self.check(
                    check_extra=check_extra,
                    run_check_acceptability=run_check_acceptability,
                    strict_uvw_antpos_check=strict_uvw_antpos_check,
                )
