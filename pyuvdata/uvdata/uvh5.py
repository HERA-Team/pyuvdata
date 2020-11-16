# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Class for reading and writing UVH5 files."""
import numpy as np
import os
import warnings
import h5py
from ast import literal_eval

from .uvdata import UVData
from .. import utils as uvutils

__all__ = ["UVH5"]


# define HDF5 type for interpreting HERA correlator outputs (integers) as
# complex numbers
_hera_corr_dtype = np.dtype([("r", "<i4"), ("i", "<i4")])


def _check_uvh5_dtype(dtype):
    """
    Check that a specified custom datatype conforms to UVH5 standards.

    According to the UVH5 spec, the data type for the data array must be a
    compound datatype with an "r" field and an "i" field. Additionally, both
    datatypes must be the same (e.g., "<i4", "<r8", etc.).

    Parameters
    ----------
    dtype : numpy dtype
        A numpy dtype object with an "r" field and an "i" field.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        This is raised if dtype is not a numpy dtype, if the dtype does not have
        an "r" field and an "i" field, or if the "r" field and "i" fields have
        different types.
    """
    if not isinstance(dtype, np.dtype):
        raise ValueError("dtype in a uvh5 file must be a numpy dtype")
    if "r" not in dtype.names or "i" not in dtype.names:
        raise ValueError(
            "dtype must be a compound datatype with an 'r' field and an 'i' field"
        )
    rkind = dtype["r"].kind
    ikind = dtype["i"].kind
    if rkind != ikind:
        raise ValueError(
            "dtype must have the same kind ('i4', 'r8', etc.) for both real "
            "and imaginary fields"
        )
    return


def _read_complex_astype(dset, indices, dtype_out=np.complex64):
    """
    Read the given data set of a specified type to floating point complex data.

    Parameters
    ----------
    dset : h5py dataset
        A reference to an HDF5 dataset on disk.
    indices : tuple
        The indices to extract. Should be either lists of indices or numpy
        slice objects.
    dtype_out : str or numpy dtype
        The datatype of the output array. One of (complex, np.complex64,
        np.complex128). Default is np.complex64 (single-precision real and
        imaginary floats).

    Returns
    -------
    output_array : ndarray
        The array referenced in the dataset cast to complex values.

    Raises
    ------
    ValueError
        This is raised if dtype_out is not an acceptable value.
    """
    if dtype_out not in (complex, np.complex64, np.complex128):
        raise ValueError(
            "output datatype must be one of (complex, np.complex64, np.complex128)"
        )
    dset_shape = _get_dset_shape(dset, indices)
    output_array = np.empty(dset_shape, dtype=dtype_out)
    dtype_in = dset.dtype
    with dset.astype(dtype_in):
        # dset is indexed in native dtype, but is upcast upon assignment
        output_array.real = _index_dset(dset["r"], indices)
        output_array.imag = _index_dset(dset["i"], indices)

    return output_array


def _write_complex_astype(data, dset, indices):
    """
    Write floating point complex data as a specified type.

    Parameters
    ----------
    data : ndarray
        The data array to write out. Should be a complex-valued array that
        supports the .real and .imag attributes for accessing real and imaginary
        components.
    dset : h5py dataset
        A reference to an HDF5 dataset on disk.
    indices : tuple
        A 4-tuple representing indices to write data to. Should be either lists
        of indices or numpy slice objects.

    Returns
    -------
    None
    """
    # get datatype from dataset
    dtype_out = dset.dtype
    # make doubly sure dtype is valid; should be unless user is pathological
    _check_uvh5_dtype(dtype_out)
    with dset.astype(dtype_out):
        dset[indices[0], indices[1], indices[2], indices[3], "r"] = data.real
        dset[indices[0], indices[1], indices[2], indices[3], "i"] = data.imag
    return


def _convert_to_slices(indices, max_nslice_frac=0.1):
    """
    Convert list of indices to a list of slices.

    Parameters
    ----------
    indices : list
        A 1D list of integers for array indexing.
    max_nslice_frac : float
        A float from 0 -- 1. If the number of slices
        needed to represent input 'indices' divided by len(indices)
        exceeds this fraction, then we determine that we cannot
        easily represent 'indices' with a list of slices.

    Returns
    -------
    list
        list of slice objects used to represent indices
    bool
        If True, indices is easily represented by slices
        (max_nslice_frac condition met), otherwise False

    Notes
    -----
    Example:
        if: indices = [1, 2, 3, 4, 10, 11, 12, 13, 14]
        then: slices = [slice(1, 5, 1), slice(11, 15, 1)]
    """
    # check for integer index
    if isinstance(indices, (int, np.integer)):
        indices = [indices]
    # check for already a slice
    if isinstance(indices, slice):
        return [indices], True
    # assert indices is longer than 2, or return trivial solutions
    if len(indices) == 0:
        return [slice(0, 0, 0)], False
    elif len(indices) == 1:
        return [slice(indices[0], indices[0] + 1, 1)], True
    elif len(indices) == 2:
        return [slice(indices[0], indices[1] + 1, indices[1] - indices[0])], True

    # setup empty slices list
    Ninds = len(indices)
    slices = []

    # iterate over indices
    for i, ind in enumerate(indices):
        if i == 0:
            # start the first slice object
            start = ind
            last_step = indices[i + 1] - ind
            continue

        # calculate step from previous index
        step = ind - indices[i - 1]

        # if step != last_step, this ends the slice
        if step != last_step:
            # append to list
            slices.append(slice(start, indices[i - 1] + 1, last_step))

            # check if this is the last element
            if i == Ninds - 1:
                # append last element
                slices.append(slice(ind, ind + 1, 1))
                continue

            # setup next step
            start = ind
            last_step = indices[i + 1] - ind

        # check if this is the last element
        elif i == Ninds - 1:
            # end slice and append
            slices.append(slice(start, ind + 1, step))

    # determine whether slices are a reasonable representation
    Nslices = len(slices)
    passed = (float(Nslices) / len(indices)) < max_nslice_frac

    return slices, passed


def _get_slice_len(s, axlen):
    """
    Get length of a slice s into array of len axlen.

    Parameters
    ----------
    s : slice object
        Slice object to index with
    axlen : int
        Length of axis s slices into

    Returns
    -------
    int
        Length of slice object
    """
    if s.start is None:
        start = 0
    else:
        start = s.start
    if s.stop is None:
        stop = axlen
    else:
        stop = np.min([s.stop, axlen])
    if s.step is None:
        step = 1
    else:
        step = s.step

    return ((stop - 1 - start) // step) + 1


def _get_dset_shape(dset, indices):
    """
    Given a 4-tuple of indices, determine the indexed array shape.

    Parameters
    ----------
    dset : h5py dataset
        A reference to an HDF5 dataset on disk.
    indices : tuple
        A 4-tuple with the indices to extract along each dimension of dset.
        Each element should contain a list of indices, a slice element,
        or a list of slice elements that will be concatenated after slicing.

    Returns
    -------
    tuple
        4-tuple with the shape of the indexed array
    """
    dset_shape = list(dset.shape)
    for i, inds in enumerate(indices):
        # check for integer
        if isinstance(inds, (int, np.integer)):
            dset_shape[i] = 1
        # check for slice object
        if isinstance(inds, slice):
            dset_shape[i] = _get_slice_len(inds, dset_shape[i])
        # check for list
        if isinstance(inds, list):
            # check for list of integers
            if isinstance(inds[0], (int, np.integer)):
                dset_shape[i] = len(inds)
            elif isinstance(inds[0], slice):
                dset_shape[i] = sum((_get_slice_len(s, dset_shape[i]) for s in inds))

    return dset_shape


def _index_dset(dset, indices):
    """
    Index a UVH5 data, flags or nsamples h5py dataset.

    Parameters
    ----------
    dset : h5py dataset
        A reference to an HDF5 dataset on disk.
    indices : tuple
        A 4-tuple with the indices to extract along each dimension of dset.
        Each element should contain a list of indices, a slice element,
        or a list of slice elements that will be concatenated after slicing.
        Indices must be provided such that all dimensions can be indexed
        simultaneously.

    Returns
    -------
    ndarray
        The indexed dset

    Notes
    -----
    This makes and fills an empty array with dset indices.
    For trivial indexing, (e.g. a trivial slice), constructing
    a new array and filling it is suboptimal over direct
    indexing, e.g. dset[indices].
    This function specializes in repeated slices over the same axis,
    e.g. if indices is [[slice(0, 5), slice(10, 15), ...], ..., ]
    """
    # get dset and arr shape
    dset_shape = dset.shape
    arr_shape = _get_dset_shape(dset, indices)

    # create empty array of dset dtype
    arr = np.empty(arr_shape, dtype=dset.dtype)

    # get arr and dset indices for each dimension in indices
    dset_indices = []
    arr_indices = []
    for i, dset_inds in enumerate(indices):
        if isinstance(dset_inds, (int, np.integer)):
            # this dimension is len 1, so slice is fine
            arr_indices.append([slice(None)])
            dset_indices.append([[dset_inds]])

        elif isinstance(dset_inds, slice):
            # this dimension is just a slice, so slice is fine
            arr_indices.append([slice(None)])
            dset_indices.append([dset_inds])

        elif isinstance(dset_inds, list):
            if isinstance(dset_inds[0], (int, np.integer)):
                # this is a list of integers, append slice
                arr_indices.append([slice(None)])
                dset_indices.append([dset_inds])
            elif isinstance(dset_inds[0], slice):
                # this is a list of slices, need list of slice lens
                slens = [_get_slice_len(s, dset_shape[i]) for s in dset_inds]
                ssums = [sum(slens[:j]) for j in range(len(slens))]
                arr_inds = [slice(s, s + l) for s, l in zip(ssums, slens)]
                arr_indices.append(arr_inds)
                dset_indices.append(dset_inds)

    # iterate over each of the 4 axes and fill the array
    for blt_arr, blt_dset in zip(arr_indices[0], dset_indices[0]):
        for spw_arr, spw_dset in zip(arr_indices[1], dset_indices[1]):
            for freq_arr, freq_dset in zip(arr_indices[2], dset_indices[2]):
                for pol_arr, pol_dset in zip(arr_indices[3], dset_indices[3]):
                    # index dset and assign to arr
                    arr[blt_arr, spw_arr, freq_arr, pol_arr] = dset[
                        blt_dset, spw_dset, freq_dset, pol_dset
                    ]

    return arr


class UVH5(UVData):
    """
    A class for UVH5 file objects.

    This class defines an HDF5-specific subclass of UVData for reading and
    writing UVH5 files. This class should not be interacted with directly,
    instead use the read_uvh5 and write_uvh5 methods on the UVData class.
    """

    def _read_header(
        self, header, filename, run_check_acceptability=True, background_lsts=True
    ):
        """
        Read header information from a UVH5 file.

        This is an internal function called by the user-space methods.
        Properties of the UVData object are updated as the file is processed.

        Parameters
        ----------
        header : h5py datagroup
            A reference to an h5py data group that contains the header
            information. Should be "/Header" for UVH5 files conforming to spec.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            reading in the file.
        background_lsts : bool
            When set to True, the lst_array is calculated in a background thread.

        Returns
        -------
        None
        """
        # get telescope information
        latitude = header["latitude"][()]
        longitude = header["longitude"][()]
        altitude = header["altitude"][()]
        self.telescope_location_lat_lon_alt_degrees = (latitude, longitude, altitude)
        self.instrument = bytes(header["instrument"][()]).decode("utf8")
        self.telescope_name = bytes(header["telescope_name"][()]).decode("utf8")

        # set history appropriately
        self.history = bytes(header["history"][()]).decode("utf8")
        if not uvutils._check_history_version(self.history, self.pyuvdata_version_str):
            self.history += self.pyuvdata_version_str

        # check for vis_units
        if "vis_units" in header:
            self.vis_units = bytes(header["vis_units"][()]).decode("utf8")
        else:
            # default to uncalibrated data
            self.vis_units = "UNCALIB"

        # check for optional values
        if "dut1" in header:
            self.dut1 = float(header["dut1"][()])
        if "earth_omega" in header:
            self.earth_omega = float(header["earth_omega"][()])
        if "gst0" in header:
            self.gst0 = float(header["gst0"][()])
        if "rdate" in header:
            self.rdate = bytes(header["rdate"][()]).decode("utf8")
        if "timesys" in header:
            self.timesys = bytes(header["timesys"][()]).decode("utf8")
        if "x_orientation" in header:
            self.x_orientation = bytes(header["x_orientation"][()]).decode("utf8")
        if "blt_order" in header:
            blt_order_str = bytes(header["blt_order"][()]).decode("utf8")
            self.blt_order = tuple(blt_order_str.split(", "))
            if self.blt_order == ("bda",):
                self._blt_order.form = (1,)

        if "antenna_diameters" in header:
            self.antenna_diameters = header["antenna_diameters"][()]
        if "uvplane_reference_time" in header:
            self.uvplane_reference_time = int(header["uvplane_reference_time"][()])
        if "eq_coeffs" in header:
            self.eq_coeffs = header["eq_coeffs"][()]
        if "eq_coeffs_convention" in header:
            self.eq_coeffs_convention = bytes(
                header["eq_coeffs_convention"][()]
            ).decode("utf8")

        # We've added a few new keywords that did not exist before, so check to see if
        # it is in the header, and if not, mark the data set as being "regular" (e.g.,
        # not a flexible spectral window setup, single source only).
        if "flex_spw" in header:
            if bool(header["flex_spw"][()]):
                self._set_flex_spw()
        if "flex_spw_id_array" in header:
            self.flex_spw_id_array = header["flex_spw_id_array"][:]
        if "multi_object" in header:
            if bool(header["multi_object"][()]):
                self._set_phased()
                self._set_multi_object(preserve_source_info=False)
        if "Nobjects" in header:
            self.Nobjects = int(header["Nobjects"][()])

        self.phase_type = bytes(header["phase_type"][()]).decode("utf8")
        # Here is where we start handing object information.  If we have a multi-object
        # dataset, this information is going to be handled a bit differently, since
        # all of these entries are expected to be required arrays.
        if self.multi_object:
            self.object_name = [
                bytes(name).decode("utf8") for name in header["object_name"][:]
            ]
            self.object_id_array = header["object_id_array"][:]
        else:
            self.object_name = bytes(header["object_name"][()]).decode("utf8")

        # check for phasing information
        if self.phase_type == "phased":
            self.phase_center_ra = float(header["phase_center_ra"][()])
            self.phase_center_dec = float(header["phase_center_dec"][()])
            self.phase_center_epoch = float(header["phase_center_epoch"][()])
            self._set_phased()
            if "phase_center_app_ra" in header and "phase_center_app_dec" in header:
                self.phase_center_app_ra = header["phase_center_app_ra"][:]
                self.phase_center_app_dec = header["phase_center_app_dec"][:]
        elif self.phase_type == "drift":
            self._set_drift()
        else:
            self._set_unknown_phase_type()

        # Here is where we collect the other optional source/phasing info
        if "object_dict" in header:
            self.object_dict = {}
            for key in header["object_dict"].keys():
                self.object_dict[key] = literal_eval(
                    bytes(header["object_dict"][key][()]).decode("utf8")
                )
        if "phase_center_frame" in header:
            self.phase_center_frame = bytes(header["phase_center_frame"][()]).decode(
                "utf8"
            )

        # get antenna arrays
        # cast to native python int type
        self.Nants_data = int(header["Nants_data"][()])
        self.Nants_telescope = int(header["Nants_telescope"][()])
        self.ant_1_array = header["ant_1_array"][:]
        self.ant_2_array = header["ant_2_array"][:]
        self.antenna_names = [
            bytes(n).decode("utf8") for n in header["antenna_names"][:]
        ]
        self.antenna_numbers = header["antenna_numbers"][:]
        self.antenna_positions = header["antenna_positions"][:]

        # set telescope params
        try:
            self.set_telescope_params()
        except ValueError as ve:
            warnings.warn(str(ve))

        # get baseline array
        self.baseline_array = self.antnums_to_baseline(
            self.ant_1_array, self.ant_2_array
        )
        self.Nbls = len(np.unique(self.baseline_array))

        # get uvw array
        self.uvw_array = header["uvw_array"][:, :]

        # get time information
        self.time_array = header["time_array"][:]
        integration_time = header["integration_time"]
        self.integration_time = integration_time[:]
        proc = None
        if "lst_array" in header:
            self.lst_array = header["lst_array"][:]
            # check that lst_array in file is self-consistent
            if run_check_acceptability:
                (
                    latitude,
                    longitude,
                    altitude,
                ) = self.telescope_location_lat_lon_alt_degrees

                lst_array = uvutils.get_lst_for_time(
                    self.time_array, latitude, longitude, altitude,
                )

                if not np.all(
                    np.isclose(
                        self.lst_array,
                        lst_array,
                        rtol=self._lst_array.tols[0],
                        atol=self._lst_array.tols[1],
                    )
                ):
                    warnings.warn(
                        "LST values stored in {file} are not self-consistent "
                        "with time_array and telescope location. Consider "
                        "recomputing with utils.get_lst_for_time.".format(file=filename)
                    )
        else:
            # compute lst_array from time_array and telescope location
            proc = self.set_lsts_from_time_array(background=background_lsts)

        # get frequency information
        self.freq_array = header["freq_array"][:, :]
        self.spw_array = header["spw_array"][:]

        # Pull in the channel_width parameter as either an array or as a single float,
        # depending on whether or not the data is stored with a flexible spw.
        if self.flex_spw:
            self.channel_width = header["channel_width"][:]
        else:
            self.channel_width = float(header["channel_width"][()])

        # get polarization information
        self.polarization_array = header["polarization_array"][:]

        # get data shapes
        self.Nfreqs = int(header["Nfreqs"][()])
        self.Npols = int(header["Npols"][()])
        self.Ntimes = int(header["Ntimes"][()])
        self.Nblts = int(header["Nblts"][()])
        self.Nspws = int(header["Nspws"][()])

        # get extra_keywords
        if "extra_keywords" in header:
            self.extra_keywords = {}
            for key in header["extra_keywords"].keys():
                if header["extra_keywords"][key].dtype.type in (np.string_, np.object_):
                    self.extra_keywords[key] = bytes(
                        header["extra_keywords"][key][()]
                    ).decode("utf8")
                else:
                    self.extra_keywords[key] = header["extra_keywords"][key][()]

        if proc is not None:
            # if lsts are in the background wait for them to return
            proc.join()

        # Finally, backfill the apparent coords if they aren't in the original datafile
        if self.phase_type == "phased" and (
            (self.phase_center_app_ra is None) or (self.phase_center_app_dec is None)
        ):
            self._set_app_coords_helper()

        return

    def _get_data(
        self,
        dgrp,
        antenna_nums,
        antenna_names,
        ant_str,
        bls,
        frequencies,
        freq_chans,
        times,
        time_range,
        polarizations,
        blt_inds,
        data_array_dtype,
        keep_all_metadata,
        multidim_index,
        run_check,
        check_extra,
        run_check_acceptability,
        strict_uvw_antpos_check,
    ):
        """
        Read the data-size arrays (data, flags, nsamples) from a file.

        This is an internal function to read just the visibility, flag, and
        nsample data of the UVH5 file. This is separated from full read so that
        header/metadata and data can be read independently. See the
        documentation of `read_uvh5` for a full description of most of the
        descriptions of parameters. Below we only include a description of args
        unique to this function.

        Parameters
        ----------
        dgrp : h5py datagroup
            The HDF5 datagroup containing the datasets. Should be "/Data" for
            UVH5 files conforming to spec.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            This is raised if the data array read from the file is not a complex
            datatype (np.complex64 or np.complex128).
        """
        # figure out what data to read in
        blt_inds, freq_inds, pol_inds, history_update_string = self._select_preprocess(
            antenna_nums,
            antenna_names,
            ant_str,
            bls,
            frequencies,
            freq_chans,
            times,
            time_range,
            polarizations,
            blt_inds,
        )

        # figure out which axis is the most selective
        if blt_inds is not None:
            blt_frac = len(blt_inds) / float(self.Nblts)
        else:
            blt_frac = 1

        if freq_inds is not None:
            freq_frac = len(freq_inds) / float(self.Nfreqs)
        else:
            freq_frac = 1

        if pol_inds is not None:
            pol_frac = len(pol_inds) / float(self.Npols)
        else:
            pol_frac = 1

        min_frac = np.min([blt_frac, freq_frac, pol_frac])

        # get the fundamental datatype of the visdata; if integers, we need to
        # cast to floats
        visdata_dtype = dgrp["visdata"].dtype
        if visdata_dtype not in ("complex64", "complex128"):
            _check_uvh5_dtype(visdata_dtype)
            if data_array_dtype not in (np.complex64, np.complex128):
                raise ValueError(
                    "data_array_dtype must be np.complex64 or np.complex128"
                )
            custom_dtype = True
        else:
            custom_dtype = False

        if min_frac == 1:
            # no select, read in all the data
            inds = (np.s_[:], np.s_[:], np.s_[:], np.s_[:])
            if custom_dtype:
                self.data_array = _read_complex_astype(
                    dgrp["visdata"], inds, data_array_dtype
                )
            else:
                self.data_array = _index_dset(dgrp["visdata"], inds)
            self.flag_array = _index_dset(dgrp["flags"], inds)
            self.nsample_array = _index_dset(dgrp["nsamples"], inds)
        else:
            # do select operations on everything except data_array, flag_array
            # and nsample_array
            self._select_metadata(
                blt_inds, freq_inds, pol_inds, history_update_string, keep_all_metadata
            )

            # determine which axes can be sliced, rather than fancy indexed
            # max_nslice_frac of 0.1 yields slice speedup over fancy index for HERA data
            # See pyuvdata PR #805
            if blt_inds is not None:
                blt_slices, blt_sliceable = _convert_to_slices(
                    blt_inds, max_nslice_frac=0.1
                )
            else:
                blt_inds, blt_slices = np.s_[:], np.s_[:]
                blt_sliceable = True

            if freq_inds is not None:
                freq_slices, freq_sliceable = _convert_to_slices(
                    freq_inds, max_nslice_frac=0.1
                )
            else:
                freq_inds, freq_slices = np.s_[:], np.s_[:]
                freq_sliceable = True

            if pol_inds is not None:
                pol_slices, pol_sliceable = _convert_to_slices(
                    pol_inds, max_nslice_frac=0.5
                )
            else:
                pol_inds, pol_slices = np.s_[:], np.s_[:]
                pol_sliceable = True

            # open references to datasets
            visdata_dset = dgrp["visdata"]
            flags_dset = dgrp["flags"]
            nsamples_dset = dgrp["nsamples"]

            # check that multidim_index is appropriate
            if multidim_index:
                # if more than one dim is not sliceable, then not appropriate
                if sum([blt_sliceable, freq_sliceable, pol_sliceable]) < 2:
                    multidim_index = False

            # just read in the right portions of the data and flag arrays
            if blt_frac == min_frac:
                # construct inds list given simultaneous sliceability
                inds = [blt_inds, np.s_[:], np.s_[:], np.s_[:]]
                if blt_sliceable:
                    inds[0] = blt_slices
                if multidim_index:
                    if freq_sliceable:
                        inds[2] = freq_slices
                    else:
                        inds[2] = freq_inds
                if multidim_index:
                    if pol_sliceable:
                        inds[3] = pol_slices
                    else:
                        inds[3] = pol_inds

                inds = tuple(inds)

                # index datasets
                if custom_dtype:
                    visdata = _read_complex_astype(visdata_dset, inds, data_array_dtype)
                else:
                    visdata = _index_dset(visdata_dset, inds)
                flags = _index_dset(flags_dset, inds)
                nsamples = _index_dset(nsamples_dset, inds)

                # down select on other dimensions if necessary
                # use indices not slices here: generally not the bottleneck
                if not multidim_index and freq_frac < 1:
                    visdata = visdata[:, :, freq_inds, :]
                    flags = flags[:, :, freq_inds, :]
                    nsamples = nsamples[:, :, freq_inds, :]
                if not multidim_index and pol_frac < 1:
                    visdata = visdata[:, :, :, pol_inds]
                    flags = flags[:, :, :, pol_inds]
                    nsamples = nsamples[:, :, :, pol_inds]

            elif freq_frac == min_frac:
                # construct inds list given simultaneous sliceability
                inds = [np.s_[:], np.s_[:], freq_inds, np.s_[:]]
                if freq_sliceable:
                    inds[2] = freq_slices
                if multidim_index:
                    if blt_sliceable:
                        inds[0] = blt_slices
                    else:
                        inds[0] = blt_inds
                if multidim_index:
                    if pol_sliceable:
                        inds[3] = pol_slices
                    else:
                        inds[3] = pol_inds

                inds = tuple(inds)

                # index datasets
                if custom_dtype:
                    visdata = _read_complex_astype(visdata_dset, inds, data_array_dtype)
                else:
                    visdata = _index_dset(visdata_dset, inds)
                flags = _index_dset(flags_dset, inds)
                nsamples = _index_dset(nsamples_dset, inds)

                # down select on other dimensions if necessary
                # use indices not slices here: generally not the bottleneck
                if not multidim_index and blt_frac < 1:
                    visdata = visdata[blt_inds, :, :, :]
                    flags = flags[blt_inds, :, :, :]
                    nsamples = nsamples[blt_inds, :, :, :]
                if not multidim_index and pol_frac < 1:
                    visdata = visdata[:, :, :, pol_inds]
                    flags = flags[:, :, :, pol_inds]
                    nsamples = nsamples[:, :, :, pol_inds]

            else:
                # construct inds list given simultaneous sliceability
                inds = [np.s_[:], np.s_[:], np.s_[:], pol_inds]
                if pol_sliceable:
                    inds[3] = pol_slices
                if multidim_index:
                    if blt_sliceable:
                        inds[0] = blt_slices
                    else:
                        inds[0] = blt_inds
                if multidim_index:
                    if freq_sliceable:
                        inds[2] = freq_slices
                    else:
                        inds[2] = freq_inds

                inds = tuple(inds)

                # index datasets
                if custom_dtype:
                    visdata = _read_complex_astype(visdata_dset, inds, data_array_dtype)
                else:
                    visdata = _index_dset(visdata_dset, inds)
                flags = _index_dset(flags_dset, inds)
                nsamples = _index_dset(nsamples_dset, inds)

                # down select on other dimensions if necessary
                # use indices not slices here: generally not the bottleneck
                if not multidim_index and blt_frac < 1:
                    visdata = visdata[blt_inds, :, :, :]
                    flags = flags[blt_inds, :, :, :]
                    nsamples = nsamples[blt_inds, :, :, :]
                if not multidim_index and freq_frac < 1:
                    visdata = visdata[:, :, freq_inds, :]
                    flags = flags[:, :, freq_inds, :]
                    nsamples = nsamples[:, :, freq_inds, :]

            # save arrays in object
            self.data_array = visdata
            self.flag_array = flags
            self.nsample_array = nsamples

        # check if object has all required UVParameters set
        if run_check:
            self.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
            )

        return

    def read_uvh5(
        self,
        filename,
        antenna_nums=None,
        antenna_names=None,
        ant_str=None,
        bls=None,
        frequencies=None,
        freq_chans=None,
        times=None,
        time_range=None,
        polarizations=None,
        blt_inds=None,
        keep_all_metadata=True,
        read_data=True,
        data_array_dtype=np.complex128,
        multidim_index=False,
        background_lsts=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
    ):
        """
        Read in data from a UVH5 file.

        Parameters
        ----------
        filename : str
             The UVH5 file to read from.
        antenna_nums : array_like of int, optional
            The antennas numbers to include when reading data into the object
            (antenna positions and names for the removed antennas will be retained
            unless `keep_all_metadata` is False). This cannot be provided if
            `antenna_names` is also provided. Ignored if read_data is False.
        antenna_names : array_like of str, optional
            The antennas names to include when reading data into the object
            (antenna positions and names for the removed antennas will be retained
            unless `keep_all_metadata` is False). This cannot be provided if
            `antenna_nums` is also provided. Ignored if read_data is False.
        bls : list of tuple, optional
            A list of antenna number tuples (e.g. [(0, 1), (3, 2)]) or a list of
            baseline 3-tuples (e.g. [(0, 1, 'xx'), (2, 3, 'yy')]) specifying baselines
            to include when reading data into the object. For length-2 tuples,
            the ordering of the numbers within the tuple does not matter. For
            length-3 tuples, the polarization string is in the order of the two
            antennas. If length-3 tuples are provided, `polarizations` must be
            None. Ignored if read_data is False.
        ant_str : str, optional
            A string containing information about what antenna numbers
            and polarizations to include when reading data into the object.
            Can be 'auto', 'cross', 'all', or combinations of antenna numbers
            and polarizations (e.g. '1', '1_2', '1x_2y').  See tutorial for more
            examples of valid strings and the behavior of different forms for ant_str.
            If '1x_2y,2y_3y' is passed, both polarizations 'xy' and 'yy' will
            be kept for both baselines (1, 2) and (2, 3) to return a valid
            pyuvdata object.
            An ant_str cannot be passed in addition to any of `antenna_nums`,
            `antenna_names`, `bls` args or the `polarizations` parameters,
            if it is a ValueError will be raised. Ignored if read_data is False.
        frequencies : array_like of float, optional
            The frequencies to include when reading data into the object, each
            value passed here should exist in the freq_array. Ignored if
            read_data is False.
        freq_chans : array_like of int, optional
            The frequency channel numbers to include when reading data into the
            object. Ignored if read_data is False.
        times : array_like of float, optional
            The times to include when reading data into the object, each value
            passed here should exist in the time_array. Cannot be used with
            `time_range`.
        time_range : array_like of float, optional
            The time range in Julian Date to keep in the object, must be
            length 2. Some of the times in the object should fall between the
            first and last elements. Cannot be used with `times`.
        polarizations : array_like of int, optional
            The polarizations numbers to include when reading data into the
            object, each value passed here should exist in the polarization_array.
            Ignored if read_data is False.
        blt_inds : array_like of int, optional
            The baseline-time indices to include when reading data into the
            object. This is not commonly used. Ignored if read_data is False.
        keep_all_metadata : bool
            Option to keep all the metadata associated with antennas, even those
            that do not have data associated with them after the select option.
        read_data : bool
            Read in the visibility, nsample and flag data. If set to False, only
            the metadata will be read in. Setting read_data to False results in
            a metadata only object.
        data_array_dtype : numpy dtype
            Datatype to store the output data_array as. Must be either
            np.complex64 (single-precision real and imaginary) or np.complex128 (double-
            precision real and imaginary). Only used if the datatype of the visibility
            data on-disk is not 'c8' or 'c16'.
        multidim_index : bool
            If True, attempt to index the HDF5 dataset
            simultaneously along all data axes. Otherwise index one axis at-a-time.
            This only works if data selection is sliceable along all but one axis.
            If indices are not well-matched to data chunks, this can be slow.
        background_lsts : bool
            When set to True, the lst_array is calculated in a background thread.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after after reading in the file (the default is True,
            meaning the check will be run). Ignored if read_data is False.
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
            Ignored if read_data is False.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            reading in the file (the default is True, meaning the acceptable
            range check will be done). Ignored if read_data is False.
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.

        Returns
        -------
            None

        Raises
        ------
        IOError
            If filename doesn't exist.
        ValueError
            If the data_array_dtype is not a complex dtype.
            If incompatible select keywords are set (e.g. `ant_str` with other
            antenna selectors, `times` and `time_range`) or select keywords
            exclude all data or if keywords are set to the wrong type.

        """
        if not os.path.exists(filename):
            raise IOError(filename + " not found")

        # open hdf5 file for reading
        with h5py.File(filename, "r") as f:
            # extract header information
            header = f["/Header"]
            self._read_header(
                header,
                filename,
                run_check_acceptability=run_check_acceptability,
                background_lsts=background_lsts,
            )

            if not read_data:
                # don't read in the data. This means the object is incomplete,
                # but that may not matter for many purposes.
                return

            # Now read in the data
            dgrp = f["/Data"]
            self._get_data(
                dgrp,
                antenna_nums,
                antenna_names,
                ant_str,
                bls,
                frequencies,
                freq_chans,
                times,
                time_range,
                polarizations,
                blt_inds,
                data_array_dtype,
                keep_all_metadata,
                multidim_index,
                run_check,
                check_extra,
                run_check_acceptability,
                strict_uvw_antpos_check,
            )

        return

    def _write_header(self, header):
        """
        Write data to the header datagroup of a UVH5 file.

        Parameters
        ----------
        header : h5py datagroup
            The datagroup to write the header information to. For a UVH5 file
            conforming to the spec, it should be "/Header"

        Returns
        -------
        None
        """
        # write out telescope and source information
        header["latitude"] = self.telescope_location_lat_lon_alt_degrees[0]
        header["longitude"] = self.telescope_location_lat_lon_alt_degrees[1]
        header["altitude"] = self.telescope_location_lat_lon_alt_degrees[2]
        header["telescope_name"] = np.string_(self.telescope_name)
        header["instrument"] = np.string_(self.instrument)
        header["object_name"] = np.string_(self.object_name)

        # write out required UVParameters
        header["Nants_data"] = self.Nants_data
        header["Nants_telescope"] = self.Nants_telescope
        header["Nbls"] = self.Nbls
        header["Nblts"] = self.Nblts
        header["Nfreqs"] = self.Nfreqs
        header["Npols"] = self.Npols
        header["Nspws"] = self.Nspws
        header["Ntimes"] = self.Ntimes
        header["antenna_numbers"] = self.antenna_numbers
        header["uvw_array"] = self.uvw_array
        header["vis_units"] = np.string_(self.vis_units)
        header["channel_width"] = self.channel_width
        header["time_array"] = self.time_array
        header["freq_array"] = self.freq_array
        header["integration_time"] = self.integration_time
        header["lst_array"] = self.lst_array
        header["polarization_array"] = self.polarization_array
        header["spw_array"] = self.spw_array
        header["ant_1_array"] = self.ant_1_array
        header["ant_2_array"] = self.ant_2_array
        header["antenna_positions"] = self.antenna_positions
        header["flex_spw"] = self.flex_spw
        header["multi_object"] = self.multi_object
        # handle antenna_names; works for lists or arrays
        header["antenna_names"] = np.asarray(self.antenna_names, dtype="bytes")

        # write out phasing information
        header["phase_type"] = np.string_(self.phase_type)
        if self.phase_center_ra is not None:
            header["phase_center_ra"] = self.phase_center_ra
        if self.phase_center_app_ra is not None:
            header["phase_center_app_ra"] = self.phase_center_app_ra
        if self.phase_center_dec is not None:
            header["phase_center_dec"] = self.phase_center_dec
        if self.phase_center_app_dec is not None:
            header["phase_center_app_dec"] = self.phase_center_app_dec
        if self.phase_center_epoch is not None:
            header["phase_center_epoch"] = self.phase_center_epoch
        if self.phase_center_frame is not None:
            header["phase_center_frame"] = np.string_(self.phase_center_frame)

        # write out optional parameters
        if self.dut1 is not None:
            header["dut1"] = self.dut1
        if self.earth_omega is not None:
            header["earth_omega"] = self.earth_omega
        if self.gst0 is not None:
            header["gst0"] = self.gst0
        if self.rdate is not None:
            header["rdate"] = np.string_(self.rdate)
        if self.timesys is not None:
            header["timesys"] = np.string_(self.timesys)
        if self.x_orientation is not None:
            header["x_orientation"] = np.string_(self.x_orientation)
        if self.blt_order is not None:
            header["blt_order"] = np.string_(", ".join(self.blt_order))
        if self.antenna_diameters is not None:
            header["antenna_diameters"] = self.antenna_diameters
        if self.uvplane_reference_time is not None:
            header["uvplane_reference_time"] = self.uvplane_reference_time
        if self.eq_coeffs is not None:
            header["eq_coeffs"] = self.eq_coeffs
        if self.eq_coeffs_convention is not None:
            header["eq_coeffs_convention"] = np.string_(self.eq_coeffs_convention)
        if self.flex_spw_id_array is not None:
            header["flex_spw_id_array"] = self.flex_spw_id_array
        if self.object_id_array is not None:
            header["object_id_array"] = self.object_id_array
        if self.Nobjects is not None:
            header["Nobjects"] = self.Nobjects

        # Write out object dictionary, if available
        if self.object_dict:
            object_dict = header.create_group("object_dict")
            for k in self.object_dict.keys():
                # Dictionary entries can be written out as strings, in what is
                # effectively JSON format. The str promotion up front is needed
                # because otherwise np.string_ trucncates the dictionary
                object_dict[k] = np.string_(str(self.object_dict[k]))

        # write out extra keywords if it exists and has elements
        if self.extra_keywords:
            extra_keywords = header.create_group("extra_keywords")
            for k in self.extra_keywords.keys():
                if isinstance(self.extra_keywords[k], str):
                    extra_keywords[k] = np.string_(self.extra_keywords[k])
                else:
                    extra_keywords[k] = self.extra_keywords[k]

        # write out history
        header["history"] = np.string_(self.history)

        return

    def write_uvh5(
        self,
        filename,
        clobber=False,
        chunks=True,
        data_compression=None,
        flags_compression="lzf",
        nsample_compression="lzf",
        data_write_dtype=None,
        add_to_history=None,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
    ):
        """
        Write an in-memory UVData object to a UVH5 file.

        Parameters
        ----------
        filename : str
            The UVH5 file to write to.
        clobber : bool
            Option to overwrite the file if it already exists.
        chunks : tuple or bool
            h5py.create_dataset chunks keyword. Tuple for chunk shape,
            True for auto-chunking, None for no chunking. Default is True.
        data_compression : str
            HDF5 filter to apply when writing the data_array. Default is None
            (no filter/compression). Dataset must be chunked.
        flags_compression : str
            HDF5 filter to apply when writing the flags_array. Default is the
            LZF filter. Dataset must be chunked.
        nsample_compression : str
            HDF5 filter to apply when writing the nsample_array. Default is the
            LZF filter. Dataset must be chunked.
        data_write_dtype : numpy dtype
            The datatype of output visibility data. If 'None', then the same
            datatype as data_array will be used. The user may specify 'c8' for
            single-precision floats or 'c16' for double-presicion. Otherwise, a
            numpy dtype object must be specified with an 'r' field and an 'i'
            field for real and imaginary parts, respectively. See uvh5.py for
            an example of defining such a datatype.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            before writing the file.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters before
            writing the file.
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.

        Returns
        -------
        None

        Raises
        ------
        IOError
            If the file located at `filename` already exists and clobber=False,
            an IOError is raised.

        Notes
        -----
        The HDF5 library allows for the application of "filters" when writing
        data, which can provide moderate to significant levels of compression
        for the datasets in question.  Testing has shown that for some typical
        cases of UVData objects (empty/sparse flag_array objects, and/or uniform
        nsample_arrays), the built-in LZF filter provides significant
        compression for minimal computational overhead.

        Note that for typical HERA data files written after mid-2020, the
        bitshuffle filter was applied to the data_array. Because of the lack of
        portability, it is not included as an option here; in the future, it may
        be added. Note that as long as bitshuffle is installed on the system in
        a way that h5py can find it, no action needs to be taken to _read_ a
        data_array encoded with bitshuffle (or an error will be raised).
        """
        if run_check:
            self.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
            )

        if os.path.exists(filename):
            if clobber:
                print("File exists; clobbering")
            else:
                raise IOError("File exists; skipping")

        # open file for writing
        with h5py.File(filename, "w") as f:
            # write header
            header = f.create_group("Header")
            self._write_header(header)

            # write out data, flags, and nsample arrays
            dgrp = f.create_group("Data")
            if data_write_dtype is None:
                if self.data_array.dtype == "complex64":
                    data_write_dtype = "c8"
                else:
                    data_write_dtype = "c16"
            if data_write_dtype not in ("c8", "c16"):
                _check_uvh5_dtype(data_write_dtype)
                visdata = dgrp.create_dataset(
                    "visdata",
                    self.data_array.shape,
                    chunks=chunks,
                    compression=data_compression,
                    dtype=data_write_dtype,
                )
                indices = (np.s_[:], np.s_[:], np.s_[:], np.s_[:])
                _write_complex_astype(self.data_array, visdata, indices)
            else:
                visdata = dgrp.create_dataset(
                    "visdata",
                    chunks=chunks,
                    data=self.data_array,
                    compression=data_compression,
                    dtype=data_write_dtype,
                )
            dgrp.create_dataset(
                "flags",
                chunks=chunks,
                data=self.flag_array,
                compression=flags_compression,
            )
            dgrp.create_dataset(
                "nsamples",
                chunks=chunks,
                data=self.nsample_array.astype(np.float32),
                compression=nsample_compression,
            )

        return

    def initialize_uvh5_file(
        self,
        filename,
        clobber=False,
        chunks=True,
        data_compression=None,
        flags_compression="lzf",
        nsample_compression="lzf",
        data_write_dtype=None,
    ):
        """
        Initialize a UVH5 file on disk to be written to in parts.

        Parameters
        ----------
        filename : str
            The UVH5 file to write to.
        clobber : bool
            Option to overwrite the file if it already exists.
        chunks : tuple or bool
            h5py.create_dataset chunks keyword. Tuple for chunk shape,
            True for auto-chunking, None for no chunking. Default is True.
        data_compression : str
            HDF5 filter to apply when writing the data_array. Default is None
            (no filter/compression). Dataset must be chunked.
        flags_compression : str
            HDF5 filter to apply when writing the flags_array. Default is the
            LZF filter. Dataset must be chunked.
        nsample_compression : str
            HDF5 filter to apply when writing the nsample_array. Default is the
            LZF filter. Dataset must be chunked.
        data_write_dtype : str or numpy dtype
            The datatype of output visibility data. If 'None', then double-
            precision floats will be used. The user may specify 'c8' for
            single-precision floats or 'c16' for double-presicion. Otherwise, a
            numpy dtype object must be specified with an 'r' field and an 'i'
            field for real and imaginary parts, respectively. See uvh5.py for
            an example of defining such a datatype.

        Returns
        -------
        None

        Raises
        ------
        IOError
            If the file located at `filename` already exists and clobber=False,
            an IOError is raised.

        Notes
        -----
        When partially writing out data, this function should be called first to
        initialize the file on disk. The data is then actually written by
        calling the write_uvh5_part method, with the same filename as the one
        specified in this function. See the tutorial for a worked example.

        The HDF5 library allows for the application of "filters" when writing
        data, which can provide moderate to significant levels of compression
        for the datasets in question.  Testing has shown that for some typical
        cases of UVData objects (empty/sparse flag_array objects, and/or uniform
        nsample_arrays), the built-in LZF filter provides significant
        compression for minimal computational overhead.

        Note that for typical HERA data files written after mid-2018, the
        bitshuffle filter was applied to the data_array. Because of the lack of
        portability, it is not included as an option here; in the future, it may
        be added. Note that as long as bitshuffle is installed on the system in
        a way that h5py can find it, no action needs to be taken to _read_ a
        data_array encoded with bitshuffle (or an error will be raised).
        """
        if os.path.exists(filename):
            if clobber:
                print("File exists; clobbering")
            else:
                raise IOError("File exists; skipping")

        # write header and empty arrays to file
        with h5py.File(filename, "w") as f:
            # write header
            header = f.create_group("Header")
            self._write_header(header)

            # initialize the data groups on disk
            # TODO: Spw axis to be collapsed in future release
            data_size = (self.Nblts, 1, self.Nfreqs, self.Npols)
            dgrp = f.create_group("Data")
            if data_write_dtype is None:
                # we don't know what kind of data we'll get--default to double-precision
                data_write_dtype = "c16"
            if data_write_dtype not in ("c8", "c16"):
                # make sure the data type is correct
                _check_uvh5_dtype(data_write_dtype)
            dgrp.create_dataset(
                "visdata",
                data_size,
                chunks=chunks,
                dtype=data_write_dtype,
                compression=data_compression,
            )
            dgrp.create_dataset(
                "flags",
                data_size,
                chunks=chunks,
                dtype="b1",
                compression=flags_compression,
            )
            dgrp.create_dataset(
                "nsamples",
                data_size,
                chunks=chunks,
                dtype="f4",
                compression=nsample_compression,
            )

        return

    def _check_header(
        self, filename, run_check_acceptability=True, background_lsts=True
    ):
        """
        Check that the metadata in a file header matches the object's metadata.

        Parameters
        ----------
        header : h5py datagroup
            A reference to an h5py data group that contains the header
            information. For UVH5 files conforming to the spec, this should be
            "/Header".
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            reading in the file.
        background_lsts : bool
            When set to True, the lst_array is calculated in a background thread.

        Returns
        -------
        None

        Notes
        -----
        This function creates a new UVData object an reads in the header
        information saved on disk to compare with the object in memory. Note
        that this adds some small memory overhead, but this amount is typically
        much smaller than the size of the data.
        """
        uvd_file = UVH5()
        with h5py.File(filename, "r") as f:
            header = f["/Header"]
            uvd_file._read_header(
                header,
                filename,
                run_check_acceptability=run_check_acceptability,
                background_lsts=background_lsts,
            )

        # temporarily remove data, flag, and nsample arrays, so we only check metadata
        if self.data_array is not None:
            data_array = self.data_array
            self.data_array = None
            replace_data = True
        else:
            replace_data = False
        if self.flag_array is not None:
            flag_array = self.flag_array
            self.flag_array = None
            replace_flags = True
        else:
            replace_flags = False
        if self.nsample_array is not None:
            nsample_array = self.nsample_array
            self.nsample_array = None
            replace_nsamples = True
        else:
            replace_nsamples = False

        if self != uvd_file:
            raise AssertionError(
                "The object metadata in memory and metadata on disk are different"
            )
        else:
            # clean up after ourselves
            if replace_data:
                self.data_array = data_array
            if replace_flags:
                self.flag_array = flag_array
            if replace_nsamples:
                self.nsample_array = nsample_array
            del uvd_file
        return

    def write_uvh5_part(
        self,
        filename,
        data_array,
        flag_array,
        nsample_array,
        check_header=True,
        antenna_nums=None,
        antenna_names=None,
        ant_str=None,
        bls=None,
        frequencies=None,
        freq_chans=None,
        times=None,
        time_range=None,
        polarizations=None,
        blt_inds=None,
        run_check_acceptability=True,
        add_to_history=None,
    ):
        """
        Write out a part of a UVH5 file that has been previously initialized.

        Parameters
        ----------
        filename : str
            The file on disk to write data to. It must already exist,
            and is assumed to have been initialized with initialize_uvh5_file.
        data_array : array of float
            The data to write to disk. A check is done to ensure that
            the dimensions of the data passed in conform to the ones specified by
            the "selection" arguments.
        flag_array : array of bool
            The flags array to write to disk. A check is done to ensure
            that the dimensions of the data passed in conform to the ones specified
            by the "selection" arguments.
        nsample_array : array of float
            The nsample array to write to disk. A check is done to ensure
            that the dimensions fo the data passed in conform to the ones specified
            by the "selection" arguments.
        check_header : bool
            Option to check that the metadata present in the header
            on disk matches that in the object.
        run_check_acceptability : bool
            If check_header, additional option to check
            acceptable range of the values of parameters after reading in the file.
        antenna_nums : array_like of int, optional
            The antennas numbers to include when writing data into
            the object (antenna positions and names for the excluded antennas
            will be retained). This cannot be provided if antenna_names is
            also provided.
        antenna_names : array_like of str, optional
            The antennas names to include when writing data into
            the object (antenna positions and names for the excluded antennas
            will be retained). This cannot be provided if antenna_nums is
            also provided.
        bls : list of tuples, optional
            A list of antenna number tuples (e.g. [(0, 1), (3, 2)]) or a list of
            baseline 3-tuples (e.g. [(0, 1, 'xx'), (2, 3, 'yy')]) specifying baselines
            to write to the file. For length-2 tuples, the ordering of the numbers
            within the tuple does not matter. For length-3 tuples, the polarization
            string is in the order of the two antennas. If length-3 tuples are provided,
            the polarizations argument below must be None.
        ant_str : str, optional
            A string containing information about what antenna numbers
            and polarizations to include when writing data into the object.
            Can be 'auto', 'cross', 'all', or combinations of antenna numbers
            and polarizations (e.g. '1', '1_2', '1x_2y').
            See tutorial for more examples of valid strings and
            the behavior of different forms for ant_str.
            If '1x_2y,2y_3y' is passed, both polarizations 'xy' and 'yy' will
            be written for both baselines (1, 2) and (2, 3) to reflect a valid
            pyuvdata object.
            An ant_str cannot be passed in addition to any of the above antenna
            args or the polarizations arg.
        frequencies : array_like of float, optional
            The frequencies to include when writing data to the file.
        freq_chans : array_like of int, optional
            The frequency channel numbers to include when writing data to the file.
        times : array_like of float, optional
            The times in Julian Day to include when writing data to the file.
        time_range : array_like of float, optional
            The time range in Julian Date to include when writing data to the
            file, must be length 2. Some of the times in the object should fall
            between the first and last elements. Cannot be used with `times`.
        polarizations : array_like of int, optional
            The polarizations to include when writing data to the file.
        blt_inds : array_like of int, optional
            The baseline-time indices to include when writing data to the file.
            This is not commonly used.
        add_to_history : str
            String to append to history before write out. Default is no appending.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            An AsserionError is raised if: (1) the location specified by
            `filename` does not exist; (2) the data_array, flag_array, and
            nsample_array do not all have the same shape; (3) the shape of the
            data arrays do not correspond to the sizes specified by the
            properties to write out.

        Notes
        -----
        When partially writing out data, this function should be called after
        calling initialize_uvh5_file. The same filename is passed in, with an
        optional check to ensure that the object's metadata in-memory matches
        the header on-disk. See the tutorial for a worked example.
        """
        # check that the file already exists
        if not os.path.exists(filename):
            raise AssertionError(
                "{0} does not exists; please first initialize it with "
                "initialize_uvh5_file".format(filename)
            )

        if check_header:
            self._check_header(
                filename, run_check_acceptability=run_check_acceptability
            )

        # figure out which "full file" indices to write data to
        blt_inds, freq_inds, pol_inds, _ = self._select_preprocess(
            antenna_nums,
            antenna_names,
            ant_str,
            bls,
            frequencies,
            freq_chans,
            times,
            time_range,
            polarizations,
            blt_inds,
        )

        # make sure that the dimensions of the data to write are correct
        if data_array.shape != flag_array.shape:
            raise AssertionError("data_array and flag_array must have the same shape")
        if data_array.shape != nsample_array.shape:
            raise AssertionError(
                "data_array and nsample_array must have the same shape"
            )

        # check what part of each dimension to grab
        # we can use numpy slice objects to index the h5py indices
        if blt_inds is not None:
            Nblts = len(blt_inds)

            # test if blts are regularly spaced
            if len(set(np.ediff1d(blt_inds))) <= 1:
                blt_reg_spaced = True
                blt_start = blt_inds[0]
                blt_end = blt_inds[-1] + 1
                if len(blt_inds) == 1:
                    d_blt = 1
                else:
                    d_blt = blt_inds[1] - blt_inds[0]
                blt_inds = np.s_[blt_start:blt_end:d_blt]
            else:
                blt_reg_spaced = False
        else:
            Nblts = self.Nblts
            blt_reg_spaced = True
            blt_inds = np.s_[:]
        if freq_inds is not None:
            Nfreqs = len(freq_inds)

            # test if frequencies are regularly spaced
            if len(set(np.ediff1d(freq_inds))) <= 1:
                freq_reg_spaced = True
                freq_start = freq_inds[0]
                freq_end = freq_inds[-1] + 1
                if len(freq_inds) == 1:
                    d_freq = 1
                else:
                    d_freq = freq_inds[1] - freq_inds[0]
                freq_inds = np.s_[freq_start:freq_end:d_freq]
            else:
                freq_reg_spaced = False
        else:
            Nfreqs = self.Nfreqs
            freq_reg_spaced = True
            freq_inds = np.s_[:]
        if pol_inds is not None:
            Npols = len(pol_inds)

            # test if pols are regularly spaced
            if len(set(np.ediff1d(pol_inds))) <= 1:
                pol_reg_spaced = True
                pol_start = pol_inds[0]
                pol_end = pol_inds[-1] + 1
                if len(pol_inds) == 1:
                    d_pol = 1
                else:
                    d_pol = pol_inds[1] - pol_inds[0]
                pol_inds = np.s_[pol_start:pol_end:d_pol]
            else:
                pol_reg_spaced = False
        else:
            Npols = self.Npols
            pol_reg_spaced = True
            pol_inds = np.s_[:]

        # check for proper size of input arrays
        proper_shape = (Nblts, 1, Nfreqs, Npols)
        if data_array.shape != proper_shape:
            raise AssertionError(
                "data_array has shape {0}; was expecting {1}".format(
                    data_array.shape, proper_shape
                )
            )

        # actually write the data
        with h5py.File(filename, "r+") as f:
            dgrp = f["/Data"]
            visdata_dset = dgrp["visdata"]
            flags_dset = dgrp["flags"]
            nsamples_dset = dgrp["nsamples"]
            visdata_dtype = visdata_dset.dtype
            if visdata_dtype not in ("complex64", "complex128"):
                custom_dtype = True
            else:
                custom_dtype = False

            # check if we can do fancy indexing
            # as long as at least 2 out of 3 axes can be written as slices,
            # we can be fancy
            n_reg_spaced = np.count_nonzero(
                [blt_reg_spaced, freq_reg_spaced, pol_reg_spaced]
            )
            if n_reg_spaced >= 2:
                if custom_dtype:
                    indices = (blt_inds, np.s_[:], freq_inds, pol_inds)
                    _write_complex_astype(data_array, visdata_dset, indices)
                else:
                    visdata_dset[blt_inds, :, freq_inds, pol_inds] = data_array
                flags_dset[blt_inds, :, freq_inds, pol_inds] = flag_array
                nsamples_dset[blt_inds, :, freq_inds, pol_inds] = nsample_array
            elif n_reg_spaced == 1:
                # figure out which axis is regularly spaced
                if blt_reg_spaced:
                    for ifreq, freq_idx in enumerate(freq_inds):
                        for ipol, pol_idx in enumerate(pol_inds):
                            if custom_dtype:
                                indices = (blt_inds, np.s_[:], freq_idx, pol_idx)
                                _write_complex_astype(
                                    data_array[:, :, ifreq, ipol], visdata_dset, indices
                                )
                            else:
                                visdata_dset[
                                    blt_inds, :, freq_idx, pol_idx
                                ] = data_array[:, :, ifreq, ipol]
                            flags_dset[blt_inds, :, freq_idx, pol_idx] = flag_array[
                                :, :, ifreq, ipol
                            ]
                            nsamples_dset[
                                blt_inds, :, freq_idx, pol_idx
                            ] = nsample_array[:, :, ifreq, ipol]
                elif freq_reg_spaced:
                    for iblt, blt_idx in enumerate(blt_inds):
                        for ipol, pol_idx in enumerate(pol_inds):
                            if custom_dtype:
                                indices = (blt_idx, np.s_[:], freq_inds, pol_idx)
                                _write_complex_astype(
                                    data_array[iblt, :, :, ipol], visdata_dset, indices
                                )
                            else:
                                visdata_dset[
                                    blt_idx, :, freq_inds, pol_idx
                                ] = data_array[iblt, :, :, ipol]
                            flags_dset[blt_idx, :, freq_inds, pol_idx] = flag_array[
                                iblt, :, :, ipol
                            ]
                            nsamples_dset[
                                blt_idx, :, freq_inds, pol_idx
                            ] = nsample_array[iblt, :, :, ipol]
                else:  # pol_reg_spaced
                    for iblt, blt_idx in enumerate(blt_inds):
                        for ifreq, freq_idx in enumerate(freq_inds):
                            if custom_dtype:
                                indices = (blt_idx, np.s_[:], freq_idx, pol_inds)
                                _write_complex_astype(
                                    data_array[iblt, :, ifreq, :], visdata_dset, indices
                                )
                            else:
                                visdata_dset[
                                    blt_idx, :, freq_idx, pol_inds
                                ] = data_array[iblt, :, ifreq, :]
                            flags_dset[blt_idx, :, freq_idx, pol_inds] = flag_array[
                                iblt, :, ifreq, :
                            ]
                            nsamples_dset[
                                blt_idx, :, freq_idx, pol_inds
                            ] = nsample_array[iblt, :, ifreq, :]
            else:
                # all axes irregularly spaced
                # perform a triple loop -- probably very slow!
                for iblt, blt_idx in enumerate(blt_inds):
                    for ifreq, freq_idx in enumerate(freq_inds):
                        for ipol, pol_idx in enumerate(pol_inds):
                            if custom_dtype:
                                indices = (blt_idx, np.s_[:], freq_idx, pol_idx)
                                _write_complex_astype(
                                    data_array[iblt, :, ifreq, ipol],
                                    visdata_dset,
                                    indices,
                                )
                            else:
                                visdata_dset[
                                    blt_idx, :, freq_idx, pol_idx
                                ] = data_array[iblt, :, ifreq, ipol]
                            flags_dset[blt_idx, :, freq_idx, pol_idx] = flag_array[
                                iblt, :, ifreq, ipol
                            ]
                            nsamples_dset[
                                blt_idx, :, freq_idx, pol_idx
                            ] = nsample_array[iblt, :, ifreq, ipol]

            # append to history if desired
            if add_to_history is not None:
                history = np.string_(self.history) + np.string_(add_to_history)
                if "history" in f["Header"]:
                    # erase dataset first b/c it has fixed-length string datatype
                    del f["Header"]["history"]
                f["Header"]["history"] = np.string_(history)

        return
