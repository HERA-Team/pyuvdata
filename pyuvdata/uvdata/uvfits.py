# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Class for reading and writing uvfits files."""
import warnings

import numpy as np
from astropy import constants as const
from astropy.time import Time
from astropy.io import fits

from .uvdata import UVData
from .. import utils as uvutils

__all__ = ["UVFITS"]


class UVFITS(UVData):
    """
    Defines a uvfits-specific subclass of UVData for reading and writing uvfits.

    This class should not be interacted with directly, instead use the read_uvfits
    and write_uvfits methods on the UVData class.

    Attributes
    ----------
    uvfits_required_extra : list of str
        Names of optional UVParameters that are required for uvfits.

    """

    uvfits_required_extra = [
        "antenna_positions",
        "gst0",
        "rdate",
        "earth_omega",
        "dut1",
        "timesys",
    ]

    def _get_parameter_data(
        self, vis_hdu, run_check_acceptability, background_lsts=True, use_astropy=False,
    ):
        """
        Read just the random parameters portion of the uvfits file ("metadata").

        Separated from full read so that header, metadata and data can be read
        independently.
        """
        # astropy.io fits reader scales date according to relevant PZER0 (?)
        # uvfits standard is to have 2 DATE parameters, both floats:
        # DATE (full day) and _DATE (fractional day)
        # cotter uvfits files have one DATE that is a double
        # using data.par('date') is general -- it will add them together if there are 2
        self.time_array = vis_hdu.data.par("date")

        self.Ntimes = len(np.unique(self.time_array))

        # check if lst array is saved. It's not a standard metadata item in uvfits,
        # but if the file was written with pyuvdata it may be present
        # (depending on pyuvdata version)
        proc = None
        if "LST" in vis_hdu.data.parnames:
            # angles in uvfits files are stored in degrees, so convert to radians
            self.lst_array = np.deg2rad(vis_hdu.data.par("lst"))
            if run_check_acceptability:
                (
                    latitude,
                    longitude,
                    altitude,
                ) = self.telescope_location_lat_lon_alt_degrees
                lst_array = uvutils.get_lst_for_time(
                    self.time_array, latitude, longitude, altitude
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
                        "LST values stored in this file are not "
                        "self-consistent with time_array and telescope "
                        "location. Consider recomputing with "
                        "utils.get_lst_for_time."
                    )

        else:
            proc = self.set_lsts_from_time_array(
                background=background_lsts, use_astropy=use_astropy
            )

        # if antenna arrays are present, use them. otherwise use baseline array
        if "ANTENNA1" in vis_hdu.data.parnames and "ANTENNA2" in vis_hdu.data.parnames:
            # Note: uvfits antennas are 1 indexed,
            # need to subtract one to get to 0-indexed
            self.ant_1_array = np.int32(vis_hdu.data.par("ANTENNA1")) - 1
            self.ant_2_array = np.int32(vis_hdu.data.par("ANTENNA2")) - 1
            subarray = np.int32(vis_hdu.data.par("SUBARRAY")) - 1
            # error on files with multiple subarrays
            if len(set(subarray)) > 1:
                raise ValueError(
                    "This file appears to have multiple subarray "
                    "values; only files with one subarray are "
                    "supported."
                )
        else:
            # cannot set this to be the baseline array because it uses the
            # 256 convention, not our 2048 convention
            bl_input_array = np.int64(vis_hdu.data.par("BASELINE"))

            # get antenna arrays based on uvfits baseline array
            self.ant_1_array, self.ant_2_array = self.baseline_to_antnums(
                bl_input_array
            )

        # check for multi source files
        if "SOURCE" in vis_hdu.data.parnames:
            # Preserve the source info just in case the AIPS SU table is missing, and
            # we need to revert things back.
            self._set_multi_object(preserve_source_info=True)
            source = vis_hdu.data.par("SOURCE")
            if len(set(source)) > 1:
                raise ValueError(
                    "This file has multiple sources. Only single "
                    "source observations are supported."
                )

        # get self.baseline_array using our convention
        self.baseline_array = self.antnums_to_baseline(
            self.ant_1_array, self.ant_2_array
        )
        self.Nbls = len(np.unique(self.baseline_array))

        # initialize internal variables based on the antenna lists
        self.Nants_data = int(np.union1d(self.ant_1_array, self.ant_2_array).size)

        # read baseline vectors in units of seconds, return in meters
        # FITS uvw direction convention is opposite ours and Miriad's.
        # So conjugate the visibilities and flip the uvws:
        self.uvw_array = (-1) * (
            np.array(
                np.stack(
                    (
                        vis_hdu.data.par("UU"),
                        vis_hdu.data.par("VV"),
                        vis_hdu.data.par("WW"),
                    )
                )
            )
            * const.c.to("m/s").value
        ).T

        if "INTTIM" in vis_hdu.data.parnames:
            self.integration_time = np.asarray(
                vis_hdu.data.par("INTTIM"), dtype=np.float64
            )
        else:
            if self.Ntimes > 1:
                # assume that all integration times in the file are the same
                int_time = self._calc_single_integration_time()
                self.integration_time = (
                    np.ones_like(self.time_array, dtype=np.float64) * int_time
                )
            else:
                raise ValueError(
                    "integration time not specified and only " "one time present"
                )

        if proc is not None:
            proc.join()

    def _get_data(
        self,
        vis_hdu,
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
        read_metadata,
        keep_all_metadata,
        run_check,
        check_extra,
        run_check_acceptability,
        strict_uvw_antpos_check,
    ):
        """
        Read just the visibility and flag data of the uvfits file.

        Separated from full read so header and metadata can be read without data.
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

        if blt_inds is not None:
            blt_frac = len(blt_inds) / float(self.Nblts)
        else:
            blt_frac = 1

        if freq_inds is not None:
            freq_frac = len(freq_inds) * float(self.Nspws) / float(self.Nfreqs)
        else:
            freq_frac = 1

        if pol_inds is not None:
            pol_frac = len(pol_inds) / float(self.Npols)
        else:
            pol_frac = 1

        min_frac = np.min([blt_frac, freq_frac, pol_frac])

        if min_frac == 1:
            # no select, read in all the data
            if vis_hdu.header["NAXIS"] == 7:
                raw_data_array = vis_hdu.data.data[:, 0, 0, :, :, :, :]
                assert self.Nspws == raw_data_array.shape[1]

            else:
                # in many uvfits files the spw axis is left out,
                # here we put it back in so the dimensionality stays the same
                raw_data_array = vis_hdu.data.data[:, 0, 0, :, :, :]
                raw_data_array = raw_data_array[:, np.newaxis, :, :]
        else:
            # do select operations on everything except data_array, flag_array
            # and nsample_array
            self._select_metadata(
                blt_inds, freq_inds, pol_inds, history_update_string, keep_all_metadata
            )

            # just read in the right portions of the data and flag arrays
            if blt_frac == min_frac:
                if vis_hdu.header["NAXIS"] == 7:
                    raw_data_array = vis_hdu.data.data[blt_inds, :, :, :, :, :, :]
                    raw_data_array = raw_data_array[:, 0, 0, :, :, :, :]
                    assert self.Nspws == raw_data_array.shape[1]
                else:
                    # in many uvfits files the spw axis is left out,
                    # here we put it back in so the dimensionality stays the same
                    raw_data_array = vis_hdu.data.data[blt_inds, :, :, :, :, :]
                    raw_data_array = raw_data_array[:, 0, 0, :, :, :]
                    raw_data_array = raw_data_array[:, np.newaxis, :, :, :]
                if freq_frac < 1:
                    raw_data_array = raw_data_array[:, :, freq_inds, :, :]
                if pol_frac < 1:
                    raw_data_array = raw_data_array[:, :, :, pol_inds, :]
            elif freq_frac == min_frac:
                if vis_hdu.header["NAXIS"] == 7:
                    raw_data_array = vis_hdu.data.data[:, :, :, :, freq_inds, :, :]
                    raw_data_array = raw_data_array[:, 0, 0, :, :, :, :]
                    assert self.Nspws == raw_data_array.shape[1]
                else:
                    # in many uvfits files the spw axis is left out,
                    # here we put it back in so the dimensionality stays the same
                    raw_data_array = vis_hdu.data.data[:, :, :, freq_inds, :, :]
                    raw_data_array = raw_data_array[:, 0, 0, :, :, :]
                    raw_data_array = raw_data_array[:, np.newaxis, :, :, :]

                if blt_frac < 1:
                    raw_data_array = raw_data_array[blt_inds, :, :, :, :]
                if pol_frac < 1:
                    raw_data_array = raw_data_array[:, :, :, pol_inds, :]
            else:
                if vis_hdu.header["NAXIS"] == 7:
                    raw_data_array = vis_hdu.data.data[:, :, :, :, :, pol_inds, :]
                    raw_data_array = raw_data_array[:, 0, 0, :, :, :, :]
                    assert self.Nspws == raw_data_array.shape[1]
                else:
                    # in many uvfits files the spw axis is left out,
                    # here we put it back in so the dimensionality stays the same
                    raw_data_array = vis_hdu.data.data[:, :, :, :, pol_inds, :]
                    raw_data_array = raw_data_array[:, 0, 0, :, :, :]
                    raw_data_array = raw_data_array[:, np.newaxis, :, :, :]

                if blt_frac < 1:
                    raw_data_array = raw_data_array[blt_inds, :, :, :, :]
                if freq_frac < 1:
                    raw_data_array = raw_data_array[:, :, freq_inds, :, :]

        assert len(raw_data_array.shape) == 5

        # Reshape the data array to be the right size if we are working w/ multiple
        # spectral windows to be 'flex_spw' compliant
        if self.Nspws > 1:
            raw_data_array = np.reshape(
                raw_data_array,
                (self.Nblts, 1, self.Nfreqs, self.Npols, raw_data_array.shape[4]),
            )

        # FITS uvw direction convention is opposite ours and Miriad's.
        # So conjugate the visibilities and flip the uvws:
        self.data_array = (
            raw_data_array[:, :, :, :, 0] - 1j * raw_data_array[:, :, :, :, 1]
        )
        self.flag_array = raw_data_array[:, :, :, :, 2] <= 0
        self.nsample_array = np.abs(raw_data_array[:, :, :, :, 2])

        # check if object has all required UVParameters set
        if run_check:
            self.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
            )

    def read_uvfits(
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
        background_lsts=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
        use_astropy=False,
    ):
        """
        Read in header, metadata and data from a uvfits file.

        Supports reading only selected portions of the data.

        Parameters
        ----------
        filename : str
            The uvfits file to read from.
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
            passed here should exist in the time_array.
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


        Raises
        ------
        IOError
            If filename doesn't exist.
        ValueError
            If incompatible select keywords are set (e.g. `ant_str` with other
            antenna selectors, `times` and `time_range`) or select keywords
            exclude all data or if keywords are set to the wrong type.
            If the data are multi source.
            If the data have multi spw with different channel widths.
            If the metadata are not internally consistent or missing.

        """
        with fits.open(filename, memmap=True) as hdu_list:
            vis_hdu = hdu_list[0]  # assumes the visibilities are in the primary hdu
            vis_hdr = vis_hdu.header.copy()
            hdunames = uvutils._fits_indexhdus(hdu_list)  # find the rest of the tables

            # First get everything we can out of the header.
            self._set_phased()
            # check if we have an spw dimension
            if vis_hdr["NAXIS"] == 7:
                self.Nspws = vis_hdr.pop("NAXIS5")
                self.spw_array = uvutils._fits_gethduaxis(vis_hdu, 5).astype(np.int) - 1

                # the axis number for phase center depends on if the spw exists
                self.phase_center_ra_degrees = np.float(vis_hdr.pop("CRVAL6"))
                self.phase_center_dec_degrees = np.float(vis_hdr.pop("CRVAL7"))
            else:
                self.Nspws = 1
                self.spw_array = np.array([np.int(0)])

                # the axis number for phase center depends on if the spw exists
                self.phase_center_ra_degrees = np.float(vis_hdr.pop("CRVAL5"))
                self.phase_center_dec_degrees = np.float(vis_hdr.pop("CRVAL6"))

            # get shapes
            self.Npols = vis_hdr.pop("NAXIS3")
            self.Nblts = vis_hdr.pop("GCOUNT")

            if self.Nspws > 1:
                # If this is multi-spw, use the 'flexibile' spectral window setup
                self._set_flex_spw()
                uvfits_nchan = vis_hdr.pop("NAXIS4")
                self.Nfreqs = uvfits_nchan * self.Nspws
                self.flex_spw_id_array = np.transpose(
                    np.tile(np.arange(self.Nspws), (uvfits_nchan, 1))
                ).flatten()
                fq_hdu = hdu_list[hdunames["AIPS FQ"]]
                assert self.Nspws == fq_hdu.header["NO_IF"]

                # TODO: This is fine for now, although I (karto) think that this
                # is relative to the ref_freq, which can be specified as part of
                # the AIPS SU table.

                # Get rest freq value
                ref_freq = uvutils._fits_gethduaxis(vis_hdu, 4)[0]
                self.channel_width = np.transpose(
                    np.tile(abs(fq_hdu.data["CH WIDTH"]), (uvfits_nchan, 1))
                ).flatten()
                self.freq_array = np.reshape(
                    np.transpose(
                        (
                            ref_freq
                            + fq_hdu.data["IF FREQ"]
                            + np.outer(np.arange(uvfits_nchan), fq_hdu.data["CH WIDTH"])
                        )
                    ),
                    (1, -1),
                )
            else:
                self.Nfreqs = vis_hdr.pop("NAXIS4")
                self.freq_array = uvutils._fits_gethduaxis(vis_hdu, 4)
                # TODO: Spw axis to be collapsed in future release
                self.freq_array.shape = (1,) + self.freq_array.shape
                self.channel_width = vis_hdr.pop("CDELT4")

            self.polarization_array = np.int32(uvutils._fits_gethduaxis(vis_hdu, 3))
            # other info -- not required but frequently used
            self.object_name = vis_hdr.pop("OBJECT", None)
            self.telescope_name = vis_hdr.pop("TELESCOP", None)
            self.instrument = vis_hdr.pop("INSTRUME", None)
            latitude_degrees = vis_hdr.pop("LAT", None)
            longitude_degrees = vis_hdr.pop("LON", None)
            altitude = vis_hdr.pop("ALT", None)
            self.x_orientation = vis_hdr.pop("XORIENT", None)
            blt_order_str = vis_hdr.pop("BLTORDER", None)
            if blt_order_str is not None:
                self.blt_order = tuple(blt_order_str.split(", "))
                if self.blt_order == ("bda",):
                    self._blt_order.form = (1,)
            self.history = str(vis_hdr.get("HISTORY", ""))
            if not uvutils._check_history_version(
                self.history, self.pyuvdata_version_str
            ):
                self.history += self.pyuvdata_version_str

            self.vis_units = vis_hdr.pop("BUNIT", "UNCALIB")
            self.phase_center_epoch = vis_hdr.pop("EPOCH", None)
            self.phase_center_frame = vis_hdr.pop("PHSFRAME", None)

            self.extra_keywords = uvutils._get_fits_extra_keywords(
                vis_hdr, keywords_to_skip=["DATE-OBS"]
            )

            # Next read the antenna table
            ant_hdu = hdu_list[hdunames["AIPS AN"]]

            # stuff in the header
            if self.telescope_name is None:
                self.telescope_name = ant_hdu.header["ARRNAM"]

            self.gst0 = ant_hdu.header["GSTIA0"]
            self.rdate = ant_hdu.header["RDATE"]
            self.earth_omega = ant_hdu.header["DEGPDY"]
            self.dut1 = ant_hdu.header["UT1UTC"]
            if "TIMESYS" in ant_hdu.header.keys():
                self.timesys = ant_hdu.header["TIMESYS"]
            else:
                # CASA misspells this one
                self.timesys = ant_hdu.header["TIMSYS"]

            if "FRAME" in ant_hdu.header.keys():
                xyz_telescope_frame = ant_hdu.header["FRAME"]
            else:
                if "COMMENT" in self.extra_keywords:
                    if uvutils._check_history_version(
                        self.extra_keywords["COMMENT"],
                        "Created by Cotter MWA preprocessor",
                    ):
                        # this is a Cotter file, which should have the frame set to ITRF
                        warnings.warn(
                            "Required Antenna frame keyword not set, but this appears "
                            "to be a Cotter file, setting to ITRF."
                        )
                        xyz_telescope_frame = "ITRF"
                else:
                    warnings.warn(
                        "Required Antenna frame keyword not set, setting to ????"
                    )
                    xyz_telescope_frame = "????"

            # get telescope location and antenna positions.
            # VLA incorrectly sets ARRAYX/ARRAYY/ARRAYZ to 0, and puts array center
            # in the antenna positions themselves
            if (
                np.isclose(ant_hdu.header["ARRAYX"], 0)
                and np.isclose(ant_hdu.header["ARRAYY"], 0)
                and np.isclose(ant_hdu.header["ARRAYZ"], 0)
            ):
                x_telescope = np.mean(ant_hdu.data["STABXYZ"][:, 0])
                y_telescope = np.mean(ant_hdu.data["STABXYZ"][:, 1])
                z_telescope = np.mean(ant_hdu.data["STABXYZ"][:, 2])
                self.antenna_positions = ant_hdu.data.field("STABXYZ") - np.array(
                    [x_telescope, y_telescope, z_telescope]
                )

            else:
                x_telescope = ant_hdu.header["ARRAYX"]
                y_telescope = ant_hdu.header["ARRAYY"]
                z_telescope = ant_hdu.header["ARRAYZ"]
                # AIPS memo #117 says that antenna_positions should be relative to
                # the array center, but in a rotated ECEF frame so that the x-axis
                # goes through the local meridian.
                rot_ecef_positions = ant_hdu.data.field("STABXYZ")
                latitude, longitude, altitude = uvutils.LatLonAlt_from_XYZ(
                    np.array([x_telescope, y_telescope, z_telescope]),
                    check_acceptability=run_check_acceptability,
                )
                self.antenna_positions = uvutils.ECEF_from_rotECEF(
                    rot_ecef_positions, longitude
                )

            if xyz_telescope_frame == "ITRF":
                self.telescope_location = np.array(
                    [x_telescope, y_telescope, z_telescope]
                )
            else:
                if (
                    latitude_degrees is not None
                    and longitude_degrees is not None
                    and altitude is not None
                ):
                    self.telescope_location_lat_lon_alt_degrees = (
                        latitude_degrees,
                        longitude_degrees,
                        altitude,
                    )

            # stuff in columns
            ant_names = ant_hdu.data.field("ANNAME").tolist()
            self.antenna_names = []
            for ant_ind, name in enumerate(ant_names):
                # Sometimes CASA writes antnames as bytes not strings.
                # If the ant name is shorter than 8 characters, the trailing
                # characters may be non-ascii.
                # This is technically a FITS violation as FITS requires ascii.
                # So we just ignore any non-ascii bytes in the decode.
                if isinstance(name, bytes):
                    ant_name_str = str(name.decode("utf-8", "ignore"))
                else:
                    ant_name_str = name
                # remove non-printing ascii characters and exclamation points
                ant_name_str = (
                    ant_name_str.replace("\x00", "")
                    .replace("\x07", "")
                    .replace("!", "")
                )
                self.antenna_names.append(ant_name_str)

            # subtract one to get to 0-indexed values rather than 1-indexed values
            self.antenna_numbers = ant_hdu.data.field("NOSTA") - 1

            self.Nants_telescope = len(self.antenna_numbers)

            if "DIAMETER" in ant_hdu.columns.names:
                self.antenna_diameters = ant_hdu.data.field("DIAMETER")

            try:
                self.set_telescope_params()
            except ValueError as ve:
                warnings.warn(str(ve))

            # Now read in the random parameter info
            self._get_parameter_data(
                vis_hdu,
                run_check_acceptability,
                background_lsts=background_lsts,
                use_astropy=use_astropy,
            )
            # If we find the source attribute in the FITS random paramter list,
            # the multi_object attribute will be set to True, and we should also
            # expect that there must be an AIPS SU table.
            if self.multi_object and "AIPS SU" not in hdunames.keys():
                warnings.warn(
                    "UVFITS file is missing AIPS SU table, which is required when "
                    "SOURCE is one of the `random paramters` in the main binary "
                    "table. Bypassing for now, but note that this file _may_ not "
                    "work correctly in UVFITS-based programs (e.g., AIPS, CASA)."
                )
                self.multi_object = False
                self._object_id_array.required = False
                self._Nobjects.required = False
                self._object_dict.required = False
                self._object_name.form = "str"
                self.object_name = self.object_name[0]
                self.Nobjects = None
                self.object_dict = None
                self.object_id_array = None
            elif self.multi_object:
                su_hdu = hdu_list[hdunames["AIPS SU"]]

                # We should have as many entries in the AIPS SU header as we have
                # unique entries in the SOURCES random paramter (checked in the call
                # to get_parameter_data above)
                assert len(su_hdu.data) == self.Nobjects

                # Set up these arrays so we can assign values to them
                self.phase_center_app_ra = np.zeros(self.Nblts)
                self.phase_center_app_dec = np.zeros(self.Nblts)

                object_dict = {}
                idx_dict = {}
                self.object_name = []

                # Alright, we are off to the races!
                for idx in range(self.Nobjects):
                    # Grab the indv source entry
                    sou_info = su_hdu.data[idx]
                    sou_id = sou_info["ID. NO."]
                    sou_name = sou_info["SOURCE"]
                    self.object_name.append(sou_name)
                    sou_ra = sou_info["RAEPO"] * (np.pi / 180.0)
                    sou_dec = sou_info["DECEPO"] * (np.pi / 180.0)
                    sou_epoch = sou_info["EPOCH"]
                    sou_frame = "fk5"

                    # Okay, now that we have all the source info, we should calculate
                    # app coords for the source, since we'll want to use those for our
                    # baseline calculations
                    app_ra, app_dec = uvutils.calc_app_coords(
                        sou_ra,
                        sou_dec,
                        sou_frame,
                        coord_epoch=sou_epoch,
                        object_type="sidereal",
                        telescope_lat=self.telescope_location_lat_lon_alt[0],
                        telescope_lon=self.telescope_location_lat_lon_alt[1],
                        telescope_alt=self.telescope_location_lat_lon_alt[2],
                        time_array=self.time_array[self.object_id_array == sou_id],
                    )

                    self.phase_center_app_ra[self.object_id_array == sou_id] = app_ra
                    self.phase_center_app_dec[self.object_id_array == sou_id] = app_dec

                    idx_dict[sou_id] = idx
                    object_dict[sou_name] = {
                        "object_type": "sidereal",
                        "object_lon": sou_ra,
                        "object_lat": sou_dec,
                        "coord_frame": sou_frame,
                        "coord_epoch": sou_epoch,
                    }
                self.object_dict = object_dict
                # Now that we've read in the sources, we need to remap object_id_array
                # so that it's zero indexed. UVFITS is 1-indexed, although technically,
                # the numbers _could_ be completely out of order/assigned arbitrary int
                # values
                self.object_id_array = np.array(
                    [idx_dict[idx] for idx in self.object_id_array], dtype=np.int
                )

            # Calculate the apparent coordinate values
            self._set_app_coords_helper()

            if not read_data:
                # don't read in the data. This means the object is a metadata
                # only object but that may not matter for many purposes.
                return

            # Now read in the data
            self._get_data(
                vis_hdu,
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
                False,
                keep_all_metadata,
                run_check,
                check_extra,
                run_check_acceptability,
                strict_uvw_antpos_check,
            )

    def write_uvfits(
        self,
        filename,
        spoof_nonessential=False,
        write_lst=True,
        force_phase=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
    ):
        """
        Write the data to a uvfits file.

        Parameters
        ----------
        filename : str
            The uvfits file to write to.
        spoof_nonessential : bool
            Option to spoof the values of optional UVParameters that are not set
            but are required for uvfits files.
        write_lst : bool
            Option to write the LSTs to the metadata (random group parameters).
        force_phase : bool
            Option to automatically phase drift scan data to zenith of the first
            timestamp.
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

        Raises
        ------
        ValueError
            The `phase_type` of the object is "drift" and the `force_phase`
            keyword is not set.
            The `phase_type` of the object is "unknown".
            If the frequencies are not evenly spaced or are separated by more
            than their channel width.
            The polarization values are not evenly spaced.
            Any of ['antenna_positions', 'gst0', 'rdate', 'earth_omega', 'dut1',
            'timesys'] are not set on the object and `spoof_nonessential` is False.
            If the `timesys` parameter is not set to "UTC".
        TypeError
            If any entry in extra_keywords is not a single string or number.

        """
        if run_check:
            self.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                check_freq_spacing=True,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
            )

        if self.phase_type == "phased":
            pass
        elif self.phase_type == "drift":
            if force_phase:
                print(
                    "The data are in drift mode and do not have a "
                    "defined phase center. Phasing to zenith of the first "
                    "timestamp."
                )
                phase_time = Time(self.time_array[0], format="jd")
                self.phase_to_time(phase_time)
            else:
                raise ValueError(
                    "The data are in drift mode. "
                    "Set force_phase to true to phase the data "
                    "to zenith of the first timestamp before "
                    "writing a uvfits file."
                )
        else:
            raise ValueError(
                "The phasing type of the data is unknown. "
                "Set the phase_type to drift or phased to "
                "reflect the phasing status of the data"
            )
        if self.flex_spw:
            # If we have a 'flexible' spectral window, we will need to evaluate the
            # frequency axis slightly differently.
            nchan_list = []
            start_freq_array = []
            delta_freq_array = []
            for idx in range(self.Nspws):
                chan_mask = self.flex_spw_id_array == idx
                nchan_list += [np.sum(chan_mask)]
                # TODO: Spw axis to be collapsed in future release
                start_freq_array += [self.freq_array[0, chan_mask][0]]
                # Need the array direction here since channel_width is always supposed
                # to be > 0, but channels can be in decending freq order
                freq_dir = np.sign(np.median(np.diff(self.freq_array[0, chan_mask])))
                delta_freq_array += [
                    np.median(self.channel_width[chan_mask]) * freq_dir
                ]

            start_freq_array = np.reshape(np.array(start_freq_array), (1, -1)).astype(
                np.float
            )

            delta_freq_array = np.reshape(np.array(delta_freq_array), (1, -1)).astype(
                np.float
            )

            # We've constructed a couple of lists with relevant values, now time to
            # check them to make sure that the data will write correctly

            # Make sure that all the windows are of the same size
            if len(np.unique(nchan_list)) != 1:
                raise IndexError(
                    "UVFITS format cannot handle spectral windows of different sizes!"
                )

            # Make sure freq values are greater zero. Note that I think _technically
            # one could write negative frequencies into the dataset, but I am pretty
            # sure that reduction packages may balk hard.
            if np.any(start_freq_array <= 0):
                raise ValueError("Frequency values must be > 0 for UVFITS!")

            # Make sure the delta values are non-zero
            if np.any(delta_freq_array == 0):
                raise ValueError("Something is wrong, frequency values not unique!")

            # If we passed all the above checks, then it's time to fill some extra
            # array values. Note that 'ref_freq' is something of a placeholder for
            # other exciting things...
            ref_freq = start_freq_array[0, 0]
        else:
            delta_freq_array = np.array([[self.channel_width]]).astype(np.float)
            ref_freq = self.freq_array[0, 0]

        if self.Npols > 1:
            pol_spacing = np.diff(self.polarization_array)
            if np.min(pol_spacing) < np.max(pol_spacing):
                raise ValueError(
                    "The polarization values are not evenly spaced (probably "
                    "because of a select operation). The uvfits format "
                    "does not support unevenly spaced polarizations."
                )
            pol_spacing = pol_spacing[0]
        else:
            pol_spacing = 1

        for p in self.extra():
            param = getattr(self, p)
            if param.name in self.uvfits_required_extra:
                if param.value is None:
                    if spoof_nonessential:
                        param.apply_spoof()
                        setattr(self, p, param)
                    else:
                        raise ValueError(
                            "Required attribute {attribute} "
                            "for uvfits not defined. Define or "
                            "set spoof_nonessential to True to "
                            "spoof this attribute.".format(attribute=p)
                        )

        # check for unflagged data with nsample = 0. Warn if any found
        wh_nsample0 = np.where(self.nsample_array == 0)
        if np.any(~self.flag_array[wh_nsample0]):
            warnings.warn(
                "Some unflagged data has nsample = 0. Flags and "
                "nsamples are combined in uvfits files such that "
                "these data will appear to be flagged."
            )

        uvfits_data_shape = (
            self.Nblts,
            1,
            1,
            self.Nspws,
            self.Nfreqs // self.Nspws,
            self.Npols,
            1,
        )

        # Reshape the arrays so that they match the uvfits conventions
        # FITS uvw direction convention is opposite ours and Miriad's.
        # So conjugate the visibilities and flip the uvws:
        data_array = np.reshape(np.conj(self.data_array), uvfits_data_shape)
        weights_array = np.reshape(
            self.nsample_array * np.where(self.flag_array, -1, 1), uvfits_data_shape,
        )

        uvfits_array_data = np.concatenate(
            [data_array.real, data_array.imag, weights_array], axis=6
        )

        # FITS uvw direction convention is opposite ours and Miriad's.
        # So conjugate the visibilities and flip the uvws:
        uvw_array_sec = -1 * self.uvw_array / const.c.to("m/s").value
        # jd_midnight = np.floor(self.time_array[0] - 0.5) + 0.5
        tzero = np.float32(self.time_array[0])

        # uvfits convention is that time_array + relevant PZERO = actual JD
        # We are setting PZERO4 = float32(first time of observation)
        time_array = np.float32(self.time_array - np.float64(tzero))

        int_time_array = self.integration_time

        baselines_use = self.antnums_to_baseline(
            self.ant_1_array, self.ant_2_array, attempt256=True
        )
        # Set up dictionaries for populating hdu
        # Note that uvfits antenna arrays are 1-indexed so we add 1
        # to our 0-indexed arrays
        group_parameter_dict = {
            "UU      ": uvw_array_sec[:, 0],
            "VV      ": uvw_array_sec[:, 1],
            "WW      ": uvw_array_sec[:, 2],
            "DATE    ": time_array,
            "BASELINE": baselines_use,
            # "SOURCE  ": np.ones(time_array.shape, dtype=np.float32),
            "FREQSEL ": np.ones_like(self.time_array, dtype=np.float32),
            "ANTENNA1": self.ant_1_array + 1,
            "ANTENNA2": self.ant_2_array + 1,
            "SUBARRAY": np.ones_like(self.ant_1_array),
            "INTTIM  ": int_time_array,
        }

        pscal_dict = {
            "UU      ": 1.0,
            "VV      ": 1.0,
            "WW      ": 1.0,
            "DATE    ": 1.0,
            "BASELINE": 1.0,
            # "SOURCE  ": 1.0,
            "FREQSEL ": 1.0,
            "ANTENNA1": 1.0,
            "ANTENNA2": 1.0,
            "SUBARRAY": 1.0,
            "INTTIM  ": 1.0,
        }
        pzero_dict = {
            "UU      ": 0.0,
            "VV      ": 0.0,
            "WW      ": 0.0,
            "DATE    ": tzero,
            "BASELINE": 0.0,
            # "SOURCE  ": 0.0,
            "FREQSEL ": 0.0,
            "ANTENNA1": 0.0,
            "ANTENNA2": 0.0,
            "SUBARRAY": 0.0,
            "INTTIM  ": 0.0,
        }

        if write_lst:
            # lst is a non-standard entry (it's not in the AIPS memo)
            # but storing it can be useful (e.g. can avoid recalculating it on read)
            # need to store it in 2 parts to get enough accuracy
            # angles in uvfits files are stored in degrees, so first convert to degrees
            lst_array_deg = np.rad2deg(self.lst_array)
            lst_array_1 = np.float32(lst_array_deg)
            lst_array_2 = np.float32(lst_array_deg - np.float64(lst_array_1))
            group_parameter_dict["LST     "] = lst_array_1
            pscal_dict["LST     "] = 1.0
            pzero_dict["LST     "] = 0.0

        # list contains arrays of [u,v,w,date,baseline];
        # each array has shape (Nblts)
        parnames_use = ["UU      ", "VV      ", "WW      ", "DATE    "]
        if np.max(self.ant_1_array) < 255 and np.max(self.ant_2_array) < 255:
            # if the number of antennas is less than 256 then include both the
            # baseline array and the antenna arrays in the group parameters.
            # Otherwise just use the antenna arrays
            parnames_use.append("BASELINE")

        # TODO Karto: Here's where to add "SOURCE " item, and potentially "FREQSEL "

        parnames_use += ["ANTENNA1", "ANTENNA2", "SUBARRAY", "INTTIM  "]

        if write_lst:
            parnames_use.append("LST     ")

        group_parameter_list = [
            group_parameter_dict[parname] for parname in parnames_use
        ]

        if write_lst:
            # add second LST array part
            parnames_use.append("LST     ")
            group_parameter_list.append(lst_array_2)

        hdu = fits.GroupData(
            uvfits_array_data,
            parnames=parnames_use,
            pardata=group_parameter_list,
            bitpix=-32,
        )
        hdu = fits.GroupsHDU(hdu)

        for i, key in enumerate(parnames_use):
            hdu.header["PSCAL" + str(i + 1) + "  "] = pscal_dict[key]
            hdu.header["PZERO" + str(i + 1) + "  "] = pzero_dict[key]

        # ISO string of first time in self.time_array
        hdu.header["DATE-OBS"] = Time(self.time_array[0], scale="utc", format="jd").isot

        hdu.header["CTYPE2  "] = "COMPLEX "
        hdu.header["CRVAL2  "] = 1.0
        hdu.header["CRPIX2  "] = 1.0
        hdu.header["CDELT2  "] = 1.0

        # Note: This axis is called STOKES to comply with the AIPS memo 117
        # However, this confusing because it is NOT a true Stokes axis,
        #   it is really the polarization axis.
        hdu.header["CTYPE3  "] = "STOKES  "
        hdu.header["CRVAL3  "] = float(self.polarization_array[0])
        hdu.header["CRPIX3  "] = 1.0
        hdu.header["CDELT3  "] = float(pol_spacing)

        hdu.header["CTYPE4  "] = "FREQ    "
        hdu.header["CRVAL4  "] = ref_freq
        hdu.header["CRPIX4  "] = 1.0
        hdu.header["CDELT4  "] = delta_freq_array[0, 0]

        hdu.header["CTYPE5  "] = "IF      "
        hdu.header["CRVAL5  "] = 1.0
        hdu.header["CRPIX5  "] = 1.0
        hdu.header["CDELT5  "] = 1.0

        hdu.header["CTYPE6  "] = "RA"
        hdu.header["CRVAL6  "] = self.phase_center_ra_degrees

        hdu.header["CTYPE7  "] = "DEC"
        hdu.header["CRVAL7  "] = self.phase_center_dec_degrees

        hdu.header["BUNIT   "] = self.vis_units
        hdu.header["BSCALE  "] = 1.0
        hdu.header["BZERO   "] = 0.0

        hdu.header["OBJECT  "] = self.object_name
        hdu.header["TELESCOP"] = self.telescope_name
        hdu.header["LAT     "] = self.telescope_location_lat_lon_alt_degrees[0]
        hdu.header["LON     "] = self.telescope_location_lat_lon_alt_degrees[1]
        hdu.header["ALT     "] = self.telescope_location_lat_lon_alt[2]
        hdu.header["INSTRUME"] = self.instrument
        hdu.header["EPOCH   "] = float(self.phase_center_epoch)
        # TODO: This is a keyword that should at some point get added for velocity
        # reference stuff, although for right now pyuvdata doesn't do any sort of
        # handling of this, so stub this out for now.
        # hdu.header["SPECSYS "] = "TOPOCENT"

        if self.phase_center_frame is not None:
            hdu.header["PHSFRAME"] = self.phase_center_frame

        if self.x_orientation is not None:
            hdu.header["XORIENT"] = self.x_orientation

        if self.blt_order is not None:
            blt_order_str = ", ".join(self.blt_order)
            hdu.header["BLTORDER"] = blt_order_str

        for line in self.history.splitlines():
            hdu.header.add_history(line)

        # end standard keywords; begin user-defined keywords
        for key, value in self.extra_keywords.items():
            # header keywords have to be 8 characters or less
            if len(str(key)) > 8:
                warnings.warn(
                    "key {key} in extra_keywords is longer than 8 "
                    "characters. It will be truncated to 8 as required "
                    "by the uvfits file format.".format(key=key)
                )
            keyword = key[:8].upper()
            if isinstance(value, (dict, list, np.ndarray)):
                raise TypeError(
                    "Extra keyword {keyword} is of {keytype}. "
                    "Only strings and numbers are "
                    "supported in uvfits.".format(keyword=key, keytype=type(value))
                )

            if keyword == "COMMENT":
                for line in value.splitlines():
                    hdu.header.add_comment(line)
            else:
                hdu.header[keyword] = value

        # ADD the ANTENNA table
        staxof = np.zeros(self.Nants_telescope)

        # 0 specifies alt-az, 6 would specify a phased array
        mntsta = np.zeros(self.Nants_telescope)

        # beware, X can mean just about anything
        poltya = np.full((self.Nants_telescope), "X", dtype=np.object_)
        polaa = [90.0] + np.zeros(self.Nants_telescope)
        poltyb = np.full((self.Nants_telescope), "Y", dtype=np.object_)
        polab = [0.0] + np.zeros(self.Nants_telescope)

        col1 = fits.Column(name="ANNAME", format="8A", array=self.antenna_names)
        # AIPS memo #117 says that antenna_positions should be relative to
        # the array center, but in a rotated ECEF frame so that the x-axis
        # goes through the local meridian.
        longitude = self.telescope_location_lat_lon_alt[1]
        rot_ecef_positions = uvutils.rotECEF_from_ECEF(
            self.antenna_positions, longitude
        )
        col2 = fits.Column(name="STABXYZ", format="3D", array=rot_ecef_positions)
        # convert to 1-indexed from 0-indexed indicies
        col3 = fits.Column(name="NOSTA", format="1J", array=self.antenna_numbers + 1)
        col4 = fits.Column(name="MNTSTA", format="1J", array=mntsta)
        col5 = fits.Column(name="STAXOF", format="1E", array=staxof)
        col6 = fits.Column(name="POLTYA", format="1A", array=poltya)
        col7 = fits.Column(name="POLAA", format="1E", array=polaa)
        # col8 = fits.Column(name='POLCALA', format='3E', array=polcala)
        col9 = fits.Column(name="POLTYB", format="1A", array=poltyb)
        col10 = fits.Column(name="POLAB", format="1E", array=polab)
        # col11 = fits.Column(name='POLCALB', format='3E', array=polcalb)
        # note ORBPARM is technically required, but we didn't put it in
        col_list = [col1, col2, col3, col4, col5, col6, col7, col9, col10]

        if self.antenna_diameters is not None:
            col12 = fits.Column(
                name="DIAMETER", format="1E", array=self.antenna_diameters
            )
            col_list.append(col12)

        cols = fits.ColDefs(col_list)

        ant_hdu = fits.BinTableHDU.from_columns(cols)

        ant_hdu.header["EXTNAME"] = "AIPS AN"
        ant_hdu.header["EXTVER"] = 1

        # write XYZ coordinates
        ant_hdu.header["ARRAYX"] = self.telescope_location[0]
        ant_hdu.header["ARRAYY"] = self.telescope_location[1]
        ant_hdu.header["ARRAYZ"] = self.telescope_location[2]
        ant_hdu.header["FRAME"] = "ITRF"
        ant_hdu.header["GSTIA0"] = self.gst0
        # TODO Karto: Do this more intelligently in the future
        ant_hdu.header["FREQ"] = self.freq_array[0, 0]
        ant_hdu.header["RDATE"] = self.rdate
        ant_hdu.header["UT1UTC"] = self.dut1

        ant_hdu.header["TIMESYS"] = self.timesys
        if self.timesys != "UTC":
            raise ValueError(
                "This file has a time system {tsys}. "
                'Only "UTC" time system files are supported'.format(tsys=self.timesys)
            )
        ant_hdu.header["ARRNAM"] = self.telescope_name
        ant_hdu.header["NO_IF"] = self.Nspws
        ant_hdu.header["DEGPDY"] = self.earth_omega
        # ant_hdu.header['IATUTC'] = 35.

        # set mandatory parameters which are not supported by this object
        # (or that we just don't understand)
        ant_hdu.header["NUMORB"] = 0

        # note: Bart had this set to 3. We've set it 0 after aips 117. -jph
        ant_hdu.header["NOPCAL"] = 0

        ant_hdu.header["POLTYPE"] = "X-Y LIN"

        # note: we do not support the concept of "frequency setups"
        # -- lists of spws given in a SU table.
        # Karto: Here might be a place to address freq setup?
        ant_hdu.header["FREQID"] = 1

        # if there are offsets in images, this could be the culprit
        ant_hdu.header["POLARX"] = 0.0
        ant_hdu.header["POLARY"] = 0.0

        ant_hdu.header["DATUTC"] = 0  # ONLY UTC SUPPORTED

        # we always output right handed coordinates
        ant_hdu.header["XYZHAND"] = "RIGHT"

        fits_tables = [hdu, ant_hdu]
        # If needed, add the FQ table
        if self.Nspws > 1:
            fmt_d = "%iD" % self.Nspws
            fmt_e = "%iE" % self.Nspws
            fmt_j = "%iJ" % self.Nspws

            # TODO Karto: Temp implementation until we fix some other things in UVData
            if_freq = start_freq_array - ref_freq
            ch_width = delta_freq_array
            tot_bw = (self.Nfreqs // self.Nspws) * np.abs(delta_freq_array)
            sideband = np.sign(delta_freq_array) * np.ones((1, self.Nspws))

            # FRQSEL is hardcoded at the moment, could think about doing this
            # at least somewhat more intelligently...
            col_list = [
                fits.Column(name="FRQSEL", format="1J", array=[1]),
                fits.Column(name="IF FREQ", unit="HZ", format=fmt_d, array=if_freq),
                fits.Column(name="CH WIDTH", unit="HZ", format=fmt_e, array=ch_width),
                fits.Column(
                    name="TOTAL BANDWIDTH", unit="HZ", format=fmt_e, array=tot_bw
                ),
                fits.Column(name="SIDEBAND", format=fmt_j, array=sideband),
            ]

            fq_hdu = fits.BinTableHDU.from_columns(fits.ColDefs(col_list))

            fq_hdu.header["EXTNAME"] = "AIPS FQ"
            fq_hdu.header["NO_IF"] = self.Nspws
            fits_tables.append(fq_hdu)

        # If needed, add the SU table
        if self.multi_object:
            fmt_d = "%iD" % self.Nspws
            fmt_e = "%iE" % self.Nspws
            fmt_j = "%iJ" % self.Nspws

            int_zeros = np.zeros(self.Nobjects, dtype=np.int)
            flt_zeros = np.zeros(self.Nobjects, dtype=np.float)
            zero_arr = np.zeros((self.Nobjects, self.Nspws))
            sou_ids = np.arange(self.Nobjects) + 1
            name_arr = np.array(self.object_name)
            cal_code = ["    "] * self.Nobjects
            # These are things we need to flip through on a source-by-source basis
            ra_arr = np.zeros(self.Nobjects, dtype=np.float)
            app_ra = np.zeros(self.Nobjects, dtype=np.float)
            dec_arr = np.zeros(self.Nobjects, dtype=np.float)
            app_dec = np.zeros(self.Nobjects, dtype=np.float)
            epo_arr = np.zeros(self.Nobjects, dtype=np.float)
            pm_ra = np.zeros(self.Nobjects, dtype=np.float)
            pm_dec = np.zeros(self.Nobjects, dtype=np.float)
            rest_freq = np.zeros((self.Nobjects, self.Nspws), dtype=np.float)
            for idx in range(self.Nobjects):
                object_dict = self.object_dict[self.object_name[idx]]
                # This is a stub for something smarter in the future
                rest_freq[idx][:] = np.mean(self.freq_array)
                pm_ra[idx] = 0.0
                pm_dec[idx] = 0.0
                if object_dict["object_type"] == "sidereal":
                    # So here's the deal -- we need all the objects to be in the same
                    # coordinate frame, although nothing in object_dict forces objects
                    # to share the same frame. So we want to make sure that everything
                    # lines up with the coordinate frame listed in phase_center_frame.
                    ra_arr[idx], dec_arr[idx] = uvutils.translate_sidereal_to_sidereal(
                        object_dict["object_lon"],
                        object_dict["object_lat"],
                        object_dict["coord_frame"],
                        "fk5",
                        coord_epoch=object_dict["coord_epoch"]
                        if "coord_epoch" in (object_dict.keys())
                        else None,
                        time_array=np.mean(self.time_array),
                    )

                    epo_arr[idx] = (
                        object_dict["coord_epoch"]
                        if "coord_epoch" in (object_dict.keys())
                        else 2000.0
                    )

                    app_ra[idx] = np.mean(
                        self.phase_center_app_ra[self.object_id_array == idx]
                    )

                    app_dec[idx] = np.mean(
                        self.phase_center_app_dec[self.object_id_array == idx]
                    )
                ra_arr *= 180.0 / np.pi
                dec_arr *= 180.0 / np.pi
                app_ra *= 180.0 / np.pi
                app_dec *= 180.0 / np.pi

            col_list = [
                fits.Column(name="ID. NO.", format="1J", array=sou_ids),
                fits.Column(name="SOURCE", format="20A", array=name_arr),
                fits.Column(name="QUAL", format="1J", array=int_zeros),
                fits.Column(name="CALCODE", format="4A", array=cal_code),
                fits.Column(name="IFLUX", format=fmt_e, unit="JY", array=zero_arr),
                fits.Column(name="QFLUX", format=fmt_e, unit="JY", array=zero_arr),
                fits.Column(name="UFLUX", format=fmt_e, unit="JY", array=zero_arr),
                fits.Column(name="VFLUX", format=fmt_e, unit="JY", array=zero_arr),
                fits.Column(name="FREQOFF", format=fmt_d, unit="HZ", array=zero_arr),
                fits.Column(name="BANDWIDTH", format="1D", unit="HZ", array=flt_zeros),
                fits.Column(name="RAEPO", format="1D", unit="DEGREES", array=ra_arr),
                fits.Column(name="DECEPO", format="1D", unit="DEGREES", array=dec_arr),
                fits.Column(name="EPOCH", format="1D", unit="YEARS", array=epo_arr),
                fits.Column(name="RAAPP", format="1D", unit="DEGREES", array=app_ra),
                fits.Column(name="DECAPP", format="1D", unit="DEGREES", array=app_dec),
                fits.Column(name="LSRVEL", format=fmt_d, unit="M/SEC", array=zero_arr),
                fits.Column(name="RESTFREQ", format=fmt_d, unit="HZ", array=rest_freq),
                fits.Column(name="PMRA", format="1D", unit="DEG/DAY", array=pm_ra),
                fits.Column(name="PMDEC", format="1D", unit="DEG/DAY", array=pm_dec),
            ]

            su_hdu = fits.BinTableHDU.from_columns(fits.ColDefs(col_list))
            su_hdu.header["EXTNAME"] = "AIPS SU"
            su_hdu.header["NO_IF"] = self.Nspws
            su_hdu.header["FREQID"] = 1
            su_hdu.header["VELDEF"] = "RADIO"
            # TODO: Eventually we want to not have this hardcoded, but pyuvdata at
            # present does not carry around any velocity information. As per usual,
            # I (Karto) am tipping my hand on what I might be working on next...
            su_hdu.header["VELTYP"] = "LSR"
            fits_tables.append(su_hdu)

        # write the file
        hdulist = fits.HDUList(hdus=fits_tables)
        hdulist.writeto(filename, overwrite=True)
        hdulist.close()
