# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Class for reading and writing Miriad files."""
import os
import shutil
import numpy as np
import itertools
import warnings
from astropy import constants as const
from astropy.coordinates import Angle, SkyCoord

from .uvdata import UVData
from .. import telescopes as uvtel
from .. import utils as uvutils

__all__ = ["Miriad"]


class Miriad(UVData):
    """
    Defines a Miriad-specific subclass of UVData for reading and writing Miriad files.

    This class should not be interacted with directly, instead use the read
    and write_miriad methods on the UVData class.

    """

    def _pol_to_ind(self, pol):
        if self.polarization_array is None:
            raise ValueError(
                "Can't index polarization {p} because "
                "polarization_array is not set".format(p=pol)
            )
        pol_ind = np.argwhere(self.polarization_array == pol)
        if len(pol_ind) != 1:
            raise ValueError(
                "multiple matches for pol={pol} in "
                "polarization_array".format(pol=pol)
            )
        return pol_ind

    def read_miriad(
        self,
        filepath,
        antenna_nums=None,
        ant_str=None,
        bls=None,
        polarizations=None,
        time_range=None,
        read_data=True,
        phase_type=None,
        correct_lat_lon=True,
        background_lsts=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
        calc_lst=True,
        multi_object=False,
    ):
        """
        Read in data from a miriad file.

        Parameters
        ----------
        filepath : str
            The miriad root directory to read from.
        antenna_nums : array_like of int, optional
            The antennas numbers to read into the object.
        bls : list of tuple, optional
            A list of antenna number tuples (e.g. [(0, 1), (3, 2)]) or a list of
            baseline 3-tuples (e.g. [(0, 1, 'xx'), (2, 3, 'yy')]) specifying baselines
            to include when reading data into the object. For length-2 tuples,
            the ordering of the numbers within the tuple does not matter. For
            length-3 tuples, the polarization string is in the order of the two
            antennas. If length-3 tuples are provided, `polarizations` must be
            None.
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
            `bls` or `polarizations` parameters, if it is a ValueError will be raised.
        polarizations : array_like of int or str, optional
            List of polarization integers or strings to read-in. e.g. ['xx', 'yy', ...]
        time_range : list of float, optional
            len-2 list containing min and max range of times in Julian Date to
            include when reading data into the object. e.g. [2458115.20, 2458115.40]
        read_data : bool
            Read in the uvws, times, visibility and flag data. If set to False,
            only the metadata that can be read quickly (without reading the data)
            will be read in. For Miriad, some of the normally required metadata
            are not fast to read in (e.g. uvws, times) so will not be read in
            if this keyword is False. Therefore, setting read_data to False
            results in an incompletely defined object (check will not pass).
        phase_type : str, optional
            Option to specify the phasing status of the data. Options are 'drift',
            'phased' or None. 'drift' means the data are zenith drift data,
            'phased' means the data are phased to a single RA/Dec. Default is None
            meaning it will be guessed at based on the file contents.
        correct_lat_lon : bool
            Option to update the latitude and longitude from the known_telescopes
            list if the altitude is missing.
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
        calc_lst : bool
            Recalculate the LST values that are present within the file, useful in
            cases where the "online" calculate values have precision or value errors.
            Default is True.
        multi_object : bool
            Forces the UVData object to be marked as multi-object type (where the
            attribute multi_object is set to True). This produces some minor changes
            in the way that the data (in particular, source info) are handled.

        Raises
        ------
        IOError
            If root file directory doesn't exist.
        ValueError
            If incompatible select keywords are set (e.g. `ant_str` with other
            antenna selectors, `times` and `time_range`) or select keywords
            exclude all data or if keywords are set to the wrong type.
            If the data are multi source or have multiple
            spectral windows.
            If the metadata are not internally consistent.

        """
        from . import aipy_extracts

        if not os.path.exists(filepath):
            raise IOError(filepath + " not found")
        uv = aipy_extracts.UV(filepath)

        # load metadata
        (
            default_miriad_variables,
            other_miriad_variables,
            extra_miriad_variables,
            check_variables,
        ) = self._read_miriad_metadata(uv, correct_lat_lon=correct_lat_lon)

        if not read_data:
            # don't read in the data. This means the object is incomplete,
            # but that may not matter for many purposes.
            return

        # read through the file and get the data
        # _source = uv["source"]  # check source of initial visibility

        history_update_string = "  Downselected to specific "
        n_selects = 0

        # select on ant_str if provided
        if ant_str is not None:
            # type check
            assert isinstance(ant_str, (str, np.str)), "ant_str must be fed as a string"
            assert (
                antenna_nums is None and bls is None
            ), "ant_str must be None if antenna_nums or bls is not None"
            aipy_extracts.uv_selector(uv, ant_str)
            if ant_str != "all":
                history_update_string += "antenna pairs"
                n_selects += 1

        # select on antenna_nums and/or bls using aipy_extracts.uv_selector
        if antenna_nums is not None or bls is not None:
            antpair_str = ""
            if antenna_nums is not None:
                # type check
                err_msg = (
                    "antenna_nums must be fed as a list of antenna number integers"
                )
                assert isinstance(antenna_nums, (np.ndarray, list)), err_msg
                assert isinstance(antenna_nums[0], (int, np.integer)), err_msg
                # get all possible combinations
                antpairs = list(
                    itertools.combinations_with_replacement(antenna_nums, 2)
                )
                # convert antenna numbers to string form required by
                # aipy_extracts.uv_selector
                antpair_str_list = ["_".join([str(a) for a in ap]) for ap in antpairs]
                history_update_string += "antennas"
                n_selects += 1

            if bls is not None:
                if isinstance(bls, tuple) and (len(bls) == 2 or len(bls) == 3):
                    bls = [bls]
                if not all(isinstance(item, tuple) for item in bls):
                    raise ValueError(
                        "bls must be a list of tuples of antenna numbers "
                        "(optionally with polarization)."
                    )
                if all(len(item) == 2 for item in bls):
                    if not all(
                        [isinstance(item[0], (int, np.integer,)) for item in bls]
                        + [isinstance(item[1], (int, np.integer,)) for item in bls]
                    ):
                        raise ValueError(
                            "bls must be a list of tuples of antenna numbers "
                            "(optionally with polarization)."
                        )
                elif all(len(item) == 3 for item in bls):
                    if polarizations is not None:
                        raise ValueError(
                            "Cannot provide length-3 tuples and also specify "
                            "polarizations."
                        )
                    if not all(isinstance(item[2], str) for item in bls):
                        raise ValueError(
                            "The third element in each bl must be a polarization string"
                        )
                else:
                    raise ValueError("bls tuples must be all length-2 or all length-3")

                bl_str_list = []
                bl_pols = set()
                for bl in bls:
                    if bl[0] <= bl[1]:
                        bl_str_list.append(str(bl[0]) + "_" + str(bl[1]))
                        if len(bl) == 3:
                            bl_pols.add(bl[2])
                    else:
                        bl_str_list.append(str(bl[1]) + "_" + str(bl[0]))
                        if len(bl) == 3:
                            bl_pols.add(uvutils.conj_pol(bl[2]))

                if n_selects > 0:
                    # combine antpair_str_list and bl_str_list with an intersection
                    antpair_str_list = list(
                        set(antpair_str_list).intersection(bl_str_list)
                    )
                else:
                    antpair_str_list = bl_str_list

                if len(bl_pols) > 0:
                    polarizations = list(bl_pols)

                if n_selects > 0:
                    history_update_string += ", baselines"
                else:
                    history_update_string += "baselines"
                n_selects += 1

            # convert antenna pair list to string form required by
            # aipy_extracts.uv_selector
            antpair_str += ",".join(antpair_str_list)
            aipy_extracts.uv_selector(uv, antpair_str)

        # select on time range
        if time_range is not None:
            # type check
            err_msg = (
                "time_range must be a len-2 list of Julian Date floats, "
                "Ex: [2458115.2, 2458115.6]"
            )
            assert isinstance(time_range, (list, np.ndarray)), err_msg
            assert len(time_range) == 2, err_msg
            assert np.array(
                [isinstance(t, (float, np.float, np.float64)) for t in time_range]
            ).all(), err_msg

            # UVData.time_array marks center of integration, while Miriad
            # 'time' marks beginning
            # assume time_range refers to the center of the integrations,
            # so subtract 1/2 an integration before using with miriad select
            time_range_use = np.array(time_range) - uv["inttime"] / (24 * 3600.0) / 2

            uv.select("time", time_range_use[0], time_range_use[1], include=True)
            if n_selects > 0:
                history_update_string += ", times"
            else:
                history_update_string += "times"
            n_selects += 1

        # select on polarizations
        if polarizations is not None:
            # type check
            err_msg = (
                "pols must be a list of polarization strings or ints, "
                "Ex: ['xx', ...] or [-5, ...]"
            )
            assert isinstance(polarizations, (list, np.ndarray)), err_msg
            assert np.array(
                map(
                    lambda p: isinstance(p, (str, np.str, int, np.integer)),
                    polarizations,
                )
            ).all(), err_msg
            # convert to pol integer if string
            polarizations = [
                p
                if isinstance(p, (int, np.integer))
                else uvutils.polstr2num(p, x_orientation=self.x_orientation)
                for p in polarizations
            ]
            # iterate through all possible pols and reject if not in pols
            pol_list = []
            for p in np.arange(-8, 5):
                if p not in polarizations:
                    uv.select("polarization", p, p, include=False)
                else:
                    pol_list.append(p)
            # assert not empty
            assert len(pol_list) > 0, "No polarizations in data matched {}".format(
                polarizations
            )
            if n_selects > 0:
                history_update_string += ", polarizations"
            else:
                history_update_string += "polarizations"
            n_selects += 1

        history_update_string += " using pyuvdata."
        if n_selects > 0:
            self.history += history_update_string

        data_accumulator = {}
        pol_list = []
        app_ra = None
        app_dec = None
        epoch = None
        sou_dict = {}
        Nobjects = 0
        record_epoch = "epoch" in uv.vartable.keys()
        record_app = ("obsra" in uv.vartable.keys()) and (
            "obsdec" in uv.vartable.keys()
        )
        # dra_list - Stubbing out for later
        # ddec_list - Stubbing out for later
        for (uvw, t, (i, j)), d, f in uv.all_data(raw=True):
            # control for the case of only a single spw not showing up in
            # the dimension
            # Note that the (i, j) tuple is calculated from a baseline number in
            # _miriad (see miriad_wrap.h). The i, j values are also adjusted by _miriad
            # to start at 0 rather than 1.
            # N.B. (Karto): So the below is right, but for the wrong reasons. The data
            # array return is of length nchan, which encompasses the total number of
            # channels in all spectral windows, where 'ischan' and 'nschan' demark the
            # starting positions and length of the individual windows. I don't think
            # this array should ever _not_ be 1D, but the below does cast the shape of
            # the data array correctly so that the vestigial spw-axis is preserved. At
            # some point, the below will need to be fixed -- I'm keeping this here so
            # that I can skip reading through the MIRIAD programmers guide yet again.
            if len(d.shape) == 1:
                d.shape = (1,) + d.shape

            if np.size(d) != self.Nfreqs:
                raise ValueError("Number of channels in spectrum has changed!")
            try:
                cnt = uv["cnt"]
            except (KeyError):
                cnt = np.ones(d.shape, dtype=np.float)
            ra = uv["ra"]
            dec = uv["dec"]
            if record_epoch:
                epoch = uv["epoch"]
            if record_app:
                app_ra = uv["obsra"]
                app_dec = uv["obsdec"]
            # NOTE: Using our lst calculator, which uses astropy,
            # instead of _miriad values which come from pyephem.
            # The differences are of order 5 seconds.
            # To use the values from the file you'd want: lst = uv['lst']
            # N.B. (Karto): So the above is true (presumably) for PAPER/HERA, although
            # not every miriad-supporting facility uses pyephem for LAST generation.
            # I've changed this to be a switch rather than always having the code
            # recalculate these values.
            lst = uv["lst"]
            inttime = uv["inttime"]
            try:
                sou_id = sou_dict[uv["source"]]
            except KeyError:
                sou_dict[uv["source"]] = Nobjects
                sou_id = Nobjects
                Nobjects += 1

            # check extra variables for changes compared with initial value
            for extra_variable in list(check_variables.keys()):
                if type(check_variables[extra_variable]) == str:
                    if uv[extra_variable] != check_variables[extra_variable]:
                        check_variables.pop(extra_variable)
                else:
                    if not np.allclose(
                        uv[extra_variable], check_variables[extra_variable]
                    ):
                        check_variables.pop(extra_variable)

            try:
                # TODO: Gotta clean this up at some point - there are going to be
                # addition things to stick into the metadata accumulator, and accessing
                # those values by index number will likely make it more difficult to
                # maintain/expand the code in the future.
                data_accumulator[uv["pol"]].append(
                    [
                        uvw,  # Entry 0
                        t,  # Entry 1
                        i,  # Entry 2
                        j,  # Entry 3
                        d,  # Entry 4
                        f,  # Entry 5
                        cnt,  # Entry 6
                        ra,  # Entry 7
                        dec,  # Entry 8
                        inttime,  # Entry 9
                        sou_id,  # Entry 10
                        epoch,  # Entry 11
                        app_ra,  # Entry 12
                        app_dec,  # Entry 13
                        lst,  # Entry 14
                    ]
                )
            except (KeyError):
                data_accumulator[uv["pol"]] = [
                    [
                        uvw,  # Entry 0
                        t,  # Entry 1
                        i,  # Entry 2
                        j,  # Entry 3
                        d,  # Entry 4
                        f,  # Entry 5
                        cnt,  # Entry 6
                        ra,  # Entry 7
                        dec,  # Entry 8
                        inttime,  # Entry 9
                        sou_id,  # Entry 10
                        epoch,  # Entry 11
                        app_ra,  # Entry 12
                        app_dec,  # Entry 13
                        lst,  # Entry 14
                    ]
                ]
                pol_list.append(uv["pol"])
                # NB: flag types in miriad are usually ints

        if len(list(data_accumulator.keys())) == 0:
            raise ValueError(
                "No data is present, probably as a result of "
                "select on read that excludes all the data"
            )

        for pol, data in data_accumulator.items():
            # make array "object" dtype because underlying array may be ragged
            data_accumulator[pol] = np.array(data, dtype="object")

        self.polarization_array = np.array(pol_list)
        if polarizations is None:
            # A select on read would make the header npols not match the pols
            # in the data
            if len(self.polarization_array) != self.Npols:
                warnings.warn(
                    "npols={npols} but found {n} pols in data file".format(
                        npols=self.Npols, n=len(self.polarization_array)
                    )
                )
        self.Npols = len(pol_list)

        # makes a data_array (and flag_array) of zeroes to be filled in by
        #   data values
        # any missing data will have zeros

        # use set to get the unique list of all times ever listed in the file
        # iterate over polarizations and all spectra (bls and times) in two
        # nested loops, then flatten into a single vector, then set
        # then list again.

        times = list(
            set(np.concatenate([[k[1] for k in d] for d in data_accumulator.values()]))
        )
        times = np.sort(times)

        ant_i_unique = list(
            set(np.concatenate([[k[2] for k in d] for d in data_accumulator.values()]))
        )
        ant_j_unique = list(
            set(np.concatenate([[k[3] for k in d] for d in data_accumulator.values()]))
        )

        sorted_unique_ants = sorted(set(ant_i_unique + ant_j_unique))
        ant_i_unique = np.array(ant_i_unique)
        ant_j_unique = np.array(ant_j_unique)

        # Determine maximum digits needed to distinguish different values
        if sorted_unique_ants[-1] > 0:
            ndig_ant = np.ceil(np.log10(sorted_unique_ants[-1])).astype(int) + 1
        else:
            ndig_ant = 1
        # Be excessive in precision because we use the floating point values as
        # dictionary keys later
        prec_t = -2 * np.floor(np.log10(self._time_array.tols[-1])).astype(int)
        ndig_t = np.ceil(np.log10(times[-1])).astype(int) + prec_t + 2
        blts = []
        for d in data_accumulator.values():
            for k in d:
                blt = [
                    "{1:.{0}f}".format(prec_t, k[1]).zfill(ndig_t),
                    str(k[2]).zfill(ndig_ant),
                    str(k[3]).zfill(ndig_ant),
                    str(k[9]).zfill(ndig_t),
                ]
                blt = "_".join(blt)
                blts.append(blt)
        unique_blts = np.unique(np.array(blts))

        reverse_inds = dict(zip(unique_blts, range(len(unique_blts))))
        self.Nants_data = len(sorted_unique_ants)

        # load antennas and antenna positions using sorted unique ants list
        self._load_antpos(uv, sorted_unique_ants=sorted_unique_ants)

        # form up a grid which indexes time and baselines along the 'long'
        # axis of the visdata array
        tij_grid = np.array([list(map(float, x.split("_"))) for x in unique_blts])
        t_grid, ant_i_grid, ant_j_grid, int_grid = tij_grid.T
        # set the data sizes
        if (
            antenna_nums is None
            and bls is None
            and ant_str is None
            and time_range is None
        ):
            try:
                self.Nblts = uv["nblts"]
                if self.Nblts != len(t_grid):
                    warnings.warn(
                        "Nblts does not match the number of unique blts in the data"
                    )
                    self.Nblts = len(t_grid)
            except (KeyError):
                self.Nblts = len(t_grid)
        else:
            # The select on read will make the header nblts not match the
            # number of unique blts
            self.Nblts = len(t_grid)
        if time_range is None:
            try:
                self.Ntimes = uv["ntimes"]
                if self.Ntimes != len(times):
                    warnings.warn(
                        "Ntimes does not match the number of unique times in the data"
                    )
                    self.Ntimes = len(times)
            except (KeyError):
                self.Ntimes = len(times)
        else:
            # The select on read will make the header ntimes not match the
            # number of unique times
            self.Ntimes = len(times)

        # UVData.time_array marks center of integration, while Miriad 'time'
        # marks beginning
        # also, int_grid is in units of seconds, so we need to convert to days
        self.time_array = t_grid + int_grid / (24 * 3600.0) / 2

        self.integration_time = np.asarray(int_grid, dtype=np.float64)

        self.ant_1_array = ant_i_grid.astype(int)
        self.ant_2_array = ant_j_grid.astype(int)

        self.baseline_array = self.antnums_to_baseline(
            ant_i_grid.astype(int), ant_j_grid.astype(int)
        )
        if antenna_nums is None and bls is None and ant_str is None:
            try:
                self.Nbls = uv["nbls"]
                if self.Nbls != len(np.unique(self.baseline_array)):
                    warnings.warn(
                        "Nbls does not match the number of unique baselines in the data"
                    )
                    self.Nbls = len(np.unique(self.baseline_array))
            except (KeyError):
                self.Nbls = len(np.unique(self.baseline_array))
        else:
            # The select on read will make the header nbls not match the
            # number of unique bls
            self.Nbls = len(np.unique(self.baseline_array))

        # slot the data into a grid
        # TODO: Spw axis to be collapsed in future release
        self.data_array = np.zeros(
            (self.Nblts, 1, self.Nfreqs, self.Npols), dtype=np.complex64
        )
        self.flag_array = np.ones(self.data_array.shape, dtype=np.bool)
        self.uvw_array = np.zeros((self.Nblts, 3))
        # NOTE: Using our lst calculator, which uses astropy,
        # instead of _miriad values which come from pyephem.
        # The differences are of order 5 seconds.
        proc = None
        if (self.telescope_location is not None) and calc_lst:
            proc = self.set_lsts_from_time_array(background=background_lsts)
        self.nsample_array = np.ones(self.data_array.shape, dtype=np.float)

        # Temporary arrays to hold polarization axis, which will be collapsed
        ra_pol_list = np.zeros((self.Nblts, self.Npols))
        dec_pol_list = np.zeros((self.Nblts, self.Npols))
        uvw_pol_list = np.zeros((self.Nblts, 3, self.Npols))
        sou_id_pol_list = np.zeros((self.Nblts, self.Npols), dtype=np.int)
        epoch_pol_list = np.zeros((self.Nblts, self.Npols))
        app_ra_pol_list = np.zeros((self.Nblts, self.Npols))
        app_dec_pol_list = np.zeros((self.Nblts, self.Npols))
        lst_pol_list = np.zeros((self.Nblts, self.Npols))

        c_ns = const.c.to("m/ns").value
        for pol, data in data_accumulator.items():
            pol_ind = self._pol_to_ind(pol)
            for ind, d in enumerate(data):
                blt = [
                    "{1:.{0}f}".format(prec_t, d[1]).zfill(ndig_t),
                    str(d[2]).zfill(ndig_ant),
                    str(d[3]).zfill(ndig_ant),
                    str(d[9]).zfill(ndig_t),
                ]
                blt = "_".join(blt)
                blt_index = reverse_inds[blt]

                self.data_array[blt_index, :, :, pol_ind] = d[4]
                self.flag_array[blt_index, :, :, pol_ind] = d[5]
                self.nsample_array[blt_index, :, :, pol_ind] = d[6]

                # because there are uvws/ra/dec for each pol, and one pol may not
                # have that visibility, we collapse along the polarization
                # axis but avoid any missing visbilities
                uvw = d[0] * c_ns
                uvw.shape = (1, 3)
                uvw_pol_list[blt_index, :, pol_ind] = uvw
                ra_pol_list[blt_index, pol_ind] = d[7]
                dec_pol_list[blt_index, pol_ind] = d[8]
                sou_id_pol_list[blt_index, pol_ind] = d[10]
                epoch_pol_list[blt_index, pol_ind] = d[11]
                app_ra_pol_list[blt_index, pol_ind] = d[12]
                app_dec_pol_list[blt_index, pol_ind] = d[13]
                lst_pol_list[blt_index, pol_ind] = d[14]

        # Collapse pol axis for ra_list, dec_list, and uvw_list
        ra_list = np.zeros(self.Nblts)
        dec_list = np.zeros(self.Nblts)
        sou_id_list = np.zeros(self.Nblts)
        epoch_list = np.zeros(self.Nblts)
        app_ra_list = np.zeros(self.Nblts)
        app_dec_list = np.zeros(self.Nblts)
        lst_list = np.zeros(self.Nblts)

        for blt_index in range(self.Nblts):
            test = ~np.all(self.flag_array[blt_index, :, :, :], axis=(0, 1))
            good_pol = np.where(test)[0]
            if len(good_pol) == 0:
                # No good pols for this blt. Fill with first one.
                good_pol = np.array([0])
            if len(good_pol) == 1:
                # Only one good pol, use it
                self.uvw_array[blt_index, :] = uvw_pol_list[blt_index, :, good_pol]
                ra_list[blt_index] = ra_pol_list[blt_index, good_pol]
                dec_list[blt_index] = dec_pol_list[blt_index, good_pol]
                sou_id_list[blt_index] = sou_id_pol_list[blt_index, good_pol]
                epoch_list[blt_index] = epoch_pol_list[blt_index, good_pol]
                app_ra_list[blt_index] = app_ra_pol_list[blt_index, good_pol]
                app_dec_list[blt_index] = app_dec_pol_list[blt_index, good_pol]
                lst_list[blt_index] = lst_pol_list[blt_index, good_pol]
            else:
                # Multiple good pols, check for consistency. pyuvdata does not
                # support pol-dependent uvw, ra, or dec.
                if np.any(np.diff(uvw_pol_list[blt_index, :, good_pol], axis=0)):
                    raise ValueError("uvw values are different by polarization.")
                else:
                    self.uvw_array[blt_index, :] = uvw_pol_list[
                        blt_index, :, good_pol[0]
                    ]
                if np.any(np.diff(ra_pol_list[blt_index, good_pol])):
                    raise ValueError("ra values are different by polarization.")
                else:
                    ra_list[blt_index] = ra_pol_list[blt_index, good_pol[0]]
                if np.any(np.diff(dec_pol_list[blt_index, good_pol])):
                    raise ValueError("dec values are different by polarization.")
                else:
                    dec_list[blt_index] = dec_pol_list[blt_index, good_pol[0]]
                # Check source IDs
                if np.any(np.diff(sou_id_pol_list[blt_index, good_pol])):
                    raise ValueError("source values are different by polarization.")
                else:
                    sou_id_list[blt_index] = sou_id_pol_list[blt_index, good_pol[0]]
                # Check epochs if needed
                if record_epoch:
                    if np.any(np.diff(epoch_pol_list[blt_index, good_pol])):
                        raise ValueError("epochs vals are different by polarization.")
                    else:
                        epoch_list[blt_index] = epoch_pol_list[blt_index, good_pol[0]]
                # Check apparent coordinates
                if record_app:
                    if np.any(np.diff(app_ra_pol_list[blt_index, good_pol])):
                        raise ValueError("app ra vals are different by polarization.")
                    else:
                        app_ra_list[blt_index] = app_ra_pol_list[blt_index, good_pol[0]]
                    if np.any(np.diff(app_dec_pol_list[blt_index, good_pol])):
                        raise ValueError("app dec vals are different by polarization.")
                    else:
                        app_dec_list[blt_index] = app_dec_pol_list[
                            blt_index, good_pol[0]
                        ]
                if np.any(np.diff(lst_pol_list[blt_index, good_pol])):
                    raise ValueError("source values are different by polarization.")
                else:
                    lst_list[blt_index] = lst_pol_list[blt_index, good_pol[0]]

        # get unflagged blts
        # If we have a 1-baseline, single integration data set, set single_ra and
        # single_time to be true, otherwise evaluate the arrays
        blt_good = np.where(~np.all(self.flag_array, axis=(1, 2, 3)))
        single_ra = (
            True
            if (len(blt_good[0]) == 1)
            else (np.isclose(np.mean(np.diff(ra_list[blt_good])), 0.0))
        )
        single_time = (
            True
            if (len(blt_good[0]) == 1)
            else (np.isclose(np.mean(np.diff(self.time_array[blt_good])), 0.0))
        )

        # first check to see if the phase_type was specified.
        if phase_type is not None:
            if phase_type == "phased":
                self._set_phased()
                if Nobjects > 1:
                    self._set_multi_object()
            elif phase_type == "drift":
                self._set_drift()
            else:
                raise ValueError(
                    "The phase_type was not recognized. "
                    'Set the phase_type to "drift" or "phased" to '
                    "reflect the phasing status of the data"
                )
        else:
            # check if ra is constant throughout file OR if there are multiple sources
            # listed; if the obs meets either of those conditions, then assume that the
            # file is tracking, and if not, assume that the  file is drift scanning,
            # checking if there's only one unflagged time
            if not single_time:
                if single_ra or ((Nobjects > 1) or multi_object):
                    self._set_phased()
                    if Nobjects > 1:
                        self._set_multi_object()
                else:
                    self._set_drift()
            else:
                # if there's only one time, checking for consistent RAs doesn't work.
                # instead check for the presence of an epoch variable, which isn't
                # really a good option, but at least it prevents crashes.
                if "epoch" in uv.vartable.keys():
                    self._set_phased()
                else:
                    self._set_drift()

        if proc is not None:
            proc.join()

        if (self.telescope_location is None) or not calc_lst:
            self.lst_array = lst_list
        if record_app and (self.phase_type == "phased"):
            self.phase_center_app_ra = app_ra_list
            self.phase_center_app_dec = app_dec_list
        if self.multi_object:
            # This presupposes that the data are already phased
            self.phase_center_ra = 0.0  # This isn't used for multi-obj data
            self.phase_center_dec = 0.0  # This isn't used for multi-obj data
            # The concept of an epoch is kinda weird in multi-obj datasets, because
            # different sources are allowed to have different epochs. Just set it to
            # the median value for now. Eventually, I think this should be used as the
            # "if your format forces me to use FK5, here's the epoch" value. Note that
            # MIRIAD, AIPS, and (I think) CASA tasks assume the coordinates are given
            # in FK5 format (well, unless epoch <= 1984, in which case MIRIAD assumes
            # in semi-Orwellian fashion that you _really_ wanted FK4/Bessel-Newcomb).
            self.phase_center_epoch = float(np.median(epoch_list))
            self.object_id_array = sou_id_list.astype(np.int)
            self.Nobjects = Nobjects
            flip_dict = {sou_dict[key]: key for key in sou_dict.keys()}
            self.object_name = [flip_dict[key] for key in range(Nobjects)]
            # Here is where we should package up sources
            object_dict = {}
            for idx in range(Nobjects):
                select_mask = sou_id_list == idx
                object_dict[self.object_name[idx]] = {
                    "object_type": "sidereal",
                    "object_lon": np.median(ra_list[select_mask]),
                    "object_lat": np.median(dec_list[select_mask]),
                    "coord_frame": "fk5",
                    "coord_epoch": np.median(epoch_list[select_mask]),
                }
            self.object_dict = object_dict
        elif self.phase_type == "phased":
            # check that the RA values do not vary
            if not single_ra:
                raise ValueError(
                    'phase_type is "phased" but the RA values are varying.'
                )
            self.phase_center_ra = float(ra_list[0])
            self.phase_center_dec = float(dec_list[0])
            self.phase_center_epoch = uv["epoch"]
            if "phsframe" in uv.vartable.keys():
                self.phase_center_frame = uv["phsframe"].replace("\x00", "")
        else:
            # check that the RA values are not constant (if more than one time present)
            if single_ra and not single_time:
                raise ValueError(
                    'phase_type is "drift" but the RA values are constant.'
                )

            # use skycoord to simplify calculating sky separations.
            # Note, this should be done in the TEE frame, which isn't supported
            # by astropy. Frame doesn't really matter, though, because this is just
            # geometrical, so use icrs.
            pointing_coords = SkyCoord(
                ra=ra_list, dec=dec_list, unit="radian", frame="icrs"
            )
            zenith_coord = SkyCoord(
                ra=self.lst_array,
                dec=self.telescope_location_lat_lon_alt[0],
                unit="radian",
                frame="icrs",
            )

            separation_angles = pointing_coords.separation(zenith_coord)
            acceptable_offset = Angle("1d")
            if np.max(separation_angles.rad) > acceptable_offset.rad:
                warnings.warn(
                    "drift RA, Dec is off from lst, latitude by more than {}, "
                    "so it appears that it is not a zenith drift scan. "
                    'Setting phase_type to "unknown"'.format(acceptable_offset)
                )
                self._set_unknown_phase_type()

        # close out now that we're done
        uv.close()

        if not record_app and (self.phase_type == "phased"):
            # TODO: If we don't have apparent coordinates, figure out whether or not we
            # want to try and generate them (similar to how LSTs are generated)
            self._set_app_coords_helper()

        try:
            self.set_telescope_params()
        except ValueError as ve:
            warnings.warn(str(ve))

        # if blt_order is defined, reorder data to match that order
        # this is required because the data are ordered by (time, baseline) on the read
        if self.blt_order is not None:
            if len(list(self.blt_order)) == 2:
                order, minor_order = self.blt_order
            else:
                order = self.blt_order[0]
                minor_order = None
            self.reorder_blts(order=order, minor_order=minor_order)

        # check if object has all required uv_properties set
        if run_check:
            self.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
            )

    def write_miriad(
        self,
        filepath,
        clobber=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
        no_antnums=False,
    ):
        """
        Write the data to a miriad file.

        Parameters
        ----------
        filename : str
            The miriad root directory to write to.
        clobber : bool
            Option to overwrite the filename if the file already exists.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after before writing the file (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters before
            writing the file (the default is True, meaning the acceptable
            range check will be done).
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.
        no_antnums : bool
            Option to not write the antnums variable to the file.
            Should only be used for testing purposes.

        Raises
        ------
        ValueError
            If the frequencies are not evenly spaced or are separated by more
            than their channel width.
            The `phase_type` of the object is "unknown".
        TypeError
            If any entry in extra_keywords is not a single string or number.

        """
        from . import aipy_extracts

        # change time_array and lst_array to mark beginning of integration,
        # per Miriad format
        miriad_time_array = self.time_array - self.integration_time / (24 * 3600.0) / 2
        if self.telescope_location is not None:
            latitude, longitude, altitude = self.telescope_location_lat_lon_alt_degrees
            miriad_lsts = uvutils.get_lst_for_time(
                miriad_time_array, latitude, longitude, altitude
            )

        # Miriad requires j>i which we call ant1<ant2
        self.conjugate_bls(convention="ant1<ant2")

        if run_check:
            self.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                check_freq_spacing=True,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
            )

        if os.path.exists(filepath):
            if clobber:
                print("File exists: clobbering")
                shutil.rmtree(filepath)
            else:
                raise IOError("File exists: skipping")

        uv = aipy_extracts.UV(filepath, status="new")

        # initialize header variables
        uv._wrhd("obstype", "mixed-auto-cross")
        # avoid inserting extra \n.
        if not self.history[-1] == "\n":
            self.history += "\n"
        uv._wrhd("history", self.history)

        # recognized miriad variables
        uv.add_var("nchan", "i")
        uv["nchan"] = self.Nfreqs
        uv.add_var("npol", "i")
        uv["npol"] = self.Npols

        #####################################################
        # Frequency information here
        uv.add_var("nspect", "i")
        uv["nspect"] = self.Nspws

        if self.flex_spw:
            win_start_pos = np.insert(
                np.where(self.flex_spw_id_array[1:] != self.flex_spw_id_array[:-1])[0]
                + 1,
                0,
                0,
            ).astype(int)

            uv.add_var("ischan", "i")  # Starting chan of window
            uv["ischan"] = win_start_pos + 1  # Miriad is 1-based indexed

            uv.add_var("nschan", "i")  # Number of chan per window
            uv["nschan"] = np.diff(np.append(win_start_pos, self.Nfreqs))

            uv.add_var("sfreq", "d")  # Freq of first channel of the window, Hz -> GHz
            uv["sfreq"] = (self.freq_array[0][win_start_pos] / 1e9).astype(np.double)

            # Need the array direction here since channel_width is always supposed
            # to be > 0, but channels can be in decending freq order
            freq_dir = np.sign(
                self.freq_array[0][np.append(win_start_pos[1:] - 1, self.Nfreqs - 1)]
                - self.freq_array[0][win_start_pos]
            )

            uv.add_var("sdf", "d")  # Channel width, Hz -> GHz
            uv["sdf"] = self.channel_width[win_start_pos] * freq_dir / 1e9
        else:
            uv.add_var("ischan", "i")  # Starting chan of window
            uv["ischan"] = 1  # Miriad is 1-based indexed

            uv.add_var("nschan", "i")  # Number of chan per window
            uv["nschan"] = self.Nfreqs

            # Need the array direction here since channel_width is always supposed
            # to be > 0, but channels can be in decending freq order
            freq_dir = np.sign(np.diff(self.freq_array[0][([0, -1])]))

            uv.add_var("sfreq", "d")  # Freq of first channel of the window, in GHz
            uv["sfreq"] = (self.freq_array[0, 0] / 1e9).astype(np.double)  # Hz -> GHz

            uv.add_var("sdf", "d")  # Channel width, in GHz
            uv["sdf"] = self.channel_width * freq_dir / 1e9  # Hz -> GHz

        # NB: restfreq should go in here at some point
        #####################################################
        if not self.multi_object:
            uv.add_var("source", "a")
            uv["source"] = self.object_name
        uv.add_var("telescop", "a")
        uv["telescop"] = self.telescope_name
        uv.add_var("latitud", "d")
        uv["latitud"] = self.telescope_location_lat_lon_alt[0].astype(np.double)
        uv.add_var("longitu", "d")
        uv["longitu"] = self.telescope_location_lat_lon_alt[1].astype(np.double)
        uv.add_var("nants", "i")

        if self.antenna_diameters is not None:
            if not np.allclose(self.antenna_diameters, self.antenna_diameters[0]):
                warnings.warn(
                    "Antenna diameters are not uniform, but miriad only"
                    "supports a single diameter. Skipping."
                )
            else:
                uv.add_var("antdiam", "d")
                uv["antdiam"] = float(self.antenna_diameters[0])

        # Miriad has no way to keep track of antenna numbers, so the antenna
        # numbers are simply the index for each antenna in any array that
        # describes antenna attributes (e.g. antpos for the antenna_postions).
        # Therefore on write, nants (which gives the size of the antpos array)
        # needs to be increased to be the max value of antenna_numbers+1 and the
        # antpos array needs to be inflated with zeros at locations where we
        # don't have antenna information. These inflations need to be undone at
        # read. If the file was written by pyuvdata, then the variable antnums
        # will be present and we can use it, otherwise we need to test for zeros
        # in the antpos array and/or antennas with no visibilities.
        nants = np.max(self.antenna_numbers) + 1
        uv["nants"] = nants
        if self.antenna_positions is not None:
            # Miriad wants antenna_positions to be in absolute coordinates
            # (not relative to array center) in a rotated ECEF frame where the
            # x-axis goes through the local meridian.
            rel_ecef_antpos = np.zeros((nants, 3), dtype=self.antenna_positions.dtype)
            for ai, num in enumerate(self.antenna_numbers):
                rel_ecef_antpos[num, :] = self.antenna_positions[ai, :]

            # find zeros so antpos can be zeroed there too
            antpos_length = np.sqrt(np.sum(np.abs(rel_ecef_antpos) ** 2, axis=1))

            ecef_antpos = rel_ecef_antpos + self.telescope_location
            longitude = self.telescope_location_lat_lon_alt[1]
            antpos = uvutils.rotECEF_from_ECEF(ecef_antpos, longitude)

            # zero out bad locations (these are checked on read)
            antpos[np.where(antpos_length == 0), :] = [0, 0, 0]

            uv.add_var("antpos", "d")
            # Miriad stores antpos values in units of ns, pyuvdata uses meters.
            uv["antpos"] = (antpos.T.flatten() / const.c.to("m/ns").value).astype(
                np.double
            )

        if self.phase_type == "phased":
            uv.add_var("epoch", "r")
            uv["epoch"] = self.phase_center_epoch
            if self.phase_center_frame is not None:
                uv.add_var("phsframe", "a")
                uv["phsframe"] = self.phase_center_frame

        # required pyuvdata variables that are not recognized miriad variables
        uv.add_var("ntimes", "i")
        uv["ntimes"] = self.Ntimes
        uv.add_var("nbls", "i")
        uv["nbls"] = self.Nbls
        uv.add_var("nblts", "i")
        uv["nblts"] = self.Nblts
        uv.add_var("visunits", "a")
        uv["visunits"] = self.vis_units
        uv.add_var("instrume", "a")
        uv["instrume"] = self.instrument
        uv.add_var("altitude", "d")
        uv["altitude"] = self.telescope_location_lat_lon_alt[2].astype(np.double)

        # optional pyuvdata variables that are not recognized miriad variables
        if self.dut1 is not None:
            uv.add_var("dut1", "d")
            uv["dut1"] = self.dut1
        if self.earth_omega is not None:
            uv.add_var("degpdy", "d")
            uv["degpdy"] = self.earth_omega
        if self.gst0 is not None:
            uv.add_var("gst0", "d")
            uv["gst0"] = self.gst0
        if self.rdate is not None:
            uv.add_var("rdate", "a")
            uv["rdate"] = self.rdate
        if self.timesys is not None:
            uv.add_var("timesys", "a")
            uv["timesys"] = self.timesys
        if self.x_orientation is not None:
            uv.add_var("xorient", "a")
            uv["xorient"] = self.x_orientation
        if self.blt_order is not None:
            blt_order_str = ", ".join(self.blt_order)
            uv.add_var("bltorder", "a")
            uv["bltorder"] = blt_order_str

        # other extra keywords
        # set up dictionaries to map common python types to miriad types
        # NB: arrays/lists/dicts could potentially be written as strings or 1D
        # vectors.  This is not supported at present!
        # NB: complex numbers *should* be supportable, but are not currently
        # supported due to unexplained errors in _miriad and/or its underlying libraries
        numpy_types = {
            np.int8: int,
            np.int16: int,
            np.int32: int,
            np.int64: int,
            np.uint8: int,
            np.uint16: int,
            np.uint32: int,
            np.uint64: int,
            np.float16: float,
            np.float32: float,
            np.float64: float,
            np.float128: float,
        }
        types = {
            str: "a",
            int: "i",
            float: "d",
            bool: "a",  # booleans are stored as strings and changed back on read
        }
        for key, value in self.extra_keywords.items():
            if type(value) in numpy_types.keys():
                if numpy_types[type(value)] == int:
                    value = int(value)
                elif numpy_types[type(value)] == float:
                    value = float(value)
            elif type(value) == bool:
                value = str(value)
            elif type(value) not in types.keys():
                raise TypeError(
                    "Extra keyword {keyword} is of {keytype}. "
                    "Only strings and real numbers are "
                    "supported in miriad.".format(keyword=key, keytype=type(value))
                )

            if len(str(key)) > 8:
                warnings.warn(
                    "key {key} in extra_keywords is longer than 8 "
                    "characters. It will be truncated to 8 as required "
                    "by the miriad file format.".format(key=key)
                )

            uvkeyname = str(key)[:8]  # name must be string, max 8 letters
            typestring = types[type(value)]
            uv.add_var(uvkeyname, typestring)
            uv[uvkeyname] = value

        if not no_antnums:
            # Add in the antenna_numbers so we have them if we read this file back in.
            # For some reason Miriad doesn't handle an array of integers properly,
            # so convert to floats here and integers on read.
            uv.add_var("antnums", "d")
            uv["antnums"] = self.antenna_numbers.astype(np.float64)

        # antenna names is a foreign concept in miriad but required in other formats.
        # Miriad can't handle arrays of strings, so we make it into one long
        # comma-separated string and convert back on read.
        ant_name_str = "[" + ", ".join(self.antenna_names) + "]"
        uv.add_var("antnames", "a")
        uv["antnames"] = ant_name_str

        # variables that can get updated with every visibility
        uv.add_var("pol", "i")
        uv.add_var("lst", "d")
        uv.add_var("cnt", "d")
        if self.multi_object:
            uv.add_var("source", "a")
        uv.add_var("ra", "d")
        uv.add_var("dec", "d")
        uv.add_var("inttime", "d")

        app_record = False
        if (self.phase_center_app_dec is not None) and (
            self.phase_center_app_ra is not None
        ):
            app_record = True
            uv.add_var("obsra", "d")
            uv.add_var("obsdec", "d")

        # write data
        c_ns = const.c.to("m/ns").value
        for viscnt, blt in enumerate(self.data_array):
            uvw = (self.uvw_array[viscnt] / c_ns).astype(np.double)
            t = miriad_time_array[viscnt]
            i = self.ant_1_array[viscnt]
            j = self.ant_2_array[viscnt]

            uv["lst"] = miriad_lsts[viscnt].astype(np.double)
            uv["inttime"] = self.integration_time[viscnt].astype(np.double)
            if self.phase_type == "phased":
                if self.multi_object:
                    object_name = self.object_name[self.object_id_array[viscnt]]
                    uv["source"] = object_name
                    uv["ra"] = self.object_dict[object_name]["object_lon"]
                    uv["dec"] = self.object_dict[object_name]["object_lat"]
                else:
                    uv["ra"] = self.phase_center_ra
                    uv["dec"] = self.phase_center_dec
                if app_record:
                    uv["obsra"] = self.phase_center_app_ra[viscnt]
                    uv["obsdec"] = self.phase_center_app_dec[viscnt]
            elif self.phase_type == "drift":
                uv["ra"] = miriad_lsts[viscnt].astype(np.double)
                uv["dec"] = self.telescope_location_lat_lon_alt[0].astype(np.double)
            else:
                raise ValueError(
                    "The phasing type of the data is unknown. "
                    'Set the phase_type to "drift" or "phased" to '
                    "reflect the phasing status of the data"
                )

            # NOTE only writing spw 0, not supporting multiple spws for write
            for polcnt, pol in enumerate(self.polarization_array):
                uv["pol"] = pol.astype(np.int)
                uv["cnt"] = self.nsample_array[viscnt, 0, :, polcnt].astype(np.double)

                data = self.data_array[viscnt, 0, :, polcnt]
                flags = self.flag_array[viscnt, 0, :, polcnt]
                assert j >= i, (
                    "Miriad requires ant1<ant2 which should be "
                    "guaranteed by prior conjugate_bls call"
                )
                preamble = (uvw, t, (i, j))

                uv.write(preamble, data, flags)

        # close out now that we're done
        uv.close()

        return

    def _read_miriad_metadata(self, uv, correct_lat_lon=True):
        """
        Read in metadata (parameter info) but not data from a miriad file.

        Parameters
        ----------
        uv : aipy.miriad.UV
            aipy object to load metadata from.
        correct_lat_lon : bool
            Option to update the latitude and longitude from the known_telescopes
            list if the altitude is missing.

        Returns
        -------
        default_miriad_variables : list of str
            list of default miriad variables
        other_miriad_variables: list of str
            list of other miriad variables
        extra_miriad_variables: list of str
            list of extra, non-standard variables
        check_variables: dict
            dict of extra miriad variables to add to extra_keywords parameter.

        """
        # load miriad variables
        (
            default_miriad_variables,
            other_miriad_variables,
            extra_miriad_variables,
        ) = self._load_miriad_variables(uv)

        # dict of extra variables
        check_variables = {}
        for extra_variable in extra_miriad_variables:
            check_variables[extra_variable] = uv[extra_variable]

        # keep all single valued extra_variables as extra_keywords
        for key in check_variables.keys():
            if type(check_variables[key]) == str:
                value = check_variables[key].replace("\x00", "")
                # check for booleans encoded as strings
                if value == "True":
                    value = True
                elif value == "False":
                    value = False
                self.extra_keywords[key] = value
            else:
                self.extra_keywords[key] = check_variables[key]

        # Check for items in itemtable to put into extra_keywords
        # These will end up as variables in written files, but is internally consistent.
        for key in uv.items():
            # A few items that are not needed, we read elsewhere, or is not supported
            # when downselecting, so we don't read here.
            if (
                key not in ["vartable", "history", "obstype"]
                and key not in other_miriad_variables
            ):
                if type(uv[key]) == str:
                    value = uv[key].replace("\x00", "")
                    value = uv[key].replace("\x01", "")
                    if value == "True":
                        value = True
                    elif value == "False":
                        value = False
                    self.extra_keywords[key] = value
                else:
                    self.extra_keywords[key] = uv[key]

        # load telescope coords
        self._load_telescope_coords(uv, correct_lat_lon=correct_lat_lon)

        # load antenna positions
        self._load_antpos(uv)

        return (
            default_miriad_variables,
            other_miriad_variables,
            extra_miriad_variables,
            check_variables,
        )

    def _load_miriad_variables(self, uv):
        """
        Load miriad variables from an aipy.miriad UV descriptor onto self.

        Parameters
        ----------
        uv : aipy.miriad.UV
            aipy object to load variables from.

        Returns
        -------
        default_miriad_variables : list of str
            list of default miriad variables
        other_miriad_variables: list of str
            list of other miriad variables
        extra_miriad_variables: list of str
            list of extra, non-standard variables

        """
        # list of miriad variables always read
        # NB: this includes variables in try/except (i.e. not all variables are
        # necessarily present in the miriad file)
        default_miriad_variables = [
            "nchan",
            "npol",
            "inttime",
            "sdf",
            "source",
            "telescop",
            "latitud",
            "longitu",
            "altitude",
            "history",
            "visunits",
            "instrume",
            "dut1",
            "gst0",
            "rdate",
            "timesys",
            "xorient",
            "cnt",
            "ra",
            "dec",
            "lst",
            "pol",
            "nants",
            "antnames",
            "nblts",
            "ntimes",
            "nbls",
            "sfreq",
            "epoch",
            "antpos",
            "antnums",
            "degpdy",
            "antdiam",
            "phsframe",
            "xorient",
            "bltorder",
        ]
        # list of miriad variables not read, but also not interesting
        # NB: nspect (I think) is number of spectral windows, will want one day
        # NB: xyphase & xyamp are "On-line X Y phase/amplitude measurements"
        #     which we may want in a calibration object some day
        # NB: systemp, xtsys & ytsys are "System temperatures of the antenna/X/Y feed"
        #     which we may want in a calibration object some day
        # NB: freqs, leakage and bandpass may be part of a calibration object some day
        other_miriad_variables = [
            "nspect",
            "obsdec",
            "vsource",
            "ischan",
            "restfreq",
            "nschan",
            "corr",
            "freq",
            "freqs",
            "leakage",
            "bandpass",
            "tscale",
            "coord",
            "veldop",
            "time",
            "obsra",
            "operator",
            "version",
            "axismax",
            "axisrms",
            "xyphase",
            "xyamp",
            "systemp",
            "xtsys",
            "ytsys",
            "baseline",
        ]

        extra_miriad_variables = []
        for variable in uv.variables():
            if (
                variable not in default_miriad_variables
                and variable not in other_miriad_variables
            ):
                extra_miriad_variables.append(variable)

        miriad_header_data = {
            "Nfreqs": "nchan",
            "Nspws": "nspect",
            "Npols": "npol",
            "channel_width": "sdf",  # in Ghz!
            "object_name": "source",
            "telescope_name": "telescop",
        }
        for item in miriad_header_data:
            if isinstance(uv[miriad_header_data[item]], str):
                header_value = uv[miriad_header_data[item]].replace("\x00", "")
            else:
                header_value = uv[miriad_header_data[item]]
            setattr(self, item, header_value)

        # Do the units and potential sign conversion for channel_width
        self.channel_width = np.abs(self.channel_width * 1e9)  # change from GHz to Hz

        # Deal with the spectral axis now
        if self.Nspws > 1:
            self._set_flex_spw()
            # Channel widths are described per spw, just need to expand it out to be
            # for each frequency channel.
            self.channel_width = (
                np.concatenate(
                    tuple(
                        np.abs(chan_width) + np.zeros(nchan)
                        for (chan_width, nchan) in zip(uv["sdf"] * 1e9, uv["nschan"])
                    )
                )
                .flatten()
                .astype(np.float)
            )
            # Now setup frequency array
            # TODO: Spw axis to be collapsed in future release
            self.freq_array = np.reshape(
                np.concatenate(
                    tuple(
                        chan_width * np.arange(nchan) + sfreq
                        for (chan_width, nchan, sfreq) in zip(
                            uv["sdf"] * 1e9, uv["nschan"], uv["sfreq"] * 1e9
                        )
                    )
                )
                .flatten()
                .astype(np.float),
                (1, -1),
            )
            # TODO: Fix this to capture unsorted spectra
            self.flex_spw_id_array = (
                np.concatenate(
                    tuple(
                        idx + np.zeros(nchan)
                        for (idx, nchan) in zip(range(self.Nspws), uv["nschan"])
                    )
                )
                .flatten()
                .astype(np.int)
            )
        else:
            self.freq_array = np.reshape(
                np.arange(self.Nfreqs) * self.channel_width + uv["sfreq"] * 1e9, (1, -1)
            )
            self.channel_width = np.float(self.channel_width)

        self.spw_array = np.arange(self.Nspws)

        self.history = uv["history"]
        if not uvutils._check_history_version(self.history, self.pyuvdata_version_str):
            self.history += self.pyuvdata_version_str

        # check for pyuvdata variables that are not recognized miriad variables
        if "visunits" in uv.vartable.keys():
            self.vis_units = uv["visunits"].replace("\x00", "")
        else:
            self.vis_units = "UNCALIB"  # assume no calibration
        if "instrume" in uv.vartable.keys():
            self.instrument = uv["instrume"].replace("\x00", "")
        else:
            self.instrument = self.telescope_name  # set instrument = telescope

        if "dut1" in uv.vartable.keys():
            self.dut1 = uv["dut1"]
        if "degpdy" in uv.vartable.keys():
            self.earth_omega = uv["degpdy"]
        if "gst0" in uv.vartable.keys():
            self.gst0 = uv["gst0"]
        if "rdate" in uv.vartable.keys():
            self.rdate = uv["rdate"].replace("\x00", "")
        if "timesys" in uv.vartable.keys():
            self.timesys = uv["timesys"].replace("\x00", "")
        if "xorient" in uv.vartable.keys():
            self.x_orientation = uv["xorient"].replace("\x00", "")
        if "bltorder" in uv.vartable.keys():
            blt_order_str = uv["bltorder"].replace("\x00", "")
            self.blt_order = tuple(blt_order_str.split(", "))
            if self.blt_order == ("bda",):
                self._blt_order.form = (1,)

        return default_miriad_variables, other_miriad_variables, extra_miriad_variables

    def _load_telescope_coords(self, uv, correct_lat_lon=True):
        """
        Load telescope lat, lon, alt coordinates from aipy.miriad UV descriptor.

        Parameters
        ----------
        uv : aipy.miriad.UV
            aipy object to load lat, lon, alt coordinates from.
        correct_lat_lon : bool
            Option to update the latitude and longitude from the known_telescopes
            list if the altitude is missing.

        """
        # check if telescope name is present
        if self.telescope_name is None:
            self._load_miriad_variables(uv)

        latitude = uv["latitud"]  # in units of radians
        longitude = uv["longitu"]
        try:
            altitude = uv["altitude"]
            self.telescope_location_lat_lon_alt = (latitude, longitude, altitude)
        except (KeyError):
            # get info from known telescopes.
            # Check to make sure the lat/lon values match reasonably well
            telescope_obj = uvtel.get_telescope(self.telescope_name)
            if telescope_obj is not False:

                tol = 2 * np.pi * 1e-3 / (60.0 * 60.0 * 24.0)  # 1mas in radians
                lat_close = np.isclose(
                    telescope_obj.telescope_location_lat_lon_alt[0],
                    latitude,
                    rtol=0,
                    atol=tol,
                )
                lon_close = np.isclose(
                    telescope_obj.telescope_location_lat_lon_alt[1],
                    longitude,
                    rtol=0,
                    atol=tol,
                )
                if correct_lat_lon:
                    self.telescope_location_lat_lon_alt = (
                        telescope_obj.telescope_location_lat_lon_alt
                    )
                else:
                    self.telescope_location_lat_lon_alt = (
                        latitude,
                        longitude,
                        telescope_obj.telescope_location_lat_lon_alt[2],
                    )
                if lat_close and lon_close:
                    if correct_lat_lon:
                        warnings.warn(
                            "Altitude is not present in Miriad file, "
                            "using known location values for "
                            "{telescope_name}.".format(
                                telescope_name=telescope_obj.telescope_name
                            )
                        )
                    else:
                        warnings.warn(
                            "Altitude is not present in Miriad file, "
                            "using known location altitude value "
                            "for {telescope_name} and lat/lon from "
                            "file.".format(telescope_name=telescope_obj.telescope_name)
                        )
                else:
                    warn_string = "Altitude is not present in file "
                    if not lat_close and not lon_close:
                        warn_string = (
                            warn_string
                            + "and latitude and longitude values do not match values "
                        )
                    else:
                        if not lat_close:
                            warn_string = (
                                warn_string + "and latitude value does not match value "
                            )
                        else:
                            warn_string = (
                                warn_string
                                + "and longitude value does not match value "
                            )
                    if correct_lat_lon:
                        warn_string = (
                            warn_string + "for {telescope_name} in known "
                            "telescopes. Using values from known "
                            "telescopes.".format(
                                telescope_name=telescope_obj.telescope_name
                            )
                        )
                        warnings.warn(warn_string)
                    else:
                        warn_string = (
                            warn_string + "for {telescope_name} in known "
                            "telescopes. Using altitude value from known "
                            "telescopes and lat/lon from "
                            "file.".format(telescope_name=telescope_obj.telescope_name)
                        )
                        warnings.warn(warn_string)
            else:
                warnings.warn(
                    "Altitude is not present in Miriad file, and "
                    "telescope {telescope_name} is not in "
                    "known_telescopes. Telescope location will be "
                    "set using antenna positions.".format(
                        telescope_name=self.telescope_name
                    )
                )

    def _load_antpos(self, uv, sorted_unique_ants=None, correct_lat_lon=True):
        """
        Load antennas and their positions from a Miriad UV descriptor.

        Parameters
        ----------
        uv : aipy.miriad.UV
            aipy object to antennas and positions from.
        sorted_unique_ants : list of ints, optional
            List of unique antennas.
        correct_lat_lon : bool
            Option to update the latitude and longitude from the known_telescopes
            list if the altitude is missing.

        """
        # check if telescope coords exist
        if self.telescope_location_lat_lon_alt is None:
            self._load_telescope_coords(uv, correct_lat_lon=correct_lat_lon)

        latitude = uv["latitud"]  # in units of radians
        longitude = uv["longitu"]

        # Miriad has no way to keep track of antenna numbers, so the antenna
        # numbers are simply the index for each antenna in any array that
        # describes antenna attributes (e.g. antpos for the antenna_postions).
        # Therefore on write, nants (which gives the size of the antpos array)
        # needs to be increased to be the max value of antenna_numbers+1 and the
        # antpos array needs to be inflated with zeros at locations where we
        # don't have antenna information. These inflations need to be undone at
        # read. If the file was written by pyuvdata, then the variable antnums
        # will be present and we can use it, otherwise we need to test for zeros
        # in the antpos array and/or antennas with no visibilities.
        try:
            # The antnums variable will only exist if the file was written with
            # pyuvdata.
            # For some reason Miriad doesn't handle an array of integers properly,
            # so we convert to floats on write and back here
            self.antenna_numbers = uv["antnums"].astype(int)
            self.Nants_telescope = len(self.antenna_numbers)
        except (KeyError):
            self.antenna_numbers = None
            self.Nants_telescope = None

        nants = uv["nants"]
        try:
            # Miriad stores antpos values in units of ns, pyuvdata uses meters.
            antpos = uv["antpos"].reshape(3, nants).T * const.c.to("m/ns").value

            # first figure out what are good antenna positions so we can only
            # use the non-zero ones to evaluate position information
            antpos_length = np.sqrt(np.sum(np.abs(antpos) ** 2, axis=1))
            good_antpos = np.where(antpos_length > 0)[0]
            mean_antpos_length = np.mean(antpos_length[good_antpos])
            if mean_antpos_length > 6.35e6 and mean_antpos_length < 6.39e6:
                absolute_positions = True
            else:
                absolute_positions = False

            # Miriad stores antpos values in a rotated ECEF coordinate system
            # where the x-axis goes through the local meridan. Need to convert
            # these positions back to standard ECEF and if they are absolute
            # positions, subtract off the telescope position to make them
            # relative to the array center.
            ecef_antpos = uvutils.ECEF_from_rotECEF(antpos, longitude)

            if self.telescope_location is not None:
                if absolute_positions:
                    rel_ecef_antpos = ecef_antpos - self.telescope_location
                    # maintain zeros because they mark missing data
                    rel_ecef_antpos[np.where(antpos_length == 0)[0]] = ecef_antpos[
                        np.where(antpos_length == 0)[0]
                    ]
                else:
                    rel_ecef_antpos = ecef_antpos
            else:
                self.telescope_location = np.mean(ecef_antpos[good_antpos, :], axis=0)
                valid_location = self._telescope_location.check_acceptability()[0]

                # check to see if this could be a valid telescope_location
                if valid_location:
                    mean_lat, mean_lon, mean_alt = self.telescope_location_lat_lon_alt
                    tol = 2 * np.pi / (60.0 * 60.0 * 24.0)  # 1 arcsecond in radians
                    mean_lat_close = np.isclose(mean_lat, latitude, rtol=0, atol=tol)
                    mean_lon_close = np.isclose(mean_lon, longitude, rtol=0, atol=tol)

                    if mean_lat_close and mean_lon_close:
                        # this looks like a valid telescope_location, and the
                        # mean antenna lat & lon values are close. Set the
                        # telescope_location using the file lat/lons and the mean alt.
                        # Then subtract it off of the antenna positions
                        warnings.warn(
                            "Telescope location is not set, but antenna "
                            "positions are present. Mean antenna latitude and "
                            "longitude values match file values, so "
                            "telescope_position will be set using the "
                            "mean of the antenna altitudes"
                        )
                        self.telescope_location_lat_lon_alt = (
                            latitude,
                            longitude,
                            mean_alt,
                        )
                        rel_ecef_antpos = ecef_antpos - self.telescope_location

                    else:
                        # this looks like a valid telescope_location, but the
                        # mean antenna lat & lon values are not close. Set the
                        # telescope_location using the file lat/lons at sea level.
                        # Then subtract it off of the antenna positions
                        self.telescope_location_lat_lon_alt = (latitude, longitude, 0)
                        warn_string = (
                            "Telescope location is set at sealevel at "
                            "the file lat/lon coordinates. Antenna "
                            "positions are present, but the mean "
                            "antenna "
                        )
                        rel_ecef_antpos = ecef_antpos - self.telescope_location

                        if not mean_lat_close and not mean_lon_close:
                            warn_string += (
                                "latitude and longitude values do not "
                                "match file values so they are not used "
                                "for altiude."
                            )
                        elif not mean_lat_close:
                            warn_string += (
                                "latitude value does not "
                                "match file values so they are not used "
                                "for altiude."
                            )
                        else:
                            warn_string += (
                                "longitude value does not "
                                "match file values so they are not used "
                                "for altiude."
                            )

                        warnings.warn(warn_string)

                else:
                    # This does not give a valid telescope_location. Instead
                    # calculate it from the file lat/lon and sea level for altiude
                    self.telescope_location_lat_lon_alt = (latitude, longitude, 0)
                    warn_string = (
                        "Telescope location is set at sealevel at "
                        "the file lat/lon coordinates. Antenna "
                        "positions are present, but the mean "
                        "antenna "
                    )

                    warn_string += (
                        "position does not give a "
                        "telescope_location on the surface of the "
                        "earth."
                    )
                    if absolute_positions:
                        rel_ecef_antpos = ecef_antpos - self.telescope_location
                    else:
                        warn_string += (
                            " Antenna positions do not appear to be "
                            "on the surface of the earth and will be treated "
                            "as relative."
                        )
                        rel_ecef_antpos = ecef_antpos

                    warnings.warn(warn_string)

            if self.Nants_telescope is not None:
                # in this case there is an antnums variable
                # (meaning that the file was written with pyuvdata), so we'll use it
                if nants == self.Nants_telescope:
                    # no inflation, so just use the positions
                    self.antenna_positions = rel_ecef_antpos
                else:
                    # there is some inflation, just use the ones that appear in antnums
                    self.antenna_positions = np.zeros(
                        (self.Nants_telescope, 3), dtype=antpos.dtype
                    )
                    for ai, num in enumerate(self.antenna_numbers):
                        self.antenna_positions[ai, :] = rel_ecef_antpos[num, :]
            else:
                # there is no antnums variable (meaning that this file was not
                # written by pyuvdata), so we test for antennas with non-zero
                # positions and/or that appear in the visibility data
                # (meaning that they have entries in ant_1_array or ant_2_array)
                antpos_length = np.sqrt(np.sum(np.abs(antpos) ** 2, axis=1))
                good_antpos = np.where(antpos_length > 0)[0]
                # take the union of the antennas with good positions (good_antpos)
                # and the antennas that have visisbilities (sorted_unique_ants)
                # if there are antennas with visibilities but zeroed positions
                # we issue a warning below
                if sorted_unique_ants is not None:
                    ants_use = set(good_antpos).union(sorted_unique_ants)
                else:
                    ants_use = set(good_antpos)
                # ants_use are the antennas we'll keep track of in the UVData
                # object, so they dictate Nants_telescope
                self.Nants_telescope = len(ants_use)
                self.antenna_numbers = np.array(list(ants_use))
                self.antenna_positions = np.zeros(
                    (self.Nants_telescope, 3), dtype=rel_ecef_antpos.dtype
                )
                for ai, num in enumerate(self.antenna_numbers):
                    if antpos_length[num] == 0:
                        warnings.warn(
                            "antenna number {n} has visibilities "
                            "associated with it, but it has a position"
                            " of (0,0,0)".format(n=num)
                        )
                    else:
                        # leave bad locations as zeros to make them obvious
                        self.antenna_positions[ai, :] = rel_ecef_antpos[num, :]

        except (KeyError):
            # there is no antpos variable
            warnings.warn("Antenna positions are not present in the file.")
            self.antenna_positions = None

        if self.antenna_numbers is None:
            # there are no antenna_numbers or antenna_positions, so just use
            # the antennas present in the visibilities
            # (Nants_data will therefore match Nants_telescope)
            if sorted_unique_ants is not None:
                self.antenna_numbers = np.array(sorted_unique_ants)
                self.Nants_telescope = len(self.antenna_numbers)

        # antenna names is a foreign concept in miriad but required in other formats.
        try:
            # Here we deal with the way pyuvdata tacks it on to keep the
            # name information if we have it:
            # make it into one long comma-separated string
            ant_name_var = uv["antnames"]
            ant_name_str = ant_name_var.replace("\x00", "")
            ant_name_list = ant_name_str[1:-1].split(", ")
            self.antenna_names = ant_name_list
        except (KeyError):
            self.antenna_names = self.antenna_numbers.astype(str).tolist()

        # check for antenna diameters
        try:
            self.antenna_diameters = uv["antdiam"]
        except (KeyError):
            # backwards compatibility for when keyword was 'diameter'
            try:
                self.antenna_diameters = uv["diameter"]
                # if we find it, we need to remove it from extra_keywords to
                # keep from writing it out
                self.extra_keywords.pop("diameter")
            except (KeyError):
                pass
        if self.antenna_diameters is not None:
            self.antenna_diameters = self.antenna_diameters * np.ones(
                self.Nants_telescope, dtype=np.float
            )
