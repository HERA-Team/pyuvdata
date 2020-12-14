# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Primary container for radio interferometer flag manipulation."""
import numpy as np
import os
import warnings
import h5py
import pathlib

from ..uvbase import UVBase
from .. import parameter as uvp
from ..uvdata import UVData
from ..uvcal import UVCal
from .. import utils as uvutils
from .. import telescopes as uvtel

__all__ = ["UVFlag", "flags2waterfall", "and_rows_cols", "lst_from_uv"]


def and_rows_cols(waterfall):
    """Perform logical and over rows and cols of a waterfall.

    For a 2D flag waterfall, flag pixels only if fully flagged along
    time and/or frequency

    Parameters
    ----------
    waterfall : 2D boolean array of shape (Ntimes, Nfreqs)

    Returns
    -------
    wf : 2D array
        A 2D array (size same as input) where only times/integrations
        that were fully flagged are flagged.

    """
    wf = np.zeros_like(waterfall, dtype=np.bool)
    Ntimes, Nfreqs = waterfall.shape
    wf[:, (np.sum(waterfall, axis=0) / Ntimes) == 1] = True
    wf[(np.sum(waterfall, axis=1) / Nfreqs) == 1] = True
    return wf


def lst_from_uv(uv):
    """Calculate the lst_array for a UVData or UVCal object.

    Parameters
    ----------
    uv : a UVData or UVCal object.
        Object from which lsts are calculated

    Returns
    -------
    lst_array: array of float
        lst_array corresponding to time_array and at telescope location.
        Units are radian.

    """
    if not isinstance(uv, (UVCal, UVData)):
        raise ValueError(
            "Function lst_from_uv can only operate on " "UVCal or UVData object."
        )

    tel = uvtel.get_telescope(uv.telescope_name)
    lat, lon, alt = tel.telescope_location_lat_lon_alt_degrees
    lst_array = uvutils.get_lst_for_time(uv.time_array, lat, lon, alt)
    return lst_array


def flags2waterfall(uv, flag_array=None, keep_pol=False):
    """Convert a flag array to a 2D waterfall of dimensions (Ntimes, Nfreqs).

    Averages over baselines and polarizations (in the case of visibility data),
    or antennas and jones parameters (in case of calibrationd data).

    Parameters
    ----------
    uv : A UVData or UVCal object
        Object defines the times and frequencies, and supplies the
        flag_array to convert (if flag_array not specified)
    flag_array :  Optional,
        flag array to convert instead of uv.flag_array.
        Must have same dimensions as uv.flag_array.
    keep_pol : bool
        Option to keep the polarization axis intact.

    Returns
    -------
    waterfall : 2D array or 3D array
        Waterfall of averaged flags, for example fraction of baselines
        which are flagged for every time and frequency (in case of UVData input)
        Size is (Ntimes, Nfreqs) or (Ntimes, Nfreqs, Npols).

    """
    if not isinstance(uv, (UVData, UVCal)):
        raise ValueError(
            "flags2waterfall() requires a UVData or UVCal object as "
            "the first argument."
        )
    if flag_array is None:
        flag_array = uv.flag_array
    if uv.flag_array.shape != flag_array.shape:
        raise ValueError("Flag array must align with UVData or UVCal object.")

    if isinstance(uv, UVCal):
        if keep_pol:
            waterfall = np.swapaxes(np.mean(flag_array, axis=(0, 1)), 0, 1)
        else:
            waterfall = np.mean(flag_array, axis=(0, 1, 4)).T
    else:
        if keep_pol:
            waterfall = np.zeros((uv.Ntimes, uv.Nfreqs, uv.Npols))
            for i, t in enumerate(np.unique(uv.time_array)):
                waterfall[i, :] = np.mean(
                    flag_array[uv.time_array == t, 0, :, :], axis=0
                )
        else:
            waterfall = np.zeros((uv.Ntimes, uv.Nfreqs))
            for i, t in enumerate(np.unique(uv.time_array)):
                waterfall[i, :] = np.mean(
                    flag_array[uv.time_array == t, 0, :, :], axis=(0, 2)
                )

    return waterfall


class UVFlag(UVBase):
    """Object to handle flag arrays and waterfalls for interferometric datasets.

    Supports reading/writing, and stores all relevant information to combine
    flags and apply to data.
    Initialization of the UVFlag object requires some parameters. Metadata is
    copied from indata object. If indata is subclass of UVData or UVCal,
    the weights_array will be set to all ones.
    Lists or tuples are iterated through, treating each entry with an
    individual UVFlag init.

    Parameters
    ----------
    indata : UVData, UVCal, str, pathlib.Path, list of compatible combination
        Input to initialize UVFlag object. If str, assumed to be path to previously
        saved UVFlag object. UVData and UVCal objects cannot be directly combined,
        unless waterfall is True.
    mode : {"metric", "flag"}, optional
        The mode determines whether the object has a floating point metric_array
        or a boolean flag_array.
    copy_flags : bool, optional
        Whether to copy flags from indata to new UVFlag object
    waterfall : bool, optional
        Whether to immediately initialize as a waterfall object, with flag/metric
        axes: time, frequency, polarization.
    history : str, optional
        History string to attach to object.
    extra_keywords : dict, optional
        A dictionary of metadata values not explicitly specified by another
        parameter.
    label: str, optional
        String used for labeling the object (e.g. 'FM').
    run_check : bool
        Option to check for the existence and proper shapes of parameters
        after creating UVFlag object.
    check_extra : bool
        Option to check optional parameters as well as required ones (the
        default is True, meaning the optional parameters will be checked).
    run_check_acceptability : bool
        Option to check acceptable range of the values of parameters after
        creating UVFlag object.

    Attributes
    ----------
     UVParameter objects :
        For full list see the UVFlag Parameters Documentation.
        (https://pyuvdata.readthedocs.io/en/latest/uvflag_parameters.html)
        Some are always required, some are required for certain phase_types
        and others are always optional.


    """

    def __init__(
        self,
        indata=None,
        mode="metric",
        copy_flags=False,
        waterfall=False,
        history="",
        label="",
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """Initialize the object."""
        # standard angle tolerance: 10 mas in radians.
        # Should perhaps be decreased to 1 mas in the future
        radian_tol = 10 * 2 * np.pi * 1e-3 / (60.0 * 60.0 * 360.0)

        desc = (
            "The mode determines whether the object has a "
            "floating point metric_array or a boolean flag_array. "
            'Options: {"metric", "flag"}. Default is "metric".'
        )
        self._mode = uvp.UVParameter(
            "mode",
            description=desc,
            form="str",
            expected_type=str,
            acceptable_vals=["metric", "flag"],
        )

        desc = (
            "String used for labeling the object (e.g. 'FM'). "
            "Default is empty string."
        )
        self._label = uvp.UVParameter(
            "label", description=desc, form="str", expected_type=str
        )

        desc = (
            "The type of object defines the form of some arrays "
            " and also how metrics/flags are combined. "
            'Accepted types:"waterfall", "baseline", "antenna"'
        )
        self._type = uvp.UVParameter(
            "type",
            description=desc,
            form="str",
            expected_type=str,
            acceptable_vals=["antenna", "baseline", "waterfall"],
        )

        self._Ntimes = uvp.UVParameter(
            "Ntimes", description="Number of times", expected_type=int
        )
        desc = "Number of baselines. " 'Only Required for "baseline" type objects.'
        self._Nbls = uvp.UVParameter(
            "Nbls", description=desc, expected_type=int, required=False
        )
        self._Nblts = uvp.UVParameter(
            "Nblts",
            description="Number of baseline-times "
            "(i.e. number of spectra). Not necessarily "
            "equal to Nbls * Ntimes",
            expected_type=int,
        )
        self._Nspws = uvp.UVParameter(
            "Nspws",
            description="Number of spectral windows "
            "(ie non-contiguous spectral chunks). "
            "More than one spectral window is not "
            "currently supported.",
            expected_type=int,
            required=False,
        )
        self._Nfreqs = uvp.UVParameter(
            "Nfreqs", description="Number of frequency channels", expected_type=int
        )
        self._Npols = uvp.UVParameter(
            "Npols", description="Number of polarizations", expected_type=int
        )

        desc = (
            "Floating point metric information, only availble in metric mode. "
            "shape (Nblts, Nspws, Nfreq, Npols)."
        )
        self._metric_array = uvp.UVParameter(
            "metric_array",
            description=desc,
            form=("Nblts", "Nspws", "Nfreqs", "Npols"),
            expected_type=np.float,
            required=False,
        )

        desc = (
            "Boolean flag, True is flagged, only availble in flag mode. "
            "shape (Nblts, Nspws, Nfreq, Npols)."
        )
        self._flag_array = uvp.UVParameter(
            "flag_array",
            description=desc,
            form=("Nblts", "Nspws", "Nfreqs", "Npols"),
            expected_type=np.bool,
            required=False,
        )

        desc = "Floating point weight information, shape (Nblts, Nspws, Nfreq, Npols)."
        self._weights_array = uvp.UVParameter(
            "weights_array",
            description=desc,
            form=("Nblts", "Nspws", "Nfreqs", "Npols"),
            expected_type=np.float,
        )

        desc = (
            "Floating point weight information about sum of squares of weights"
            " when weighted data converted from baseline to waterfall  mode."
        )
        self._weights_square_array = uvp.UVParameter(
            "weights_square_array",
            description=desc,
            form=("Nblts", "Nspws", "Nfreqs", "Npols"),
            expected_type=np.float,
            required=False,
        )

        desc = (
            "Array of times, center of integration, shape (Nblts), " "units Julian Date"
        )
        self._time_array = uvp.UVParameter(
            "time_array",
            description=desc,
            form=("Nblts",),
            expected_type=np.float,
            tols=1e-3 / (60.0 * 60.0 * 24.0),
        )  # 1 ms in days

        desc = "Array of lsts, center of integration, shape (Nblts), " "units radians"
        self._lst_array = uvp.UVParameter(
            "lst_array",
            description=desc,
            form=("Nblts",),
            expected_type=np.float,
            tols=radian_tol,
        )

        desc = (
            "Array of first antenna indices, shape (Nblts). "
            'Only available for "baseline" type objects. '
            "type = int, 0 indexed"
        )
        self._ant_1_array = uvp.UVParameter(
            "ant_1_array", description=desc, expected_type=int, form=("Nblts",)
        )
        desc = (
            "Array of second antenna indices, shape (Nblts). "
            'Only available for "baseline" type objects. '
            "type = int, 0 indexed"
        )
        self._ant_2_array = uvp.UVParameter(
            "ant_2_array", description=desc, expected_type=int, form=("Nblts",)
        )

        desc = (
            "Array of antenna numbers, shape (Nants_data), "
            'Only available for "antenna" type objects. '
            "type = int, 0 indexed"
        )
        self._ant_array = uvp.UVParameter(
            "ant_array", description=desc, expected_type=int, form=("Nants_data",)
        )

        desc = (
            "Array of baseline indices, shape (Nblts). "
            'Only available for "baseline" type objects. '
            "type = int; baseline = 2048 * (ant1+1) + (ant2+1) + 2^16"
        )
        self._baseline_array = uvp.UVParameter(
            "baseline_array", description=desc, expected_type=int, form=("Nblts",)
        )

        desc = (
            "Array of frequencies, center of the channel, "
            "shape (Nspws, Nfreqs), units Hz"
        )
        self._freq_array = uvp.UVParameter(
            "freq_array",
            description=desc,
            form=("Nspws", "Nfreqs"),
            expected_type=np.float,
            tols=1e-3,
        )  # mHz

        desc = (
            "Array of polarization integers, shape (Npols). "
            "AIPS Memo 117 says: pseudo-stokes 1:4 (pI, pQ, pU, pV);  "
            "circular -1:-4 (RR, LL, RL, LR); linear -5:-8 (XX, YY, XY, YX). "
            "NOTE: AIPS Memo 117 actually calls the pseudo-Stokes polarizations "
            '"Stokes", but this is inaccurate as visibilities cannot be in '
            "true Stokes polarizations for physical antennas. We adopt the "
            "term pseudo-Stokes to refer to linear combinations of instrumental "
            "visibility polarizations (e.g. pI = xx + yy)."
        )
        self._polarization_array = uvp.UVParameter(
            "polarization_array",
            description=desc,
            expected_type=int,
            acceptable_vals=list(np.arange(-8, 0)) + list(np.arange(1, 5)),
            form=("Npols",),
        )

        self._history = uvp.UVParameter(
            "history",
            description="String of history, units English",
            form="str",
            expected_type=str,
        )

        desc = (
            "Any user supplied extra keywords, type=dict."
            "Use the special key 'comment' for long multi-line string comments."
            "Default is an empty dictionary."
        )
        self._extra_keywords = uvp.UVParameter(
            "extra_keywords",
            required=False,
            description=desc,
            value={},
            spoof_val={},
            expected_type=dict,
        )

        # ---antenna information ---
        desc = (
            "Number of antennas in the array. "
            'Only available for "baseline" type objects. '
            "May be larger than the number of antennas with data."
        )
        self._Nants_telescope = uvp.UVParameter(
            "Nants_telescope", description=desc, expected_type=int, required=False
        )
        desc = (
            "Number of antennas with data present. "
            'Only available for "baseline" or "antenna" type objects.'
            "May be smaller than the number of antennas in the array"
        )
        self._Nants_data = uvp.UVParameter(
            "Nants_data", description=desc, expected_type=int, required=True
        )
        #  --extra information ---
        desc = (
            "Orientation of the physical dipole corresponding to what is "
            'labelled as the x polarization. Options are "east" '
            '(indicating east/west orientation) and "north" (indicating '
            "north/south orientation)"
        )
        self._x_orientation = uvp.UVParameter(
            "x_orientation",
            description=desc,
            required=False,
            expected_type=str,
            acceptable_vals=["east", "north"],
        )

        # initialize the underlying UVBase properties
        super(UVFlag, self).__init__()

        self.history = ""  # Added to at the end

        self.label = ""  # Added to at the end
        if isinstance(indata, (list, tuple)):
            self.__init__(
                indata[0],
                mode=mode,
                copy_flags=copy_flags,
                waterfall=waterfall,
                history=history,
                label=label,
                run_check=run_check,
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
            )
            if len(indata) > 1:
                for i in indata[1:]:
                    fobj = UVFlag(
                        i,
                        mode=mode,
                        copy_flags=copy_flags,
                        waterfall=waterfall,
                        history=history,
                        run_check=run_check,
                        check_extra=check_extra,
                        run_check_acceptability=run_check_acceptability,
                    )
                    self.__add__(
                        fobj,
                        run_check=run_check,
                        inplace=True,
                        check_extra=check_extra,
                        run_check_acceptability=run_check_acceptability,
                    )
                del fobj

        elif issubclass(indata.__class__, (str, pathlib.Path)):
            # Given a path, read indata
            self.read(
                indata,
                history,
                run_check=run_check,
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
            )
        elif issubclass(indata.__class__, UVData):
            self.from_uvdata(
                indata,
                mode=mode,
                copy_flags=copy_flags,
                waterfall=waterfall,
                history=history,
                label=label,
                run_check=run_check,
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
            )

        elif issubclass(indata.__class__, UVCal):
            self.from_uvcal(
                indata,
                mode=mode,
                copy_flags=copy_flags,
                waterfall=waterfall,
                history=history,
                label=label,
                run_check=run_check,
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
            )

        elif indata is not None:
            raise ValueError(
                "input to UVFlag.__init__ must be one of: "
                "list, tuple, string, pathlib.Path, UVData, or UVCal."
            )

    @property
    def _data_params(self):
        """List of strings giving the data-like parameters."""
        if not hasattr(self, "mode") or self.mode is None:
            return None
        elif self.mode == "flag":
            return ["flag_array"]
        elif self.mode == "metric":
            if self.weights_square_array is None:
                return ["metric_array", "weights_array"]
            else:
                return ["metric_array", "weights_array", "weights_square_array"]
        else:
            raise ValueError(
                "Invalid mode. Mode must be one of "
                + ", ".join(["{}"] * len(self._mode.acceptable_vals)).format(
                    *self._mode.acceptable_vals
                )
            )

    @property
    def data_like_parameters(self):
        """Return iterator of defined parameters which are data-like."""
        for key in self._data_params:
            if hasattr(self, key):
                yield getattr(self, key)

    @property
    def pol_collapsed(self):
        """Determine if this object has had pols collapsed."""
        if not hasattr(self, "polarization_array") or self.polarization_array is None:
            return False
        elif isinstance(self.polarization_array.item(0), str):
            return True
        else:
            return False

    def _check_pol_state(self):
        if self.pol_collapsed:
            # collapsed pol objects have a different type for
            # the polarization array.
            self._polarization_array.expected_type = str
            self._polarization_array.acceptable_vals = None
        else:
            self._polarization_array.expected_type = int
            self._polarization_array.acceptable_vals = list(np.arange(-8, 0)) + list(
                np.arange(1, 5)
            )

    def _set_mode_flag(self):
        """Set the mode and required parameters consistent with a flag object."""
        self.mode = "flag"
        self._flag_array.required = True
        self._metric_array.required = False
        self._weights_array.required = False
        if self.weights_square_array is not None:
            self.weights_square_array = None

        return

    def _set_mode_metric(self):
        """Set the mode and required parameters consistent with a metric object."""
        self.mode = "metric"
        self._flag_array.required = False
        self._metric_array.required = True
        self._weights_array.required = True

        if self.weights_array is None and self.metric_array is not None:
            self.weights_array = np.ones_like(self.metric_array, dtype=float)

        return

    def _set_type_antenna(self):
        """Set the type and required propertis consistent with an antenna object."""
        self.type = "antenna"
        self._ant_array.required = True
        self._baseline_array.required = False
        self._ant_1_array.required = False
        self._ant_2_array.required = False
        self._Nants_telescope.required = False
        self._Nants_data.required = True
        self._Nbls.required = False
        self._Nspws.required = True
        self._Nblts.required = False

        desc = (
            "Floating point metric information, "
            "has shape (Nants_data, Nspws, Nfreqs, Ntimes, Npols)."
        )
        self._metric_array.desc = desc
        self._metric_array.form = ("Nants_data", "Nspws", "Nfreqs", "Ntimes", "Npols")

        desc = (
            "Boolean flag, True is flagged, "
            "has shape (Nants_data, Nspws, Nfreqs, Ntimes, Npols)."
        )
        self._flag_array.desc = desc
        self._flag_array.form = ("Nants_data", "Nspws", "Nfreqs", "Ntimes", "Npols")

        desc = (
            "Floating point weight information, "
            "has shape (Nants_data, Nspws, Nfreqs, Ntimes, Npols)."
        )
        self._weights_array.desc = desc
        self._weights_array.form = ("Nants_data", "Nspws", "Nfreqs", "Ntimes", "Npols")

        desc = (
            "Array of unique times, center of integration, shape (Ntimes), "
            "units Julian Date"
        )
        self._time_array.form = ("Ntimes",)

        desc = (
            "Array of unique lsts, center of integration, shape (Ntimes), "
            "units radians"
        )
        self._lst_array.form = ("Ntimes",)

        desc = (
            "Array of frequencies, center of the channel, "
            "shape (Nspws, Nfreqs), units Hz"
        )
        self._freq_array.form = ("Nspws", "Nfreqs")

    def _set_type_baseline(self):
        """Set the type and required propertis consistent with a baseline object."""
        self.type = "baseline"
        self._ant_array.required = False
        self._baseline_array.required = True
        self._ant_1_array.required = True
        self._ant_2_array.required = True
        self._Nants_telescope.required = True
        self._Nants_data.required = True
        self._Nbls.required = True
        self._Nblts.required = True
        self._Nspws.required = True

        if self.time_array is not None:
            self.Nblts = len(self.time_array)

        desc = "Floating point metric information, shape (Nblts, Nspws, Nfreqs, Npols)."
        self._metric_array.desc = desc
        self._metric_array.form = ("Nblts", "Nspws", "Nfreqs", "Npols")

        desc = "Boolean flag, True is flagged, shape (Nblts, Nfreqs, Npols)"
        self._flag_array.desc = desc
        self._flag_array.form = ("Nblts", "Nspws", "Nfreqs", "Npols")

        desc = "Floating point weight information, has shape (Nblts, Nfreqs, Npols)."
        self._weights_array.desc = desc
        self._weights_array.form = ("Nblts", "Nspws", "Nfreqs", "Npols")

        desc = (
            "Array of unique times, center of integration, shape (Ntimes), "
            "units Julian Date"
        )
        self._time_array.form = ("Nblts",)

        desc = (
            "Array of unique lsts, center of integration, shape (Ntimes), "
            "units radians"
        )
        self._lst_array.form = ("Nblts",)

        desc = (
            "Array of frequencies, center of the channel, "
            "shape (Nspws, Nfreqs), units Hz"
        )
        self._freq_array.form = ("Nspws", "Nfreqs")

    def _set_type_waterfall(self):
        """Set the type and required propertis consistent with a waterfall object."""
        self.type = "waterfall"
        self._ant_array.required = False
        self._baseline_array.required = False
        self._ant_1_array.required = False
        self._ant_2_array.required = False
        self._Nants_telescope.required = False
        self._Nants_data.required = False
        self._Nbls.required = False
        self._Nspws.required = False
        self._Nblts.required = False

        desc = "Floating point metric information, shape (Ntimes, Nfreqs, Npols)."
        self._metric_array.desc = desc
        self._metric_array.form = ("Ntimes", "Nfreqs", "Npols")

        desc = "Boolean flag, True is flagged, shape (Ntimes, Nfreqs, Npols)"
        self._flag_array.desc = desc
        self._flag_array.form = ("Ntimes", "Nfreqs", "Npols")

        desc = "Floating point weight information, has shape (Ntimes, Nfreqs, Npols)."
        self._weights_array.desc = desc
        self._weights_array.form = ("Ntimes", "Nfreqs", "Npols")

        desc = (
            "Floating point weight information about sum of squares of weights"
            " when weighted data converted from baseline to waterfall  mode."
            " Has shape (Ntimes, Nfreqs, Npols)."
        )
        self._weights_square_array.desc = desc
        self._weights_square_array.form = ("Ntimes", "Nfreqs", "Npols")

        desc = (
            "Array of unique times, center of integration, shape (Ntimes), "
            "units Julian Date"
        )
        self._time_array.form = ("Ntimes",)

        desc = (
            "Array of unique lsts, center of integration, shape (Ntimes), "
            "units radians"
        )
        self._lst_array.form = ("Ntimes",)

        desc = (
            "Array of frequencies, center of the channel, " "shape (Nfreqs), units Hz"
        )
        self._freq_array.form = ("Nfreqs",)

    def clear_unused_attributes(self):
        """Remove unused attributes.

        Useful when changing type or mode or to save memory.
        Will set all non-required attributes to None, except x_orientation,
        extra_keywords and weights_square_array.

        """
        for p in self:
            attr = getattr(self, p)
            if (
                not attr.required
                and attr.value is not None
                and attr.name != "x_orientation"
                and attr.name != "weights_square_array"
                and attr.name != "extra_keywords"
            ):
                attr.value = None
                setattr(self, p, attr)

    def __eq__(self, other, check_history=True, check_extra=True):
        """Check Equality of two UVFlag objects.

        Parameters
        ----------
        other: UVFlag
            object to check against
        check_history : bool
            Include the history keyword when comparing UVFlag objects.
        check_extra : bool
            Include non-required parameters when comparing UVFlag objects.

        """
        if check_history:
            return super(UVFlag, self).__eq__(other, check_extra=check_extra)

        else:
            # initial check that the classes are the same
            # then strip the histories
            if isinstance(other, self.__class__):
                _h1 = self.history
                self.history = None

                _h2 = other.history
                other.history = None

                truth = super(UVFlag, self).__eq__(other, check_extra=check_extra)

                self.history = _h1
                other.history = _h2

                return truth
            else:
                print("Classes do not match")
                return False

    def __ne__(self, other, check_history=True, check_extra=True):
        """Not Equal."""
        return not self.__eq__(
            other, check_history=check_history, check_extra=check_extra
        )

    def antpair2ind(self, ant1, ant2):
        """Get blt indices for given (ordered) antenna pair.

        Parameters
        ----------
        ant1 : int or array_like of int
            Number of the first antenna
        ant2 : int or array_like of int
            Number of the second antenna

        Returns
        -------
        int or array_like of int
            baseline number(s) corresponding to the input antenna number

        """
        if self.type != "baseline":
            raise ValueError(
                "UVFlag object of type " + self.type + " does not "
                "contain antenna pairs to index."
            )
        return np.where((self.ant_1_array == ant1) & (self.ant_2_array == ant2))[0]

    def baseline_to_antnums(self, baseline):
        """Get the antenna numbers corresponding to a given baseline number.

        Parameters
        ----------
        baseline : int
            baseline number

        Returns
        -------
        tuple
            Antenna numbers corresponding to baseline.

        """
        assert self.type == "baseline", 'Must be "baseline" type UVFlag object.'
        return uvutils.baseline_to_antnums(baseline, self.Nants_telescope)

    def get_baseline_nums(self):
        """Return numpy array of unique baseline numbers in data."""
        assert self.type == "baseline", 'Must be "baseline" type UVFlag object.'
        return np.unique(self.baseline_array)

    def get_antpairs(self):
        """Return list of unique antpair tuples (ant1, ant2) in data."""
        assert self.type == "baseline", 'Must be "baseline" type UVFlag object.'
        return [self.baseline_to_antnums(bl) for bl in self.get_baseline_nums()]

    def get_ants(self):
        """
        Get the unique antennas that have data associated with them.

        Returns
        -------
        ndarray of int
            Array of unique antennas with data associated with them.
        """
        if self.type == "baseline":
            return np.unique(np.append(self.ant_1_array, self.ant_2_array))
        elif self.type == "antenna":
            return np.unique(self.ant_array)
        elif self.type == "waterfall":
            raise ValueError("A waterfall type UVFlag object has no sense of antennas.")

    def get_pols(self):
        """
        Get the polarizations in the data.

        Returns
        -------
        list of str
            list of polarizations (as strings) in the data.
        """
        return uvutils.polnum2str(
            self.polarization_array, x_orientation=self.x_orientation
        )

    def parse_ants(self, ant_str, print_toggle=False):
        """
        Get antpair and polarization from parsing an aipy-style ant string.

        Used to support the the select function.
        This function is only useable when the UVFlag type is 'baseline'.
        Generates two lists of antenna pair tuples and polarization indices based
        on parsing of the string ant_str.  If no valid polarizations (pseudo-Stokes
        params, or combinations of [lr] or [xy]) or antenna numbers are found in
        ant_str, ant_pairs_nums and polarizations are returned as None.

        Parameters
        ----------
        ant_str : str
            String containing antenna information to parse. Can be 'all',
            'auto', 'cross', or combinations of antenna numbers and polarization
            indicators 'l' and 'r' or 'x' and 'y'.  Minus signs can also be used
            in front of an antenna number or baseline to exclude it from being
            output in ant_pairs_nums. If ant_str has a minus sign as the first
            character, 'all,' will be appended to the beginning of the string.
            See the tutorial for examples of valid strings and their behavior.
        print_toggle : bool
            Boolean for printing parsed baselines for a visual user check.

        Returns
        -------
        ant_pairs_nums : list of tuples of int or None
            List of tuples containing the parsed pairs of antenna numbers, or
            None if ant_str is 'all' or a pseudo-Stokes polarizations.
        polarizations : list of int or None
            List of desired polarizations or None if ant_str does not contain a
            polarization specification.

        """
        if self.type != "baseline":
            raise ValueError(
                "UVFlag objects can only call 'parse_ants' function "
                "if type is 'baseline'."
            )
        return uvutils.parse_ants(
            self,
            ant_str=ant_str,
            print_toggle=print_toggle,
            x_orientation=self.x_orientation,
        )

    def collapse_pol(
        self,
        method="quadmean",
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """Collapse the polarization axis using a given method.

        If the original UVFlag object has more than one polarization,
        the resulting polarization_array will be a single element array with a
        comma separated string encoding the original polarizations.

        Parameters
        ----------
        method : str, {"quadmean", "absmean", "mean", "or", "and"}
            How to collapse the dimension(s).
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after collapsing polarizations.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            collapsing polarizations.

        """
        method = method.lower()
        if self.mode == "flag":
            darr = self.flag_array
        else:
            darr = self.metric_array
        if len(self.polarization_array) > 1:
            if self.mode == "metric":
                _weights = self.weights_array
            else:
                _weights = np.ones_like(darr)
            # Collapse pol dimension. But note we retain a polarization axis.
            d, w = uvutils.collapse(
                darr, method, axis=-1, weights=_weights, return_weights=True
            )
            darr = np.expand_dims(d, axis=d.ndim)

            if self.mode == "metric":
                self.weights_array = np.expand_dims(w, axis=w.ndim)

            self.polarization_array = np.array(
                [",".join(map(str, self.polarization_array))], dtype=np.str_
            )

            self.Npols = len(self.polarization_array)
            self._check_pol_state()
        else:
            warnings.warn(
                "Cannot collapse polarization axis when only one pol present."
            )
            return
        if ((method == "or") or (method == "and")) and (self.mode == "flag"):
            self.flag_array = darr
        else:
            self.metric_array = darr
            self._set_mode_metric()
        self.clear_unused_attributes()
        self.history += "Pol axis collapse. "

        if not uvutils._check_history_version(self.history, self.pyuvdata_version_str):
            self.history += self.pyuvdata_version_str

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

    def to_waterfall(
        self,
        method="quadmean",
        keep_pol=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        return_weights_square=False,
    ):
        """Convert an 'antenna' or 'baseline' type object to waterfall.

        Parameters
        ----------
        method : str, {"quadmean", "absmean", "mean", "or", "and"}
            How to collapse the dimension(s).
        keep_pol : bool
            Whether to also collapse the polarization dimension
            If keep_pol is False, and the original UVFlag object has more
            than one polarization, the resulting polarization_array
            will be a single element array with a comma separated string
            encoding the original polarizations.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after converting to waterfall type.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            converting to waterfall type.
        return_weights_square: bool
            Option to compute the sum of the squares of the weights when
            collapsing baseline object to waterfall. Not used if type is not
            baseline to begin with. Fills an optional parameter if so.

        """
        method = method.lower()
        if self.type == "waterfall" and (
            keep_pol or (len(self.polarization_array) == 1)
        ):
            warnings.warn("This object is already a waterfall. Nothing to change.")
            return
        if (not keep_pol) and (len(self.polarization_array) > 1):
            self.collapse_pol(method)

        if self.mode == "flag":
            darr = self.flag_array
        else:
            darr = self.metric_array

        if self.type == "antenna":
            d, w = uvutils.collapse(
                darr,
                method,
                axis=(0, 1),
                weights=self.weights_array,
                return_weights=True,
            )
            darr = np.swapaxes(d, 0, 1)
            if self.mode == "metric":
                self.weights_array = np.swapaxes(w, 0, 1)
        elif self.type == "baseline":
            Nt = len(np.unique(self.time_array))
            Nf = len(self.freq_array[0, :])
            Np = len(self.polarization_array)
            d = np.zeros((Nt, Nf, Np))
            w = np.zeros((Nt, Nf, Np))
            if return_weights_square:
                ws = np.zeros((Nt, Nf, Np))
            for i, t in enumerate(np.unique(self.time_array)):
                ind = self.time_array == t
                if self.mode == "metric":
                    _weights = self.weights_array[ind, :, :]
                else:
                    _weights = np.ones_like(darr[ind, :, :], dtype=float)
                if return_weights_square:
                    d[i, :, :], w[i, :, :], ws[i, :, :] = uvutils.collapse(
                        darr[ind, :, :],
                        method,
                        axis=0,
                        weights=_weights,
                        return_weights=True,
                        return_weights_square=return_weights_square,
                    )
                else:
                    d[i, :, :], w[i, :, :] = uvutils.collapse(
                        darr[ind, :, :],
                        method,
                        axis=0,
                        weights=_weights,
                        return_weights=True,
                        return_weights_square=return_weights_square,
                    )
            darr = d
            if self.mode == "metric":
                self.weights_array = w
                if return_weights_square:
                    self.weights_square_array = ws
            self.time_array, ri = np.unique(self.time_array, return_index=True)
            self.lst_array = self.lst_array[ri]
        if ((method == "or") or (method == "and")) and (self.mode == "flag"):
            # If using a boolean operation (AND/OR) and in flag mode, stay in flag
            # flags should be bool, but somehow it is cast as float64
            # is reacasting to bool like this best?
            self.flag_array = darr.astype(bool)
        else:
            # Otherwise change to (or stay in) metric
            self.metric_array = darr
            self._set_mode_metric()
        self.freq_array = self.freq_array.flatten()
        self.Nspws = None
        self._set_type_waterfall()
        self.history += 'Collapsed to type "waterfall". '  # + self.pyuvdata_version_str

        if not uvutils._check_history_version(self.history, self.pyuvdata_version_str):
            self.history += self.pyuvdata_version_str

        self.clear_unused_attributes()
        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

    def to_baseline(
        self,
        uv,
        force_pol=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """Convert a UVFlag object of type "waterfall" to type "baseline".

        Broadcasts the flag array to all baselines.
        This function does NOT apply flags to uv.

        Parameters
        ----------
        uv : UVData or UVFlag object
            Objcet with type baseline to match.
        force_pol : bool
            If True, will use 1 pol to broadcast to any other pol.
            Otherwise, will require polarizations match.
            For example, this keyword is useful if one flags on all
            pols combined, and wants to broadcast back to individual pols.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after converting to baseline type.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            converting to baseline type.

        """
        if self.type == "baseline":
            return
        if not (
            issubclass(uv.__class__, UVData)
            or (isinstance(uv, UVFlag) and uv.type == "baseline")
        ):
            raise ValueError(
                "Must pass in UVData object or UVFlag object of type "
                '"baseline" to match.'
            )

        # Deal with polarization
        if force_pol and self.polarization_array.size == 1:
            # Use single pol for all pols, regardless
            self.polarization_array = uv.polarization_array
            # Broadcast arrays
            if self.mode == "flag":
                self.flag_array = self.flag_array.repeat(
                    self.polarization_array.size, axis=-1
                )
            else:
                self.metric_array = self.metric_array.repeat(
                    self.polarization_array.size, axis=-1
                )
                self.weights_array = self.weights_array.repeat(
                    self.polarization_array.size, axis=-1
                )
            self.Npols = len(self.polarization_array)
            self._check_pol_state()

        # Now the pol axes should match regardless of force_pol.
        if not np.array_equal(uv.polarization_array, self.polarization_array):
            if self.polarization_array.size == 1:
                raise ValueError(
                    "Polarizations do not match. Try keyword force_pol"
                    + " if you wish to broadcast to all polarizations."
                )
            else:
                raise ValueError("Polarizations could not be made to match.")
        if self.type == "waterfall":
            # Populate arrays
            if self.mode == "flag":
                if issubclass(uv.__class__, UVData) and uv.future_array_shapes:
                    arr = np.zeros_like(uv.flag_array[:, np.newaxis, :, :])
                else:
                    arr = np.zeros_like(uv.flag_array)
                sarr = self.flag_array
            elif self.mode == "metric":
                if issubclass(uv.__class__, UVData) and uv.future_array_shapes:
                    arr = np.zeros_like(uv.flag_array[:, np.newaxis, :, :], dtype=float)
                    warr = np.zeros_like(
                        uv.flag_array[:, np.newaxis, :, :], dtype=np.float
                    )
                else:
                    arr = np.zeros_like(uv.flag_array, dtype=float)
                    warr = np.zeros_like(uv.flag_array, dtype=np.float)
                sarr = self.metric_array
            for i, t in enumerate(np.unique(self.time_array)):
                ti = np.where(uv.time_array == t)
                arr[ti, :, :, :] = sarr[i, :, :][np.newaxis, np.newaxis, :, :]
                if self.mode == "metric":
                    warr[ti, :, :, :] = self.weights_array[i, :, :][
                        np.newaxis, np.newaxis, :, :
                    ]
            if self.mode == "flag":
                self.flag_array = arr
            elif self.mode == "metric":
                self.metric_array = arr
                self.weights_array = warr

        elif self.type == "antenna":
            if self.mode == "metric":
                raise NotImplementedError(
                    "Cannot currently convert from "
                    "antenna type, metric mode to "
                    "baseline type UVFlag object."
                )
            else:
                ants_data = np.unique(uv.ant_1_array.tolist() + uv.ant_2_array.tolist())
                new_ants = np.setdiff1d(ants_data, self.ant_array)
                if new_ants.size > 0:
                    self.ant_array = np.append(self.ant_array, new_ants).tolist()
                    # make new flags of the same shape but with first axis the
                    # size of the new ants
                    flag_shape = list(self.flag_array.shape)
                    flag_shape[0] = new_ants.size
                    new_flags = np.full(flag_shape, True, dtype=bool)
                    self.flag_array = np.append(self.flag_array, new_flags, axis=0)

                baseline_flags = np.full(
                    (uv.Nblts, self.Nspws, self.Nfreqs, self.Npols), True, dtype=bool
                )

                for t_index, bl in enumerate(uv.baseline_array):
                    uvf_t_index = np.nonzero(uv.time_array[t_index] == self.time_array)[
                        0
                    ]
                    if uvf_t_index.size > 0:
                        # if the time is found in the array
                        # input the or'ed data from each antenna
                        ant1, ant2 = uv.baseline_to_antnums(bl)
                        ant1_index = np.nonzero(np.array(self.ant_array) == ant1)
                        ant2_index = np.nonzero(np.array(self.ant_array) == ant2)
                        or_flag = np.logical_or(
                            self.flag_array[ant1_index, :, :, uvf_t_index, :],
                            self.flag_array[ant2_index, :, :, uvf_t_index, :],
                        )

                        baseline_flags[t_index, :, :, :] = or_flag.copy()

                self.flag_array = baseline_flags

        # Check the frequency array for Nspws, otherwise broadcast to 1,Nfreqs
        self.freq_array = np.atleast_2d(self.freq_array)
        self.Nspws = self.freq_array.shape[0]

        self.baseline_array = uv.baseline_array
        self.Nbls = np.unique(self.baseline_array).size
        self.ant_1_array = uv.ant_1_array
        self.ant_2_array = uv.ant_2_array
        self.Nants_data = int(np.union1d(self.ant_1_array, self.ant_2_array).size)

        self.time_array = uv.time_array
        self.lst_array = uv.lst_array
        self.Nblts = self.time_array.size

        self.Nants_telescope = int(uv.Nants_telescope)
        self._set_type_baseline()
        self.clear_unused_attributes()
        self.history += 'Broadcast to type "baseline". '

        if not uvutils._check_history_version(self.history, self.pyuvdata_version_str):
            self.history += self.pyuvdata_version_str

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

    def to_antenna(
        self,
        uv,
        force_pol=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """Convert a UVFlag object of type "waterfall" to type "antenna".

        Broadcasts the flag array to all antennas.
        This function does NOT apply flags to uv.

        Parameters
        ----------
        uv : UVCal or UVFlag object
            object of type antenna to match.
        force_pol : bool
            If True, will use 1 pol to broadcast to any other pol.
            Otherwise, will require polarizations match.
            For example, this keyword is useful if one flags on all
            pols combined, and wants to broadcast back to individual pols.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after converting to antenna type.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            converting to antenna type.

        """
        if self.type == "antenna":
            return
        if not (
            issubclass(uv.__class__, UVCal)
            or (isinstance(uv, UVFlag) and uv.type == "antenna")
        ):
            raise ValueError(
                "Must pass in UVCal object or UVFlag object of type "
                '"antenna" to match.'
            )
        if self.type != "waterfall":
            raise ValueError(
                'Cannot convert from type "' + self.type + '" to "antenna".'
            )
        # Deal with polarization
        if issubclass(uv.__class__, UVCal):
            polarr = uv.jones_array
        else:
            polarr = uv.polarization_array
        if force_pol and self.polarization_array.size == 1:
            # Use single pol for all pols, regardless
            self.polarization_array = polarr
            # Broadcast arrays
            if self.mode == "flag":
                self.flag_array = self.flag_array.repeat(
                    self.polarization_array.size, axis=-1
                )
            else:
                self.metric_array = self.metric_array.repeat(
                    self.polarization_array.size, axis=-1
                )
                self.weights_array = self.weights_array.repeat(
                    self.polarization_array.size, axis=-1
                )
            self.Npols = len(self.polarization_array)
            self._check_pol_state()

        # Now the pol axes should match regardless of force_pol.
        if not np.array_equal(polarr, self.polarization_array):
            if self.polarization_array.size == 1:
                raise ValueError(
                    "Polarizations do not match. Try keyword force_pol"
                    + "if you wish to broadcast to all polarizations."
                )
            else:
                raise ValueError("Polarizations could not be made to match.")
        # Populate arrays
        if self.mode == "flag":
            self.flag_array = np.swapaxes(self.flag_array, 0, 1)[
                np.newaxis, np.newaxis, :, :, :
            ]
            self.flag_array = self.flag_array.repeat(len(uv.ant_array), axis=0)
        elif self.mode == "metric":
            self.metric_array = np.swapaxes(self.metric_array, 0, 1)[
                np.newaxis, np.newaxis, :, :, :
            ]
            self.metric_array = self.metric_array.repeat(len(uv.ant_array), axis=0)
            self.weights_array = np.swapaxes(self.weights_array, 0, 1)[
                np.newaxis, np.newaxis, :, :, :
            ]
            self.weights_array = self.weights_array.repeat(len(uv.ant_array), axis=0)
        self.ant_array = uv.ant_array
        self.Nants_data = len(uv.ant_array)
        # Check the frequency array for Nspws, otherwise broadcast to 1,Nfreqs
        self.freq_array = np.atleast_2d(self.freq_array)
        self.Nspws = self.freq_array.shape[0]

        self._set_type_antenna()
        self.history += 'Broadcast to type "antenna". '

        if not uvutils._check_history_version(self.history, self.pyuvdata_version_str):
            self.history += self.pyuvdata_version_str

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

    def to_flag(
        self,
        threshold=np.inf,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """Convert to flag mode.

        This function is NOT SMART. Removes metric_array and creates a
        flag_array from a simple threshold on the metric values.

        Parameters
        ----------
        threshold : float
            Metric value over which the corresponding flag is
            set to True. Default is np.inf, which results in flags of all False.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after converting to flag mode.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            converting to flag mode.

        """
        if self.mode == "flag":
            return
        elif self.mode == "metric":
            self.flag_array = np.where(self.metric_array >= threshold, True, False)
            self._set_mode_flag()
        else:
            raise ValueError(
                "Unknown UVFlag mode: " + self.mode + ". Cannot convert to flag."
            )
        self.history += 'Converted to mode "flag". '
        if not uvutils._check_history_version(self.history, self.pyuvdata_version_str):
            self.history += self.pyuvdata_version_str
        self.clear_unused_attributes()

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

    def to_metric(
        self,
        convert_wgts=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """Convert to metric mode.

        This function is NOT SMART. Simply recasts flag_array as float
        and uses this as the metric array.

        Parameters
        ----------
        convert_wgts : bool
            if True convert self.weights_array to ones
            unless a column or row is completely flagged, in which case
            convert those pixels to zero. This is used when reinterpretting
            flags as metrics to calculate flag fraction. Zero weighting
            completely flagged rows/columns prevents those from counting
            against a threshold along the other dimension.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after converting to metric mode.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            converting to metric mode.

        """
        if self.mode == "metric":
            return
        elif self.mode == "flag":
            self.metric_array = self.flag_array.astype(np.float)
            self._set_mode_metric()

            if convert_wgts:
                self.weights_array = np.ones_like(self.weights_array)
                if self.type == "waterfall":
                    for i, pol in enumerate(self.polarization_array):
                        self.weights_array[:, :, i] *= ~and_rows_cols(
                            self.flag_array[:, :, i]
                        )
                elif self.type == "baseline":
                    for i, pol in enumerate(self.polarization_array):
                        for j, ap in enumerate(self.get_antpairs()):
                            inds = self.antpair2ind(*ap)
                            self.weights_array[inds, 0, :, i] *= ~and_rows_cols(
                                self.flag_array[inds, 0, :, i]
                            )
                elif self.type == "antenna":
                    for i, pol in enumerate(self.polarization_array):
                        for j in range(self.weights_array.shape[0]):
                            self.weights_array[j, 0, :, :, i] *= ~and_rows_cols(
                                self.flag_array[j, 0, :, :, i]
                            )
        else:
            raise ValueError(
                "Unknown UVFlag mode: " + self.mode + ". Cannot convert to metric."
            )
        self.history += 'Converted to mode "metric". '

        if not uvutils._check_history_version(self.history, self.pyuvdata_version_str):
            self.history += self.pyuvdata_version_str
        self.clear_unused_attributes()

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

    def __add__(
        self,
        other,
        inplace=False,
        axis="time",
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """Add two UVFlag objects together along a given axis.

        Parameters
        ----------
        other : UVFlag
            object to combine with self.
        axis : str
            Axis along which to combine UVFlag objects.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after combining two objects.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            combining two objects.
        inplace : bool
            Option to perform the select directly on self or return a new UVData
            object with just the selected data.

        Returns
        -------
        uvf : UVFlag
            If inplace==False, return new UVFlag object.

        """
        # Handle in place
        if inplace:
            this = self
        else:
            this = self.copy()

        # Check that objects are compatible
        if not isinstance(other, this.__class__):
            raise ValueError("Only UVFlag objects can be added to a UVFlag object")
        if this.type != other.type:
            raise ValueError(
                "UVFlag object of type " + other.type + " cannot be "
                "added to object of type " + this.type + "."
            )
        if this.mode != other.mode:
            raise ValueError(
                "UVFlag object of mode " + other.mode + " cannot be "
                "added to object of mode " + this.type + "."
            )

        # Simplify axis referencing
        axis = axis.lower()
        type_nums = {"waterfall": 0, "baseline": 1, "antenna": 2}
        axis_nums = {
            "time": [0, 0, 3],
            "baseline": [None, 0, None],
            "antenna": [None, None, 0],
            "frequency": [1, 2, 2],
            "polarization": [2, 3, 4],
            "pol": [2, 3, 4],
            "jones": [2, 3, 4],
        }
        ax = axis_nums[axis][type_nums[self.type]]
        if axis == "time":
            this.time_array = np.concatenate([this.time_array, other.time_array])
            this.lst_array = np.concatenate([this.lst_array, other.lst_array])
            if this.type == "baseline":
                this.baseline_array = np.concatenate(
                    [this.baseline_array, other.baseline_array]
                )
                this.ant_1_array = np.concatenate([this.ant_1_array, other.ant_1_array])
                this.ant_2_array = np.concatenate([this.ant_2_array, other.ant_2_array])
                this.Nants_data = int(
                    np.union1d(this.ant_1_array, this.ant_2_array).size
                )

            this.Ntimes = np.unique(this.time_array).size
            this.Nblts = len(this.time_array)

        elif axis == "baseline":
            if self.type != "baseline":
                raise ValueError(
                    "Flag object of type " + self.type + " cannot be "
                    "concatenated along baseline axis."
                )
            this.time_array = np.concatenate([this.time_array, other.time_array])
            this.lst_array = np.concatenate([this.lst_array, other.lst_array])
            this.baseline_array = np.concatenate(
                [this.baseline_array, other.baseline_array]
            )
            this.ant_1_array = np.concatenate([this.ant_1_array, other.ant_1_array])
            this.ant_2_array = np.concatenate([this.ant_2_array, other.ant_2_array])
            this.Nants_data = int(np.union1d(self.ant_1_array, self.ant_2_array).size)

            this.Nbls = np.unique(this.baseline_array).size
            this.Nblts = len(this.baseline_array)

        elif axis == "antenna":
            if self.type != "antenna":
                raise ValueError(
                    "Flag object of type " + self.type + " cannot be "
                    "concatenated along antenna axis."
                )
            this.ant_array = np.concatenate([this.ant_array, other.ant_array])
            this.Nants_data = len(this.ant_array)
        elif axis == "frequency":
            this.freq_array = np.concatenate(
                [this.freq_array, other.freq_array], axis=-1
            )
            this.Nfreqs = np.unique(this.freq_array.flatten()).size
        elif axis in ["polarization", "pol", "jones"]:
            if this.pol_collapsed:
                raise NotImplementedError(
                    "Two UVFlag objects with their "
                    "polarizations collapsed cannot be "
                    "added along the polarization axis "
                    "at this time."
                )
            this.polarization_array = np.concatenate(
                [this.polarization_array, other.polarization_array]
            )
            this.Npols = len(this.polarization_array)

        for attr in this._data_params:
            # Check that 'other' also has the attribute filled
            if getattr(other, attr) is not None:
                setattr(
                    this,
                    attr,
                    np.concatenate(
                        [getattr(this, attr), getattr(other, attr)], axis=ax
                    ),
                )
            # May 21, 2020 - should only happen for weights_square_array attr
            else:
                raise ValueError(
                    f"{attr} optional parameter is missing from second UVFlag"
                    f" object. To concatenate two {this.mode} objects, they"
                    " must both contain the same optional parameters set."
                )

        this.history += "Data combined along " + axis + " axis. "
        if not uvutils._check_history_version(this.history, this.pyuvdata_version_str):
            this.history += this.pyuvdata_version_str

        this.Ntimes = np.unique(this.time_array).size

        if run_check:
            this.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )
        if not inplace:
            return this

    def __iadd__(
        self,
        other,
        axis="time",
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """In place add.

        Parameters
        ----------
        other : UVFlag
            object to combine with self.
        axis : str
            Axis along which to combine UVFlag objects.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after combining two objects.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            combining two objects.

        """
        self.__add__(
            other,
            inplace=True,
            axis=axis,
            run_check=True,
            check_extra=True,
            run_check_acceptability=True,
        )
        return self

    def __or__(
        self,
        other,
        inplace=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """Combine two UVFlag objects in "flag" mode by "OR"-ing their flags.

        Parameters
        ----------
        other : UVFlag
            object to combine with self.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after combining two objects.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            combining two objects.
        inplace : bool
            Option to perform the select directly on self or return a new UVData
            object with just the selected data.

        Returns
        -------
        uvf : UVFlag
            If inplace==False, return new UVFlag object.

        """
        if (self.mode != "flag") or (other.mode != "flag"):
            raise ValueError(
                'UVFlag object must be in "flag" mode to use "or" function.'
            )

        # Handle in place
        if inplace:
            this = self
        else:
            this = self.copy()
        this.flag_array += other.flag_array
        if other.history not in this.history:
            this.history += "Flags OR'd with: " + other.history

        if not uvutils._check_history_version(this.history, this.pyuvdata_version_str):
            this.history += this.pyuvdata_version_str

        if run_check:
            this.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )
        if not inplace:
            return this

    def __ior__(
        self, other, run_check=True, check_extra=True, run_check_acceptability=True
    ):
        """Perform an inplace logical or.

        Parameters
        ----------
        other : UVFlag
            object to combine with self.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after combining two objects.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            combining two objects.

        """
        self.__or__(
            other,
            inplace=True,
            run_check=True,
            check_extra=True,
            run_check_acceptability=True,
        )
        return self

    def combine_metrics(
        self,
        others,
        method="quadmean",
        inplace=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """Combine metric arrays between different UVFlag objects together.

        Parameters
        ----------
        others : UVFlag or list of UVFlags
            Other UVFlag objects to combine metrics with this one.
        method : str, {"quadmean", "absmean", "mean", "or", "and"}
            Method to combine metrics.
        inplace : bool, optional
            Perform combination in place.

        Returns
        -------
        uvf : UVFlag
            If inplace==False, return new UVFlag object with combined metrics.

        """
        # Ensure others is iterable (in case of single UVFlag object)
        # cannot use uvutils._get_iterable because the object itself is iterable
        if not isinstance(others, (list, tuple, np.ndarray)):
            others = [others]

        if np.any([not isinstance(other, UVFlag) for other in others]):
            raise ValueError('"others" must be UVFlag or list of UVFlag objects')
        if (self.mode != "metric") or np.any(
            [other.mode != "metric" for other in others]
        ):
            raise ValueError(
                'UVFlag object and "others" must be in "metric" mode '
                'to use "add_metrics" function.'
            )
        if inplace:
            this = self
        else:
            this = self.copy()
        method = method.lower()
        darray = np.expand_dims(this.metric_array, 0)
        warray = np.expand_dims(this.weights_array, 0)
        for other in others:
            if this.metric_array.shape != other.metric_array.shape:
                raise ValueError("UVFlag metric array shapes do not match.")
            darray = np.vstack([darray, np.expand_dims(other.metric_array, 0)])
            warray = np.vstack([warray, np.expand_dims(other.weights_array, 0)])
        darray, warray = uvutils.collapse(
            darray, method, weights=warray, axis=0, return_weights=True
        )
        this.metric_array = darray
        this.weights_array = warray
        this.history += "Combined metric arrays. "

        if not uvutils._check_history_version(this.history, this.pyuvdata_version_str):
            this.history += this.pyuvdata_version_str

        if run_check:
            this.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )
        if not inplace:
            return this

    def _select_preprocess(
        self,
        antenna_nums,
        ant_str,
        bls,
        frequencies,
        freq_chans,
        times,
        polarizations,
        blt_inds,
        ant_inds,
    ):
        """Build up blt_inds, freq_inds, pol_inds and history_update_string for select.

        Parameters
        ----------
        antenna_nums : array_like of int, optional
            The antennas numbers to keep in the object (antenna positions and
            names for the removed antennas will be retained unless
            `keep_all_metadata` is False).
        bls : list of tuple, optional
            A list of antenna number tuples (e.g. [(0,1), (3,2)]) or a list of
            baseline 3-tuples (e.g. [(0,1,'xx'), (2,3,'yy')]) specifying baselines
            to keep in the object. For length-2 tuples, the ordering of the numbers
            within the tuple does not matter. For length-3 tuples, the polarization
            string is in the order of the two antennas. If length-3 tuples are
            provided, `polarizations` must be None.
        ant_str : str, optional
            A string containing information about what antenna numbers
            and polarizations to keep in the object.  Can be 'auto', 'cross', 'all',
            or combinations of antenna numbers and polarizations (e.g. '1',
            '1_2', '1x_2y').  See tutorial for more examples of valid strings and
            the behavior of different forms for ant_str.
            If '1x_2y,2y_3y' is passed, both polarizations 'xy' and 'yy' will
            be kept for both baselines (1, 2) and (2, 3) to return a valid
            pyuvdata object.
            An ant_str cannot be passed in addition to any of `antenna_nums`,
            `bls` args or the `polarizations` parameters,
            if it is a ValueError will be raised.
        frequencies : array_like of float, optional
            The frequencies to keep in the object, each value passed here should
            exist in the freq_array.
        freq_chans : array_like of int, optional
            The frequency channel numbers to keep in the object.
        times : array_like of float, optional
            The times to keep in the object, each value passed here should
            exist in the time_array.
        polarizations : array_like of int, optional
            The polarizations numbers to keep in the object, each value passed
            here should exist in the polarization_array.
        blt_inds : array_like of int, optional
            The baseline-time indices to keep in the object. This is
            not commonly used.
        ant_inds : array_like of int, optional
            The antenna indices to keep in the object. This is
            not commonly used.

        Returns
        -------
        blt_inds : list of int
            list of baseline-time indices to keep. Can be None (to keep everything).
        ant_inds : list of int
            list of antenna number indices to keep. Can be None
            (keep all; only valid for "antenna" mode).
        freq_inds : list of int
            list of frequency indices to keep. Can be None (to keep everything).
        pol_inds : list of int
            list of polarization indices to keep. Can be None (to keep everything).
        history_update_string : str
            string to append to the end of the history.

        """
        # build up history string as we go
        history_update_string = "  Downselected to specific "
        n_selects = 0

        if self.type == "waterfall":
            if antenna_nums is not None:
                raise ValueError(
                    "Cannot select on antenna_nums with waterfall type "
                    "UVFlag objects."
                )
            if bls is not None:
                raise ValueError(
                    "Cannot select on bls with waterfall type " "UVFlag objects."
                )

        if ant_str is not None:
            if not (antenna_nums is None and bls is None and polarizations is None):
                raise ValueError(
                    "Cannot provide ant_str with antenna_nums, bls, or polarizations."
                )
            else:
                bls, polarizations = self.parse_ants(ant_str)
                if bls is not None and len(bls) == 0:
                    raise ValueError(
                        f"There is no data matching ant_str={ant_str} in this object."
                    )

        # Antennas, times and blt_inds all need to be combined into a set of
        # blts indices to keep.

        # test for blt_inds presence before adding inds from antennas & times
        if blt_inds is not None:
            blt_inds = uvutils._get_iterable(blt_inds)
            if np.array(blt_inds).ndim > 1:
                blt_inds = np.array(blt_inds).flatten()
            if self.type == "baseline":
                history_update_string += "baseline-times"
            else:
                history_update_string += "times"
            n_selects += 1

        if antenna_nums is not None:
            antenna_nums = uvutils._get_iterable(antenna_nums)
            if np.array(antenna_nums).ndim > 1:
                antenna_nums = np.array(antenna_nums).flatten()
            if n_selects > 0:
                history_update_string += ", antennas"
            else:
                history_update_string += "antennas"
            n_selects += 1

            if self.type == "baseline":
                inds1 = np.zeros(0, dtype=np.int)
                inds2 = np.zeros(0, dtype=np.int)
                for ant in antenna_nums:
                    if ant in self.ant_1_array or ant in self.ant_2_array:
                        wh1 = np.where(self.ant_1_array == ant)[0]
                        wh2 = np.where(self.ant_2_array == ant)[0]
                        if len(wh1) > 0:
                            inds1 = np.append(inds1, list(wh1))
                        if len(wh2) > 0:
                            inds2 = np.append(inds2, list(wh2))
                    else:
                        raise ValueError(
                            "Antenna number {a} is not present in the "
                            "ant_1_array or ant_2_array".format(a=ant)
                        )
                ant_blt_inds = set(inds1).intersection(inds2)

            if self.type == "antenna":
                ant_blt_inds = None
                ant_inds = np.zeros(0, dtype=np.int)
                for ant in antenna_nums:
                    if ant in self.ant_array:
                        wh = np.nonzero(self.ant_array == ant)[0]
                        if len(wh) > 0:
                            ant_inds = np.append(ant_inds, list(wh))
                    else:
                        raise ValueError(
                            "Antenna number {a} is not present in the "
                            "ant_array".format(a=ant)
                        )

        else:
            ant_blt_inds = None

        if bls is not None:
            if self.type != "baseline":
                raise ValueError(
                    'Only "baseline" mode UVFlag objects may select'
                    " along the baseline axis"
                )
            if isinstance(bls, tuple) and (len(bls) == 2 or len(bls) == 3):
                bls = [bls]
            if not all(isinstance(item, tuple) for item in bls):
                raise ValueError(
                    "bls must be a list of tuples of antenna numbers "
                    "(optionally with polarization)."
                )
            if not all(
                [isinstance(item[0], (int, np.integer,)) for item in bls]
                + [isinstance(item[1], (int, np.integer,)) for item in bls]
            ):
                raise ValueError(
                    "bls must be a list of tuples of integer antenna numbers "
                    "(optionally with polarization)."
                )
            if all(len(item) == 3 for item in bls):
                if polarizations is not None:
                    raise ValueError(
                        "Cannot provide length-3 tuples and also specify polarizations."
                    )
                if not all(isinstance(item[2], str) for item in bls):
                    raise ValueError(
                        "The third element in each bl must be a polarization string"
                    )

            if n_selects > 0:
                history_update_string += ", baselines"
            else:
                history_update_string += "baselines"

            n_selects += 1
            bls_blt_inds = np.zeros(0, dtype=np.int)
            bl_pols = set()
            for bl in bls:
                if not (bl[0] in self.ant_1_array or bl[0] in self.ant_2_array):
                    raise ValueError(
                        "Antenna number {a} is not present in the "
                        "ant_1_array or ant_2_array".format(a=bl[0])
                    )
                if not (bl[1] in self.ant_1_array or bl[1] in self.ant_2_array):
                    raise ValueError(
                        "Antenna number {a} is not present in the "
                        "ant_1_array or ant_2_array".format(a=bl[1])
                    )
                wh1 = np.where(
                    np.logical_and(self.ant_1_array == bl[0], self.ant_2_array == bl[1])
                )[0]
                wh2 = np.where(
                    np.logical_and(self.ant_1_array == bl[1], self.ant_2_array == bl[0])
                )[0]
                if len(wh1) > 0:
                    bls_blt_inds = np.append(bls_blt_inds, list(wh1))
                    if len(bl) == 3:
                        bl_pols.add(bl[2])
                elif len(wh2) > 0:
                    bls_blt_inds = np.append(bls_blt_inds, list(wh2))
                    if len(bl) == 3:
                        bl_pols.add(uvutils.conj_pol(bl[2]))
                else:
                    raise ValueError(
                        "Antenna pair {p} does not have any data "
                        "associated with it.".format(p=bl)
                    )
            if len(bl_pols) > 0:
                polarizations = list(bl_pols)

            if ant_blt_inds is not None:
                # Use intersection (and) to join antenna_names/nums & ant_pairs_nums
                ant_blt_inds = set(ant_blt_inds).intersection(bls_blt_inds)
            else:
                ant_blt_inds = bls_blt_inds

        if ant_blt_inds is not None:
            if blt_inds is not None:
                # Use intersection (and) to join
                # antenna_names/nums/ant_pairs_nums with blt_inds
                blt_inds = set(blt_inds).intersection(ant_blt_inds)
            else:
                blt_inds = ant_blt_inds

        if times is not None:
            times = uvutils._get_iterable(times)
            if np.array(times).ndim > 1:
                times = np.array(times).flatten()

            if n_selects > 0:
                if (
                    self.type != "baseline" and "times" not in history_update_string
                ) or self.type == "baseline":

                    history_update_string += ", times"
            else:
                history_update_string += "times"

            n_selects += 1

            time_blt_inds = np.zeros(0, dtype=np.int)
            for jd in times:
                if jd in self.time_array:
                    time_blt_inds = np.append(
                        time_blt_inds, np.where(self.time_array == jd)[0]
                    )
                else:
                    raise ValueError(
                        "Time {t} is not present in the time_array".format(t=jd)
                    )

            if blt_inds is not None:
                # Use intesection (and) to join
                # antenna_names/nums/ant_pairs_nums/blt_inds with times
                blt_inds = set(blt_inds).intersection(time_blt_inds)
            else:
                blt_inds = time_blt_inds

        if blt_inds is not None:
            if len(blt_inds) == 0:
                raise ValueError("No baseline-times were found that match criteria")

            if self.type == "baseline":
                compare_length = self.Nblts
            else:
                compare_length = self.Ntimes

            if max(blt_inds) >= compare_length:
                raise ValueError("blt_inds contains indices that are too large")
            if min(blt_inds) < 0:
                raise ValueError("blt_inds contains indices that are negative")

            blt_inds = sorted(set(blt_inds))

        if freq_chans is not None:
            freq_chans = uvutils._get_iterable(freq_chans)
            if np.array(freq_chans).ndim > 1:
                freq_chans = np.array(freq_chans).flatten()
            if frequencies is None:
                if self.type != "waterfall":
                    frequencies = self.freq_array[0, freq_chans]
                else:
                    frequencies = self.freq_array[freq_chans]

            else:
                frequencies = uvutils._get_iterable(frequencies)
                if self.type != "waterfall":
                    frequencies = np.sort(
                        list(set(frequencies) | set(self.freq_array[0, freq_chans]))
                    )
                else:
                    frequencies = np.sort(
                        list(set(frequencies) | set(self.freq_array[freq_chans]))
                    )

        if frequencies is not None:
            frequencies = uvutils._get_iterable(frequencies)
            if np.array(frequencies).ndim > 1:
                frequencies = np.array(frequencies).flatten()
            if n_selects > 0:
                history_update_string += ", frequencies"
            else:
                history_update_string += "frequencies"
            n_selects += 1

            freq_inds = np.zeros(0, dtype=np.int)
            # this works because we only allow one SPW. This will have to be
            # reworked when we support more.
            if self.type != "waterfall":
                freq_arr_use = self.freq_array[0, :]
            else:
                freq_arr_use = self.freq_array[:]
            for f in frequencies:
                if f in freq_arr_use:
                    freq_inds = np.append(freq_inds, np.where(freq_arr_use == f)[0])
                else:
                    raise ValueError(
                        "Frequency {f} is not present in the freq_array".format(f=f)
                    )

            freq_inds = sorted(set(freq_inds))
        else:
            freq_inds = None

        if polarizations is not None:
            polarizations = uvutils._get_iterable(polarizations)
            if np.array(polarizations).ndim > 1:
                polarizations = np.array(polarizations).flatten()
            if n_selects > 0:
                history_update_string += ", polarizations"
            else:
                history_update_string += "polarizations"
            n_selects += 1

            pol_inds = np.zeros(0, dtype=np.int)
            for p in polarizations:
                if isinstance(p, str):
                    p_num = uvutils.polstr2num(p, x_orientation=self.x_orientation)
                else:
                    p_num = p
                if p_num in self.polarization_array:
                    pol_inds = np.append(
                        pol_inds, np.where(self.polarization_array == p_num)[0]
                    )
                else:
                    raise ValueError(
                        "Polarization {p} is not present in the "
                        "polarization_array".format(p=p)
                    )

            pol_inds = sorted(set(pol_inds))
        else:
            pol_inds = None

        history_update_string += " using pyuvdata."

        return blt_inds, ant_inds, freq_inds, pol_inds, history_update_string

    def _select_metadata(
        self, blt_inds, ant_inds, freq_inds, pol_inds, history_update_string
    ):
        """Perform select on everything except the data-sized arrays.

        Parameters
        ----------
        blt_inds : list of int
            list of baseline-time indices to keep. Can be None (to keep everything).
        freq_inds : list of int
            list of frequency indices to keep. Can be None (to keep everything).
        pol_inds : list of int
            list of polarization indices to keep. Can be None (to keep everything).
        history_update_string : str
            string to append to the end of the history.
        keep_all_metadata : bool
            Option to keep metadata for antennas that are no longer in the dataset.

        """
        if blt_inds is not None:
            if self.type == "baseline":
                self.Nblts = len(blt_inds)
                self.baseline_array = self.baseline_array[blt_inds]
                self.Nbls = len(np.unique(self.baseline_array))
                self.ant_1_array = self.ant_1_array[blt_inds]
                self.ant_2_array = self.ant_2_array[blt_inds]
                self.Nants_data = int(
                    np.union1d(self.ant_1_array, self.ant_2_array).size
                )

            self.time_array = self.time_array[blt_inds]
            self.lst_array = self.lst_array[blt_inds]
            self.Ntimes = len(np.unique(self.time_array))

        if self.type == "antenna":
            if ant_inds is not None:
                self.ant_array = self.ant_array[ant_inds]
                self.Nants_data = int(len(self.ant_array))

        if freq_inds is not None:
            self.Nfreqs = len(freq_inds)
            if self.type != "waterfall":
                self.freq_array = self.freq_array[:, freq_inds]
            else:
                self.freq_array = self.freq_array[freq_inds]

        if pol_inds is not None:
            self.Npols = len(pol_inds)
            self.polarization_array = self.polarization_array[pol_inds]

        self.history = self.history + history_update_string

    def select(
        self,
        antenna_nums=None,
        ant_inds=None,
        bls=None,
        ant_str=None,
        frequencies=None,
        freq_chans=None,
        times=None,
        polarizations=None,
        blt_inds=None,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        inplace=True,
    ):
        """
        Downselect data to keep on the object along various axes.

        Axes that can be selected along depend on the current type of the object.
        However some axis may always be selected upon, these include frequencies,
        times and polarizations.
        In "baseline" and "antenna" modes, antenna numbers may be selected.
        In "baseline" mode, antenna pairs may be selected.
        Specific baseline-time indices can also be selected in "baseline" mode,
        but this is not commonly used.
        The history attribute on the object will be updated to identify the
        operations performed.

        Parameters
        ----------
        antenna_nums : array_like of int, optional
            The antennas numbers to keep in the object (antenna positions and
            names for the removed antennas will be retained unless
            `keep_all_metadata` is False). This cannot be provided if
            `antenna_names` is also provided.
        bls : list of tuple, optional
            A list of antenna number tuples (e.g. [(0,1), (3,2)]) or a list of
            baseline 3-tuples (e.g. [(0,1,'xx'), (2,3,'yy')]) specifying baselines
            to keep in the object. For length-2 tuples, the ordering of the numbers
            within the tuple does not matter. For length-3 tuples, the polarization
            string is in the order of the two antennas. If length-3 tuples are
            provided, `polarizations` must be None.
        ant_str : str, optional
            A string containing information about what antenna numbers
            and polarizations to keep in the object.  Can be 'auto', 'cross', 'all',
            or combinations of antenna numbers and polarizations (e.g. '1',
            '1_2', '1x_2y').  See tutorial for more examples of valid strings and
            the behavior of different forms for ant_str.
            If '1x_2y,2y_3y' is passed, both polarizations 'xy' and 'yy' will
            be kept for both baselines (1, 2) and (2, 3) to return a valid
            pyuvdata object.
            An ant_str cannot be passed in addition to any of `antenna_nums`,
            `antenna_names`, `bls` args or the `polarizations` parameters,
            if it is a ValueError will be raised.
        frequencies : array_like of float, optional
            The frequencies to keep in the object, each value passed here should
            exist in the freq_array.
        freq_chans : array_like of int, optional
            The frequency channel numbers to keep in the object.
        times : array_like of float, optional
            The times to keep in the object, each value passed here should
            exist in the time_array.
        polarizations : array_like of int, optional
            The polarizations numbers to keep in the object, each value passed
            here should exist in the polarization_array.
        blt_inds : array_like of int, optional
            The baseline-time indices to keep in the object. This is
            not commonly used.
        ant_inds : array_like of int, optional
            The antenna indices to keep in the object. This is
            not commonly used.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after downselecting data on this object.
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            downselecting data on this object.
        inplace : bool
            Option to perform the select directly on self or return a new UVData
            object with just the selected data.

        Returns
        -------
        UVData object or None
            None is returned if inplace is True, otherwise a new UVData object
            with just the selected data is returned

        Raises
        ------
        ValueError
            If any of the parameters are set to inappropriate values.

        """
        if inplace:
            uv_object = self
        else:
            uv_object = self.copy()

        (
            blt_inds,
            ant_inds,
            freq_inds,
            pol_inds,
            history_update_string,
        ) = uv_object._select_preprocess(
            antenna_nums=antenna_nums,
            ant_str=ant_str,
            bls=bls,
            frequencies=frequencies,
            freq_chans=freq_chans,
            times=times,
            polarizations=polarizations,
            blt_inds=blt_inds,
            ant_inds=ant_inds,
        )

        # do select operations on everything except data_array, flag_array
        # and nsample_array
        uv_object._select_metadata(
            blt_inds, ant_inds, freq_inds, pol_inds, history_update_string
        )

        if blt_inds is not None:
            if self.type == "baseline":
                for param_name, param in zip(
                    self._data_params, uv_object.data_like_parameters
                ):
                    setattr(uv_object, param_name, param[blt_inds, :, :, :])
            if self.type == "waterfall":
                for param_name, param in zip(
                    self._data_params, uv_object.data_like_parameters
                ):
                    setattr(uv_object, param_name, param[blt_inds, :, :])
            if self.type == "antenna":
                for param_name, param in zip(
                    self._data_params, uv_object.data_like_parameters
                ):
                    setattr(uv_object, param_name, param[:, :, :, blt_inds, :])

        if ant_inds is not None and self.type == "antenna":
            for param_name, param in zip(
                self._data_params, uv_object.data_like_parameters
            ):
                setattr(uv_object, param_name, param[ant_inds, :, :, :])

        if freq_inds is not None:
            if self.type == "baseline":
                for param_name, param in zip(
                    self._data_params, uv_object.data_like_parameters
                ):
                    setattr(uv_object, param_name, param[:, :, freq_inds, :])
            if self.type == "waterfall":
                for param_name, param in zip(
                    self._data_params, uv_object.data_like_parameters
                ):
                    setattr(uv_object, param_name, param[:, freq_inds, :])
            if self.type == "antenna":
                for param_name, param in zip(
                    self._data_params, uv_object.data_like_parameters
                ):
                    setattr(uv_object, param_name, param[:, :, freq_inds, :, :])

        if pol_inds is not None:
            if self.type == "baseline":
                for param_name, param in zip(
                    self._data_params, uv_object.data_like_parameters
                ):
                    setattr(uv_object, param_name, param[:, :, :, pol_inds])
            if self.type == "waterfall":
                for param_name, param in zip(
                    self._data_params, uv_object.data_like_parameters
                ):
                    setattr(uv_object, param_name, param[:, :, pol_inds])
            if self.type == "antenna":
                for param_name, param in zip(
                    self._data_params, uv_object.data_like_parameters
                ):
                    setattr(uv_object, param_name, param[:, :, :, :, pol_inds])

        # check if object is uv_object-consistent
        if run_check:
            uv_object.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

        if not inplace:
            return uv_object

    def read(
        self,
        filename,
        history="",
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """Read in flag/metric data from a HDF5 file.

        Parameters
        ----------
        filename : str or pathlib.Path
            The file name to read.
        history : str
            History string to append to UVFlag history attribute.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after reading data.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            reading data.

        """
        if isinstance(filename, (tuple, list)):
            self.read(filename[0])
            if len(filename) > 1:
                for f in filename[1:]:
                    f2 = UVFlag(f, history=history)
                    self += f2
                del f2

        else:
            if not os.path.exists(filename):
                raise IOError(filename + " not found.")

            # Open file for reading
            with h5py.File(filename, "r") as f:
                header = f["/Header"]

                self.type = header["type"][()].decode("utf8")
                if self.type == "antenna":
                    self._set_type_antenna()
                elif self.type == "baseline":
                    self._set_type_baseline()
                elif self.type == "waterfall":
                    self._set_type_waterfall()
                else:
                    raise ValueError(
                        "File cannot be read. Received type "
                        "parameter: {receive} but "
                        "must be within acceptable values: "
                        "{expect}".format(
                            receive=self.type,
                            expect=(", ").join(self._type.acceptable_vals),
                        )
                    )

                self.mode = header["mode"][()].decode("utf8")

                if self.mode == "metric":
                    self._set_mode_metric()
                elif self.mode == "flag":
                    self._set_mode_flag()
                else:
                    raise ValueError(
                        "File cannot be read. Received mode "
                        "parameter: {receive} but "
                        "must be within acceptable values: "
                        "{expect}".format(
                            receive=self.mode,
                            expect=(", ").join(self._mode.acceptable_vals),
                        )
                    )

                if "x_orientation" in header.keys():
                    self.x_orientation = header["x_orientation"][()].decode("utf8")

                self.time_array = header["time_array"][()]
                if "Ntimes" in header.keys():
                    self.Ntimes = int(header["Ntimes"][()])
                else:
                    self.Ntimes = np.unique(self.time_array).size

                self.lst_array = header["lst_array"][()]

                self.freq_array = header["freq_array"][()]
                # older save files will not have this spws axis
                # at least_2d will preserve shape of 2d arrays and
                # promote 1D to (1, Nfreqs)
                if self.type != "waterfall":
                    self.freq_array = np.atleast_2d(self.freq_array)

                if "Nfreqs" in header.keys():
                    self.Nfreqs = int(header["Nfreqs"][()])
                else:
                    self.Nfreqs = np.unique(self.freq_array).size

                self.history = header["history"][()].decode("utf8")

                self.history += history

                if not uvutils._check_history_version(
                    self.history, self.pyuvdata_version_str
                ):
                    self.history += self.pyuvdata_version_str

                # get extra_keywords
                if "extra_keywords" in header.keys():
                    self.extra_keywords = {}
                    for key in header["extra_keywords"].keys():
                        if header["extra_keywords"][key].dtype.type in (
                            np.string_,
                            np.object_,
                        ):
                            self.extra_keywords[key] = bytes(
                                header["extra_keywords"][key][()]
                            ).decode("utf8")
                        else:
                            self.extra_keywords[key] = header["extra_keywords"][key][()]
                else:
                    self.extra_keywords = {}

                if "label" in header.keys():
                    self.label = header["label"][()].decode("utf8")

                polarization_array = header["polarization_array"][()]
                if isinstance(polarization_array[0], np.string_):
                    polarization_array = np.asarray(polarization_array, dtype=np.str_)
                self.polarization_array = polarization_array
                self._check_pol_state()

                if "Npols" in header.keys():
                    self.Npols = int(header["Npols"][()])
                else:
                    self.Npols = len(self.polarization_array)

                if self.type == "baseline":

                    self.baseline_array = header["baseline_array"][()]

                    if "Nblts" in header.keys():
                        self.Nblts = int(header["Nblts"][()])
                    else:
                        self.Nblts = len(self.baseline_array)

                    if "Nbls" in header.keys():
                        self.Nbls = int(header["Nbls"][()])
                    else:
                        self.Nbls = np.unique(self.baseline_array).size

                    self.ant_1_array = header["ant_1_array"][()]
                    self.ant_2_array = header["ant_2_array"][()]

                    try:
                        self.Nants_telescope = int(header["Nants_telescope"][()])
                    except KeyError:
                        warnings.warn(
                            "Nants_telescope not available in file, " "assuming < 2048."
                        )
                        self.Nants_telescope = 2047

                    if "Nants_data" in header.keys():
                        self.Nants_data = int(header["Nants_data"][()])
                    else:
                        self.Nants_data = int(
                            np.unique(
                                np.union1d(self.ant_1_array, self.ant_2_array)
                            ).size
                        )

                    if "Nspws" in header.keys():
                        self.Nspws = int(header["Nspws"][()])
                    else:
                        self.Nspws = np.shape(self.freq_array)[0]

                elif self.type == "antenna":
                    self.ant_array = header["ant_array"][()]
                    try:
                        self.Nants_data = int(header["Nants_data"][()])
                    except KeyError:
                        warnings.warn(
                            "Nants_data not available in file, "
                            "attempting to calculate from ant_array."
                        )
                        self.Nants_data = len(self.ant_array)

                    if "Nspws" in header.keys():
                        self.Nspws = int(header["Nspws"][()])
                    else:
                        self.Nspws = np.shape(self.freq_array)[0]

                dgrp = f["/Data"]
                if self.mode == "metric":
                    self.metric_array = dgrp["metric_array"][()]
                    self.weights_array = dgrp["weights_array"][()]
                    if "weights_square_array" in dgrp:
                        self.weights_square_array = dgrp["weights_square_array"][()]
                elif self.mode == "flag":
                    self.flag_array = dgrp["flag_array"][()]
            self.clear_unused_attributes()

            if run_check:
                self.check(
                    check_extra=check_extra,
                    run_check_acceptability=run_check_acceptability,
                )

    def write(self, filename, clobber=False, data_compression="lzf"):
        """Write a UVFlag object to a hdf5 file.

        Parameters
        ----------
        filename : str
            The file to write to.
        clobber : bool
            Option to overwrite the file if it already exists.
        data_compression : str
            HDF5 filter to apply when writing the data_array.
            If no compression is wanted, set to None.

        """
        if os.path.exists(filename):
            if clobber:
                print("File " + filename + " exists; clobbering")
            else:
                raise ValueError("File " + filename + " exists; skipping")

        with h5py.File(filename, "w") as f:
            header = f.create_group("Header")

            # write out metadata
            header["type"] = np.string_(self.type)
            header["mode"] = np.string_(self.mode)

            header["Ntimes"] = self.Ntimes
            header["time_array"] = self.time_array
            header["lst_array"] = self.lst_array

            header["freq_array"] = self.freq_array
            header["Nfreqs"] = self.Nfreqs

            header["Npols"] = self.Npols

            if self.x_orientation is not None:
                header["x_orientation"] = np.string_(self.x_orientation)

            if isinstance(self.polarization_array.item(0), str):
                polarization_array = np.asarray(
                    self.polarization_array, dtype=np.string_
                )
            else:
                polarization_array = self.polarization_array
            header["polarization_array"] = polarization_array

            if not uvutils._check_history_version(
                self.history, self.pyuvdata_version_str
            ):
                self.history += self.pyuvdata_version_str

            # write out extra keywords if it exists and has elements
            if self.extra_keywords:
                extra_keywords = header.create_group(
                    "extra_keywords"
                )  # create spot in header
                for k in self.extra_keywords.keys():
                    if isinstance(self.extra_keywords[k], str):
                        extra_keywords[k] = np.string_(self.extra_keywords[k])
                    else:
                        extra_keywords[k] = self.extra_keywords[k]

            header["history"] = np.string_(self.history)

            header["label"] = np.string_(self.label)

            if self.type == "baseline":
                header["baseline_array"] = self.baseline_array
                header["Nbls"] = self.Nbls
                header["Nblts"] = self.Nblts
                header["ant_1_array"] = self.ant_1_array
                header["ant_2_array"] = self.ant_2_array
                header["Nants_data"] = self.Nants_data
                header["Nants_telescope"] = self.Nants_telescope
                header["Nspws"] = self.Nspws

            elif self.type == "antenna":
                header["ant_array"] = self.ant_array
                header["Nants_data"] = self.Nants_data
                header["Nspws"] = self.Nspws

            dgrp = f.create_group("Data")
            if self.mode == "metric":
                dgrp.create_dataset(
                    "metric_array",
                    chunks=True,
                    data=self.metric_array,
                    compression=data_compression,
                )
                dgrp.create_dataset(
                    "weights_array",
                    chunks=True,
                    data=self.weights_array,
                    compression=data_compression,
                )
                if self.weights_square_array is not None:
                    dgrp.create_dataset(
                        "weights_square_array",
                        chunks=True,
                        data=self.weights_square_array,
                        compression=data_compression,
                    )
            elif self.mode == "flag":
                dgrp.create_dataset(
                    "flag_array",
                    chunks=True,
                    data=self.flag_array,
                    compression=data_compression,
                )

    def from_uvdata(
        self,
        indata,
        mode="metric",
        copy_flags=False,
        waterfall=False,
        history="",
        label="",
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """Construct a UVFlag object from a UVData object.

        Parameters
        ----------
        indata : UVData
            Input to initialize UVFlag object.
        mode : {"metric", "flag"}, optional
            The mode determines whether the object has a floating point metric_array
            or a boolean flag_array.
        copy_flags : bool, optional
            Whether to copy flags from indata to new UVFlag object
        waterfall : bool, optional
            Whether to immediately initialize as a waterfall object, with flag/metric
            axes: time, frequency, polarization.
        history : str, optional
            History string to attach to object.
        label: str, optional
            String used for labeling the object (e.g. 'FM').
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after creating UVFlag object.
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            creating UVFlag object.

        """
        if not issubclass(indata.__class__, UVData):
            raise ValueError(
                "from_uvdata can only initialize a UVFlag object from an input "
                "UVData object or a subclass of a UVData object."
            )

        if mode.lower() == "metric":
            self._set_mode_metric()
        elif mode.lower() == "flag":
            self._set_mode_flag()
        else:
            raise ValueError(
                "Input mode must be within acceptable values: "
                "{}".format((", ").join(self._mode.acceptable_vals))
            )

        if waterfall:
            self._set_type_waterfall()
            self.history += 'Flag object with type "waterfall" created. '
            if not uvutils._check_history_version(
                self.history, self.pyuvdata_version_str
            ):
                self.history += self.pyuvdata_version_str

            self.time_array, ri = np.unique(indata.time_array, return_index=True)
            self.Ntimes = len(self.time_array)
            if indata.future_array_shapes:
                self.freq_array = indata.freq_array
            else:
                self.freq_array = indata.freq_array[0, :]
            self.Nspws = None
            self.Nfreqs = len(self.freq_array)
            self.Nblts = len(self.time_array)
            self.polarization_array = indata.polarization_array
            self.Npols = len(self.polarization_array)
            self.lst_array = indata.lst_array[ri]
            if copy_flags:
                raise NotImplementedError(
                    "Cannot copy flags when initializing waterfall UVFlag from "
                    "UVData or UVCal."
                )
            else:
                if self.mode == "flag":
                    self.flag_array = np.zeros(
                        (
                            len(self.time_array),
                            len(self.freq_array),
                            len(self.polarization_array),
                        ),
                        np.bool,
                    )
                elif self.mode == "metric":
                    self.metric_array = np.zeros(
                        (
                            len(self.time_array),
                            len(self.freq_array),
                            len(self.polarization_array),
                        )
                    )

        else:
            self._set_type_baseline()
            self.history += 'Flag object with type "baseline" created. '
            if not uvutils._check_history_version(
                self.history, self.pyuvdata_version_str
            ):
                self.history += self.pyuvdata_version_str

            self.baseline_array = indata.baseline_array
            self.Nbls = np.unique(self.baseline_array).size
            self.Nblts = len(self.baseline_array)
            self.ant_1_array = indata.ant_1_array
            self.ant_2_array = indata.ant_2_array
            self.Nants_data = indata.Nants_data

            self.time_array = indata.time_array
            self.lst_array = indata.lst_array
            self.Ntimes = np.unique(self.time_array).size

            if indata.future_array_shapes:
                self.freq_array = indata.freq_array[np.newaxis, :]
            else:
                self.freq_array = indata.freq_array
            self.Nfreqs = indata.Nfreqs
            self.Nspws = indata.Nspws

            self.polarization_array = indata.polarization_array
            self.Npols = len(self.polarization_array)
            self.Nants_telescope = indata.Nants_telescope
            if copy_flags:
                if indata.future_array_shapes:
                    self.flag_array = indata.flag_array[:, np.newaxis, :, :]
                else:
                    self.flag_array = indata.flag_array
                self.history += (
                    " Flags copied from " + str(indata.__class__) + " object."
                )
                if self.mode == "metric":
                    warnings.warn(
                        'Copying flags to type=="baseline" ' 'results in mode=="flag".'
                    )
                    self._set_mode_flag()
            else:
                array_shape = (self.Nblts, self.Nspws, self.Nfreqs, self.Npols)
                if self.mode == "flag":
                    self.flag_array = np.zeros(array_shape, dtype=np.bool)
                elif self.mode == "metric":
                    self.metric_array = np.zeros(array_shape, dtype=np.float64)

        if indata.x_orientation is not None:
            self.x_orientation = indata.x_orientation

        if self.mode == "metric":
            self.weights_array = np.ones(self.metric_array.shape)

        if indata.extra_keywords is not None:
            self.extra_keywords = indata.extra_keywords

        if history not in self.history:
            self.history += history
        self.label += label

        self.clear_unused_attributes()

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )
        return

    def from_uvcal(
        self,
        indata,
        mode="metric",
        copy_flags=False,
        waterfall=False,
        history="",
        label="",
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """Construct a UVFlag object from a UVCal object.

        Parameters
        ----------
        indata : UVData
            Input to initialize UVFlag object.
        mode : {"metric", "flag"}, optional
            The mode determines whether the object has a floating point metric_array
            or a boolean flag_array.
        copy_flags : bool, optional
            Whether to copy flags from indata to new UVFlag object
        waterfall : bool, optional
            Whether to immediately initialize as a waterfall object, with flag/metric
            axes: time, frequency, polarization.
        history : str, optional
            History string to attach to object.
        label: str, optional
            String used for labeling the object (e.g. 'FM').
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after creating UVFlag object.
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            creating UVFlag object.

        """
        if not issubclass(indata.__class__, UVCal):
            raise ValueError(
                "from_uvcal can only initialize a UVFlag object from an input "
                "UVCal object or a subclass of a UVCal object."
            )

        if mode.lower() == "metric":
            self._set_mode_metric()
        elif mode.lower() == "flag":
            self._set_mode_flag()
        else:
            raise ValueError(
                "Input mode must be within acceptable values: "
                "{}".format((", ").join(self._mode.acceptable_vals))
            )

        if waterfall:
            self._set_type_waterfall()
            self.history += 'Flag object with type "waterfall" created. '
            if not uvutils._check_history_version(
                self.history, self.pyuvdata_version_str
            ):
                self.history += self.pyuvdata_version_str

            self.time_array, ri = np.unique(indata.time_array, return_index=True)
            self.Ntimes = len(self.time_array)
            self.freq_array = indata.freq_array[0, :]
            self.Nspws = None
            self.Nfreqs = len(self.freq_array)
            self.Nblts = len(self.time_array)
            self.polarization_array = indata.jones_array
            self.Npols = len(self.polarization_array)
            self.lst_array = lst_from_uv(indata)[ri]
            if copy_flags:
                raise NotImplementedError(
                    "Cannot copy flags when "
                    "initializing waterfall UVFlag "
                    "from UVData or UVCal."
                )
            else:
                if self.mode == "flag":
                    self.flag_array = np.zeros(
                        (
                            len(self.time_array),
                            len(self.freq_array),
                            len(self.polarization_array),
                        ),
                        np.bool,
                    )
                elif self.mode == "metric":
                    self.metric_array = np.zeros(
                        (
                            len(self.time_array),
                            len(self.freq_array),
                            len(self.polarization_array),
                        )
                    )

        else:
            self._set_type_antenna()
            self.history += 'Flag object with type "antenna" created. '
            if not uvutils._check_history_version(
                self.history, self.pyuvdata_version_str
            ):
                self.history += self.pyuvdata_version_str
            self.ant_array = indata.ant_array
            self.Nants_data = len(self.ant_array)

            self.time_array = indata.time_array
            self.lst_array = lst_from_uv(indata)
            self.Ntimes = np.unique(self.time_array).size
            self.Nblts = self.Ntimes

            self.freq_array = indata.freq_array
            self.Nspws = indata.Nspws
            self.Nfreqs = np.unique(self.freq_array).size

            self.polarization_array = indata.jones_array
            self.Npols = len(self.polarization_array)
            if copy_flags:
                self.flag_array = indata.flag_array
                self.history += (
                    " Flags copied from " + str(indata.__class__) + " object."
                )
                if self.mode == "metric":
                    warnings.warn(
                        'Copying flags to type=="antenna" ' 'results in mode=="flag".'
                    )
                    self._set_mode_flag()
            else:
                if self.mode == "flag":
                    self.flag_array = np.zeros_like(indata.flag_array)
                elif self.mode == "metric":
                    self.metric_array = np.zeros_like(indata.flag_array).astype(
                        np.float
                    )
        if self.mode == "metric":
            self.weights_array = np.ones(self.metric_array.shape)

        if indata.x_orientation is not None:
            self.x_orientation = indata.x_orientation

        if history not in self.history:
            self.history += history
        self.label += label

        self.clear_unused_attributes()

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )
        return
