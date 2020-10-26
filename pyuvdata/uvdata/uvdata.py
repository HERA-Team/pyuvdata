# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Primary container for radio interferometer datasets."""
import os
import copy
from collections.abc import Iterable
import warnings
import threading

import numpy as np
from astropy import constants as const
import astropy.units as units
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, FK5, Angle
from astropy import coordinates as coord

from ..uvbase import UVBase
from .. import parameter as uvp
from .. import telescopes as uvtel
from .. import utils as uvutils

__all__ = ["UVData"]


class UVData(UVBase):
    """
    A class for defining a radio interferometer dataset.

    Currently supported file types: uvfits, miriad, fhd.
    Provides phasing functions.

    Attributes
    ----------
    UVParameter objects :
        For full list see UVData Parameters
        (http://pyuvdata.readthedocs.io/en/latest/uvdata_parameters.html).
        Some are always required, some are required for certain phase_types
        and others are always optional.
    """

    def __init__(self):
        """Create a new UVData object."""
        # add the UVParameters to the class

        # standard angle tolerance: 10 mas in radians.
        # Should perhaps be decreased to 1 mas in the future
        radian_tol = 10 * 2 * np.pi * 1e-3 / (60.0 * 60.0 * 360.0)

        self._Ntimes = uvp.UVParameter(
            "Ntimes", description="Number of times", expected_type=int
        )
        self._Nbls = uvp.UVParameter(
            "Nbls", description="Number of baselines", expected_type=int
        )
        self._Nblts = uvp.UVParameter(
            "Nblts",
            description="Number of baseline-times "
            "(i.e. number of spectra). Not necessarily "
            "equal to Nbls * Ntimes",
            expected_type=int,
        )
        self._Nfreqs = uvp.UVParameter(
            "Nfreqs", description="Number of frequency channels", expected_type=int
        )
        self._Npols = uvp.UVParameter(
            "Npols", description="Number of polarizations", expected_type=int
        )

        desc = (
            "Array of the visibility data, shape: (Nblts, 1, Nfreqs, "
            "Npols), type = complex float, in units of self.vis_units"
        )
        # TODO: Spw axis to be collapsed in future release
        self._data_array = uvp.UVParameter(
            "data_array",
            description=desc,
            form=("Nblts", 1, "Nfreqs", "Npols"),
            expected_type=np.complex,
        )

        desc = 'Visibility units, options are: "uncalib", "Jy" or "K str"'
        self._vis_units = uvp.UVParameter(
            "vis_units",
            description=desc,
            form="str",
            expected_type=str,
            acceptable_vals=["uncalib", "Jy", "K str"],
        )

        desc = (
            "Number of data points averaged into each data element, "
            "NOT required to be an integer, type = float, same shape as data_array."
            "The product of the integration_time and the nsample_array "
            "value for a visibility reflects the total amount of time "
            "that went into the visibility. Best practice is for the "
            "nsample_array to be used to track flagging within an integration_time "
            "(leading to a decrease of the nsample array value below 1) and "
            "LST averaging (leading to an increase in the nsample array "
            "value). So datasets that have not been LST averaged should "
            "have nsample array values less than or equal to 1."
            "Note that many files do not follow this convention, but it is "
            "safe to assume that the product of the integration_time and "
            "the nsample_array is the total amount of time included in a visibility."
        )
        self._nsample_array = uvp.UVParameter(
            "nsample_array",
            description=desc,
            form=("Nblts", 1, "Nfreqs", "Npols"),
            expected_type=(np.float),
        )

        desc = "Boolean flag, True is flagged, same shape as data_array."
        self._flag_array = uvp.UVParameter(
            "flag_array",
            description=desc,
            form=("Nblts", 1, "Nfreqs", "Npols"),
            expected_type=np.bool,
        )

        self._Nspws = uvp.UVParameter(
            "Nspws",
            description="Number of spectral windows "
            "(ie non-contiguous spectral chunks). "
            "More than one spectral window is not "
            "currently supported.",
            expected_type=int,
        )

        self._spw_array = uvp.UVParameter(
            "spw_array",
            description="Array of spectral window numbers, shape (Nspws)",
            form=("Nspws",),
            expected_type=int,
        )

        desc = (
            "Projected baseline vectors relative to phase center, "
            "shape (Nblts, 3), units meters. "
            "Convention is: uvw = xyz(ant2) - xyz(ant1)."
            "Note that this is the Miriad convention but it is different "
            "from the AIPS/FITS convention (where uvw = xyz(ant1) - xyz(ant2))."
        )
        self._uvw_array = uvp.UVParameter(
            "uvw_array",
            description=desc,
            form=("Nblts", 3),
            expected_type=np.float,
            acceptable_range=(0, 1e8),
            tols=1e-3,
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

        desc = (
            "Array of local apparent sidereal times (LAST) at the center of "
            "integration, shape (Nblts), units radians."
        )
        self._lst_array = uvp.UVParameter(
            "lst_array",
            description=desc,
            form=("Nblts",),
            expected_type=np.float,
            tols=radian_tol,
        )

        desc = "Array of first antenna indices, shape (Nblts), " "type = int, 0 indexed"
        self._ant_1_array = uvp.UVParameter(
            "ant_1_array", description=desc, expected_type=int, form=("Nblts",)
        )
        desc = (
            "Array of second antenna indices, shape (Nblts), " "type = int, 0 indexed"
        )
        self._ant_2_array = uvp.UVParameter(
            "ant_2_array", description=desc, expected_type=int, form=("Nblts",)
        )

        desc = (
            "Array of baseline numbers, shape (Nblts), "
            "type = int; baseline = 2048 * (ant1+1) + (ant2+1) + 2^16"
        )
        self._baseline_array = uvp.UVParameter(
            "baseline_array", description=desc, expected_type=int, form=("Nblts",)
        )

        # this dimensionality of freq_array does not allow for different spws
        # to have different dimensions
        desc = (
            "Array of frequencies, center of the channel, "
            "shape (1, Nfreqs), units Hz"
        )
        # TODO: Spw axis to be collapsed in future release
        self._freq_array = uvp.UVParameter(
            "freq_array",
            description=desc,
            form=(1, "Nfreqs"),
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

        desc = (
            "Length of the integration in seconds, shape (Nblts). "
            "The product of the integration_time and the nsample_array "
            "value for a visibility reflects the total amount of time "
            "that went into the visibility. Best practice is for the "
            "integration_time to reflect the length of time a visibility "
            "was integrated over (so it should vary in the case of "
            "baseline-dependent averaging and be a way to do selections "
            "for differently integrated baselines)."
            "Note that many files do not follow this convention, but it is "
            "safe to assume that the product of the integration_time and "
            "the nsample_array is the total amount of time included in a visibility."
        )
        self._integration_time = uvp.UVParameter(
            "integration_time",
            description=desc,
            form=("Nblts",),
            expected_type=np.float,
            tols=1e-3,
        )  # 1 ms

        desc = (
            "Width of frequency channels (Hz). If flex_spw = False, then it is a "
            "single value of type = float, otherwise it is an array of shape "
            "(Nfreqs), type = float."
        )
        self._channel_width = uvp.UVParameter(
            "channel_width", description=desc, expected_type=np.float, tols=1e-3,
        )  # 1 mHz

        desc = (
            "Name(s) of source(s) or field(s) observed. Type string, or if "
            "multi_object = True, a list of strings of length (Nobjects,)."
        )
        self._object_name = uvp.UVParameter(
            "object_name", description=desc, form="str", expected_type=str,
        )

        # --- multi-object handling ---
        desc = (
            'Only relevant if phase_type = "phased". Specifies the that the data set '
            "contains multiple sources within it."
        )
        self._multi_object = uvp.UVParameter(
            "multi_object", description=desc, expected_type=np.bool, value=False,
        )

        desc = (
            "Required if multi_object = True. Specifies the number of sources "
            "contained within the data set."
        )

        self._Nobjects = uvp.UVParameter(
            "Nobjects", description=desc, expected_type=np.int, required=False,
        )

        desc = (
            "Only relevant if multi_object = True. Dictionary containing information "
            "individual objects, including coordinate information. Keys are matched "
            "items in the object_name list attribute."
        )
        self._object_dict = uvp.UVParameter(
            "object_dict", description=desc, expected_type=dict, required=False,
        )

        self._telescope_name = uvp.UVParameter(
            "telescope_name",
            description="Name of telescope " "(string)",
            form="str",
            expected_type=str,
        )

        self._instrument = uvp.UVParameter(
            "instrument",
            description="Receiver or backend. " "Sometimes identical to telescope_name",
            form="str",
            expected_type=str,
        )

        desc = (
            "Telescope location: xyz in ITRF (earth-centered frame). "
            "Can also be accessed using telescope_location_lat_lon_alt or "
            "telescope_location_lat_lon_alt_degrees properties"
        )
        self._telescope_location = uvp.LocationParameter(
            "telescope_location",
            description=desc,
            acceptable_range=(6.35e6, 6.39e6),
            tols=1e-3,
        )

        self._history = uvp.UVParameter(
            "history",
            description="String of history, units English",
            form="str",
            expected_type=str,
        )

        # --- flexible spectral window information ---

        desc = (
            'Option to construct a "flexible spectral window", which stores'
            "all spectral channels across the frequency axis of data_array. "
            "Allows for spectral windows of variable sizes, and channels of "
            "varying widths."
        )
        self._flex_spw = uvp.UVParameter(
            "flex_spw", description=desc, expected_type=np.bool, value=False,
        )

        desc = (
            "Required if flex_spw = True. Maps individual channels along the "
            "frequency axis to individual spectral windows, as listed in the "
            "spw_array (zero-indexed). Shape (Nfreqs), type = int."
        )
        self._flex_spw_id_array = uvp.UVParameter(
            "flex_spw_id_array",
            description=desc,
            form=("Nfreqs",),
            expected_type=np.int,
            required=False,
        )

        # --- phasing information ---
        desc = (
            'String indicating phasing type. Allowed values are "drift", '
            '"phased" and "unknown"'
        )
        self._phase_type = uvp.UVParameter(
            "phase_type",
            form="str",
            expected_type=str,
            description=desc,
            value="unknown",
            acceptable_vals=["drift", "phased", "unknown"],
        )

        desc = (
            'Required if phase_type = "phased". Epoch year of the phase '
            "applied to the data (eg 2000.)"
        )
        self._phase_center_epoch = uvp.UVParameter(
            "phase_center_epoch",
            required=False,
            description=desc,
            expected_type=np.float,
        )

        desc = (
            "Required if phase_type = 'phased'. Right ascension of phase "
            "center (see uvw_array), units radians. Can also be accessed using "
            "phase_center_ra_degrees."
        )
        self._phase_center_ra = uvp.AngleParameter(
            "phase_center_ra",
            required=False,
            description=desc,
            expected_type=np.float,
            tols=radian_tol,
        )

        desc = (
            'Required if phase_type = "phased". Declination of phase center '
            "(see uvw_array), units radians. Can also be accessed using "
            "phase_center_dec_degrees."
        )
        self._phase_center_dec = uvp.AngleParameter(
            "phase_center_dec",
            required=False,
            description=desc,
            expected_type=np.float,
            tols=radian_tol,
        )

        desc = (
            'Required if phase_type = "phased". Apparent right ascension of phase '
            "center in the topocentric frame of the observatory, units radians."
            "Shape (Nblts,), type = float."
        )
        self._phase_center_app_ra = uvp.AngleParameter(
            "phase_center_app_ra",
            required=False,
            form=("Nblts",),
            expected_type=np.float64,
            description=desc,
            tols=radian_tol,
        )

        desc = (
            'Required if phase_type = "phased". Declination of phase center '
            "in the topocentric frame of the observatory, units radians. "
            "Shape (Nblts,), type = float."
        )
        self._phase_center_app_dec = uvp.AngleParameter(
            "phase_center_app_dec",
            required=False,
            form=("Nblts",),
            expected_type=np.float,
            description=desc,
            tols=radian_tol,
        )

        desc = (
            'Only relevant if phase_type = "phased". Specifies the frame the'
            ' data and uvw_array are phased to. Options are "gcrs" and "icrs",'
            ' default is "icrs"'
        )
        self._phase_center_frame = uvp.UVParameter(
            "phase_center_frame",
            required=False,
            description=desc,
            expected_type=str,
            acceptable_vals=["icrs", "gcrs"],
        )

        desc = (
            "Required if multi_object = True. Maps individual entries along the Nblt "
            "axis to a specific object in object_name (zero-indexed). Shape (Nblts), "
            "type = int."
        )
        self._object_id_array = uvp.UVParameter(
            "object_id_array",
            description=desc,
            form=("Nblts",),
            expected_type=np.int,
            required=False,
        )

        # --- antenna information ----
        desc = (
            "Number of antennas with data present (i.e. number of unique "
            "entries in ant_1_array and ant_2_array). May be smaller "
            "than the number of antennas in the array"
        )
        self._Nants_data = uvp.UVParameter(
            "Nants_data", description=desc, expected_type=int
        )

        desc = (
            "Number of antennas in the array. May be larger "
            "than the number of antennas with data"
        )
        self._Nants_telescope = uvp.UVParameter(
            "Nants_telescope", description=desc, expected_type=int
        )

        desc = (
            "List of antenna names, shape (Nants_telescope), "
            "with numbers given by antenna_numbers (which can be matched "
            "to ant_1_array and ant_2_array). There must be one entry "
            "here for each unique entry in ant_1_array and "
            "ant_2_array, but there may be extras as well."
        )
        self._antenna_names = uvp.UVParameter(
            "antenna_names",
            description=desc,
            form=("Nants_telescope",),
            expected_type=str,
        )

        desc = (
            "List of integer antenna numbers corresponding to antenna_names, "
            "shape (Nants_telescope). There must be one "
            "entry here for each unique entry in ant_1_array and "
            "ant_2_array, but there may be extras as well."
        )
        self._antenna_numbers = uvp.UVParameter(
            "antenna_numbers",
            description=desc,
            form=("Nants_telescope",),
            expected_type=int,
        )

        desc = (
            "Array giving coordinates of antennas relative to "
            "telescope_location (ITRF frame), shape (Nants_telescope, 3), "
            "units meters. See the tutorial page in the documentation "
            "for an example of how to convert this to topocentric frame."
        )
        self._antenna_positions = uvp.UVParameter(
            "antenna_positions",
            description=desc,
            form=("Nants_telescope", 3),
            expected_type=np.float,
            tols=1e-3,  # 1 mm
        )

        # -------- extra, non-required parameters ----------
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

        blt_order_options = ["time", "baseline", "ant1", "ant2", "bda"]
        desc = (
            "Ordering of the data array along the blt axis. A tuple with "
            'the major and minor order (minor order is omitted if order is "bda"). '
            "The allowed values are: "
            + " ,".join([str(val) for val in blt_order_options])
        )
        self._blt_order = uvp.UVParameter(
            "blt_order",
            description=desc,
            form=(2,),
            required=False,
            expected_type=str,
            acceptable_vals=blt_order_options,
        )

        desc = (
            "Any user supplied extra keywords, type=dict. Keys should be "
            "8 character or less strings if writing to uvfits or miriad files. "
            'Use the special key "comment" for long multi-line string comments.'
        )
        self._extra_keywords = uvp.UVParameter(
            "extra_keywords",
            required=False,
            description=desc,
            value={},
            spoof_val={},
            expected_type=dict,
        )

        desc = (
            "Array of antenna diameters in meters. Used by CASA to "
            "construct a default beam if no beam is supplied."
        )
        self._antenna_diameters = uvp.UVParameter(
            "antenna_diameters",
            required=False,
            description=desc,
            form=("Nants_telescope",),
            expected_type=np.float,
            tols=1e-3,  # 1 mm
        )

        # --- other stuff ---
        # the below are copied from AIPS memo 117, but could be revised to
        # merge with other sources of data.
        self._gst0 = uvp.UVParameter(
            "gst0",
            required=False,
            description="Greenwich sidereal time at " "midnight on reference date",
            spoof_val=0.0,
            expected_type=np.float,
        )
        self._rdate = uvp.UVParameter(
            "rdate",
            required=False,
            description="Date for which the GST0 or " "whatever... applies",
            spoof_val="",
            form="str",
        )
        self._earth_omega = uvp.UVParameter(
            "earth_omega",
            required=False,
            description="Earth's rotation rate " "in degrees per day",
            spoof_val=360.985,
            expected_type=np.float,
        )
        self._dut1 = uvp.UVParameter(
            "dut1",
            required=False,
            description="DUT1 (google it) AIPS 117 " "calls it UT1UTC",
            spoof_val=0.0,
            expected_type=np.float,
        )
        self._timesys = uvp.UVParameter(
            "timesys",
            required=False,
            description="We only support UTC",
            spoof_val="UTC",
            form="str",
        )

        desc = (
            "FHD thing we do not understand, something about the time "
            "at which the phase center is normal to the chosen UV plane "
            "for phasing"
        )
        self._uvplane_reference_time = uvp.UVParameter(
            "uvplane_reference_time", required=False, description=desc, spoof_val=0
        )

        desc = "Per-antenna and per-frequency equalization coefficients"
        self._eq_coeffs = uvp.UVParameter(
            "eq_coeffs",
            required=False,
            description=desc,
            form=("Nants_telescope", "Nfreqs"),
            expected_type=np.float,
            spoof_val=1.0,
        )

        desc = "Convention for how to remove eq_coeffs from data"
        self._eq_coeffs_convention = uvp.UVParameter(
            "eq_coeffs_convention",
            required=False,
            description=desc,
            form="str",
            spoof_val="divide",
        )

        super(UVData, self).__init__()

    def _set_flex_spw(self):
        """
        Set flex_spw to True, and adjust required parameters.

        This method should not be called directly by users; instead it is called
        by the file-reading methods to indicate that an object has multiple spectral
        windows concatenated together across the frequency axis.
        """
        # Mark once-optional arrays as now required
        self.flex_spw = True
        self._flex_spw_id_array.required = True
        # Now make sure that chan_width is set to be an array
        self._channel_width.form = ("Nfreqs",)

    def _set_multi_object(self, preserve_source_info=True):
        """
        Set multi_object to True, and adjust required paramteres.

        This method is typically not be called directly by users; instead it is called
        by the file-reading methods to indicate that an object has multiple sources
        together in the same data set.

        Parameters
        ----------
        preserve_source_info : bool
            Preserve the source information located in object_name, and for phased data
            sets, also phase_center_ra, phase_center_dec, phase_center_epoch and
            phase_center_frame. Default is True.

        Raises
        ------
        ValueError
            if the telescope_name is not in known telescopes
        """
        self.multi_object = True

        # Mark once-option arrays as now required
        self._object_id_array.required = True
        self._Nobjects.required = True
        self._object_dict.required = True

        if preserve_source_info:
            if self.phase_type == "phased":
                # "Phased" objects in the parlance of multi-object is a "sidereal"
                # source -- a celestial body that does not require an ephemeris to
                # describe the motion of.
                self.object_name = [self.object_name]
                object_dict = {
                    "object_type": "sidereal",
                    "object_name": self.object_name[0],
                    "object_ra": self.phase_center_ra,
                    "object_dec": self.phase_center_dec,
                    "coord_frame": self.phase_center_frame,
                    "coord_epoch": self.phase_center_epoch,
                }
                if object_dict["coord_frame"] is None:
                    object_dict["coord_frame"] = "icrs"
            elif self.phase_type == "drift":
                # A "drift" phase-typed is really just an unphased data set, not
                # actually what one might call a drift scan (array is co-planar to a
                # fixed az and el). In the lingo of multi-object data sets, we just
                # call this unphased.
                self.object_name = ["unphased"]
                # Unphased dictionary are simple -- they just require two elements
                object_dict = {"object_type": "unphased", "object_name": "unphased"}

            elif self.phase_type == "unknown":
                raise ValueError(
                    "Cannot preserve object properties of the existing data set if "
                    "the phasing type is unknown! Either specify phasing type, or "
                    "set /preserve_source_info=False."
                )
            # Create our object dictionary
            self.object_dict = {object_dict["object_name"]: object_dict}
            # When convering from single-source, all baselines should be phased to
            # a single target
            self.object_id_array = np.ones(self.Nblts, dtype=np.int)
            self.Nobjects = 1
        else:
            # Convert object_name into an empty list
            self.object_name = []
            # Make a dummy dict for the objects
            self.object_dict = {}
            # Mark the ID array as having no sources
            self.object_id_array = np.zeros(self.Nblts, dtype=np.int) - 1
            # Set the total number of sources to zero
            self.Nobjects = 0

    def _add_object(
        self,
        object_name,
        object_type,
        object_lon=None,
        object_lat=None,
        object_frame=None,
        object_epoch=None,
        object_pm_ra=None,
        object_pm_dec=None,
        object_parallax=None,
        object_rad_vel=None,
    ):
        """
        Add an entry to the internal object/source catalog.

        This is a helper function for adding adding a source to the internal object
        catalog, contained within the attribute object_dict.
        """
        # Check whether we should actually be doing this in the first place
        if not self.mutli_source:
            raise TypeError("Cannot add a source if multi_object != True!")

        # Object names serve as dict keys, so we need to make sure that they're unique
        if object_name in self.object_name:
            souDiffs = 0
            object_dict = self.object_dict[object_name]
            souDiffs += object_type != object_type

            if "object_lon" in object_dict.keys():
                souDiffs += object_dict["object_lon"] != object_lon
            else:
                souDiffs += object_lon is not None

            if "object_lat" in object_dict.keys():
                souDiffs += object_dict["object_lat"] != object_lat
            else:
                souDiffs += object_lat is not None

            if "object_frame" in object_dict.keys():
                souDiffs += object_dict["object_frame"] != object_frame
            else:
                souDiffs += object_frame is not None

            if "object_epoch" in object_dict.keys():
                souDiffs += object_dict["object_epoch"] != object_epoch
            else:
                souDiffs += object_epoch is not None

            if "object_pm_ra" in object_dict.keys():
                souDiffs += object_dict["object_pm_ra"] != object_pm_ra
            else:
                souDiffs += object_pm_ra is not None

            if "object_pm_dec" in object_dict.keys():
                souDiffs += object_dict["object_pm_dec"] != object_pm_dec
            else:
                souDiffs += object_pm_dec is not None

            if "object_parallax" in object_dict.keys():
                souDiffs += object_dict["object_parallax"] != object_parallax
            else:
                souDiffs += object_parallax is not None

            if "object_rad_vel" in object_dict.keys():
                souDiffs += object_dict["object_rad_vel"] != object_rad_vel
            else:
                souDiffs += object_rad_vel is not None

            # If the source does have the same name, check to see if all the
            # atributes match. If so, no problem, go about your business
            if souDiffs == 0:
                # Everything matches, return back the object ID of the matching entry
                return self.object_name.index(object_name)
            else:
                raise IndexError("Cannot add different source with an identical name!")

        # If source is unique, begin creating a dictionary for it
        if object_type == "sidereal":
            if object_lon is None:
                raise TypeError("object_lon cannot be None for sidereal object!")
            elif object_lat is None:
                raise TypeError("object_lat cannot be None for sidereal object!")
            elif object_frame is None:
                raise TypeError("object_frame cannot be None for sidereal object!")

            object_dict = {
                "object_type": "sidereal",
                "object_lon": object_lon,
                "object_lat": object_lat,
                "object_frame": object_frame,
            }

            if object_epoch:
                object_dict["object_epoch"] = object_epoch
            if object_pm_ra:
                object_dict["object_pm_ra"] = object_pm_ra
            if object_pm_dec:
                object_dict["object_pm_dec"] = object_pm_dec
            if object_rad_vel:
                object_dict["object_rad_vel"] = object_pm_dec
            if object_parallax:
                object_dict["object_parallax"] = object_parallax

        elif object_type == "ephem":
            raise NotImplementedError("Ephemeris objects are not (yet) supported!")

        elif object_type == "driftscan":
            object_dict = {
                "object_type": "driftscan",
                "object_lon": object_lon if (object_lon is not None) else np.pi,
                "object_lat": object_lat if (object_lat is not None) else (np.pi / 2),
            }

        elif object_type == "unphased":
            object_dict = {"object_type": "unphased"}

        self.object_name.append(object_name)
        self.object_dict[object_name] = object_dict
        self.Nobjects += 1
        object_id = self.Nobjects - 1
        return object_id

    def _remove_object(self, defunct_name):
        """
        Remove an entry from the internal object/source catalog.

        Removes an object entry from the attributes object_name and object_dict.
        Only allowed when the UVData object in question is a multi-source data set
        (i.e., multi_object == True).

        Parameters
        ----------
        defunct_name : str
            Name of the source to be removed

        Raises
        ------
        TypeError
            If multi_object is not set to True
        IndexError
            If the name provided is not found in the attribute object_name
        """
        if not self.mutli_object:
            raise TypeError("Cannot remove an object if multi_object != True.")

        if defunct_name not in self.object_name:
            raise IndexError("No source by that name contained in the catalog!")

        self.object_name.remove(defunct_name)
        del self.object_dict[defunct_name]

    def _clear_objects(self):
        """
        Remove objects dictionaries and names that are no longer in use.

        Remoeves
        """
        if not self.mutli_object:
            raise TypeError("Cannot remove an object if multi_object != True.")

        unique_object_ids = self.object_id_array
        defunct_list = []
        good_map = {}
        Nobjects = 0
        for idx in range(len(self.object_name)):
            if idx in unique_object_ids:
                good_map[idx] = Nobjects
                Nobjects += 1
            else:
                defunct_list.append(self.object_name[idx])

        # Check the number of "good" sources we have -- if we haven't dropped any,
        # then we are free to bail, otherwise update the Nobjects attribute
        if Nobjects == self.Nobjects:
            return
        else:
            self.Nobjects = Nobjects

        # Time to kill the entries that are no longer in the source stack
        for defunct_object in defunct_list:
            self.object_name.remove(defunct_object)
            del self.object_dict[defunct_object]

        # Finally, we need to remap the object_id_array, to reflect the fact that we
        # have mucked the primary indexing in the object_name list
        self.object_id_array = np.array(
            [[good_map[object_idx] for object_idx in self.object_id_array]],
            dtype=np.int,
        )

    def _check_for_unphased_objects(self):
        """
        Check which Nblts are unphased in a multi-object dataset.

        This convenience method returns back a boolean mask to identify which data
        along the Blt axis contains unphased objects (which is only applicable when
        multi_object=True)

        Returns
        -------
        blt_mask : ndarray of bool
            A boolean mask for identifying which elements contain unphased objects
        """
        # Gotta be a multi-object data set for this operation to even make sense
        if not self.mutli_object:
            raise TypeError("Cannot remove an object if multi_object != True.")

        # Check and see if we have any unphased objects, in which case
        # their w-values should be zeroed out.
        nophase_dict = {
            idx: self.object_dict[self.object_name[idx]]["object_type"] == "unphased"
            for idx in range(self.Nobjects)
        }

        # Dictionaries make for quick hash access, so us it to construct a bool array
        blt_mask = np.array(
            [nophase_dict[idx] for idx in self.object_id_array], dtype=bool
        )

        return blt_mask

    def _set_drift(self):
        """
        Set phase_type to 'drift' and adjust required parameters.

        This method should not be called directly by users; instead it is called
        by phasing methods and file-reading methods to indicate the object has a
        `phase_type` of "drift" and define which metadata are required.
        """
        self.phase_type = "drift"
        self._phase_center_epoch.required = False
        self._phase_center_ra.required = False
        self._phase_center_dec.required = False

    def set_drift(self):
        """
        Set phase_type to 'drift' and adjust required parameters.

        This method is deprecated, and will be removed in pyuvdata v2.2. Use
        `_set_drift` instead.
        """
        warnings.warn(
            "`set_drift` is deprecated, and will be removed in pyuvdata version "
            "2.2. Use `_set_drift` instead.",
            DeprecationWarning,
        )
        self._set_drift()

    def _set_phased(self):
        """
        Set phase_type to 'phased' and adjust required parameters.

        This method should not be called directly by users; instead it is called
        by phasing methods and file-reading methods to indicate the object has a
        `phase_type` of "phased" and define which metadata are required.
        """
        self.phase_type = "phased"
        self._phase_center_epoch.required = True
        self._phase_center_ra.required = True
        self._phase_center_dec.required = True

    def set_phased(self):
        """
        Set phase_type to 'phased' and adjust required parameters.

        This method is deprecated, and will be removed in pyuvdata v2.2. Use
        `_set_phased` instead.
        """
        warnings.warn(
            "`set_phased` is deprecated, and will be removed in pyuvdata version "
            "2.2. Use `_set_phased` instead.",
            DeprecationWarning,
        )
        self._set_phased()

    def _set_unknown_phase_type(self):
        """
        Set phase_type to 'unknown' and adjust required parameters.

        This method should not be called directly by users; instead it is called
        by file-reading methods to indicate the `phase_type` is "unknown" and
        define which metadata are required.
        """
        self.phase_type = "unknown"
        self._phase_center_epoch.required = False
        self._phase_center_ra.required = False
        self._phase_center_dec.required = False

    def set_unknown_phase_type(self):
        """
        Set phase_type to 'unknown' and adjust required parameters.

        This method is deprecated, and will be removed in pyuvdata v2.2. Use
        `_set_unknown_phase_type` instead.
        """
        warnings.warn(
            "`set_unknown_phase_type` is deprecated, and will be removed in "
            "pyuvdata version 2.2. Use `_set_unknown_phase_type` instead.",
            DeprecationWarning,
        )
        self._set_unknown_phase_type()

    @property
    def _data_params(self):
        """List of strings giving the data-like parameters."""
        return ["data_array", "nsample_array", "flag_array"]

    @property
    def data_like_parameters(self):
        """Iterate defined parameters which are data-like (not metadata-like)."""
        for key in self._data_params:
            if hasattr(self, key):
                yield getattr(self, key)

    @property
    def metadata_only(self):
        """
        Property that determines whether this is a metadata only object.

        An object is metadata only if data_array, nsample_array and flag_array
        are all None.
        """
        metadata_only = all(d is None for d in self.data_like_parameters)

        for param_name in self._data_params:
            getattr(self, "_" + param_name).required = not metadata_only

        return metadata_only

    def known_telescopes(self):
        """
        Get a list of telescopes known to pyuvdata.

        This is just a shortcut to uvdata.telescopes.known_telescopes()

        Returns
        -------
        list of str
            List of names of known telescopes
        """
        return uvtel.known_telescopes()

    def set_telescope_params(self, overwrite=False):
        """
        Set telescope related parameters.

        If the telescope_name is in the known_telescopes, set any missing
        telescope-associated parameters (e.g. telescope location) to the value
        for the known telescope.

        Parameters
        ----------
        overwrite : bool
            Option to overwrite existing telescope-associated parameters with
            the values from the known telescope.

        Raises
        ------
        ValueError
            if the telescope_name is not in known telescopes
        """
        telescope_obj = uvtel.get_telescope(self.telescope_name)
        if telescope_obj is not False:
            params_set = []
            for p in telescope_obj:
                telescope_param = getattr(telescope_obj, p)
                self_param = getattr(self, p)
                if telescope_param.value is not None and (
                    overwrite is True or self_param.value is None
                ):
                    telescope_shape = telescope_param.expected_shape(telescope_obj)
                    self_shape = self_param.expected_shape(self)
                    if telescope_shape == self_shape:
                        params_set.append(self_param.name)
                        prop_name = self_param.name
                        setattr(self, prop_name, getattr(telescope_obj, prop_name))
                    else:
                        # expected shapes aren't equal. This can happen
                        # e.g. with diameters,
                        # which is a single value on the telescope object but is
                        # an array of length Nants_telescope on the UVData object

                        # use an assert here because we want an error if this condition
                        # isn't true, but it's really an internal consistency check.
                        # This will error if there are changes to the Telescope
                        # object definition, but nothing that a normal user
                        # does will cause an error
                        assert telescope_shape == () and self_shape != "str"
                        # this parameter is as of this comment most likely a float
                        # since only diameters and antenna positions will probably
                        # trigger this else statement
                        # assign float64 as the type of the array
                        array_val = (
                            np.zeros(self_shape, dtype=np.float64,)
                            + telescope_param.value
                        )
                        params_set.append(self_param.name)
                        prop_name = self_param.name
                        setattr(self, prop_name, array_val)

            if len(params_set) > 0:
                params_set_str = ", ".join(params_set)
                warnings.warn(
                    "{params} is not set. Using known values "
                    "for {telescope_name}.".format(
                        params=params_set_str,
                        telescope_name=telescope_obj.telescope_name,
                    )
                )
        else:
            raise ValueError(
                f"Telescope {self.telescope_name} is not in known_telescopes."
            )

    def _calc_single_integration_time(self):
        """
        Calculate a single integration time in seconds when not otherwise specified.

        This function computes the shortest time difference present in the
        time_array, and returns it to be used as the integration time for all
        samples.

        Returns
        -------
        int_time : int
            integration time in seconds to be assigned to all samples in the data.

        """
        # The time_array is in units of days, and integration_time has units of
        # seconds, so we need to convert.
        return np.diff(np.sort(list(set(self.time_array))))[0] * 86400

    def _set_lsts_helper(self):
        latitude, longitude, altitude = self.telescope_location_lat_lon_alt_degrees
        unique_times, inverse_inds = np.unique(self.time_array, return_inverse=True)
        unique_lst_array = uvutils.get_lst_for_time(
            unique_times, latitude, longitude, altitude
        )
        self.lst_array = unique_lst_array[inverse_inds]
        return

    def _set_app_coords_helper(self, use_astropy=False):
        """
        Set values for the apparent coordinate arrays.

        This is an internal helper function, which is not designed to be called by
        users, but rather individual read/write functions for the UVData object.
        Users should use the phase() method for updating/adjusting coordiante values.

        Parameters
        ----------
        use_astropy : bool, False
            Use astropy for the calculation of apparent coordiantes (setting this to
            false will instead call NOVAS).
        """
        if self.multi_source:
            pass
        else:
            # So this is actually the easier of the two cases -- just use the object
            # properties to fill in the relevant data
            app_ra, app_dec = uvutils.calc_app_coords(
                self.phase_center_ra,
                self.phase_center_dec,
                self.phase_center_frame,
                coord_epoch=self.phase_center_epoch,
                time_array=self.time_array,
                lst_array=self.lst_array,
                telescope_lat=self.telescope_location_lat_lon_alt[0],
                telescope_lon=self.telescope_location_lat_lon_alt[1],
                telescope_alt=self.telescope_location_lat_lon_alt[2],
                use_astropy=use_astropy,
                object_type="sidereal",
            )

    def set_lsts_from_time_array(self, background=False):
        """Set the lst_array based from the time_array.

        Parameters
        ----------
        background : bool, False
            When set to True, start the calculation on a threading.Thread in the
            background and return the thread to the user.

        Returns
        -------
        proc : None or threading.Thread instance
            When background is set to True, a thread is returned which must be
            joined before the lst_array exists on the UVData object.

        """
        if not background:
            self._set_lsts_helper()
            return
        else:
            proc = threading.Thread(target=self._set_lsts_helper)
            proc.start()
            return proc

    def _check_flex_spw_contiguous(self):
        """
        Check if the spectral windows are contiguous for flex_spw datasets.

        This checks the flex_spw_id_array to make sure that all channels for each
        spectral window are together in one block, versus being interspersed (e.g.,
        channel #1 and #3 is in spw #1, channels #2 and #4 are in spw #2). In theory,
        UVH5 and UVData objects can handle this, but MIRIAD, MIR, UVFITS, and MS file
        formats cannot, so we just consider it forbidden.
        """
        if self.flex_spw:
            exp_spw_ids = np.arange(self.Nspws, dtype=np.int)
            # This is an internal consistency check to make sure that the indexes match
            # up as expected -- this shouldn't error unless someone is mucking with
            # settings they shouldn't be.
            assert np.all(np.unique(self.flex_spw_id_array) == exp_spw_ids)

            n_breaks = np.sum(self.flex_spw_id_array[1:] != self.flex_spw_id_array[:-1])
            if (n_breaks + 1) != self.Nspws:
                raise ValueError(
                    "Channels from different spectral windows are interspersed with "
                    "one another, rather than being grouped together along the "
                    "frequency axis. Most file formats do not support such "
                    "non-grouping of data."
                )
        else:
            # If this isn't a flex_spw data set, then there is only 1 spectral window,
            # which means that the check always passes
            pass
        return True

    def _check_freq_spacing(self):
        """
        Check if frequencies are evenly spaced and separated by their channel width.

        This is a requirement for writing uvfits & miriad files.
        """
        raise_spacing_error = False
        raise_chanwidth_error = False
        # TODO: Spw axis to be collapsed in future release
        freq_spacing = np.diff(self.freq_array, axis=1)
        if self.Nfreqs == 1:
            # Skip all of this if there is only 1 channel
            pass
        elif self.flex_spw:
            # Check to make sure that the flexible spectral window has indicies set up
            # correctly (grouped together) for this check
            self._check_flex_spw_contiguous()
            diff_chanwidth = np.diff(self.channel_width)
            freq_dir = []
            for idx in range(self.Nspws):
                chan_mask = self.flex_spw_id_array == idx
                freq_dir += [
                    np.sign(np.mean(np.diff(self.freq_array[0][chan_mask])))
                ] * np.sum(chan_mask)

            # Pop off the first entry, since the above arrays are diff'd
            # (and thus one element shorter)
            freq_dir = np.array(freq_dir[1:])
            # Ignore cases where looking at the boundaries of spectral windows
            bypass_check = self.flex_spw_id_array[1:] != self.flex_spw_id_array[:-1]
            if not np.all(
                np.logical_or(
                    bypass_check,
                    np.isclose(
                        diff_chanwidth,
                        0.0,
                        rtol=self._freq_array.tols[0],
                        atol=self._freq_array.tols[1],
                    ),
                )
            ):
                raise_spacing_error = True
            if not np.all(
                np.logical_or(
                    bypass_check,
                    np.isclose(
                        freq_spacing,
                        self.channel_width[1:] * freq_dir,
                        rtol=self._freq_array.tols[0],
                        atol=self._freq_array.tols[1],
                    ),
                )
            ):
                raise_chanwidth_error = True
        else:
            freq_dir = np.sign(np.mean(freq_spacing))
            if not np.isclose(
                np.min(freq_spacing),
                np.max(freq_spacing),
                rtol=self._freq_array.tols[0],
                atol=self._freq_array.tols[1],
            ):
                raise_spacing_error = True
            # TODO: Spw axis to be collapsed in future release
            if not np.isclose(
                np.mean(freq_spacing[0]),
                self.channel_width * freq_dir,
                rtol=self._channel_width.tols[0],
                atol=self._channel_width.tols[1],
            ):
                raise_chanwidth_error = True
        if raise_spacing_error:
            raise ValueError(
                "The frequencies are not evenly spaced (probably "
                "because of a select operation) or has differing "
                "values of channel widths. Some file formats "
                "(e.g. uvfits, miriad) and methods (frequency_average) "
                "do not support unevenly spaced frequencies."
            )
        if raise_chanwidth_error:
            raise ValueError(
                "The frequencies are separated by more than their "
                "channel width (probably because of a select operation). "
                "Some file formats (e.g. uvfits, miriad) and "
                "methods (frequency_average) do not support "
                "frequencies that are spaced by more than their "
                "channel width."
            )

        return True

    def _calc_nants_data(self):
        """Calculate the number of antennas from ant_1_array and ant_2_array arrays."""
        return int(np.union1d(self.ant_1_array, self.ant_2_array).size)

    def check(
        self,
        check_extra=True,
        run_check_acceptability=True,
        check_freq_spacing=False,
        strict_uvw_antpos_check=False,
    ):
        """
        Add some extra checks on top of checks on UVBase class.

        Check that required parameters exist. Check that parameters have
        appropriate shapes and optionally that the values are acceptable.

        Parameters
        ----------
        check_extra : bool
            If true, check all parameters, otherwise only check required parameters.
        run_check_acceptability : bool
            Option to check if values in parameters are acceptable.
        check_freq_spacing :  bool
            Option to check if frequencies are evenly spaced and the spacing is
            equal to their channel_width. This is not required for UVData
            objects in general but is required to write to uvfits and miriad files.
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.

        Returns
        -------
        bool
            True if check passes

        Raises
        ------
        ValueError
            if parameter shapes or types are wrong or do not have acceptable
            values (if run_check_acceptability is True)
        """
        # first run the basic check from UVBase
        # set the phase type based on object's value
        if self.phase_type == "phased":
            self._set_phased()
        elif self.phase_type == "drift":
            self._set_drift()
        else:
            self._set_unknown_phase_type()

        super(UVData, self).check(
            check_extra=check_extra, run_check_acceptability=run_check_acceptability
        )

        # Check internal consistency of numbers which don't explicitly correspond
        # to the shape of another array.
        if self.Nants_data != self._calc_nants_data():
            raise ValueError(
                "Nants_data must be equal to the number of unique "
                "values in ant_1_array and ant_2_array"
            )

        if self.Nbls != len(np.unique(self.baseline_array)):
            raise ValueError(
                "Nbls must be equal to the number of unique "
                "baselines in the data_array"
            )

        if self.Ntimes != len(np.unique(self.time_array)):
            raise ValueError(
                "Ntimes must be equal to the number of unique "
                "times in the time_array"
            )

        # require that all entries in ant_1_array and ant_2_array exist in
        # antenna_numbers
        if not set(np.unique(self.ant_1_array)).issubset(self.antenna_numbers):
            raise ValueError("All antennas in ant_1_array must be in antenna_numbers.")
        if not set(np.unique(self.ant_2_array)).issubset(self.antenna_numbers):
            raise ValueError("All antennas in ant_2_array must be in antenna_numbers.")

        # issue warning if extra_keywords keys are longer than 8 characters
        for key in self.extra_keywords.keys():
            if len(key) > 8:
                warnings.warn(
                    "key {key} in extra_keywords is longer than 8 "
                    "characters. It will be truncated to 8 if written "
                    "to uvfits or miriad file formats.".format(key=key)
                )

        # issue warning if extra_keywords values are lists, arrays or dicts
        for key, value in self.extra_keywords.items():
            if isinstance(value, (list, dict, np.ndarray)):
                warnings.warn(
                    "{key} in extra_keywords is a list, array or dict, "
                    "which will raise an error when writing uvfits or "
                    "miriad file types".format(key=key)
                )

        if run_check_acceptability:
            # check that the uvws make sense given the antenna positions
            # make a metadata only copy of this object to properly calculate uvws
            temp_obj = self.copy(metadata_only=True)

            if temp_obj.phase_center_frame is not None:
                output_phase_frame = temp_obj.phase_center_frame
            else:
                output_phase_frame = "icrs"

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                temp_obj.set_uvws_from_antenna_positions(
                    allow_phasing=True, output_phase_frame=output_phase_frame
                )

            if not np.allclose(temp_obj.uvw_array, self.uvw_array, atol=1):
                max_diff = np.max(np.abs(temp_obj.uvw_array - self.uvw_array))
                if not strict_uvw_antpos_check:
                    warnings.warn(
                        "The uvw_array does not match the expected values given "
                        "the antenna positions. The largest discrepancy is "
                        f"{max_diff} meters. This is a fairly common situation "
                        "but might indicate an error in the antenna positions, "
                        "the uvws or the phasing."
                    )
                else:
                    raise ValueError(
                        "The uvw_array does not match the expected values given "
                        "the antenna positions. The largest discrepancy is "
                        f"{max_diff} meters."
                    )

            # check auto and cross-corrs have sensible uvws
            autos = np.isclose(self.ant_1_array - self.ant_2_array, 0.0)
            if not np.all(
                np.isclose(
                    self.uvw_array[autos],
                    0.0,
                    rtol=self._uvw_array.tols[0],
                    atol=self._uvw_array.tols[1],
                )
            ):
                raise ValueError(
                    "Some auto-correlations have non-zero uvw_array coordinates."
                )
            if np.any(
                np.isclose(
                    # this line used to use np.linalg.norm but it turns out
                    # squaring and sqrt is slightly more efficient unless the array
                    # is "very large".
                    np.sqrt(
                        self.uvw_array[~autos, 0] ** 2
                        + self.uvw_array[~autos, 1] ** 2
                        + self.uvw_array[~autos, 2] ** 2
                    ),
                    0.0,
                    rtol=self._uvw_array.tols[0],
                    atol=self._uvw_array.tols[1],
                )
            ):
                raise ValueError(
                    "Some cross-correlations have near-zero uvw_array magnitudes."
                )

        if check_freq_spacing:
            self._check_freq_spacing()

        return True

    def copy(self, metadata_only=False):
        """
        Make and return a copy of the UVData object.

        Parameters
        ----------
        metadata_only : bool
            If True, only copy the metadata of the object.

        Returns
        -------
        UVData
            Copy of self.
        """
        if not metadata_only:
            return super(UVData, self).copy()
        else:
            uv = UVData()
            # include all attributes, not just UVParameter ones.
            for attr in self.__iter__(uvparams_only=False):
                # skip properties
                if isinstance(getattr(type(self), attr, None), property):
                    continue

                # skip data like parameters
                # parameter names have a leading underscore we want to ignore
                if attr.lstrip("_") in self._data_params:
                    continue
                setattr(uv, attr, copy.deepcopy(getattr(self, attr)))

            return uv

    def baseline_to_antnums(self, baseline):
        """
        Get the antenna numbers corresponding to a given baseline number.

        Parameters
        ----------
        baseline : int or array_like of int
            baseline number

        Returns
        -------
        int or array_like of int
            first antenna number(s)
        int or array_like of int
            second antenna number(s)
        """
        return uvutils.baseline_to_antnums(baseline, self.Nants_telescope)

    def antnums_to_baseline(self, ant1, ant2, attempt256=False):
        """
        Get the baseline number corresponding to two given antenna numbers.

        Parameters
        ----------
        ant1 : int or array_like of int
            first antenna number
        ant2 : int or array_like of int
            second antenna number
        attempt256 : bool
            Option to try to use the older 256 standard used in many uvfits files
            (will use 2048 standard if there are more than 256 antennas).

        Returns
        -------
        int or array of int
            baseline number corresponding to the two antenna numbers.
        """
        return uvutils.antnums_to_baseline(
            ant1, ant2, self.Nants_telescope, attempt256=attempt256
        )

    def antpair2ind(self, ant1, ant2=None, ordered=True):
        """
        Get indices along the baseline-time axis for a given antenna pair.

        This will search for either the key as specified, or the key and its
        conjugate.

        Parameters
        ----------
        ant1, ant2 : int
            Either an antenna-pair key, or key expanded as arguments,
            e.g. antpair2ind( (10, 20) ) or antpair2ind(10, 20)
        ordered : bool
            If True, search for antpair as provided, else search for it and
            its conjugate.

        Returns
        -------
        inds : ndarray of int-64
            indices of the antpair along the baseline-time axis.
        """
        # check for expanded antpair or key
        if ant2 is None:
            if not isinstance(ant1, tuple):
                raise ValueError(
                    "antpair2ind must be fed an antpair tuple "
                    "or expand it as arguments"
                )
            ant2 = ant1[1]
            ant1 = ant1[0]
        else:
            if not isinstance(ant1, (int, np.integer)):
                raise ValueError(
                    "antpair2ind must be fed an antpair tuple or "
                    "expand it as arguments"
                )
        if not isinstance(ordered, (bool, np.bool)):
            raise ValueError("ordered must be a boolean")

        # if getting auto-corr, ordered must be True
        if ant1 == ant2:
            ordered = True

        # get indices
        inds = np.where((self.ant_1_array == ant1) & (self.ant_2_array == ant2))[0]
        if ordered:
            return inds
        else:
            ind2 = np.where((self.ant_1_array == ant2) & (self.ant_2_array == ant1))[0]
            inds = np.asarray(np.append(inds, ind2), dtype=np.int64)
            return inds

    def _key2inds(self, key):
        """
        Interpret user specified key as antenna pair and/or polarization.

        Parameters
        ----------
        key : tuple of int
            Identifier of data. Key can be length 1, 2, or 3:

            if len(key) == 1:
                if (key < 5) or (type(key) is str):  interpreted as a
                             polarization number/name, return all blts for that pol.
                else: interpreted as a baseline number. Return all times and
                      polarizations for that baseline.

            if len(key) == 2: interpreted as an antenna pair. Return all
                times and pols for that baseline.

            if len(key) == 3: interpreted as antenna pair and pol (ant1, ant2, pol).
                Return all times for that baseline, pol. pol may be a string.

        Returns
        -------
        blt_ind1 : ndarray of int
            blt indices for antenna pair.
        blt_ind2 : ndarray of int
            blt indices for conjugate antenna pair.
            Note if a cross-pol baseline is requested, the polarization will
            also be reversed so the appropriate correlations are returned.
            e.g. asking for (1, 2, 'xy') may return conj(2, 1, 'yx'), which
            is equivalent to the requesting baseline. See utils.conj_pol() for
            complete conjugation mapping.
        pol_ind : tuple of ndarray of int
            polarization indices for blt_ind1 and blt_ind2

        """
        key = uvutils._get_iterable(key)
        if type(key) is str:
            # Single string given, assume it is polarization
            pol_ind1 = np.where(
                self.polarization_array
                == uvutils.polstr2num(key, x_orientation=self.x_orientation)
            )[0]
            if len(pol_ind1) > 0:
                blt_ind1 = np.arange(self.Nblts, dtype=np.int64)
                blt_ind2 = np.array([], dtype=np.int64)
                pol_ind2 = np.array([], dtype=np.int64)
                pol_ind = (pol_ind1, pol_ind2)
            else:
                raise KeyError("Polarization {pol} not found in data.".format(pol=key))
        elif len(key) == 1:
            key = key[0]  # For simplicity
            if isinstance(key, Iterable):
                # Nested tuple. Call function again.
                blt_ind1, blt_ind2, pol_ind = self._key2inds(key)
            elif key < 5:
                # Small number, assume it is a polarization number a la AIPS memo
                pol_ind1 = np.where(self.polarization_array == key)[0]
                if len(pol_ind1) > 0:
                    blt_ind1 = np.arange(self.Nblts)
                    blt_ind2 = np.array([], dtype=np.int64)
                    pol_ind2 = np.array([], dtype=np.int64)
                    pol_ind = (pol_ind1, pol_ind2)
                else:
                    raise KeyError(
                        "Polarization {pol} not found in data.".format(pol=key)
                    )
            else:
                # Larger number, assume it is a baseline number
                inv_bl = self.antnums_to_baseline(
                    self.baseline_to_antnums(key)[1], self.baseline_to_antnums(key)[0]
                )
                blt_ind1 = np.where(self.baseline_array == key)[0]
                blt_ind2 = np.where(self.baseline_array == inv_bl)[0]
                if len(blt_ind1) + len(blt_ind2) == 0:
                    raise KeyError("Baseline {bl} not found in data.".format(bl=key))
                if len(blt_ind1) > 0:
                    pol_ind1 = np.arange(self.Npols)
                else:
                    pol_ind1 = np.array([], dtype=np.int64)
                if len(blt_ind2) > 0:
                    try:
                        pol_ind2 = uvutils.reorder_conj_pols(self.polarization_array)
                    except ValueError:
                        if len(blt_ind1) == 0:
                            raise KeyError(
                                f"Baseline {key} not found for polarization "
                                "array in data."
                            )
                        else:
                            pol_ind2 = np.array([], dtype=np.int64)
                            blt_ind2 = np.array([], dtype=np.int64)
                else:
                    pol_ind2 = np.array([], dtype=np.int64)
                pol_ind = (pol_ind1, pol_ind2)
        elif len(key) == 2:
            # Key is an antenna pair
            blt_ind1 = self.antpair2ind(key[0], key[1])
            blt_ind2 = self.antpair2ind(key[1], key[0])
            if len(blt_ind1) + len(blt_ind2) == 0:
                raise KeyError("Antenna pair {pair} not found in data".format(pair=key))
            if len(blt_ind1) > 0:
                pol_ind1 = np.arange(self.Npols)
            else:
                pol_ind1 = np.array([], dtype=np.int64)
            if len(blt_ind2) > 0:
                try:
                    pol_ind2 = uvutils.reorder_conj_pols(self.polarization_array)
                except ValueError:
                    if len(blt_ind1) == 0:
                        raise KeyError(
                            f"Baseline {key} not found for polarization array in data."
                        )
                    else:
                        pol_ind2 = np.array([], dtype=np.int64)
                        blt_ind2 = np.array([], dtype=np.int64)
            else:
                pol_ind2 = np.array([], dtype=np.int64)
            pol_ind = (pol_ind1, pol_ind2)
        elif len(key) == 3:
            # Key is an antenna pair + pol
            blt_ind1 = self.antpair2ind(key[0], key[1])
            blt_ind2 = self.antpair2ind(key[1], key[0])
            if len(blt_ind1) + len(blt_ind2) == 0:
                raise KeyError(
                    "Antenna pair {pair} not found in "
                    "data".format(pair=(key[0], key[1]))
                )
            if type(key[2]) is str:
                # pol is str
                if len(blt_ind1) > 0:
                    pol_ind1 = np.where(
                        self.polarization_array
                        == uvutils.polstr2num(key[2], x_orientation=self.x_orientation)
                    )[0]
                else:
                    pol_ind1 = np.array([], dtype=np.int64)
                if len(blt_ind2) > 0:
                    pol_ind2 = np.where(
                        self.polarization_array
                        == uvutils.polstr2num(
                            uvutils.conj_pol(key[2]), x_orientation=self.x_orientation
                        )
                    )[0]
                else:
                    pol_ind2 = np.array([], dtype=np.int64)
            else:
                # polarization number a la AIPS memo
                if len(blt_ind1) > 0:
                    pol_ind1 = np.where(self.polarization_array == key[2])[0]
                else:
                    pol_ind1 = np.array([], dtype=np.int64)
                if len(blt_ind2) > 0:
                    pol_ind2 = np.where(
                        self.polarization_array == uvutils.conj_pol(key[2])
                    )[0]
                else:
                    pol_ind2 = np.array([], dtype=np.int64)
            pol_ind = (pol_ind1, pol_ind2)
            if len(blt_ind1) * len(pol_ind[0]) + len(blt_ind2) * len(pol_ind[1]) == 0:
                raise KeyError(
                    "Polarization {pol} not found in data.".format(pol=key[2])
                )
        # Catch autos
        if np.array_equal(blt_ind1, blt_ind2):
            blt_ind2 = np.array([], dtype=np.int64)
        return (blt_ind1, blt_ind2, pol_ind)

    def _smart_slicing(
        self, data, ind1, ind2, indp, squeeze="default", force_copy=False
    ):
        """
        Quickly get the relevant section of a data-like array.

        Used in get_data, get_flags and get_nsamples.

        Parameters
        ----------
        data : ndarray
            4-dimensional array shaped like self.data_array
        ind1 : array_like of int
            blt indices for antenna pair (e.g. from self._key2inds)
        ind2 : array_like of int
            blt indices for conjugate antenna pair. (e.g. from self._key2inds)
        indp : tuple array_like of int
            polarization indices for ind1 and ind2 (e.g. from self._key2inds)
        squeeze : str
            string specifying how to squeeze the returned array. Options are:
            'default': squeeze pol and spw dimensions if possible;
            'none': no squeezing of resulting numpy array;
            'full': squeeze all length 1 dimensions.
        force_copy : bool
            Option to explicitly make a copy of the data.

        Returns
        -------
        ndarray
            copy (or if possible, a read-only view) of relevant section of data
        """
        p_reg_spaced = [False, False]
        p_start = [0, 0]
        p_stop = [0, 0]
        dp = [1, 1]
        for i, pi in enumerate(indp):
            if len(pi) == 0:
                continue
            if len(set(np.ediff1d(pi))) <= 1:
                p_reg_spaced[i] = True
                p_start[i] = pi[0]
                p_stop[i] = pi[-1] + 1
                if len(pi) != 1:
                    dp[i] = pi[1] - pi[0]

        if len(ind2) == 0:
            # only unconjugated baselines
            if len(set(np.ediff1d(ind1))) <= 1:
                blt_start = ind1[0]
                blt_stop = ind1[-1] + 1
                if len(ind1) == 1:
                    dblt = 1
                else:
                    dblt = ind1[1] - ind1[0]
                if p_reg_spaced[0]:
                    out = data[
                        blt_start:blt_stop:dblt, :, :, p_start[0] : p_stop[0] : dp[0]
                    ]
                else:
                    out = data[blt_start:blt_stop:dblt, :, :, indp[0]]
            else:
                out = data[ind1, :, :, :]
                if p_reg_spaced[0]:
                    out = out[:, :, :, p_start[0] : p_stop[0] : dp[0]]
                else:
                    out = out[:, :, :, indp[0]]
        elif len(ind1) == 0:
            # only conjugated baselines
            if len(set(np.ediff1d(ind2))) <= 1:
                blt_start = ind2[0]
                blt_stop = ind2[-1] + 1
                if len(ind2) == 1:
                    dblt = 1
                else:
                    dblt = ind2[1] - ind2[0]
                if p_reg_spaced[1]:
                    out = np.conj(
                        data[
                            blt_start:blt_stop:dblt,
                            :,
                            :,
                            p_start[1] : p_stop[1] : dp[1],
                        ]
                    )
                else:
                    out = np.conj(data[blt_start:blt_stop:dblt, :, :, indp[1]])
            else:
                out = data[ind2, :, :, :]
                if p_reg_spaced[1]:
                    out = np.conj(out[:, :, :, p_start[1] : p_stop[1] : dp[1]])
                else:
                    out = np.conj(out[:, :, :, indp[1]])
        else:
            # both conjugated and unconjugated baselines
            out = (data[ind1, :, :, :], np.conj(data[ind2, :, :, :]))
            if p_reg_spaced[0] and p_reg_spaced[1]:
                out = np.append(
                    out[0][:, :, :, p_start[0] : p_stop[0] : dp[0]],
                    out[1][:, :, :, p_start[1] : p_stop[1] : dp[1]],
                    axis=0,
                )
            else:
                out = np.append(
                    out[0][:, :, :, indp[0]], out[1][:, :, :, indp[1]], axis=0
                )

        if squeeze == "full":
            out = np.squeeze(out)
        elif squeeze == "default":
            if out.shape[3] == 1:
                # one polarization dimension
                out = np.squeeze(out, axis=3)
            if out.shape[1] == 1:
                # one spw dimension
                out = np.squeeze(out, axis=1)
        elif squeeze != "none":
            raise ValueError(
                '"' + str(squeeze) + '" is not a valid option for squeeze.'
                'Only "default", "none", or "full" are allowed.'
            )

        if force_copy:
            out = np.array(out)
        elif out.base is not None:
            # if out is a view rather than a copy, make it read-only
            out.flags.writeable = False

        return out

    def get_ants(self):
        """
        Get the unique antennas that have data associated with them.

        Returns
        -------
        ndarray of int
            Array of unique antennas with data associated with them.
        """
        return np.unique(np.append(self.ant_1_array, self.ant_2_array))

    def get_baseline_nums(self):
        """
        Get the unique baselines that have data associated with them.

        Returns
        -------
        ndarray of int
            Array of unique baselines with data associated with them.
        """
        return np.unique(self.baseline_array)

    def get_antpairs(self):
        """
        Get the unique antpair tuples that have data associated with them.

        Returns
        -------
        list of tuples of int
            list of unique antpair tuples (ant1, ant2) with data associated with them.
        """
        return [self.baseline_to_antnums(bl) for bl in self.get_baseline_nums()]

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

    def get_antpairpols(self):
        """
        Get the unique antpair + pol tuples that have data associated with them.

        Returns
        -------
        list of tuples of int
            list of unique antpair + pol tuples (ant1, ant2, pol) with data
            associated with them.
        """
        pols = self.get_pols()
        bls = self.get_antpairs()
        return [(bl) + (pol,) for bl in bls for pol in pols]

    def get_feedpols(self):
        """
        Get the unique antenna feed polarizations in the data.

        Returns
        -------
        list of str
            list of antenna feed polarizations (e.g. ['X', 'Y']) in the data.

        Raises
        ------
        ValueError
            If any pseudo-Stokes visibilities are present
        """
        if np.any(self.polarization_array > 0):
            raise ValueError(
                "Pseudo-Stokes visibilities cannot be interpreted as feed polarizations"
            )
        else:
            return list(set("".join(self.get_pols())))

    def get_data(self, key1, key2=None, key3=None, squeeze="default", force_copy=False):
        """
        Get the data corresonding to a baseline and/or polarization.

        Parameters
        ----------
        key1, key2, key3 : int or tuple of ints
            Identifier of which data to get, can be passed as 1, 2, or 3 arguments
            or as a single tuple of length 1, 2, or 3. These are collectively
            called the key.

            If key is length 1:
                if (key < 5) or (type(key) is str):
                    interpreted as a polarization number/name, get all data for
                    that pol.
                else:
                    interpreted as a baseline number, get all data for that baseline.

            if key is length 2: interpreted as an antenna pair, get all data
                for that baseline.

            if key is length 3: interpreted as antenna pair and pol (ant1, ant2, pol),
                get all data for that baseline, pol. pol may be a string or int.
        squeeze : str
            string specifying how to squeeze the returned array. Options are:
            'default': squeeze pol and spw dimensions if possible;
            'none': no squeezing of resulting numpy array;
            'full': squeeze all length 1 dimensions.
        force_copy : bool
            Option to explicitly make a copy of the data.

        Returns
        -------
        ndarray
            copy (or if possible, a read-only view) of relevant section of data.
            If data exists conjugate to requested antenna pair, it will be conjugated
            before returning.
        """
        key = []
        for val in [key1, key2, key3]:
            if isinstance(val, str):
                key.append(val)
            elif val is not None:
                key += list(uvutils._get_iterable(val))
        if len(key) > 3:
            raise ValueError("no more than 3 key values can be passed")
        ind1, ind2, indp = self._key2inds(key)
        out = self._smart_slicing(
            self.data_array, ind1, ind2, indp, squeeze=squeeze, force_copy=force_copy
        )
        return out

    def get_flags(
        self, key1, key2=None, key3=None, squeeze="default", force_copy=False
    ):
        """
        Get the flags corresonding to a baseline and/or polarization.

        Parameters
        ----------
        key1, key2, key3 : int or tuple of ints
            Identifier of which data to get, can be passed as 1, 2, or 3 arguments
            or as a single tuple of length 1, 2, or 3. These are collectively
            called the key.

            If key is length 1:
                if (key < 5) or (type(key) is str):
                    interpreted as a polarization number/name, get all flags for
                    that pol.
                else:
                    interpreted as a baseline number, get all flags for that baseline.

            if key is length 2: interpreted as an antenna pair, get all flags
                for that baseline.

            if key is length 3: interpreted as antenna pair and pol (ant1, ant2, pol),
                get all flags for that baseline, pol. pol may be a string or int.
        squeeze : str
            string specifying how to squeeze the returned array. Options are:
            'default': squeeze pol and spw dimensions if possible;
            'none': no squeezing of resulting numpy array;
            'full': squeeze all length 1 dimensions.
        force_copy : bool
            Option to explicitly make a copy of the data.

        Returns
        -------
        ndarray
            copy (or if possible, a read-only view) of relevant section of flags.
        """
        key = []
        for val in [key1, key2, key3]:
            if isinstance(val, str):
                key.append(val)
            elif val is not None:
                key += list(uvutils._get_iterable(val))
        if len(key) > 3:
            raise ValueError("no more than 3 key values can be passed")
        ind1, ind2, indp = self._key2inds(key)
        out = self._smart_slicing(
            self.flag_array, ind1, ind2, indp, squeeze=squeeze, force_copy=force_copy
        ).astype(np.bool)
        return out

    def get_nsamples(
        self, key1, key2=None, key3=None, squeeze="default", force_copy=False
    ):
        """
        Get the nsamples corresonding to a baseline and/or polarization.

        Parameters
        ----------
        key1, key2, key3 : int or tuple of ints
            Identifier of which data to get, can be passed as 1, 2, or 3 arguments
            or as a single tuple of length 1, 2, or 3. These are collectively
            called the key.

            If key is length 1:
                if (key < 5) or (type(key) is str):
                    interpreted as a polarization number/name, get all nsamples for
                    that pol.
                else:
                    interpreted as a baseline number, get all nsamples for that
                    baseline.

            if key is length 2: interpreted as an antenna pair, get all nsamples
                for that baseline.

            if key is length 3: interpreted as antenna pair and pol (ant1, ant2, pol),
                get all nsamples for that baseline, pol. pol may be a string or int.
        squeeze : str
            string specifying how to squeeze the returned array. Options are:
            'default': squeeze pol and spw dimensions if possible;
            'none': no squeezing of resulting numpy array;
            'full': squeeze all length 1 dimensions.
        force_copy : bool
            Option to explicitly make a copy of the data.

        Returns
        -------
        ndarray
            copy (or if possible, a read-only view) of relevant section of
            nsample_array.
        """
        key = []
        for val in [key1, key2, key3]:
            if isinstance(val, str):
                key.append(val)
            elif val is not None:
                key += list(uvutils._get_iterable(val))
        if len(key) > 3:
            raise ValueError("no more than 3 key values can be passed")
        ind1, ind2, indp = self._key2inds(key)
        out = self._smart_slicing(
            self.nsample_array, ind1, ind2, indp, squeeze=squeeze, force_copy=force_copy
        )
        return out

    def get_times(self, key1, key2=None, key3=None):
        """
        Get the times for a given antpair or baseline number.

        Meant to be used in conjunction with get_data function.

        Parameters
        ----------
        key1, key2, key3 : int or tuple of ints
            Identifier of which data to get, can be passed as 1, 2, or 3 arguments
            or as a single tuple of length 1, 2, or 3. These are collectively
            called the key.

            If key is length 1:
                if (key < 5) or (type(key) is str):
                    interpreted as a polarization number/name, get all times.
                else:
                    interpreted as a baseline number, get all times for that baseline.

            if key is length 2: interpreted as an antenna pair, get all times
                for that baseline.

            if key is length 3: interpreted as antenna pair and pol (ant1, ant2, pol),
                get all times for that baseline.

        Returns
        -------
        ndarray
            times from the time_array for the given antpair or baseline.
        """
        key = []
        for val in [key1, key2, key3]:
            if isinstance(val, str):
                key.append(val)
            elif val is not None:
                key += list(uvutils._get_iterable(val))
        if len(key) > 3:
            raise ValueError("no more than 3 key values can be passed")
        inds1, inds2, indp = self._key2inds(key)
        return self.time_array[np.append(inds1, inds2)]

    def get_ENU_antpos(self, center=False, pick_data_ants=False):
        """
        Get antenna positions in ENU (topocentric) coordinates in units of meters.

        Parameters
        ----------
        center : bool
            If True, subtract median of array position from antpos
        pick_data_ants : bool
            If True, return only antennas found in data

        Returns
        -------
        antpos : ndarray
            Antenna positions in ENU (topocentric) coordinates in units of
            meters, shape=(Nants, 3)
        ants : ndarray
            Antenna numbers matching ordering of antpos, shape=(Nants,)

        """
        antpos = uvutils.ENU_from_ECEF(
            (self.antenna_positions + self.telescope_location),
            *self.telescope_location_lat_lon_alt,
        )
        ants = self.antenna_numbers

        if pick_data_ants:
            data_ants = np.unique(np.concatenate([self.ant_1_array, self.ant_2_array]))
            telescope_ants = self.antenna_numbers
            select = [x in data_ants for x in telescope_ants]
            antpos = antpos[select, :]
            ants = telescope_ants[select]

        if center is True:
            antpos -= np.median(antpos, axis=0)

        return antpos, ants

    def antpairpol_iter(self, squeeze="default"):
        """
        Iterate the data for each antpair, polarization combination.

        Parameters
        ----------
        squeeze : str
            string specifying how to squeeze the returned array. Options are:
            'default': squeeze pol and spw dimensions if possible;
            'none': no squeezing of resulting numpy array;
            'full': squeeze all length 1 dimensions.

        Yields
        ------
        key : tuple
            antenna1, antenna2, and polarization string
        data : ndarray of complex
            data for the ant pair and polarization specified in key
        """
        antpairpols = self.get_antpairpols()
        for key in antpairpols:
            yield (key, self.get_data(key, squeeze=squeeze))

    def conjugate_bls(self, convention="ant1<ant2", use_enu=True, uvw_tol=0.0):
        """
        Conjugate baselines according to one of the supported conventions.

        This will fail if only one of the cross pols is present (because
        conjugation requires changing the polarization number for cross pols).

        Parameters
        ----------
        convention : str or array_like of int
            A convention for the directions of the baselines, options are:
            'ant1<ant2', 'ant2<ant1', 'u<0', 'u>0', 'v<0', 'v>0' or an
            index array of blt indices to conjugate.
        use_enu : bool
            Use true antenna positions to determine uv location (as opposed to
            uvw array). Only applies if `convention` is 'u<0', 'u>0', 'v<0', 'v>0'.
            Set to False to use uvw array values.
        uvw_tol : float
            Defines a tolerance on uvw coordinates for setting the
            u>0, u<0, v>0, or v<0 conventions. Defaults to 0m.

        Raises
        ------
        ValueError
            If convention is not an allowed value or if not all conjugate pols exist.

        """
        if isinstance(convention, (np.ndarray, list, tuple)):
            convention = np.array(convention)
            if (
                np.max(convention) >= self.Nblts
                or np.min(convention) < 0
                or convention.dtype not in [int, np.int, np.int32, np.int64]
            ):
                raise ValueError(
                    "If convention is an index array, it must "
                    "contain integers and have values greater "
                    "than zero and less than NBlts"
                )
        else:
            if convention not in ["ant1<ant2", "ant2<ant1", "u<0", "u>0", "v<0", "v>0"]:
                raise ValueError(
                    "convention must be one of 'ant1<ant2', "
                    "'ant2<ant1', 'u<0', 'u>0', 'v<0', 'v>0' or "
                    "an index array with values less than NBlts"
                )

        if isinstance(convention, str):
            if convention in ["u<0", "u>0", "v<0", "v>0"]:
                if use_enu is True:
                    enu, anum = self.get_ENU_antpos()
                    anum = anum.tolist()
                    uvw_array_use = np.zeros_like(self.uvw_array)
                    for i, bl in enumerate(self.baseline_array):
                        a1, a2 = self.ant_1_array[i], self.ant_2_array[i]
                        i1, i2 = anum.index(a1), anum.index(a2)
                        uvw_array_use[i, :] = enu[i2] - enu[i1]
                else:
                    uvw_array_use = copy.copy(self.uvw_array)

            if convention == "ant1<ant2":
                index_array = np.asarray(self.ant_1_array > self.ant_2_array).nonzero()
            elif convention == "ant2<ant1":
                index_array = np.asarray(self.ant_2_array > self.ant_1_array).nonzero()
            elif convention == "u<0":
                index_array = np.asarray(
                    (uvw_array_use[:, 0] > uvw_tol)
                    | (uvw_array_use[:, 1] > uvw_tol)
                    & np.isclose(uvw_array_use[:, 0], 0, atol=uvw_tol)
                    | (uvw_array_use[:, 2] > uvw_tol)
                    & np.isclose(uvw_array_use[:, 0], 0, atol=uvw_tol)
                    & np.isclose(uvw_array_use[:, 1], 0, atol=uvw_tol)
                ).nonzero()
            elif convention == "u>0":
                index_array = np.asarray(
                    (uvw_array_use[:, 0] < -uvw_tol)
                    | (
                        (uvw_array_use[:, 1] < -uvw_tol)
                        & np.isclose(uvw_array_use[:, 0], 0, atol=uvw_tol)
                    )
                    | (
                        (uvw_array_use[:, 2] < -uvw_tol)
                        & np.isclose(uvw_array_use[:, 0], 0, atol=uvw_tol)
                        & np.isclose(uvw_array_use[:, 1], 0, atol=uvw_tol)
                    )
                ).nonzero()
            elif convention == "v<0":
                index_array = np.asarray(
                    (uvw_array_use[:, 1] > uvw_tol)
                    | (uvw_array_use[:, 0] > uvw_tol)
                    & np.isclose(uvw_array_use[:, 1], 0, atol=uvw_tol)
                    | (uvw_array_use[:, 2] > uvw_tol)
                    & np.isclose(uvw_array_use[:, 0], 0, atol=uvw_tol)
                    & np.isclose(uvw_array_use[:, 1], 0, atol=uvw_tol)
                ).nonzero()
            elif convention == "v>0":
                index_array = np.asarray(
                    (uvw_array_use[:, 1] < -uvw_tol)
                    | (uvw_array_use[:, 0] < -uvw_tol)
                    & np.isclose(uvw_array_use[:, 1], 0, atol=uvw_tol)
                    | (uvw_array_use[:, 2] < -uvw_tol)
                    & np.isclose(uvw_array_use[:, 0], 0, atol=uvw_tol)
                    & np.isclose(uvw_array_use[:, 1], 0, atol=uvw_tol)
                ).nonzero()
        else:
            index_array = convention

        if index_array[0].size > 0:
            new_pol_inds = uvutils.reorder_conj_pols(self.polarization_array)

            self.uvw_array[index_array] *= -1

            if not self.metadata_only:
                orig_data_array = copy.copy(self.data_array)
                for pol_ind in np.arange(self.Npols):
                    self.data_array[index_array, :, :, new_pol_inds[pol_ind]] = np.conj(
                        orig_data_array[index_array, :, :, pol_ind]
                    )

            ant_1_vals = self.ant_1_array[index_array]
            ant_2_vals = self.ant_2_array[index_array]
            self.ant_1_array[index_array] = ant_2_vals
            self.ant_2_array[index_array] = ant_1_vals
            self.baseline_array[index_array] = self.antnums_to_baseline(
                self.ant_1_array[index_array], self.ant_2_array[index_array]
            )
            self.Nbls = np.unique(self.baseline_array).size

    def reorder_pols(
        self,
        order="AIPS",
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
    ):
        """
        Rearrange polarizations in the event they are not uvfits compatible.

        Parameters
        ----------
        order : str
            Either a string specifying a cannonical ordering ('AIPS' or 'CASA')
            or an index array of length Npols that specifies how to shuffle the
            data (this is not the desired final pol order).
            CASA ordering has cross-pols in between (e.g. XX,XY,YX,YY)
            AIPS ordering has auto-pols followed by cross-pols (e.g. XX,YY,XY,YX)
            Default ('AIPS') will sort by absolute value of pol values.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after reordering.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            reordering.
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.

        Raises
        ------
        ValueError
            If the order is not one of the allowed values.

        """
        if isinstance(order, (np.ndarray, list, tuple)):
            order = np.array(order)
            if (
                order.size != self.Npols
                or order.dtype not in [int, np.int, np.int32, np.int64]
                or np.min(order) < 0
                or np.max(order) >= self.Npols
            ):
                raise ValueError(
                    "If order is an index array, it must "
                    "contain integers and be length Npols."
                )
            index_array = order
        elif order == "AIPS":
            index_array = np.argsort(np.abs(self.polarization_array))
        elif order == "CASA":
            casa_order = np.array([1, 2, 3, 4, -1, -3, -4, -2, -5, -7, -8, -6])
            pol_inds = []
            for pol in self.polarization_array:
                pol_inds.append(np.where(casa_order == pol)[0][0])
            index_array = np.argsort(pol_inds)
        else:
            raise ValueError(
                "order must be one of: 'AIPS', 'CASA', or an "
                "index array of length Npols"
            )

        self.polarization_array = self.polarization_array[index_array]
        # data array is special and large, take is faster here
        self.data_array = np.take(self.data_array, index_array, axis=3)
        self.nsample_array = self.nsample_array[:, :, :, index_array]
        self.flag_array = self.flag_array[:, :, :, index_array]

        # check if object is self-consistent
        if run_check:
            self.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
            )

    def reorder_blts(
        self,
        order="time",
        minor_order=None,
        conj_convention=None,
        uvw_tol=0.0,
        conj_convention_use_enu=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
    ):
        """
        Arrange blt axis according to desired order.

        Optionally conjugate some baselines.

        Parameters
        ----------
        order : str or array_like of int
            A string describing the desired order along the blt axis.
            Options are: `time`, `baseline`, `ant1`, `ant2`, `bda` or an
            index array of length Nblts that specifies the new order.
        minor_order : str
            Optionally specify a secondary ordering. Default depends on how
            order is set: if order is 'time', this defaults to `baseline`,
            if order is `ant1`, or `ant2` this defaults to the other antenna,
            if order is `baseline` the only allowed value is `time`. Ignored if
            order is `bda` If this is the same as order, it is reset to the default.
        conj_convention : str or array_like of int
            Optionally conjugate baselines to make the baselines have the
            desired orientation. See conjugate_bls for allowed values and details.
        uvw_tol : float
            If conjugating baselines, sets a tolerance for determining the signs
            of u,v, and w, and whether or not they are zero.
            See conjugate_bls for details.
        conj_convention_use_enu: bool
            If `conj_convention` is set, this is passed to conjugate_bls, see that
            method for details.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after reordering.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            reordering.
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.

        Raises
        ------
        ValueError
            If parameter values are inappropriate

        """
        if isinstance(order, (np.ndarray, list, tuple)):
            order = np.array(order)
            if order.size != self.Nblts or order.dtype not in [
                int,
                np.int,
                np.int32,
                np.int64,
            ]:
                raise ValueError(
                    "If order is an index array, it must "
                    "contain integers and be length Nblts."
                )
            if minor_order is not None:
                raise ValueError(
                    "Minor order cannot be set if order is an index array."
                )
        else:
            if order not in ["time", "baseline", "ant1", "ant2", "bda"]:
                raise ValueError(
                    "order must be one of 'time', 'baseline', "
                    "'ant1', 'ant2', 'bda' or an index array of "
                    "length Nblts"
                )

            if minor_order == order:
                minor_order = None

            if minor_order is not None:
                if minor_order not in ["time", "baseline", "ant1", "ant2"]:
                    raise ValueError(
                        "minor_order can only be one of 'time', "
                        "'baseline', 'ant1', 'ant2'"
                    )
                if isinstance(order, np.ndarray) or order == "bda":
                    raise ValueError(
                        "minor_order cannot be specified if order is "
                        "'bda' or an index array."
                    )
                if order == "baseline":
                    if minor_order in ["ant1", "ant2"]:
                        raise ValueError("minor_order conflicts with order")
            else:
                if order == "time":
                    minor_order = "baseline"
                elif order == "ant1":
                    minor_order = "ant2"
                elif order == "ant2":
                    minor_order = "ant1"
                elif order == "baseline":
                    minor_order = "time"

        if conj_convention is not None:
            self.conjugate_bls(
                convention=conj_convention,
                use_enu=conj_convention_use_enu,
                uvw_tol=uvw_tol,
            )

        if isinstance(order, str):
            if minor_order is None:
                self.blt_order = (order,)
                self._blt_order.form = (1,)
            else:
                self.blt_order = (order, minor_order)
                # set it back to the right shape in case it was set differently before
                self._blt_order.form = (2,)
        else:
            self.blt_order = None

        if not isinstance(order, np.ndarray):
            # Use lexsort to sort along different arrays in defined order.
            if order == "time":
                arr1 = self.time_array
                if minor_order == "ant1":
                    arr2 = self.ant_1_array
                    arr3 = self.ant_2_array
                elif minor_order == "ant2":
                    arr2 = self.ant_2_array
                    arr3 = self.ant_1_array
                else:
                    # minor_order is baseline
                    arr2 = self.baseline_array
                    arr3 = self.baseline_array
            elif order == "ant1":
                arr1 = self.ant_1_array
                if minor_order == "time":
                    arr2 = self.time_array
                    arr3 = self.ant_2_array
                elif minor_order == "ant2":
                    arr2 = self.ant_2_array
                    arr3 = self.time_array
                else:  # minor_order is baseline
                    arr2 = self.baseline_array
                    arr3 = self.time_array
            elif order == "ant2":
                arr1 = self.ant_2_array
                if minor_order == "time":
                    arr2 = self.time_array
                    arr3 = self.ant_1_array
                elif minor_order == "ant1":
                    arr2 = self.ant_1_array
                    arr3 = self.time_array
                else:
                    # minor_order is baseline
                    arr2 = self.baseline_array
                    arr3 = self.time_array
            elif order == "baseline":
                arr1 = self.baseline_array
                # only allowed minor order is time
                arr2 = self.time_array
                arr3 = self.time_array
            elif order == "bda":
                arr1 = self.integration_time
                # only allowed minor order is time
                arr2 = self.baseline_array
                arr3 = self.time_array

            # lexsort uses the listed arrays from last to first
            # (so the primary sort is on the last one)
            index_array = np.lexsort((arr3, arr2, arr1))
        else:
            index_array = order

        # actually do the reordering
        self.ant_1_array = self.ant_1_array[index_array]
        self.ant_2_array = self.ant_2_array[index_array]
        self.baseline_array = self.baseline_array[index_array]
        self.uvw_array = self.uvw_array[index_array, :]
        self.time_array = self.time_array[index_array]
        self.lst_array = self.lst_array[index_array]
        self.integration_time = self.integration_time[index_array]
        if self.phase_center_app_ra is not None:
            self.phase_center_app_ra = self.phase_center_app_ra[index_array]
        if self.phase_center_app_dec is not None:
            self.phase_center_app_ra = self.phase_center_app_dec[index_array]
        if self.multi_object:
            self.object_id_array = self.object_id_array[index_array]

        if not self.metadata_only:
            self.data_array = self.data_array[index_array, :, :, :]
            self.flag_array = self.flag_array[index_array, :, :, :]
            self.nsample_array = self.nsample_array[index_array, :, :, :]

        # check if object is self-consistent
        if run_check:
            self.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
            )

    def remove_eq_coeffs(self):
        """
        Remove equalization coefficients from the data.

        Some telescopes, e.g. HERA, apply per-antenna, per-frequency gain
        coefficients as part of the signal chain. These are stored in the
        `eq_coeffs` attribute of the object. This method will remove them, so
        that the data are in "unnormalized" raw units.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        ValueError
            Raised if eq_coeffs or eq_coeffs_convention are not defined on the
            object, or if eq_coeffs_convention is not one of "multiply" or "divide".
        """
        if self.eq_coeffs is None:
            raise ValueError(
                "The eq_coeffs attribute must be defined on the object to apply them."
            )
        if self.eq_coeffs_convention is None:
            raise ValueError(
                "The eq_coeffs_convention attribute must be defined on the object "
                "to apply them."
            )
        if self.eq_coeffs_convention not in ("multiply", "divide"):
            raise ValueError(
                "Got unknown convention {}. Must be one of: "
                '"multiply", "divide"'.format(self.eq_coeffs_convention)
            )

        # apply coefficients for each baseline
        for key in self.get_antpairs():
            # get indices for this key
            blt_inds = self.antpair2ind(key)

            ant1_index = np.asarray(self.antenna_numbers == key[0]).nonzero()[0][0]
            ant2_index = np.asarray(self.antenna_numbers == key[1]).nonzero()[0][0]

            eq_coeff1 = self.eq_coeffs[ant1_index, :]
            eq_coeff2 = self.eq_coeffs[ant2_index, :]

            # make sure coefficients are the right size to broadcast
            eq_coeff1 = np.repeat(eq_coeff1[:, np.newaxis], self.Npols, axis=1)
            eq_coeff2 = np.repeat(eq_coeff2[:, np.newaxis], self.Npols, axis=1)

            if self.eq_coeffs_convention == "multiply":
                self.data_array[blt_inds, 0, :, :] *= eq_coeff1 * eq_coeff2
            else:
                self.data_array[blt_inds, 0, :, :] /= eq_coeff1 * eq_coeff2

        return

    def unphase_to_drift(
        self,
        phase_frame=None,
        use_ant_pos=False,
        use_old_phase=False,
        auto_select_phase=True,
    ):
        """
        Convert from a phased dataset to a drift dataset.

        See the phasing memo under docs/references for more documentation.

        Parameters
        ----------
        phase_frame : str
            The astropy frame to phase from. Either 'icrs' or 'gcrs'.
            'gcrs' accounts for precession & nutation, 'icrs' also includes abberation.
            Defaults to using the 'phase_center_frame' attribute or 'icrs'
            if that attribute is None.
        use_ant_pos : bool
            If True, calculate the uvws directly from the antenna positions
            rather than from the existing uvws.
        use_old_phase : bool
            If True, uses the 'old' way of calculating phases. Default is False.
        auto_select_phase : bool
            Allow the phase method to determine whether or not use the old phasing
            method or not. Default is True, overrides the input to use_old_phase.

        Raises
        ------
        ValueError
            If the phase_type is not 'phased'
        """
        # Basically, the only time we need to use the 'old' phase method is if we are
        # working with a 'phased' data set, where the apparent coordinates have not
        # been calculated.
        if auto_select_phase:
            use_old_phase = (self.phase_type == "phased") and (
                self.phase_center_app_ra is None or self.phase_center_app_dec is None
            )

        if self.phase_type == "phased":
            pass
        elif self.phase_type == "drift":
            raise ValueError(
                "The data is already drift scanning; can only " "unphase phased data."
            )
        else:
            raise ValueError(
                "The phasing type of the data is unknown. "
                "Set the phase_type to drift or phased to "
                "reflect the phasing status of the data"
            )

        if not use_old_phase:
            # Check to make sure that these attributes are actually filled. Otherwise,
            # you probably want to use the old phase method.
            if (self.phase_center_app_ra is None) or (
                self.phase_center_app_dec is None
            ):
                raise AttributeError(
                    "Object missing phase_center_ra_app or phase_center_dec_app, "
                    "which implies that the data were phased using the 'old' "
                    "method for phasing (which is not compatible with the new "
                    "version of the code). Please run unphase_to_drift with "
                    "use_old_phase=True to continue."
                )
            old_w_vals = self.uvw_array[:, 2].copy()
            telescope_location = self.telescope_location_lat_lon_alt
            if self.multi_object:
                # Check and see if we have any unphased objects, in which case
                # their w-values should be zeroed out.
                old_w_vals[self._check_for_unphased_objects] = 0.0

            new_uvw = uvutils.calc_uvw(
                lst_array=self.lst_array,
                use_ant_pos=use_ant_pos,
                uvw_array=self.uvw_array,
                antenna_positions=self.antenna_positions,
                ant_1_array=self.ant_1_array,
                ant_2_array=self.ant_2_array,
                old_app_ra=self.phase_center_app_ra,
                old_app_dec=self.phase_center_app_dec,
                telescope_lat=telescope_location[0],
                telescope_lon=telescope_location[1],
                to_enu=True,
            )

            self._apply_w_proj(0.0, old_w_vals)
            self.uvw_array = new_uvw

            # remove/update phase center
            if self.multi_object:
                self.object_id_array[:] = self._add_object("unphased", "unphased")
                self.phase_center_app_ra = self.lst_array.copy()
                self.phaes_center_app_dec[:] = self.telescope_location[0]
            else:
                self.phase_center_frame = None
                self.phase_center_ra = None
                self.phase_center_dec = None
                self.phase_center_epoch = None
                self.phase_center_app_ra = None
                self.phase_center_app_dec = None
                self._set_drift()

        # If you are a multi-object data set, there's no valid reason to be going
        # back to the old phase method. Time to bail!
        if self.multi_object:
            raise NotImplementedError(
                "Multi-object data sets are not compatible with the old phasing "
                "method, please set use_old_phase=False."
            )

        warnings.warn(
            "The original `phase` method will be deprecated in a future release. "
            "Note that the old and new phase methods are NOT compatible with one "
            "another, so if you have phased using the old method, you should use "
            "the unphase_to_drift method with use_old_phase=True to undo the old "
            "corrections before using the new version of the phase method.",
            DeprecationWarning,
        )
        if phase_frame is None:
            if self.phase_center_frame is not None:
                phase_frame = self.phase_center_frame
            else:
                phase_frame = "icrs"

        icrs_coord = SkyCoord(
            ra=self.phase_center_ra,
            dec=self.phase_center_dec,
            unit="radian",
            frame="icrs",
        )
        if phase_frame == "icrs":
            frame_phase_center = icrs_coord
        else:
            # use center of observation for obstime for gcrs
            center_time = np.mean([np.max(self.time_array), np.min(self.time_array)])
            icrs_coord.obstime = Time(center_time, format="jd")
            frame_phase_center = icrs_coord.transform_to("gcrs")

        # This promotion is REQUIRED to get the right answer when we
        # add in the telescope location for ICRS
        # In some cases, the uvws are already float64, but sometimes they're not
        self.uvw_array = np.float64(self.uvw_array)

        # apply -w phasor
        if not self.metadata_only:
            w_lambda = (
                self.uvw_array[:, 2].reshape(self.Nblts, 1)
                / const.c.to("m/s").value
                * self.freq_array.reshape(1, self.Nfreqs)
            )
            phs = np.exp(-1j * 2 * np.pi * (-1) * w_lambda[:, None, :, None])
            self.data_array *= phs

        unique_times, unique_inds = np.unique(self.time_array, return_index=True)

        telescope_location = EarthLocation.from_geocentric(
            *self.telescope_location, unit=units.m
        )
        obs_times = Time(unique_times, format="jd")
        itrs_telescope_locations = telescope_location.get_itrs(obstime=obs_times)
        itrs_telescope_locations = SkyCoord(itrs_telescope_locations)
        # just calling transform_to(coord.GCRS) will delete the obstime information
        # need to re-add obstimes for a GCRS transformation
        if phase_frame == "gcrs":
            frame_telescope_locations = itrs_telescope_locations.transform_to(
                getattr(coord, f"{phase_frame}".upper())(obstime=obs_times)
            )
        else:
            frame_telescope_locations = itrs_telescope_locations.transform_to(
                getattr(coord, f"{phase_frame}".upper())
            )

        frame_telescope_locations.representation_type = "cartesian"

        for ind, jd in enumerate(unique_times):
            inds = np.where(self.time_array == jd)[0]

            obs_time = obs_times[ind]

            if use_ant_pos:
                ant_uvw = uvutils.phase_uvw(
                    self.telescope_location_lat_lon_alt[1],
                    self.telescope_location_lat_lon_alt[0],
                    self.antenna_positions,
                )
                # instead of looping through every ind, find the spot in antenna number
                # array where ant_num <= ant1 < ant_number and similarly for ant2
                # for all baselines in inds
                # then find the uvw coordinate for all at the same time

                # antenna_numbers does not necessarily need to be in order on the object
                # but needs to be in order for the searchsorted to work.
                # ant1_index and ant2_index arrays will preserve the order of blts
                ant_sort = np.argsort(self.antenna_numbers)
                ant1_index = np.searchsorted(
                    self.antenna_numbers[ant_sort], self.ant_1_array[inds]
                )
                ant2_index = np.searchsorted(
                    self.antenna_numbers[ant_sort], self.ant_2_array[inds]
                )
                self.uvw_array[inds] = (
                    ant_uvw[ant_sort][ant2_index, :] - ant_uvw[ant_sort][ant1_index, :]
                )

            else:
                frame_telescope_location = frame_telescope_locations[ind]
                itrs_lat_lon_alt = self.telescope_location_lat_lon_alt

                uvws_use = self.uvw_array[inds, :]

                uvw_rel_positions = uvutils.unphase_uvw(
                    frame_phase_center.ra.rad, frame_phase_center.dec.rad, uvws_use
                )

                frame_uvw_coord = SkyCoord(
                    x=uvw_rel_positions[:, 0] * units.m + frame_telescope_location.x,
                    y=uvw_rel_positions[:, 1] * units.m + frame_telescope_location.y,
                    z=uvw_rel_positions[:, 2] * units.m + frame_telescope_location.z,
                    frame=phase_frame,
                    obstime=obs_time,
                    representation_type="cartesian",
                )

                itrs_uvw_coord = frame_uvw_coord.transform_to("itrs")

                # now convert them to ENU, which is the space uvws are in
                self.uvw_array[inds, :] = uvutils.ENU_from_ECEF(
                    itrs_uvw_coord.cartesian.get_xyz().value.T, *itrs_lat_lon_alt
                )

        # remove phase center
        self.phase_center_frame = None
        self.phase_center_ra = None
        self.phase_center_dec = None
        self.phase_center_epoch = None
        self._set_drift()

    def phase(
        self,
        ra,
        dec,
        epoch="J2000",
        phase_frame="icrs",
        object_name=None,
        object_type="sidereal",
        pm_ra=0.0,
        pm_dec=0.0,
        parallax=0.0,
        rad_vel=0.0,
        use_ant_pos=False,  # Can we change this to default to True?
        allow_rephase=True,
        orig_phase_frame=None,
        use_astropy=False,
        use_old_phase=False,
        auto_select_phase=True,
        select_mask=None,
        cleanup_old_sources=True,
    ):
        """
        Phase a drift scan dataset to a single ra/dec at a particular epoch.

        See the phasing memo under docs/references for more documentation.

        Tested against MWA_Tools/CONV2UVFITS/convutils.

        Parameters
        ----------
        ra : float
            The ra to phase to in radians.
        dec : float
            The dec to phase to in radians.
        epoch : astropy.time.Time object or str
            The epoch to use for phasing. Either an astropy Time object or the
            string "J2000" (which is the default).
            Note that the epoch is only used to evaluate the ra & dec values,
            if the epoch is not J2000, the ra & dec values are interpreted
            as FK5 ra/dec values and translated to J2000, the data are then
            phased to the J2000 ra/dec values.
        phase_frame : str
            The astropy frame to phase to. Either 'icrs' or 'gcrs'.
            'gcrs' accounts for precession & nutation,
            'icrs' accounts for precession, nutation & abberation.
        use_ant_pos : bool
            If True, calculate the uvws directly from the antenna positions
            rather than from the existing uvws.
        allow_rephase : bool
            If True, allow unphasing and rephasing if this object is already
            phased.
        orig_phase_frame : str
            The original phase frame of this object (to use in unphasing). Only
            used if the object is already phased, `allow_rephase` is True and
            the phase_center_ra/dec of the object does not match `ra` and `dec`.
            Defaults to using the 'phase_center_frame' attribute or 'icrs' if
            that attribute is None.
        use_astropy : bool
            If True, use the astropy module for handling calculation of source
            position coordinates. Default is False, which instead invokes the
            python-novas library.
        use_old_phase : bool
            If True, use the "old" method for calculating baseline uvw-coordinates,
            which involved using astropy to move antenna positions (in ITRF) into
            the requested reference frame (either GCRS or ICRS). Default is False.
        auto_select_phase : bool
            Allow the phase method to determine whether or not use the old phasing
            method or not. Default is True, overrides the input to use_old_phase.
        select_mask : ndarray of bool
            Optional mask for selecting which data to operate on along the blt-axis,
            only used if with multi-object data sets (i.e., multi_object=True). Shape
            is (Nblts,).

        Raises
        ------
        ValueError
            If the phase_type is 'phased' and allow_rephase is False
        """
        # Basically, the only time we need to use the 'old' phase method is if we are
        # working with a 'phased' data set, where the apparent coordinates have not
        # been calculated.
        if auto_select_phase:
            use_old_phase = (self.phase_type == "phased") and (
                self.phase_center_app_ra is None or self.phase_center_app_dec is None
            )
        # If the selection mask is being used, make sure that it looks okay
        if select_mask is not None:
            if use_old_phase:
                raise ValueError(
                    "Cannot use the selection mask with the old phasing method. "
                    "Either remove the selection_mask argument, or set "
                    "use_old_phase=False to continue."
                )
            if not self.multi_object:
                raise ValueError(
                    "Cannot apply a selection mask if multi_object=False. "
                    "Remove the select_mask argument to continue."
                )
            if not len(select_mask) != self.Nblts:
                raise IndexError("Selection mask must be of length Nblts.")

        # Non multi-object datasets don't (yet) have a way of recording the 'extra'
        # source properties, so make sure that these aren't using any of those
        # if looking at a single object.
        if not self.multi_object:
            if pm_ra not in [0, None]:
                raise ValueError(
                    "Non-zero values of pm_ra not supported when multi_object!=True."
                )
            if pm_dec not in [0, None]:
                raise ValueError(
                    "Non-zero values of pm_dec not supported when multi_object!=True."
                )
            if parallax not in [0, None]:
                raise ValueError(
                    "Non-zero values of parallax not supported when multi_object!=True."
                )
            if rad_vel not in [0, None]:
                raise ValueError(
                    "Non-zero values of rad_vel not supported when multi_object!=True."
                )
            if object_type != "sidereal":
                raise ValueError(
                    "Only sidereal sources are supported when multi_object!=True"
                )

        # Right up front, we're gonna split off the piece of the code that
        # does the phasing using the "new" method, since its a lot more flexible
        # and because I think at some point, everything outside of this loop
        # can be depreciated
        if not use_old_phase:
            # Grab all the meta-data we need for the rotations
            telescope_location = self.telescope_location_lat_lon_alt
            time_array = self.time_array
            lst_array = self.lst_array
            uvw_array = self.uvw_array
            ant_1_array = self.ant_1_array
            ant_2_array = self.ant_2_array
            # Check to make sure
            if self.phase_type == "drift":
                old_w_vals = 0.0
                old_app_ra = None
                old_app_dec = None
                from_enu = True
            elif self.phase_type == "phased":
                old_w_vals = self.uvw_array[:, 2]
                # Check to make sure that this operation is permissible
                if not allow_rephase:
                    raise ValueError(
                        "The data is already phased; set allow_rephase"
                        " to True to unphase and rephase."
                    )
                if (self.phase_center_app_ra is None) or (
                    self.phase_center_app_dec is None
                ):
                    raise AttributeError(
                        "Object missing phase_center_ra_app or phase_center_dec_app, "
                        "which implies that the data were phased using the 'old' "
                        "method for phasing (which is not compatible with the new "
                        "version of the code). Please run unphase_to_drift with "
                        "use_old_phase=True to use the new method, otherwise call "
                        "phase with use_old_phase=True to continue."
                    )
                old_w_vals = self.uvw_array[:, 2].copy()
                old_app_ra = self.phase_center_app_ra
                old_app_dec = self.phase_center_app_dec
                from_enu = False
                if self.multi_object:
                    # Check and see if we have any unphased objects, in which case
                    # their w-values should be zeroed out.
                    old_w_vals[self._check_for_unphased_objects] = 0.0

            if select_mask is not None:
                time_array = time_array[select_mask]
                lst_array = lst_array[select_mask]
                uvw_array = uvw_array[select_mask, :]
                ant_1_array = ant_1_array[select_mask]
                ant_2_array = ant_2_array[select_mask]
                if isinstance(old_w_vals, np.ndarray):
                    old_w_vals = old_w_vals[select_mask]

            # Before moving forward with the heavy calculations, we should make sure
            # that the target in question is unique if this is a multi-source dataset.
            # If so, add it to the catalog, if not, check that the properties are the
            # same (and if thats not true, then add_object will throw an error).
            if self.multi_object:
                object_id = self._add_object(
                    object_name,
                    object_type,
                    object_lon=ra,
                    object_lat=dec,
                    object_frame=phase_frame,
                    object_epoch=epoch,
                    object_pm_ra=pm_ra,
                    object_pm_dec=pm_dec,
                    object_parallax=parallax,
                    object_rad_vel=rad_vel,
                )

            # We got the meta-data, now handle calculating the apparent coordinates
            new_app_ra, new_app_dec = uvutils.calc_app_coors(
                ra,
                dec,
                coord_frame=phase_frame,
                coord_epoch=epoch,
                object_type=object_type,
                time_array=time_array,
                lst_array=lst_array,
                pm_ra=pm_ra,
                pm_dec=pm_dec,
                rad_vel=rad_vel,
                parallax=parallax,
                telescope_lat=telescope_location[0],
                telescope_lon=telescope_location[1],
                telescope_alt=telescope_location[2],
                use_astropy=use_astropy,
            )

            # Now its time to do some rotations and calculate the new coordinates
            new_uvw = uvutils.calc_uvw(
                app_ra=new_app_ra,
                app_dec=new_app_dec,
                lst_array=lst_array,
                use_ant_pos=use_ant_pos,
                uvw_array=uvw_array,
                antenna_positions=self.antenna_positions,
                ant_1_array=ant_1_array,
                ant_2_array=ant_2_array,
                old_app_ra=old_app_ra,
                old_app_dec=old_app_dec,
                telescope_lat=telescope_location[0],
                telescope_lon=telescope_location[1],
                from_enu=from_enu,
            )

            # Now its time to update the raw data. This will return empty if
            # metadata_only is set to True. Note that object_type is only allowed
            # to be unphased if this is a multi_object data set.
            new_w_vals = 0.0 if (object_type == "unphased") else new_uvw[:, 2]
            self._apply_w_proj(new_w_vals, old_w_vals, select_mask=select_mask)

            # Finally, we now take it upon ourselves to update some metadata. What we
            # do here will depend a little bit on whether or not we have a selection
            # mask active, since most everything is affected by that.
            if select_mask is not None:
                self.uvw_array[select_mask] = new_uvw
                self.phase_center_app_ra[select_mask] = new_app_ra
                self.phase_center_app_dec[select_mask] = new_app_dec
                if self.mutli_source:
                    self.object_id_array[select_mask] = object_id
            else:
                self.uvw_array = new_uvw
                self.phase_center_app_ra = new_app_ra
                self.phase_center_app_dec = new_app_dec
                if self.multi_object:
                    self.object_id_array[:] = object_id

            # If not multi-object, make sure to update the ra/dec values, since
            # otherwise we'll have no record of source properties.
            if not self.multi_object:
                # Make sure this is actually marked as a phased dataset now
                self._set_phased()

                # Update the phase center properties
                self.phase_center_ra = ra
                self.phase_center_dec = dec
                self.phase_center_epoch = epoch
                self.phase_center_frame = phase_frame
                if object_name is not None:
                    self.object_name = object_name

            # All done w/ the new phase method
            return

        # If you are a multi-object data set, there's no valid reason to be going
        # back to the old phase method. Time to bail!
        if self.multi_object:
            raise NotImplementedError(
                "Multi-object data sets are not compatible with the old phasing "
                "method, please set use_old_phase=False."
            )

        warnings.warn(
            "The original `phase` method will be deprecated in a future release. "
            "Note that the old and new phase methods are NOT compatible with one "
            "another, so if you have phased using the old method, you should use "
            "the unphase_to_drift method with use_old_phase=True to undo the old "
            "corrections before using the new version of the phase method.",
            DeprecationWarning,
        )
        if self.phase_type == "drift":
            pass
        elif self.phase_type == "phased":
            if allow_rephase:
                if not np.isclose(
                    self.phase_center_ra,
                    ra,
                    rtol=self._phase_center_ra.tols[0],
                    atol=self._phase_center_ra.tols[1],
                ) or not np.isclose(
                    self.phase_center_dec,
                    dec,
                    rtol=self._phase_center_dec.tols[0],
                    atol=self._phase_center_dec.tols[1],
                ):
                    self.unphase_to_drift(
                        phase_frame=orig_phase_frame, use_ant_pos=use_ant_pos
                    )
            else:
                raise ValueError(
                    "The data is already phased; set allow_rephase"
                    " to True to unphase and rephase."
                )
        else:
            raise ValueError(
                "The phasing type of the data is unknown. "
                'Set the phase_type to "drift" or "phased" to '
                "reflect the phasing status of the data"
            )

        if phase_frame not in ["icrs", "gcrs"]:
            raise ValueError("phase_frame can only be set to icrs or gcrs.")

        if epoch == "J2000" or epoch == 2000:
            icrs_coord = SkyCoord(ra=ra, dec=dec, unit="radian", frame="icrs")
        else:
            assert isinstance(epoch, Time)
            phase_center_coord = SkyCoord(
                ra=ra, dec=dec, unit="radian", equinox=epoch, frame=FK5
            )
            # convert to icrs (i.e. J2000) to write to object
            icrs_coord = phase_center_coord.transform_to("icrs")

        self.phase_center_ra = icrs_coord.ra.radian
        self.phase_center_dec = icrs_coord.dec.radian
        self.phase_center_epoch = 2000.0

        if phase_frame == "icrs":
            frame_phase_center = icrs_coord
        else:
            # use center of observation for obstime for gcrs
            center_time = np.mean([np.max(self.time_array), np.min(self.time_array)])
            icrs_coord.obstime = Time(center_time, format="jd")
            frame_phase_center = icrs_coord.transform_to("gcrs")

        # This promotion is REQUIRED to get the right answer when we
        # add in the telescope location for ICRS
        self.uvw_array = np.float64(self.uvw_array)

        unique_times, unique_inds = np.unique(self.time_array, return_index=True)

        telescope_location = EarthLocation.from_geocentric(
            *self.telescope_location, unit=units.m
        )
        obs_times = Time(unique_times, format="jd")

        itrs_telescope_locations = telescope_location.get_itrs(obstime=obs_times)
        itrs_telescope_locations = SkyCoord(itrs_telescope_locations)
        # just calling transform_to(coord.GCRS) will delete the obstime information
        # need to re-add obstimes for a GCRS transformation
        if phase_frame == "gcrs":
            frame_telescope_locations = itrs_telescope_locations.transform_to(
                getattr(coord, f"{phase_frame}".upper())(obstime=obs_times)
            )
        else:
            frame_telescope_locations = itrs_telescope_locations.transform_to(
                getattr(coord, f"{phase_frame}".upper())
            )
        # set the representation_type to cartensian to get xyz later
        frame_telescope_locations.representation_type = "cartesian"

        for ind, jd in enumerate(unique_times):
            inds = np.where(self.time_array == jd)[0]

            obs_time = obs_times[ind]

            itrs_lat_lon_alt = self.telescope_location_lat_lon_alt

            frame_telescope_location = frame_telescope_locations[ind]

            if use_ant_pos:
                # This promotion is REQUIRED to get the right answer when we
                # add in the telescope location for ICRS
                ecef_ant_pos = (
                    np.float64(self.antenna_positions) + self.telescope_location
                )

                itrs_ant_coord = SkyCoord(
                    x=ecef_ant_pos[:, 0] * units.m,
                    y=ecef_ant_pos[:, 1] * units.m,
                    z=ecef_ant_pos[:, 2] * units.m,
                    frame="itrs",
                    obstime=obs_time,
                )

                frame_ant_coord = itrs_ant_coord.transform_to(phase_frame)

                frame_ant_rel = (
                    (frame_ant_coord.cartesian - frame_telescope_location.cartesian)
                    .get_xyz()
                    .T.value
                )

                frame_ant_uvw = uvutils.phase_uvw(
                    frame_phase_center.ra.rad, frame_phase_center.dec.rad, frame_ant_rel
                )
                # instead of looping through every ind, find the spot in antenna number
                # array where ant_num <= ant1 < ant_number and similarly for ant2
                # for all baselines in inds
                # then find the uvw coordinate for all at the same time

                # antenna_numbers does not necessarily need to be in order on the object
                # but needs to be in order for the searchsorted to work.
                # ant1_index and ant2_index arrays will preserve the order of blts
                ant_sort = np.argsort(self.antenna_numbers)
                ant1_index = np.searchsorted(
                    self.antenna_numbers[ant_sort], self.ant_1_array[inds]
                )
                ant2_index = np.searchsorted(
                    self.antenna_numbers[ant_sort], self.ant_2_array[inds]
                )
                self.uvw_array[inds] = (
                    frame_ant_uvw[ant_sort][ant2_index, :]
                    - frame_ant_uvw[ant_sort][ant1_index, :]
                )
            else:
                # Also, uvws should be thought of like ENU, not ECEF (or rotated ECEF)
                # convert them to ECEF to transform between frames
                uvws_use = self.uvw_array[inds, :]

                uvw_ecef = uvutils.ECEF_from_ENU(uvws_use, *itrs_lat_lon_alt)

                itrs_uvw_coord = SkyCoord(
                    x=uvw_ecef[:, 0] * units.m,
                    y=uvw_ecef[:, 1] * units.m,
                    z=uvw_ecef[:, 2] * units.m,
                    frame="itrs",
                    obstime=obs_time,
                )
                frame_uvw_coord = itrs_uvw_coord.transform_to(phase_frame)

                # this takes out the telescope location in the new frame,
                # so these are vectors again
                frame_rel_uvw = (
                    frame_uvw_coord.cartesian.get_xyz().value.T
                    - frame_telescope_location.cartesian.get_xyz().value
                )

                self.uvw_array[inds, :] = uvutils.phase_uvw(
                    frame_phase_center.ra.rad, frame_phase_center.dec.rad, frame_rel_uvw
                )

        # calculate data and apply phasor
        if not self.metadata_only:
            w_lambda = (
                self.uvw_array[:, 2].reshape(self.Nblts, 1)
                / const.c.to("m/s").value
                * self.freq_array.reshape(1, self.Nfreqs)
            )
            phs = np.exp(-1j * 2 * np.pi * w_lambda[:, None, :, None])
            self.data_array *= phs

        self.phase_center_frame = phase_frame
        self._set_phased()

    def phase_to_time(
        self,
        time,
        phase_frame="icrs",
        use_ant_pos=False,
        allow_rephase=True,
        orig_phase_frame=None,
    ):
        """
        Phase a drift scan dataset to the ra/dec of zenith at a particular time.

        See the phasing memo under docs/references for more documentation.

        Parameters
        ----------
        time : astropy.time.Time object or float
            The time to phase to, an astropy Time object or a float Julian Date
        phase_frame : str
            The astropy frame to phase to. Either 'icrs' or 'gcrs'.
            'gcrs' accounts for precession & nutation,
            'icrs' accounts for precession, nutation & abberation.
        use_ant_pos : bool
            If True, calculate the uvws directly from the antenna positions
            rather than from the existing uvws.
        allow_rephase : bool
            If True, allow unphasing and rephasing if this object is already
            phased.
        orig_phase_frame : str
            The original phase frame of this object (to use in unphasing). Only
            used if the object is already phased, `allow_rephase` is True and
            the phase_center_ra/dec of the object does not match `ra` and `dec`.
            Defaults to using the 'phase_center_frame' attribute or 'icrs' if
            that attribute is None.

        Raises
        ------
        ValueError
            If the phase_type is not 'drift'
        TypeError
            If time is not an astropy.time.Time object or Julian Date as a float
        """
        if isinstance(time, (float, np.floating)):
            time = Time(time, format="jd")

        if not isinstance(time, Time):
            raise TypeError("time must be an astropy.time.Time object or a float")

        # Generate ra/dec of zenith at time in the phase_frame coordinate
        # system to use for phasing
        telescope_location = EarthLocation.from_geocentric(
            *self.telescope_location, unit="m"
        )

        zenith_coord = SkyCoord(
            alt=Angle(90 * units.deg),
            az=Angle(0 * units.deg),
            obstime=time,
            frame="altaz",
            location=telescope_location,
        )

        obs_zenith_coord = zenith_coord.transform_to(phase_frame)
        zenith_ra = obs_zenith_coord.ra
        zenith_dec = obs_zenith_coord.dec

        self.phase(
            zenith_ra,
            zenith_dec,
            epoch="J2000",
            phase_frame=phase_frame,
            use_ant_pos=use_ant_pos,
            allow_rephase=allow_rephase,
            orig_phase_frame=orig_phase_frame,
        )

    def _apply_w_proj(self, new_w_vals, old_w_vals, select_mask=None):
        """
        Apply corrections based on changes to w-coord.

        Adjusts the data to account for a change along the w-axis of a baseline.

        Parameters
        ----------
        new_w_vals: float or ndarray of float
            New w-coordinates for the baselines, in units of meters. Can either be a
            solitary float (helpful for unphasing data, where new_w_vals can be set to
            0.0) or an array of shape (Nselect,) (which is Nblts if select_mask=None).
        old_w_vals: float or ndarray of float
            Old w-coordinates for the baselines, in units of meters. Can either be a
            solitary float (helpful for unphasing data, where new_w_vals can be set to
            0.0) or an array of shape (Nselect,) (which is Nblts if select_mask=None).
        select_mask: ndarray of bool
            Array is of shape (Nblts,), where the sum of all enties marked True is
            equal to Nselect (mentioned above).

        Raises
        ------
        IndexError
            If the length of new_w_vals or old_w_vals isn't compatible with
            select_mask, or if select mask isn't the right length.
        """
        # If we only have metadata, then we have no work to do. W00t!
        if not self.metadata_only:
            return

        # Promote everything to float64 ndarrays if they aren't already
        if not isinstance(new_w_vals, np.ndarray):
            new_w_vals = np.array([new_w_vals], dtype=np.float64)
        if not isinstance(new_w_vals[0], np.float64):
            new_w_vals = new_w_vals.as_type(np.float64)
        if not isinstance(old_w_vals, np.ndarray):
            old_w_vals = np.array([old_w_vals], dtype=np.float64)
        if not isinstance(old_w_vals[0], np.float64):
            old_w_vals = old_w_vals.as_type(np.float64)

        # Make sure the lengths of everything make sense
        new_val_len = len(new_w_vals)
        old_val_len = len(old_w_vals)
        select_len = self.Nblts if (select_mask is None) else np.sum(select_mask)

        if (select_mask is not None) and (len(select_mask) != self.Nblts):
            raise IndexError("select_mask must be of length Nblts!")
        if new_val_len not in [1, select_len]:
            raise IndexError(
                "The length of new_w_vals is wrong (expected 1 or %i, got %i)!"
                % (select_len, new_val_len)
            )
        if old_val_len not in [1, select_len]:
            raise IndexError(
                "The length of old_w_vals is wrong (expected 1 or %i, got %i)!"
                % (select_len, old_val_len)
            )

        # Calculate the difference in w terms as a function of freq. Note that the
        # 1/c is there to speed of processing (faster to multiply than divide)
        delta_w_lambda = (
            (new_w_vals - old_w_vals).reshape(self.Nblts, 1)
            * (1.0 / const.c.to("m/s").value)
            * self.freq_array.reshape(1, self.Nfreqs)
        )

        if select_mask is None or np.all(select_mask):
            # If all the w values are changing, it turns out to be twice as fast
            # to ditch any sort of selection mask and just do the full multiply.
            self.data_array *= np.exp(
                (-1j * 2 * np.pi) * delta_w_lambda[:, None, :, None]
            )
        elif np.any(select_mask):
            # In the case we are _not_ doing all baselines, use a selection mask to
            # only update the values we need. In the worse case, it slows down the
            # processing by ~2x, but it can save a lot on time and memory if only
            # needing to update a select number of baselines.
            self.data_array[select_mask] *= np.exp(
                (-1j * 2 * np.pi) * delta_w_lambda[:, None, :, None]
            )

    def set_uvws_from_antenna_positions(
        self,
        allow_phasing=False,
        orig_phase_frame=None,
        output_phase_frame="icrs",
        use_old_phase=False,
        auto_select_phase=True,
    ):
        """
        Calculate UVWs based on antenna_positions.

        Parameters
        ----------
        allow_phasing : bool
            Option for phased data. If data is phased and allow_phasing is set,
            data will be unphased, UVWs will be calculated, and then data will
            be rephased.
        orig_phase_frame : str
            The astropy frame to phase from. Either 'icrs' or 'gcrs'.
            Defaults to using the 'phase_center_frame' attribute or 'icrs' if
            that attribute is None. Only used if allow_phasing is True.
        output_phase_frame : str
            The astropy frame to phase to. Either 'icrs' or 'gcrs'. Only used if
            allow_phasing is True.
        use_old_phase : bool
            If set to True, uses the 'old' method of calculating baseline vectors.
        auto_select_phase : bool
            Allow the phase method to determine whether or not use the old phasing
            method or not. Default is True, overrides the input to use_old_phase.

        Raises
        ------
        ValueError
            If data is phased and allow_phasing is False.

        Warns
        -----
        UserWarning
            If the phase_type is 'phased'

        """
        # Basically, the only time we need to use the 'old' phase method is if we are
        # working with a 'phased' data set, where the apparent coordinates have not
        # been calculated.
        if auto_select_phase:
            use_old_phase = (self.phase_type == "phased") and (
                self.phase_center_app_ra is None or self.phase_center_app_dec is None
            )

        if not use_old_phase:
            telescope_location = self.telescope_location_lat_lon_alt
            new_uvw = uvutils.calc_uvw(
                app_ra=self.phase_center_app_ra,
                app_dec=self.phase_center_app_dec,
                lst_array=self.lst_array,
                use_ant_pos=True,
                antenna_positions=self.antenna_positions,
                ant_1_array=self.ant_1_array,
                ant_2_array=self.ant_2_array,
                telescope_lat=telescope_location[0],
                telescope_lon=telescope_location[1],
                to_enu=(self.phase_type != "phased"),
            )
            if self.phase_type == "phased":
                if (self.phase_center_app_ra is None) or (
                    self.phase_center_app_dec is None
                ):
                    raise AttributeError(
                        "Object missing phase_center_ra_app or phase_center_dec_app, "
                        "which implies that the data were phased using the 'old' "
                        "method for phasing (which is not compatible with the new "
                        "version of the code). Please run unphase_to_drift with "
                        "use_old_phase=True to use the new method, otherwise call "
                        "phase with use_old_phase=True to continue."
                    )
                if allow_phasing:
                    old_w_vals = self.uvw_array[:, 2].copy()
                    if self.multi_object:
                        old_w_vals[self._check_for_unphased_objects] = 0.0
                    self._apply_w_proj(new_uvw[:, 2], old_w_vals)
            # If the data are phased, we've already adjusted the phases. Now we just
            # need to update the uvw's and we are home free.
            self.uvw_array = new_uvw
            return

        phase_type = self.phase_type
        if phase_type == "phased":
            if allow_phasing:
                if not self.metadata_only:
                    warnings.warn(
                        "Data will be unphased and rephased "
                        "to calculate UVWs, which might introduce small "
                        "inaccuracies to the data."
                    )
                if orig_phase_frame not in [None, "icrs", "gcrs"]:
                    raise ValueError(
                        "Invalid parameter orig_phase_frame. "
                        'Options are "icrs", "gcrs", or None.'
                    )
                if output_phase_frame not in ["icrs", "gcrs"]:
                    raise ValueError(
                        "Invalid parameter output_phase_frame. "
                        'Options are "icrs" or "gcrs".'
                    )
                phase_center_ra = self.phase_center_ra
                phase_center_dec = self.phase_center_dec
                phase_center_epoch = self.phase_center_epoch
                self.unphase_to_drift(phase_frame=orig_phase_frame)
            else:
                raise ValueError(
                    "UVW calculation requires unphased data. "
                    "Use unphase_to_drift or set "
                    "allow_phasing=True."
                )
        antenna_locs_ENU, _ = self.get_ENU_antpos(center=False)
        # this code used to loop through every bl in the unique,
        # find the index into self.antenna_array of ant1 and ant2
        # and fill out the self.uvw_array for all matching bls.

        # instead, find the indices and reverse inds from the unique,
        # create the unique ant1 and ant2 arrays
        # use searchsorted to find the index of the antenna numbers into ant1 and ant2
        # create the unique uvw array then broadcast to self.uvw_array
        bls, unique_inds, reverse_inds = np.unique(
            self.baseline_array, return_index=True, return_inverse=True
        )

        # antenna_numbers does not necessarily need to be in order on the object
        # but needs to be in order for the searchsorted to work.
        # ant1_index and ant2_index arrays will preserve the order of blts
        ant_sort = np.argsort(self.antenna_numbers)
        ant1_index = np.searchsorted(
            self.antenna_numbers[ant_sort], self.ant_1_array[unique_inds],
        )
        ant2_index = np.searchsorted(
            self.antenna_numbers[ant_sort], self.ant_2_array[unique_inds],
        )
        _uvw_array = np.zeros((bls.size, 3))
        _uvw_array = (
            antenna_locs_ENU[ant_sort][ant2_index, :]
            - antenna_locs_ENU[ant_sort][ant1_index, :]
        )
        self.uvw_array = _uvw_array[reverse_inds]

        if phase_type == "phased":
            self.phase(
                phase_center_ra,
                phase_center_dec,
                phase_center_epoch,
                phase_frame=output_phase_frame,
            )

    def __add__(
        self,
        other,
        inplace=False,
        phase_center_radec=None,
        unphase_to_drift=False,
        phase_frame="icrs",
        orig_phase_frame=None,
        use_ant_pos=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
    ):
        """
        Combine two UVData objects along frequency, polarization and/or baseline-time.

        Parameters
        ----------
        other : UVData object
            Another UVData object which will be added to self.
        inplace : bool
            If True, overwrite self as we go, otherwise create a third object
            as the sum of the two.
        phase_center_radec : array_like of float
            The phase center to phase the files to before adding the objects in
            radians (in the ICRS frame). Note that if this keyword is not set
            and the two UVData objects are phased to different phase centers
            or if one is phased and one is drift, this method will error
            because the objects are not compatible.
        unphase_to_drift : bool
            If True, unphase the objects to drift before combining them.
        phase_frame : str
            The astropy frame to phase to. Either 'icrs' or 'gcrs'.
            'gcrs' accounts for precession & nutation,
            'icrs' accounts for precession, nutation & abberation.
            Only used if `phase_center_radec` is set.
        orig_phase_frame : str
            The original phase frame of the data (if it is already phased). Used
            for unphasing, only if `unphase_to_drift` or `phase_center_radec`
            are set. Defaults to using the 'phase_center_frame' attribute or
            'icrs' if that attribute is None.
        use_ant_pos : bool
            If True, calculate the phased or unphased uvws directly from the
            antenna positions rather than from the existing uvws.
            Only used if `unphase_to_drift` or `phase_center_radec` are set.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after combining objects.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            combining objects.
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.

        Raises
        ------
        ValueError
            If other is not a UVData object, self and other are not compatible
            or if data in self and other overlap. One way they can not be
            compatible is if they have different phasing, in that case set
            `unphase_to_drift` or `phase_center_radec` to (un)phase them so they
            are compatible.
            If `phase_center_radec` is not None and is not length 2.

        """
        if inplace:
            this = self
        else:
            this = self.copy()

        # Check that both objects are UVData and valid
        this.check(
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
            strict_uvw_antpos_check=strict_uvw_antpos_check,
        )
        if not issubclass(other.__class__, this.__class__):
            if not issubclass(this.__class__, other.__class__):
                raise ValueError(
                    "Only UVData (or subclass) objects can be "
                    "added to a UVData (or subclass) object"
                )
        other.check(
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
            strict_uvw_antpos_check=strict_uvw_antpos_check,
        )

        if phase_center_radec is not None and unphase_to_drift:
            raise ValueError(
                "phase_center_radec cannot be set if unphase_to_drift is True."
            )

        if unphase_to_drift:
            if this.phase_type != "drift":
                warnings.warn("Unphasing this UVData object to drift")
                this.unphase_to_drift(
                    phase_frame=orig_phase_frame, use_ant_pos=use_ant_pos
                )

            if other.phase_type != "drift":
                warnings.warn("Unphasing other UVData object to drift")
                other.unphase_to_drift(
                    phase_frame=orig_phase_frame, use_ant_pos=use_ant_pos
                )

        if phase_center_radec is not None:
            if np.array(phase_center_radec).size != 2:
                raise ValueError("phase_center_radec should have length 2.")

            # If this object is not phased or is not phased close to
            # phase_center_radec, (re)phase it.
            # Close is defined using the phase_center_ra/dec tolerances.
            if this.phase_type == "drift" or (
                not np.isclose(
                    this.phase_center_ra,
                    phase_center_radec[0],
                    rtol=this._phase_center_ra.tols[0],
                    atol=this._phase_center_ra.tols[1],
                )
                or not np.isclose(
                    this.phase_center_dec,
                    phase_center_radec[1],
                    rtol=this._phase_center_dec.tols[0],
                    atol=this._phase_center_dec.tols[1],
                )
            ):
                warnings.warn("Phasing this UVData object to phase_center_radec")
                this.phase(
                    phase_center_radec[0],
                    phase_center_radec[1],
                    phase_frame=phase_frame,
                    orig_phase_frame=orig_phase_frame,
                    use_ant_pos=use_ant_pos,
                    allow_rephase=True,
                )

            # If other object is not phased or is not phased close to
            # phase_center_radec, (re)phase it.
            # Close is defined using the phase_center_ra/dec tolerances.
            if other.phase_type == "drift" or (
                not np.isclose(
                    other.phase_center_ra,
                    phase_center_radec[0],
                    rtol=other._phase_center_ra.tols[0],
                    atol=other._phase_center_ra.tols[1],
                )
                or not np.isclose(
                    other.phase_center_dec,
                    phase_center_radec[1],
                    rtol=other._phase_center_dec.tols[0],
                    atol=other._phase_center_dec.tols[1],
                )
            ):
                warnings.warn("Phasing other UVData object to phase_center_radec")
                other.phase(
                    phase_center_radec[0],
                    phase_center_radec[1],
                    phase_frame=phase_frame,
                    orig_phase_frame=orig_phase_frame,
                    use_ant_pos=use_ant_pos,
                    allow_rephase=True,
                )

        # Define parameters that must be the same to add objects
        # But phase_center should be the same, even if in drift (empty parameters)
        compatibility_params = [
            "_vis_units",
            "_channel_width",
            "_object_name",
            "_telescope_name",
            "_instrument",
            "_telescope_location",
            "_phase_type",
            "_Nants_telescope",
            "_antenna_names",
            "_antenna_numbers",
            "_antenna_positions",
            "_phase_center_ra",
            "_phase_center_dec",
            "_phase_center_epoch",
        ]

        # Build up history string
        history_update_string = " Combined data along "
        n_axes = 0

        # Create blt arrays for convenience
        prec_t = -2 * np.floor(np.log10(this._time_array.tols[-1])).astype(int)
        prec_b = 8
        this_blts = np.array(
            [
                "_".join(
                    ["{1:.{0}f}".format(prec_t, blt[0]), str(blt[1]).zfill(prec_b)]
                )
                for blt in zip(this.time_array, this.baseline_array)
            ]
        )
        other_blts = np.array(
            [
                "_".join(
                    ["{1:.{0}f}".format(prec_t, blt[0]), str(blt[1]).zfill(prec_b)]
                )
                for blt in zip(other.time_array, other.baseline_array)
            ]
        )
        # Check we don't have overlapping data
        both_pol, this_pol_ind, other_pol_ind = np.intersect1d(
            this.polarization_array, other.polarization_array, return_indices=True
        )
        both_freq, this_freq_ind, other_freq_ind = np.intersect1d(
            this.freq_array[0, :], other.freq_array[0, :], return_indices=True
        )
        both_blts, this_blts_ind, other_blts_ind = np.intersect1d(
            this_blts, other_blts, return_indices=True
        )
        if not self.metadata_only and (
            len(both_pol) > 0 and len(both_freq) > 0 and len(both_blts) > 0
        ):
            # check that overlapping data is not valid
            # TODO: Spw axis to be collapsed in future release
            this_inds = np.ravel_multi_index(
                (
                    this_blts_ind[:, np.newaxis, np.newaxis, np.newaxis],
                    np.zeros((1, 1, 1, 1), dtype=np.int),
                    this_freq_ind[np.newaxis, np.newaxis, :, np.newaxis],
                    this_pol_ind[np.newaxis, np.newaxis, np.newaxis, :],
                ),
                this.data_array.shape,
            ).flatten()
            this_all_zero = np.all(this.data_array.flatten()[this_inds] == 0)
            this_all_flag = np.all(this.flag_array.flatten()[this_inds])
            # TODO: Spw axis to be collapsed in future release
            other_inds = np.ravel_multi_index(
                (
                    other_blts_ind[:, np.newaxis, np.newaxis, np.newaxis],
                    np.zeros((1, 1, 1, 1), dtype=np.int),
                    other_freq_ind[np.newaxis, np.newaxis, :, np.newaxis],
                    other_pol_ind[np.newaxis, np.newaxis, np.newaxis, :],
                ),
                other.data_array.shape,
            ).flatten()
            other_all_zero = np.all(other.data_array.flatten()[other_inds] == 0)
            other_all_flag = np.all(other.flag_array.flatten()[other_inds])

            if this_all_zero and this_all_flag:
                # we're fine to overwrite; update history accordingly
                history_update_string = " Overwrote invalid data using pyuvdata."
                this.history += history_update_string
            elif other_all_zero and other_all_flag:
                raise ValueError(
                    "To combine these data, please run the add operation again, "
                    "but with the object whose data is to be overwritten as the "
                    "first object in the add operation."
                )
            else:
                raise ValueError(
                    "These objects have overlapping data and cannot be combined."
                )

        # find the blt indices in "other" but not in "this"
        temp = np.nonzero(~np.in1d(other_blts, this_blts))[0]
        if len(temp) > 0:
            bnew_inds = temp
            new_blts = other_blts[temp]
            history_update_string += "baseline-time"
            n_axes += 1
        else:
            bnew_inds, new_blts = ([], [])
            # add metadata to be checked to compatibility params
            extra_params = ["_integration_time", "_uvw_array", "_lst_array"]
            compatibility_params.extend(extra_params)

        # find the freq indices in "other" but not in "this"
        temp = np.nonzero(~np.in1d(other.freq_array[0, :], this.freq_array[0, :]))[0]
        if len(temp) > 0:
            fnew_inds = temp
            if n_axes > 0:
                history_update_string += ", frequency"
            else:
                history_update_string += "frequency"
            n_axes += 1
        else:
            fnew_inds = []

        # find the pol indices in "other" but not in "this"
        temp = np.nonzero(~np.in1d(other.polarization_array, this.polarization_array))[
            0
        ]
        if len(temp) > 0:
            pnew_inds = temp
            if n_axes > 0:
                history_update_string += ", polarization"
            else:
                history_update_string += "polarization"
            n_axes += 1
        else:
            pnew_inds = []

        # Actually check compatibility parameters
        for a in compatibility_params:
            if a == "_integration_time":
                # only check that overlapping blt indices match
                params_match = np.allclose(
                    this.integration_time[this_blts_ind],
                    other.integration_time[other_blts_ind],
                    rtol=this._integration_time.tols[0],
                    atol=this._integration_time.tols[1],
                )
            elif a == "_uvw_array":
                # only check that overlapping blt indices match
                params_match = np.allclose(
                    this.uvw_array[this_blts_ind, :],
                    other.uvw_array[other_blts_ind, :],
                    rtol=this._uvw_array.tols[0],
                    atol=this._uvw_array.tols[1],
                )
            elif a == "_lst_array":
                # only check that overlapping blt indices match
                params_match = np.allclose(
                    this.lst_array[this_blts_ind],
                    other.lst_array[other_blts_ind],
                    rtol=this._lst_array.tols[0],
                    atol=this._lst_array.tols[1],
                )
            else:
                params_match = getattr(this, a) == getattr(other, a)
            if not params_match:
                msg = (
                    "UVParameter " + a[1:] + " does not match. Cannot combine objects."
                )
                raise ValueError(msg)

        # Pad out self to accommodate new data
        if len(bnew_inds) > 0:
            this_blts = np.concatenate((this_blts, new_blts))
            blt_order = np.argsort(this_blts)
            if not self.metadata_only:
                # TODO: Spw axis to be collapsed in future release
                zero_pad = np.zeros((len(bnew_inds), 1, this.Nfreqs, this.Npols))
                this.data_array = np.concatenate([this.data_array, zero_pad], axis=0)
                this.nsample_array = np.concatenate(
                    [this.nsample_array, zero_pad], axis=0
                )
                this.flag_array = np.concatenate(
                    [this.flag_array, 1 - zero_pad], axis=0
                ).astype(np.bool)
            this.uvw_array = np.concatenate(
                [this.uvw_array, other.uvw_array[bnew_inds, :]], axis=0
            )[blt_order, :]
            this.time_array = np.concatenate(
                [this.time_array, other.time_array[bnew_inds]]
            )[blt_order]
            this.integration_time = np.concatenate(
                [this.integration_time, other.integration_time[bnew_inds]]
            )[blt_order]
            this.lst_array = np.concatenate(
                [this.lst_array, other.lst_array[bnew_inds]]
            )[blt_order]
            this.ant_1_array = np.concatenate(
                [this.ant_1_array, other.ant_1_array[bnew_inds]]
            )[blt_order]
            this.ant_2_array = np.concatenate(
                [this.ant_2_array, other.ant_2_array[bnew_inds]]
            )[blt_order]
            this.baseline_array = np.concatenate(
                [this.baseline_array, other.baseline_array[bnew_inds]]
            )[blt_order]

        if len(fnew_inds) > 0:
            this.freq_array = np.concatenate(
                [this.freq_array, other.freq_array[:, fnew_inds]], axis=1
            )
            f_order = np.argsort(this.freq_array[0, :])
            if not self.metadata_only:
                # TODO: Spw axis to be collapsed in future release
                zero_pad = np.zeros(
                    (this.data_array.shape[0], 1, len(fnew_inds), this.Npols)
                )
                this.data_array = np.concatenate([this.data_array, zero_pad], axis=2)
                this.nsample_array = np.concatenate(
                    [this.nsample_array, zero_pad], axis=2
                )
                this.flag_array = np.concatenate(
                    [this.flag_array, 1 - zero_pad], axis=2
                ).astype(np.bool)
        if len(pnew_inds) > 0:
            this.polarization_array = np.concatenate(
                [this.polarization_array, other.polarization_array[pnew_inds]]
            )
            p_order = np.argsort(np.abs(this.polarization_array))
            if not self.metadata_only:
                # TODO: Spw axis to be collapsed in future release
                zero_pad = np.zeros(
                    (
                        this.data_array.shape[0],
                        1,
                        this.data_array.shape[2],
                        len(pnew_inds),
                    )
                )
                this.data_array = np.concatenate([this.data_array, zero_pad], axis=3)
                this.nsample_array = np.concatenate(
                    [this.nsample_array, zero_pad], axis=3
                )
                this.flag_array = np.concatenate(
                    [this.flag_array, 1 - zero_pad], axis=3
                ).astype(np.bool)

        # Now populate the data
        pol_t2o = np.nonzero(
            np.in1d(this.polarization_array, other.polarization_array)
        )[0]
        freq_t2o = np.nonzero(np.in1d(this.freq_array[0, :], other.freq_array[0, :]))[0]
        blt_t2o = np.nonzero(np.in1d(this_blts, other_blts))[0]
        if not self.metadata_only:
            this.data_array[np.ix_(blt_t2o, [0], freq_t2o, pol_t2o)] = other.data_array
            this.nsample_array[
                np.ix_(blt_t2o, [0], freq_t2o, pol_t2o)
            ] = other.nsample_array
            this.flag_array[np.ix_(blt_t2o, [0], freq_t2o, pol_t2o)] = other.flag_array

        if not self.metadata_only:
            if len(bnew_inds) > 0:
                for name, param in zip(this._data_params, this.data_like_parameters):
                    setattr(this, name, param[blt_order, :, :, :])

            if len(fnew_inds) > 0:
                for name, param in zip(this._data_params, this.data_like_parameters):
                    setattr(this, name, param[:, :, f_order, :])

            if len(pnew_inds) > 0:
                for name, param in zip(this._data_params, this.data_like_parameters):
                    setattr(this, name, param[:, :, :, p_order])

        if len(fnew_inds) > 0:
            this.freq_array = this.freq_array[:, f_order]
        if len(pnew_inds) > 0:
            this.polarization_array = this.polarization_array[p_order]

        # Update N parameters (e.g. Npols)
        this.Ntimes = len(np.unique(this.time_array))
        this.Nbls = len(np.unique(this.baseline_array))
        this.Nblts = this.uvw_array.shape[0]
        this.Nfreqs = this.freq_array.shape[1]
        this.Npols = this.polarization_array.shape[0]
        this.Nants_data = this._calc_nants_data()

        # Check specific requirements
        if this.Nfreqs > 1:
            freq_separation = np.diff(this.freq_array[0, :])
            if not np.isclose(
                np.min(freq_separation),
                np.max(freq_separation),
                rtol=this._freq_array.tols[0],
                atol=this._freq_array.tols[1],
            ):
                warnings.warn(
                    "Combined frequencies are not evenly spaced. This will "
                    "make it impossible to write this data out to some file types."
                )
            elif (
                np.max(freq_separation)
                > this.channel_width + this._channel_width.tols[1]
            ):
                warnings.warn(
                    "Combined frequencies are not contiguous. This will make "
                    "it impossible to write this data out to some file types."
                )

        if this.Npols > 2:
            pol_separation = np.diff(this.polarization_array)
            if np.min(pol_separation) < np.max(pol_separation):
                warnings.warn(
                    "Combined polarizations are not evenly spaced. This will "
                    "make it impossible to write this data out to some file types."
                )

        if n_axes > 0:
            history_update_string += " axis using pyuvdata."
            this.history += history_update_string

        this.history = uvutils._combine_histories(this.history, other.history)

        # Check final object is self-consistent
        if run_check:
            this.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
            )

        if not inplace:
            return this

    def __iadd__(
        self,
        other,
        phase_center_radec=None,
        unphase_to_drift=False,
        phase_frame="icrs",
        orig_phase_frame=None,
        use_ant_pos=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
    ):
        """
        In place add.

        Parameters
        ----------
        other : UVData object
            Another UVData object which will be added to self.
        phase_center_radec : array_like of float
            The phase center to phase the files to before adding the objects in
            radians (in the ICRS frame). Note that if this keyword is not set
            and the two UVData objects are phased to different phase centers
            or if one is phased and one is drift, this method will error
            because the objects are not compatible.
        unphase_to_drift : bool
            If True, unphase the objects to drift before combining them.
        phase_frame : str
            The astropy frame to phase to. Either 'icrs' or 'gcrs'.
            'gcrs' accounts for precession & nutation,
            'icrs' accounts for precession, nutation & abberation.
            Only used if `phase_center_radec` is set.
        orig_phase_frame : str
            The original phase frame of the data (if it is already phased). Used
            for unphasing, only if `unphase_to_drift` or `phase_center_radec`
            are set. Defaults to using the 'phase_center_frame' attribute or
            'icrs' if that attribute is None.
        use_ant_pos : bool
            If True, calculate the phased or unphased uvws directly from the
            antenna positions rather than from the existing uvws.
            Only used if `unphase_to_drift` or `phase_center_radec` are set.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after combining objects.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            combining objects.
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.

        Raises
        ------
        ValueError
            If other is not a UVData object, self and other are not compatible
            or if data in self and other overlap. One way they can not be
            compatible is if they have different phasing, in that case set
            `unphase_to_drift` or `phase_center_radec` to (un)phase them so they
            are compatible.
            If `phase_center_radec` is not None and is not length 2.

        """
        self.__add__(
            other,
            inplace=True,
            phase_center_radec=phase_center_radec,
            unphase_to_drift=unphase_to_drift,
            phase_frame=phase_frame,
            orig_phase_frame=orig_phase_frame,
            use_ant_pos=use_ant_pos,
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
            strict_uvw_antpos_check=strict_uvw_antpos_check,
        )
        return self

    def fast_concat(
        self,
        other,
        axis,
        inplace=False,
        phase_center_radec=None,
        unphase_to_drift=False,
        phase_frame="icrs",
        orig_phase_frame=None,
        use_ant_pos=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
    ):
        """
        Concatenate two UVData objects along specified axis with almost no checking.

        Warning! This method assumes all the metadata along other axes is sorted
        the same way. The __add__ method is much safer, it checks all the metadata,
        but it is slower. Some quick checks are run, but this method doesn't
        make any guarantees that the resulting object is correct.

        Parameters
        ----------
        other : UVData object or list of UVData objects
            UVData object or list of UVData objects which will be added to self.
        axis : str
            Axis to concatenate files along. This enables fast concatenation
            along the specified axis without the normal checking that all other
            metadata agrees. Allowed values are: 'blt', 'freq', 'polarization'.
        inplace : bool
            If True, overwrite self as we go, otherwise create a third object
            as the sum of the two.
        phase_center_radec : array_like of float
            The phase center to phase the files to before adding the objects in
            radians (in the ICRS frame). Note that if this keyword is not set
            and the two UVData objects are phased to different phase centers
            or if one is phased and one is drift, this method will error
            because the objects are not compatible.
        unphase_to_drift : bool
            If True, unphase the objects to drift before combining them.
        phase_frame : str
            The astropy frame to phase to. Either 'icrs' or 'gcrs'.
            'gcrs' accounts for precession & nutation,
            'icrs' accounts for precession, nutation & abberation.
            Only used if `phase_center_radec` is set.
        orig_phase_frame : str
            The original phase frame of the data (if it is already phased). Used
            for unphasing, only if `unphase_to_drift` or `phase_center_radec`
            are set. Defaults to using the 'phase_center_frame' attribute or
            'icrs' if that attribute is None.
        use_ant_pos : bool
            If True, calculate the phased or unphased uvws directly from the
            antenna positions rather than from the existing uvws.
            Only used if `unphase_to_drift` or `phase_center_radec` are set.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after combining objects.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            combining objects.
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.

        Raises
        ------
        ValueError
            If other is not a UVData object, axis is not an allowed value or if
            self and other are not compatible.

        """
        if inplace:
            this = self
        else:
            this = self.copy()
        if not isinstance(other, (list, tuple, np.ndarray)):
            # if this is a UVData object already, stick it in a list
            other = [other]
        # Check that both objects are UVData and valid
        this.check(
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
            strict_uvw_antpos_check=strict_uvw_antpos_check,
        )
        for obj in other:
            if not issubclass(obj.__class__, this.__class__):
                if not issubclass(this.__class__, obj.__class__):
                    raise ValueError(
                        "Only UVData (or subclass) objects can be "
                        "added to a UVData (or subclass) object"
                    )
            obj.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
            )

        if phase_center_radec is not None and unphase_to_drift:
            raise ValueError(
                "phase_center_radec cannot be set if unphase_to_drift is True."
            )

        if unphase_to_drift:
            if this.phase_type != "drift":
                warnings.warn("Unphasing this UVData object to drift")
                this.unphase_to_drift(
                    phase_frame=orig_phase_frame, use_ant_pos=use_ant_pos
                )

            for obj in other:
                if obj.phase_type != "drift":
                    warnings.warn("Unphasing other UVData object to drift")
                    obj.unphase_to_drift(
                        phase_frame=orig_phase_frame, use_ant_pos=use_ant_pos
                    )

        if phase_center_radec is not None:
            if np.array(phase_center_radec).size != 2:
                raise ValueError("phase_center_radec should have length 2.")

            # If this object is not phased or is not phased close to
            # phase_center_radec, (re)phase it.
            # Close is defined using the phase_center_ra/dec tolerances.
            if this.phase_type == "drift" or (
                not np.isclose(
                    this.phase_center_ra,
                    phase_center_radec[0],
                    rtol=this._phase_center_ra.tols[0],
                    atol=this._phase_center_ra.tols[1],
                )
                or not np.isclose(
                    this.phase_center_dec,
                    phase_center_radec[1],
                    rtol=this._phase_center_dec.tols[0],
                    atol=this._phase_center_dec.tols[1],
                )
            ):
                warnings.warn("Phasing this UVData object to phase_center_radec")
                this.phase(
                    phase_center_radec[0],
                    phase_center_radec[1],
                    phase_frame=phase_frame,
                    orig_phase_frame=orig_phase_frame,
                    use_ant_pos=use_ant_pos,
                    allow_rephase=True,
                )

            # If other object is not phased or is not phased close to
            # phase_center_radec, (re)phase it.
            # Close is defined using the phase_center_ra/dec tolerances.
            for obj in other:
                if obj.phase_type == "drift" or (
                    not np.isclose(
                        obj.phase_center_ra,
                        phase_center_radec[0],
                        rtol=obj._phase_center_ra.tols[0],
                        atol=obj._phase_center_ra.tols[1],
                    )
                    or not np.isclose(
                        obj.phase_center_dec,
                        phase_center_radec[1],
                        rtol=obj._phase_center_dec.tols[0],
                        atol=obj._phase_center_dec.tols[1],
                    )
                ):
                    warnings.warn("Phasing other UVData object to phase_center_radec")
                    obj.phase(
                        phase_center_radec[0],
                        phase_center_radec[1],
                        phase_frame=phase_frame,
                        orig_phase_frame=orig_phase_frame,
                        use_ant_pos=use_ant_pos,
                        allow_rephase=True,
                    )

        allowed_axes = ["blt", "freq", "polarization"]
        if axis not in allowed_axes:
            raise ValueError(
                "If axis is specifed it must be one of: " + ", ".join(allowed_axes)
            )

        compatibility_params = [
            "_vis_units",
            "_channel_width",
            "_object_name",
            "_telescope_name",
            "_instrument",
            "_telescope_location",
            "_phase_type",
            "_Nants_telescope",
            "_antenna_names",
            "_antenna_numbers",
            "_antenna_positions",
            "_phase_center_ra",
            "_phase_center_dec",
            "_phase_center_epoch",
        ]

        history_update_string = " Combined data along "

        if axis == "freq":
            history_update_string += "frequency"
            compatibility_params += [
                "_polarization_array",
                "_ant_1_array",
                "_ant_2_array",
                "_integration_time",
                "_uvw_array",
                "_lst_array",
            ]
        elif axis == "polarization":
            history_update_string += "polarization"
            compatibility_params += [
                "_freq_array",
                "_ant_1_array",
                "_ant_2_array",
                "_integration_time",
                "_uvw_array",
                "_lst_array",
            ]
        elif axis == "blt":
            history_update_string += "baseline-time"
            compatibility_params += ["_freq_array", "_polarization_array"]

        history_update_string += " axis using pyuvdata."
        this.history += history_update_string

        for obj in other:
            this.history = uvutils._combine_histories(this.history, obj.history)

        # Actually check compatibility parameters
        for obj in other:
            for a in compatibility_params:
                params_match = getattr(this, a) == getattr(obj, a)
                if not params_match:
                    msg = (
                        "UVParameter "
                        + a[1:]
                        + " does not match. Cannot combine objects."
                    )
                    raise ValueError(msg)

        if axis == "freq":
            this.freq_array = np.concatenate(
                [this.freq_array] + [obj.freq_array for obj in other], axis=1
            )
            this.Nfreqs = sum([this.Nfreqs] + [obj.Nfreqs for obj in other])

            freq_separation = np.diff(this.freq_array[0, :])
            if not np.isclose(
                np.min(freq_separation),
                np.max(freq_separation),
                rtol=this._freq_array.tols[0],
                atol=this._freq_array.tols[1],
            ):
                warnings.warn(
                    "Combined frequencies are not evenly spaced. This will "
                    "make it impossible to write this data out to some file types."
                )
            elif (
                np.max(freq_separation)
                > this.channel_width + this._channel_width.tols[1]
            ):
                warnings.warn(
                    "Combined frequencies are not contiguous. This will make "
                    "it impossible to write this data out to some file types."
                )
            if not self.metadata_only:
                this.data_array = np.concatenate(
                    [this.data_array] + [obj.data_array for obj in other], axis=2,
                )
                this.nsample_array = np.concatenate(
                    [this.nsample_array] + [obj.nsample_array for obj in other], axis=2,
                )
                this.flag_array = np.concatenate(
                    [this.flag_array] + [obj.flag_array for obj in other], axis=2,
                )
        elif axis == "polarization":
            this.polarization_array = np.concatenate(
                [this.polarization_array] + [obj.polarization_array for obj in other]
            )
            this.Npols = sum([this.Npols] + [obj.Npols for obj in other])

            pol_separation = np.diff(this.polarization_array)
            if np.min(pol_separation) < np.max(pol_separation):
                warnings.warn(
                    "Combined polarizations are not evenly spaced. This will "
                    "make it impossible to write this data out to some file types."
                )

            if not self.metadata_only:
                this.data_array = np.concatenate(
                    [this.data_array] + [obj.data_array for obj in other], axis=3,
                )
                this.nsample_array = np.concatenate(
                    [this.nsample_array] + [obj.nsample_array for obj in other], axis=3,
                )
                this.flag_array = np.concatenate(
                    [this.flag_array] + [obj.flag_array for obj in other], axis=3,
                )
        elif axis == "blt":
            this.Nblts = sum([this.Nblts] + [obj.Nblts for obj in other])
            this.ant_1_array = np.concatenate(
                [this.ant_1_array] + [obj.ant_1_array for obj in other]
            )
            this.ant_2_array = np.concatenate(
                [this.ant_2_array] + [obj.ant_2_array for obj in other]
            )
            this.Nants_data = this._calc_nants_data()
            this.uvw_array = np.concatenate(
                [this.uvw_array] + [obj.uvw_array for obj in other], axis=0
            )
            this.time_array = np.concatenate(
                [this.time_array] + [obj.time_array for obj in other]
            )
            this.Ntimes = len(np.unique(this.time_array))
            this.lst_array = np.concatenate(
                [this.lst_array] + [obj.lst_array for obj in other]
            )
            this.baseline_array = np.concatenate(
                [this.baseline_array] + [obj.baseline_array for obj in other]
            )
            this.Nbls = len(np.unique(this.baseline_array))
            this.integration_time = np.concatenate(
                [this.integration_time] + [obj.integration_time for obj in other]
            )
            if not self.metadata_only:
                this.data_array = np.concatenate(
                    [this.data_array] + [obj.data_array for obj in other], axis=0,
                )
                this.nsample_array = np.concatenate(
                    [this.nsample_array] + [obj.nsample_array for obj in other], axis=0,
                )
                this.flag_array = np.concatenate(
                    [this.flag_array] + [obj.flag_array for obj in other], axis=0,
                )

        # Check final object is self-consistent
        if run_check:
            this.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
            )

        if not inplace:
            return this

    def sum_vis(
        self,
        other,
        inplace=False,
        difference=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
        override_params=None,
    ):
        """
        Sum visibilities between two UVData objects.

        By default requires that all UVParameters are the same on the two objects
        except for `history`, `data_array`, `object_name`, and `extra_keywords`.
        The `object_name` values are concatenated if they are different. If keys
        in `extra_keywords` have different values the values from the first
        object are taken.

        Parameters
        ----------
        other : UVData object
            Another UVData object which will be added to self.
        difference : bool
            If True, differences the visibilities of the two UVData objects
            rather than summing them.
        inplace : bool
            If True, overwrite self as we go, otherwise create a third object
            as the sum of the two.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after combining objects.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            combining objects.
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.
        override_params : array_like of strings
            List of object UVParameters to omit from compatibility check. Overridden
            parameters will not be compared between the objects, and the values
            for these parameters will be taken from the first object.

        Returns
        -------
        UVData Object
            If inplace parameter is False.

        Raises
        ------
        ValueError
            If other is not a UVData object, or if self and other
            are not compatible.

        """
        if inplace:
            this = self
        else:
            this = self.copy()

        # Check that both objects are UVData and valid
        this.check(
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
            strict_uvw_antpos_check=strict_uvw_antpos_check,
        )
        if not issubclass(other.__class__, this.__class__):
            if not issubclass(this.__class__, other.__class__):
                raise ValueError(
                    "Only UVData (or subclass) objects can be "
                    "added to a UVData (or subclass) object"
                )
        other.check(
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
            strict_uvw_antpos_check=strict_uvw_antpos_check,
        )

        compatibility_params = list(this.__iter__())
        remove_params = ["_history", "_data_array", "_object_name", "_extra_keywords"]

        # Add underscores to override_params to match list from __iter__()
        # Add to parameters to be removed
        if override_params and all(isinstance(param, str) for param in override_params):
            for param in override_params:
                if param[0] != "_":
                    param = "_" + param
                if param not in compatibility_params:
                    msg = (
                        "Provided parameter " + param[1:] + " is not a recognizable "
                        "UVParameter."
                    )
                    raise ValueError(msg)
                remove_params.append(param)

        # compatibility_params should define the parameters that need to
        # be the same for objects to be summed or diffed
        compatibility_params = list(set(compatibility_params) - set(remove_params))

        # Check each UVParameter in compatibility_params
        for param in compatibility_params:
            params_match = getattr(this, param) == getattr(other, param)
            if not params_match:
                msg = (
                    "UVParameter " + param[1:] + " does not match. Cannot "
                    "combine objects."
                )
                raise ValueError(msg)

        # Merge extra keywords and object_name
        for intersection in set(this.extra_keywords.keys()) & set(
            other.extra_keywords.keys()
        ):
            if this.extra_keywords[intersection] != other.extra_keywords[intersection]:
                warnings.warn(
                    "Keyword " + intersection + " in _extra_keywords is different "
                    "in the two objects. Taking the first object's entry."
                )

        # Merge extra_keywords lists, taking values from the first object
        this.extra_keywords = dict(
            list(other.extra_keywords.items()) + list(this.extra_keywords.items())
        )

        # Merge object_name if different.
        if this.object_name != other.object_name:
            this.object_name = this.object_name + "-" + other.object_name

        # Do the summing / differencing
        if difference:
            this.data_array = this.data_array - other.data_array
            history_update_string = " Visibilities differenced using pyuvdata."
        else:
            this.data_array = this.data_array + other.data_array
            history_update_string = " Visibilities summed using pyuvdata."

        this.history = uvutils._combine_histories(this.history, other.history)
        this.history += history_update_string

        # Check final object is self-consistent
        if run_check:
            this.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
            )

        if not inplace:
            return this

    def diff_vis(
        self,
        other,
        inplace=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
        override_params=None,
    ):
        """
        Difference visibilities between two UVData objects.

        By default requires that all UVParameters are the same on the two objects
        except for `history`, `data_array`, `object_name`, and `extra_keywords`.
        The `object_name` values are concatenated if they are different. If keys
        in `extra_keywords` have different values the values from the first
        object are taken.

        Parameters
        ----------
        other : UVData object
            Another UVData object which will be added to self.
        inplace : bool
            If True, overwrite self as we go, otherwise create a third object
            as the sum of the two.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after combining objects.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            combining objects.
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.
        override_params : array_like of strings
            List of object UVParameters to omit from compatibility check. Overridden
            parameters will not be compared between the objects, and the values
            for these parameters will be taken from the first object.

        Returns
        -------
        UVData Object
            If inplace parameter is False.

        Raises
        ------
        ValueError
            If other is not a UVData object, or if self and other
            are not compatible.

        """
        if inplace:
            self.sum_vis(
                other,
                difference=True,
                inplace=inplace,
                run_check=True,
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
                override_params=override_params,
            )
        else:
            return self.sum_vis(
                other,
                difference=True,
                inplace=inplace,
                run_check=True,
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
                override_params=override_params,
            )

    def parse_ants(self, ant_str, print_toggle=False):
        """
        Get antpair and polarization from parsing an aipy-style ant string.

        Used to support the the select function.
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
        return uvutils.parse_ants(
            uv=self,
            ant_str=ant_str,
            print_toggle=print_toggle,
            x_orientation=self.x_orientation,
        )

    def _select_preprocess(
        self,
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
    ):
        """
        Build up blt_inds, freq_inds, pol_inds and history_update_string for select.

        Parameters
        ----------
        antenna_nums : array_like of int, optional
            The antennas numbers to keep in the object (antenna positions and
            names for the removed antennas will be retained unless
            `keep_all_metadata` is False). This cannot be provided if
            `antenna_names` is also provided.
        antenna_names : array_like of str, optional
            The antennas names to keep in the object (antenna positions and
            names for the removed antennas will be retained unless
            `keep_all_metadata` is False). This cannot be provided if
            `antenna_nums` is also provided.
        bls : list of tuple or list of int, optional
            A list of antenna number tuples (e.g. [(0, 1), (3, 2)]), a list of
            baseline 3-tuples (e.g. [(0, 1, 'xx'), (2, 3, 'yy')]), or a list of
            baseline numbers (e.g. [67599, 71699, 73743]) specifying baselines
            to keep in the object. For length-2 tuples, the ordering of the
            numbers within the tuple does not matter. For length-3 tuples, the
            polarization string is in the order of the two antennas. If
            length-3 tuples are provided, `polarizations` must be None.
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
            exist in the time_array. Cannot be used with `time_range`.
        time_range : array_like of float, optional
            The time range in Julian Date to keep in the object, must be
            length 2. Some of the times in the object should fall between the
            first and last elements. Cannot be used with `times`.
        polarizations : array_like of int, optional
            The polarizations numbers to keep in the object, each value passed
            here should exist in the polarization_array.
        blt_inds : array_like of int, optional
            The baseline-time indices to keep in the object. This is
            not commonly used.

        Returns
        -------
        blt_inds : list of int
            list of baseline-time indices to keep. Can be None (to keep everything).
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

        if ant_str is not None:
            if not (
                antenna_nums is None
                and antenna_names is None
                and bls is None
                and polarizations is None
            ):
                raise ValueError(
                    "Cannot provide ant_str with antenna_nums, antenna_names, "
                    "bls, or polarizations."
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
            history_update_string += "baseline-times"
            n_selects += 1

        if antenna_names is not None:
            if antenna_nums is not None:
                raise ValueError(
                    "Only one of antenna_nums and antenna_names can be provided."
                )

            if not isinstance(antenna_names, (list, tuple, np.ndarray)):
                antenna_names = (antenna_names,)
            if np.array(antenna_names).ndim > 1:
                antenna_names = np.array(antenna_names).flatten()
            antenna_nums = []
            for s in antenna_names:
                if s not in self.antenna_names:
                    raise ValueError(
                        "Antenna name {a} is not present in the antenna_names"
                        " array".format(a=s)
                    )
                antenna_nums.append(
                    self.antenna_numbers[np.where(np.array(self.antenna_names) == s)][0]
                )

        if antenna_nums is not None:
            antenna_nums = uvutils._get_iterable(antenna_nums)
            if np.array(antenna_nums).ndim > 1:
                antenna_nums = np.array(antenna_nums).flatten()
            if n_selects > 0:
                history_update_string += ", antennas"
            else:
                history_update_string += "antennas"
            n_selects += 1
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

            ant_blt_inds = np.array(list(set(inds1).intersection(inds2)), dtype=np.int)
        else:
            ant_blt_inds = None

        if bls is not None:
            if isinstance(bls, list) and all(
                isinstance(bl_ind, (int, np.integer,)) for bl_ind in bls
            ):
                for bl_ind in bls:
                    if not (bl_ind in self.baseline_array):
                        raise ValueError(
                            "Baseline number {i} is not present in the "
                            "baseline_array".format(i=bl_ind)
                        )
                bls = [self.baseline_to_antnums(bl) for bl in bls]
            elif isinstance(bls, tuple) and (len(bls) == 2 or len(bls) == 3):
                bls = [bls]
            if len(bls) == 0 or not all(isinstance(item, tuple) for item in bls):
                raise ValueError(
                    "bls must be a list of tuples of antenna numbers "
                    "(optionally with polarization) or a list of baseline numbers."
                )
            if not all(
                [isinstance(item[0], (int, np.integer,)) for item in bls]
                + [isinstance(item[1], (int, np.integer,)) for item in bls]
            ):
                raise ValueError(
                    "bls must be a list of tuples of antenna numbers "
                    "(optionally with polarization) or a list of baseline numbers."
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

            if ant_str is None:
                if n_selects > 0:
                    history_update_string += ", baselines"
                else:
                    history_update_string += "baselines"
            else:
                history_update_string += "antenna pairs"
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
                        # find conjugate polarization
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
                ant_blt_inds = np.array(
                    list(set(ant_blt_inds).intersection(bls_blt_inds))
                )
            else:
                ant_blt_inds = bls_blt_inds

        if ant_blt_inds is not None:
            if blt_inds is not None:
                # Use intersection (and) to join antenna_names/nums/ant_pairs_nums
                # with blt_inds
                blt_inds = np.array(
                    list(set(blt_inds).intersection(ant_blt_inds)), dtype=np.int
                )
            else:
                blt_inds = ant_blt_inds

        if times is not None:
            if time_range is not None:
                raise ValueError('Only one of "times" and "time_range" can be set')

            times = uvutils._get_iterable(times)
            if np.array(times).ndim > 1:
                times = np.array(times).flatten()

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

        if time_range is not None:
            if np.size(time_range) != 2:
                raise ValueError("time_range must be length 2.")

            time_blt_inds = np.nonzero(
                (self.time_array <= time_range[1]) & (self.time_array >= time_range[0])
            )[0]
            if time_blt_inds.size == 0:
                raise ValueError(
                    f"No elements in time range between {time_range[0]} and "
                    f"{time_range[1]}."
                )

        if times is not None or time_range is not None:
            if n_selects > 0:
                history_update_string += ", times"
            else:
                history_update_string += "times"
            n_selects += 1

            if blt_inds is not None:
                # Use intesection (and) to join
                # antenna_names/nums/ant_pairs_nums/blt_inds with times
                blt_inds = np.array(
                    list(set(blt_inds).intersection(time_blt_inds)), dtype=np.int
                )
            else:
                blt_inds = time_blt_inds

        if blt_inds is not None:
            if len(blt_inds) == 0:
                raise ValueError("No baseline-times were found that match criteria")
            if max(blt_inds) >= self.Nblts:
                raise ValueError("blt_inds contains indices that are too large")
            if min(blt_inds) < 0:
                raise ValueError("blt_inds contains indices that are negative")

            blt_inds = sorted(set(blt_inds))

        if freq_chans is not None:
            freq_chans = uvutils._get_iterable(freq_chans)
            if np.array(freq_chans).ndim > 1:
                freq_chans = np.array(freq_chans).flatten()
            if frequencies is None:
                frequencies = self.freq_array[0, freq_chans]
            else:
                frequencies = uvutils._get_iterable(frequencies)
                frequencies = np.sort(
                    list(set(frequencies) | set(self.freq_array[0, freq_chans]))
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
            freq_arr_use = self.freq_array[0, :]
            for f in frequencies:
                if f in freq_arr_use:
                    freq_inds = np.append(freq_inds, np.where(freq_arr_use == f)[0])
                else:
                    raise ValueError(
                        "Frequency {f} is not present in the freq_array".format(f=f)
                    )

            if len(frequencies) > 1:
                freq_ind_separation = freq_inds[1:] - freq_inds[:-1]
                if np.min(freq_ind_separation) < np.max(freq_ind_separation):
                    warnings.warn(
                        "Selected frequencies are not evenly spaced. This "
                        "will make it impossible to write this data out to "
                        "some file types"
                    )
                elif np.max(freq_ind_separation) > 1:
                    warnings.warn(
                        "Selected frequencies are not contiguous. This "
                        "will make it impossible to write this data out to "
                        "some file types."
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

            if len(pol_inds) > 2:
                pol_ind_separation = pol_inds[1:] - pol_inds[:-1]
                if np.min(pol_ind_separation) < np.max(pol_ind_separation):
                    warnings.warn(
                        "Selected polarization values are not evenly spaced. This "
                        "will make it impossible to write this data out to "
                        "some file types"
                    )

            pol_inds = sorted(set(pol_inds))
        else:
            pol_inds = None

        history_update_string += " using pyuvdata."

        return blt_inds, freq_inds, pol_inds, history_update_string

    def _select_metadata(
        self,
        blt_inds,
        freq_inds,
        pol_inds,
        history_update_string,
        keep_all_metadata=True,
    ):
        """
        Perform select on everything except the data-sized arrays.

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
            self.Nblts = len(blt_inds)
            self.baseline_array = self.baseline_array[blt_inds]
            self.Nbls = len(np.unique(self.baseline_array))
            self.time_array = self.time_array[blt_inds]
            self.integration_time = self.integration_time[blt_inds]
            self.lst_array = self.lst_array[blt_inds]
            self.uvw_array = self.uvw_array[blt_inds, :]

            self.ant_1_array = self.ant_1_array[blt_inds]
            self.ant_2_array = self.ant_2_array[blt_inds]
            self.Nants_data = self._calc_nants_data()

            if self.phase_center_app_ra is not None:
                self.phase_center_app_ra = self.phase_center_app_ra[blt_inds]
            if self.phase_center_app_dec is not None:
                self.phase_center_app_dec = self.phase_center_app_dec[blt_inds]
            if self.multi_object:
                self.object_id_array = self.object_id_array[blt_inds]

            self.Ntimes = len(np.unique(self.time_array))
            if not keep_all_metadata:
                ants_to_keep = set(np.unique(self.ant_1_array)).union(
                    np.unique(self.ant_2_array)
                )

                inds_to_keep = [
                    self.antenna_numbers.tolist().index(ant) for ant in ants_to_keep
                ]
                self.antenna_names = [self.antenna_names[ind] for ind in inds_to_keep]
                self.antenna_numbers = self.antenna_numbers[inds_to_keep]
                self.antenna_positions = self.antenna_positions[inds_to_keep, :]
                if self.antenna_diameters is not None:
                    self.antenna_diameters = self.antenna_diameters[inds_to_keep]
                self.Nants_telescope = int(len(ants_to_keep))

        if freq_inds is not None:
            self.Nfreqs = len(freq_inds)
            self.freq_array = self.freq_array[:, freq_inds]

        if pol_inds is not None:
            self.Npols = len(pol_inds)
            self.polarization_array = self.polarization_array[pol_inds]

        self.history = self.history + history_update_string

    def select(
        self,
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
        inplace=True,
        keep_all_metadata=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
    ):
        """
        Downselect data to keep on the object along various axes.

        Axes that can be selected along include antenna names or numbers,
        antenna pairs, frequencies, times and polarizations. Specific
        baseline-time indices can also be selected, but this is not commonly
        used.
        The history attribute on the object will be updated to identify the
        operations performed.

        Parameters
        ----------
        antenna_nums : array_like of int, optional
            The antennas numbers to keep in the object (antenna positions and
            names for the removed antennas will be retained unless
            `keep_all_metadata` is False). This cannot be provided if
            `antenna_names` is also provided.
        antenna_names : array_like of str, optional
            The antennas names to keep in the object (antenna positions and
            names for the removed antennas will be retained unless
            `keep_all_metadata` is False). This cannot be provided if
            `antenna_nums` is also provided.
        bls : list of tuple, optional
            A list of antenna number tuples (e.g. [(0, 1), (3, 2)]) or a list of
            baseline 3-tuples (e.g. [(0, 1, 'xx'), (2, 3, 'yy')]) specifying baselines
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
            exist in the time_array. Cannot be used with `time_range`.
        time_range : array_like of float, optional
            The time range in Julian Date to keep in the object, must be
            length 2. Some of the times in the object should fall between the
            first and last elements. Cannot be used with `times`.
        polarizations : array_like of int, optional
            The polarizations numbers to keep in the object, each value passed
            here should exist in the polarization_array.
        blt_inds : array_like of int, optional
            The baseline-time indices to keep in the object. This is
            not commonly used.
        inplace : bool
            Option to perform the select directly on self or return a new UVData
            object with just the selected data (the default is True, meaning the
            select will be done on self).
        keep_all_metadata : bool
            Option to keep all the metadata associated with antennas, even those
            that do do not have data associated with them after the select option.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after downselecting data on this object (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            downselecting data on this object (the default is True, meaning the
            acceptable range check will be done).
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.

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
            freq_inds,
            pol_inds,
            history_update_string,
        ) = uv_object._select_preprocess(
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

        # do select operations on everything except data_array, flag_array
        # and nsample_array
        uv_object._select_metadata(
            blt_inds, freq_inds, pol_inds, history_update_string, keep_all_metadata
        )

        if self.metadata_only:
            if not inplace:
                return uv_object
            else:
                return

        if blt_inds is not None:
            for param_name, param in zip(
                self._data_params, uv_object.data_like_parameters
            ):
                setattr(uv_object, param_name, param[blt_inds, :, :, :])

        if freq_inds is not None:
            for param_name, param in zip(
                self._data_params, uv_object.data_like_parameters
            ):
                setattr(uv_object, param_name, param[:, :, freq_inds, :])

        if pol_inds is not None:
            for param_name, param in zip(
                self._data_params, uv_object.data_like_parameters
            ):
                setattr(uv_object, param_name, param[:, :, :, pol_inds])

        # check if object is uv_object-consistent
        if run_check:
            uv_object.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
            )

        if not inplace:
            return uv_object

    def _harmonize_resample_arrays(
        self,
        inds_to_keep,
        temp_baseline,
        temp_time,
        temp_int_time,
        temp_data,
        temp_flag,
        temp_nsample,
    ):
        """
        Make a self-consistent object after up/downsampling.

        This function is called by both upsample_in_time and downsample_in_time.
        See those functions for more information about arguments.
        """
        self.baseline_array = self.baseline_array[inds_to_keep]
        self.time_array = self.time_array[inds_to_keep]
        self.integration_time = self.integration_time[inds_to_keep]

        self.baseline_array = np.concatenate((self.baseline_array, temp_baseline))
        self.time_array = np.concatenate((self.time_array, temp_time))
        self.integration_time = np.concatenate((self.integration_time, temp_int_time))
        if not self.metadata_only:
            self.data_array = self.data_array[inds_to_keep]
            self.flag_array = self.flag_array[inds_to_keep]
            self.nsample_array = self.nsample_array[inds_to_keep]

            # concatenate temp array with existing arrays
            self.data_array = np.concatenate((self.data_array, temp_data), axis=0)
            self.flag_array = np.concatenate((self.flag_array, temp_flag), axis=0)
            self.nsample_array = np.concatenate(
                (self.nsample_array, temp_nsample), axis=0
            )

        # set antenna arrays from baseline_array
        self.ant_1_array, self.ant_2_array = self.baseline_to_antnums(
            self.baseline_array
        )

        # update metadata
        self.Nblts = self.baseline_array.shape[0]
        self.Ntimes = np.unique(self.time_array).size
        self.uvw_array = np.zeros((self.Nblts, 3))

        # set lst array
        self.set_lsts_from_time_array()

        # temporarily store the metadata only to calculate UVWs correctly
        uv_temp = self.copy(metadata_only=True)

        # properly calculate the UVWs self-consistently
        uv_temp.set_uvws_from_antenna_positions(allow_phasing=True)
        self.uvw_array = uv_temp.uvw_array

        return

    def upsample_in_time(
        self,
        max_int_time,
        blt_order="time",
        minor_order="baseline",
        summing_correlator_mode=False,
        allow_drift=False,
    ):
        """
        Resample to a shorter integration time.

        This method will resample a UVData object such that all data samples have
        an integration time less than or equal to the `max_int_time`. The new
        samples are copied from the original samples (not interpolated).

        Parameters
        ----------
        max_int_time : float
            Maximum integration time to upsample to in seconds.
        blt_order : str
            Major baseline ordering for output object. Default is "time". See
            the documentation on the `reorder_blts` method for more info.
        minor_order : str
            Minor baseline ordering for output object. Default is "baseline".
        summing_correlator_mode : bool
            Option to split the flux from the original samples into the new
            samples rather than duplicating the original samples in all the new
            samples (undoing an integration rather than an average) to emulate
            undoing the behavior in some correlators (e.g. HERA).
        allow_drift : bool
            Option to allow resampling of drift mode data. If this is False,
            drift mode data will be phased before resampling and then unphased
            after resampling. Phasing and unphasing can introduce small errors,
            but resampling in drift mode may result in unexpected behavior.

        Returns
        -------
        None

        """
        # check that max_int_time is sensible given integration_time
        min_integration_time = np.amin(self.integration_time)
        sensible_min = 1e-2 * min_integration_time
        if max_int_time < sensible_min:
            raise ValueError(
                "Decreasing the integration time by more than a "
                "factor of 100 is not supported. Also note that "
                "max_int_time should be in seconds."
            )

        # figure out where integration_time is longer than max_int_time
        inds_to_upsample = np.nonzero(
            (self.integration_time > max_int_time)
            & (
                ~np.isclose(
                    self.integration_time,
                    max_int_time,
                    rtol=self._integration_time.tols[0],
                    atol=self._integration_time.tols[1],
                )
            )
        )
        if len(inds_to_upsample[0]) == 0:
            warnings.warn(
                "All values in the integration_time array are already "
                "longer than the value specified; doing nothing."
            )
            return

        input_phase_type = self.phase_type
        if input_phase_type == "drift":
            if allow_drift:
                print(
                    "Data are in drift mode and allow_drift is True, so "
                    "resampling will be done without phasing."
                )
            else:
                # phase to RA/dec of zenith
                print("Data are in drift mode, phasing before resampling.")
                phase_time = Time(self.time_array[0], format="jd")
                self.phase_to_time(phase_time)

        # we want the ceil of this, but we don't want to get the wrong answer
        # when the number is very close to an integer but just barely above it.
        temp_new_samples = self.integration_time[inds_to_upsample] / max_int_time
        mask_close_floor = np.isclose(temp_new_samples, np.floor(temp_new_samples))
        temp_new_samples[mask_close_floor] = np.floor(
            temp_new_samples[mask_close_floor]
        )

        n_new_samples = np.asarray(list(map(int, np.ceil(temp_new_samples))))

        temp_Nblts = np.sum(n_new_samples)

        temp_baseline = np.zeros((temp_Nblts,), dtype=np.int)
        temp_time = np.zeros((temp_Nblts,))
        temp_int_time = np.zeros((temp_Nblts,))
        if self.metadata_only:
            temp_data = None
            temp_flag = None
            temp_nsample = None
        else:
            # TODO: Spw axis to be collapsed in future release
            temp_data = np.zeros(
                (temp_Nblts, 1, self.Nfreqs, self.Npols), dtype=self.data_array.dtype,
            )
            # TODO: Spw axis to be collapsed in future release
            temp_flag = np.zeros(
                (temp_Nblts, 1, self.Nfreqs, self.Npols), dtype=self.flag_array.dtype,
            )
            # TODO: Spw axis to be collapsed in future release
            temp_nsample = np.zeros(
                (temp_Nblts, 1, self.Nfreqs, self.Npols),
                dtype=self.nsample_array.dtype,
            )

        i0 = 0
        for i, ind in enumerate(inds_to_upsample[0]):
            i1 = i0 + n_new_samples[i]
            temp_baseline[i0:i1] = self.baseline_array[ind]
            if not self.metadata_only:
                if summing_correlator_mode:
                    temp_data[i0:i1] = self.data_array[ind] / n_new_samples[i]
                else:
                    temp_data[i0:i1] = self.data_array[ind]
                temp_flag[i0:i1] = self.flag_array[ind]
                temp_nsample[i0:i1] = self.nsample_array[ind]

            # compute the new times of the upsampled array
            t0 = self.time_array[ind]
            dt = self.integration_time[ind] / n_new_samples[i]

            # `offset` will be 0.5 or 1, depending on whether n_new_samples for
            # this baseline is even or odd.
            offset = 0.5 + 0.5 * (n_new_samples[i] % 2)
            n2 = n_new_samples[i] // 2

            # Figure out the new center for sample ii taking offset into
            # account. Because `t0` is the central time for the original time
            # sample, `nt` will range from negative to positive so that
            # `temp_time` will result in the central time for the new samples.
            # `idx2` tells us how to far to shift and in what direction for each
            # new sample.
            for ii, idx in enumerate(range(i0, i1)):
                idx2 = ii + offset + n2 - n_new_samples[i]
                nt = ((t0 * units.day) + (dt * idx2 * units.s)).to(units.day).value
                temp_time[idx] = nt

            temp_int_time[i0:i1] = dt

            i0 = i1

        # harmonize temporary arrays with existing ones
        inds_to_keep = np.nonzero(self.integration_time <= max_int_time)
        self._harmonize_resample_arrays(
            inds_to_keep,
            temp_baseline,
            temp_time,
            temp_int_time,
            temp_data,
            temp_flag,
            temp_nsample,
        )

        if input_phase_type == "drift" and not allow_drift:
            print("Unphasing back to drift mode.")
            self.unphase_to_drift()

        # reorganize along blt axis
        self.reorder_blts(order=blt_order, minor_order=minor_order)

        # check the resulting object
        self.check()

        # add to the history
        history_update_string = (
            " Upsampled data to {:f} second integration time "
            "using pyuvdata.".format(max_int_time)
        )
        self.history = self.history + history_update_string

        return

    def downsample_in_time(
        self,
        min_int_time=None,
        n_times_to_avg=None,
        blt_order="time",
        minor_order="baseline",
        keep_ragged=True,
        summing_correlator_mode=False,
        allow_drift=False,
    ):
        """
        Average to a longer integration time.

        This method will average a UVData object either by an integer factor
        (by setting `n_times_to_avg`) or by a factor that can differ by
        baseline-time sample such that after averaging, the samples have an
        integration time greater than or equal to the `min_int_time` (up to the
        tolerance on the integration_time).

        Note that if the integrations for a baseline do not divide evenly by the
        `n_times_to_avg` or into the specified `min_int_time`, the final
        integrations for that baseline may have integration times less than
        `min_int_time` or be composed of fewer input integrations than `n_times_to_avg`.
        This behavior can be controlled with the `keep_ragged` argument.
        The new samples are averages of the original samples (not interpolations).

        Parameters
        ----------
        min_int_time : float
            Minimum integration time to average the UVData integration_time to
            in seconds.
        n_times_to_avg : int
            Number of time integrations to average together.
        blt_order : str
            Major baseline ordering for output object. Default is "time". See the
            documentation on the `reorder_blts` method for more details.
        minor_order : str
            Minor baseline ordering for output object. Default is "baseline".
        keep_ragged : bool
            When averaging baselines that do not evenly divide into min_int_time,
            or that have a number of integrations that do not evenly divide by
            n_times_to_avg, keep_ragged controls whether to keep the (averaged)
            integrations corresponding to the remaining samples (keep_ragged=True),
            or discard them (keep_ragged=False).
        summing_correlator_mode : bool
            Option to integrate the flux from the original samples rather than
            average the flux to emulate the behavior in some correlators (e.g. HERA).
        allow_drift : bool
            Option to allow averaging of drift mode data. If this is False,
            drift mode data will be phased before resampling and then unphased
            after resampling. Phasing and unphasing can introduce small errors,
            but averaging in drift mode may result in more decoherence.

        Returns
        -------
        None

        """
        if min_int_time is None and n_times_to_avg is None:
            raise ValueError("Either min_int_time or n_times_to_avg must be set.")

        if min_int_time is not None and n_times_to_avg is not None:
            raise ValueError("Only one of min_int_time or n_times_to_avg can be set.")

        if self.Ntimes == 1:
            raise ValueError("Only one time in this object, cannot downsample.")

        if min_int_time is not None:
            # check that min_int_time is sensible given integration_time
            max_integration_time = np.amax(self.integration_time)
            sensible_max = 1e2 * max_integration_time
            if min_int_time > sensible_max:
                raise ValueError(
                    "Increasing the integration time by more than a "
                    "factor of 100 is not supported. Also note that "
                    "min_int_time should be in seconds."
                )

            # first figure out where integration_time is shorter than min_int_time
            inds_to_downsample = np.nonzero(
                (self.integration_time < min_int_time)
                & (
                    ~np.isclose(
                        self.integration_time,
                        min_int_time,
                        rtol=self._integration_time.tols[0],
                        atol=self._integration_time.tols[1],
                    )
                )
            )

            if len(inds_to_downsample[0]) == 0:
                warnings.warn(
                    "All values in the integration_time array are already "
                    "longer than the value specified; doing nothing."
                )
                return
        else:
            if not isinstance(n_times_to_avg, (int, np.integer)):
                raise ValueError("n_times_to_avg must be an integer.")
        # If we're going to do actual work, reorder the baselines to ensure time is
        # monotonically increasing.
        # Default of reorder_blts is baseline major, time minor, which is what we want.
        self.reorder_blts()

        if min_int_time is not None:
            # now re-compute inds_to_downsample, in case things have changed
            inds_to_downsample = np.nonzero(
                (self.integration_time < min_int_time)
                & ~np.isclose(
                    self.integration_time,
                    min_int_time,
                    rtol=self._integration_time.tols[0],
                    atol=self._integration_time.tols[1],
                )
            )
            bls_to_downsample = np.unique(self.baseline_array[inds_to_downsample])
        else:
            bls_to_downsample = np.unique(self.baseline_array)

        # figure out how many baseline times we'll end up with at the end
        n_new_samples = 0
        for bl in bls_to_downsample:
            bl_inds = np.nonzero(self.baseline_array == bl)[0]
            int_times = self.integration_time[bl_inds]

            if min_int_time is not None:
                running_int_time = 0.0
                for itime, int_time in enumerate(int_times):
                    running_int_time += int_time
                    over_min_int_time = running_int_time > min_int_time or np.isclose(
                        running_int_time,
                        min_int_time,
                        rtol=self._integration_time.tols[0],
                        atol=self._integration_time.tols[1],
                    )
                    last_sample = itime == len(bl_inds) - 1
                    # We sum up all the samples found so far if we're over the
                    # target minimum time, or we've hit the end of the time
                    # samples for this baseline.
                    if over_min_int_time or last_sample:
                        if last_sample and not (over_min_int_time or keep_ragged):
                            # don't do anything -- implicitly drop these integrations
                            continue
                        n_new_samples += 1
                        running_int_time = 0.0
            else:
                n_bl_times = self.time_array[bl_inds].size
                nsample_temp = np.sum(n_bl_times / n_times_to_avg)
                if keep_ragged and not np.isclose(nsample_temp, np.floor(nsample_temp)):
                    n_new_samples += np.ceil(nsample_temp).astype(int)
                else:
                    n_new_samples += np.floor(nsample_temp).astype(int)

            # figure out if there are any time gaps in the data
            # meaning that the time differences are larger than the integration times
            # time_array is in JD, need to convert to seconds for the diff
            dtime = np.ediff1d(self.time_array[bl_inds]) * 24 * 3600
            int_times = int_times
            if len(np.unique(int_times)) == 1:
                # this baseline has all the same integration times
                if len(np.unique(dtime)) > 1 and not np.isclose(
                    np.max(dtime),
                    np.min(dtime),
                    rtol=self._integration_time.tols[0],
                    atol=self._integration_time.tols[1],
                ):
                    warnings.warn(
                        "There is a gap in the times of baseline {bl}. "
                        "The output may include averages across long "
                        "time gaps.".format(bl=self.baseline_to_antnums(bl))
                    )
                elif not np.isclose(
                    dtime[0],
                    int_times[0],
                    rtol=self._integration_time.tols[0],
                    atol=self._integration_time.tols[1],
                ):
                    warnings.warn(
                        "The time difference between integrations is "
                        "not the same as the integration time for "
                        "baseline {bl}. The output may average across "
                        "longer time intervals than "
                        "expected".format(bl=self.baseline_to_antnums(bl))
                    )

            else:
                # varying integration times for this baseline, need to be more careful
                expected_dtimes = (int_times[:-1] + int_times[1:]) / 2
                wh_diff = np.nonzero(~np.isclose(dtime, expected_dtimes))
                if wh_diff[0].size > 1:
                    warnings.warn(
                        "The time difference between integrations is "
                        "different than the expected given the "
                        "integration times for baseline {bl}. The "
                        "output may include averages across long time "
                        "gaps.".format(bl=self.baseline_to_antnums(bl))
                    )

        temp_Nblts = n_new_samples

        input_phase_type = self.phase_type
        if input_phase_type == "drift":
            if allow_drift:
                print(
                    "Data are in drift mode and allow_drift is True, so "
                    "resampling will be done without phasing."
                )
            else:
                # phase to RA/dec of zenith
                print("Data are in drift mode, phasing before resampling.")
                phase_time = Time(self.time_array[0], format="jd")
                self.phase_to_time(phase_time)

        # make temporary arrays
        temp_baseline = np.zeros((temp_Nblts,), dtype=np.int)
        temp_time = np.zeros((temp_Nblts,))
        temp_int_time = np.zeros((temp_Nblts,))
        if self.metadata_only:
            temp_data = None
            temp_flag = None
            temp_nsample = None
        else:
            temp_data = np.zeros(
                (temp_Nblts, 1, self.Nfreqs, self.Npols), dtype=self.data_array.dtype,
            )
            temp_flag = np.zeros(
                (temp_Nblts, 1, self.Nfreqs, self.Npols), dtype=self.flag_array.dtype,
            )
            temp_nsample = np.zeros(
                (temp_Nblts, 1, self.Nfreqs, self.Npols),
                dtype=self.nsample_array.dtype,
            )

        temp_idx = 0
        for bl in bls_to_downsample:
            bl_inds = np.nonzero(self.baseline_array == bl)[0]
            running_int_time = 0.0
            summing_idx = 0
            n_sum = 0
            for itime, int_time in enumerate(self.integration_time[bl_inds]):
                running_int_time += int_time
                n_sum += 1
                if min_int_time is not None:
                    over_min_int_time = running_int_time > min_int_time or np.isclose(
                        running_int_time,
                        min_int_time,
                        rtol=self._integration_time.tols[0],
                        atol=self._integration_time.tols[1],
                    )
                else:
                    over_min_int_time = n_sum >= n_times_to_avg
                last_sample = itime == len(bl_inds) - 1
                # We sum up all the samples found so far if we're over the
                # target minimum time, or we've hit the end of the time
                # samples for this baseline.
                if over_min_int_time or last_sample:
                    if last_sample and not (over_min_int_time or keep_ragged):
                        # don't do anything -- implicitly drop these integrations
                        continue
                    # sum together that number of samples
                    temp_baseline[temp_idx] = bl
                    # this might be wrong if some of the constituent times are
                    # *totally* flagged
                    averaging_idx = bl_inds[summing_idx : summing_idx + n_sum]
                    # take potential non-uniformity of integration_time into account
                    temp_time[temp_idx] = np.sum(
                        self.time_array[averaging_idx]
                        * self.integration_time[averaging_idx]
                    ) / np.sum(self.integration_time[averaging_idx])
                    temp_int_time[temp_idx] = running_int_time
                    if not self.metadata_only:
                        # if all inputs are flagged, the flag array should be True,
                        # otherwise it should be False.
                        # The sum below will be zero if it's all flagged and
                        # greater than zero otherwise
                        # Then we use a test against 0 to turn it into a Boolean
                        temp_flag[temp_idx] = (
                            np.sum(~self.flag_array[averaging_idx], axis=0) == 0
                        )

                        mask = self.flag_array[averaging_idx]
                        # need to update mask if a downsampled visibility will
                        # be flagged so that we don't set it to zero
                        if (temp_flag[temp_idx]).any():
                            ax1_inds, ax2_inds, ax3_inds = np.nonzero(
                                temp_flag[temp_idx]
                            )
                            mask[:, ax1_inds, ax2_inds, ax3_inds] = False

                        masked_data = np.ma.masked_array(
                            self.data_array[averaging_idx], mask=mask
                        )

                        # nsample array is the fraction of data that we actually kept,
                        # relative to the amount that went into the sum or average
                        nsample_dtype = self.nsample_array.dtype.type
                        # promote nsample dtype if half-precision
                        if nsample_dtype is np.float16:
                            masked_nsample_dtype = np.float32
                        else:
                            masked_nsample_dtype = nsample_dtype
                        masked_nsample = np.ma.masked_array(
                            self.nsample_array[averaging_idx],
                            mask=mask,
                            dtype=masked_nsample_dtype,
                        )

                        masked_int_time = np.ma.masked_array(
                            np.ones_like(
                                self.data_array[averaging_idx],
                                dtype=self.integration_time.dtype,
                            )
                            * self.integration_time[
                                averaging_idx, np.newaxis, np.newaxis, np.newaxis
                            ],
                            mask=mask,
                        )
                        if summing_correlator_mode:
                            temp_data[temp_idx] = np.sum(masked_data, axis=0)
                        else:
                            # take potential non-uniformity of integration_time
                            # and nsamples into account
                            weights = masked_nsample * masked_int_time
                            weighted_data = masked_data * weights
                            temp_data[temp_idx] = np.sum(
                                weighted_data, axis=0
                            ) / np.sum(weights, axis=0)

                        # output of masked array calculation should be coerced
                        # to the datatype of temp_nsample (which has the same
                        # precision as the original nsample_array)
                        temp_nsample[temp_idx] = np.sum(
                            masked_nsample * masked_int_time, axis=0
                        ) / np.sum(self.integration_time[averaging_idx])
                    # increment counters and reset values
                    temp_idx += 1
                    summing_idx += n_sum
                    running_int_time = 0.0
                    n_sum = 0

        # make sure we've populated the right number of baseline-times
        assert temp_idx == temp_Nblts, (
            "Wrong number of baselines. Got {:d},  expected {:d}. This is a bug, "
            "please make an issue at https://github.com/RadioAstronomySoftwareGroup/"
            "pyuvdata/issues".format(temp_idx, temp_Nblts)
        )

        # harmonize temporary arrays with existing ones
        if min_int_time is not None:
            bls_not_downsampled = set(self.baseline_array) - set(bls_to_downsample)
            inds_to_keep = []
            for bl in bls_not_downsampled:
                inds_to_keep += np.nonzero(self.baseline_array == bl)[0].tolist()
            inds_to_keep = np.array(inds_to_keep, dtype=np.int)
        else:
            inds_to_keep = np.array([], dtype=bool)
        self._harmonize_resample_arrays(
            inds_to_keep,
            temp_baseline,
            temp_time,
            temp_int_time,
            temp_data,
            temp_flag,
            temp_nsample,
        )

        if input_phase_type == "drift" and not allow_drift:
            print("Unphasing back to drift mode.")
            self.unphase_to_drift()

        # reorganize along blt axis
        self.reorder_blts(order=blt_order, minor_order=minor_order)

        # check the resulting object
        self.check()

        # add to the history
        if min_int_time is not None:
            history_update_string = (
                " Downsampled data to {:f} second integration "
                "time using pyuvdata.".format(min_int_time)
            )
        else:
            history_update_string = (
                " Downsampled data by a factor of {} in "
                "time using pyuvdata.".format(n_times_to_avg)
            )
        self.history = self.history + history_update_string

        return

    def resample_in_time(
        self,
        target_time,
        only_downsample=False,
        only_upsample=False,
        blt_order="time",
        minor_order="baseline",
        keep_ragged=True,
        summing_correlator_mode=False,
        allow_drift=False,
    ):
        """
        Intelligently upsample or downsample a UVData object to the target time.

        Parameters
        ----------
        target_time : float
            The target integration time to resample to, in seconds.
        only_downsample : bool
            Option to only call bda_downsample.
        only_upsample : bool
            Option to only call bda_upsample.
        blt_order : str
            Major baseline ordering for output object. Default is "time". See the
            documentation on the `reorder_blts` method for more details.
        minor_order : str
            Minor baseline ordering for output object. Default is "baseline".
        keep_ragged : bool
            When averaging baselines that do not evenly divide into min_int_time,
            keep_ragged controls whether to keep the (summed) integrations
            corresponding to the remaining samples (keep_ragged=True), or
            discard them (keep_ragged=False). Note this option only applies to the
            `bda_downsample` method.
        summing_correlator_mode : bool
            Option to integrate or split the flux from the original samples
            rather than average or duplicate the flux from the original samples
            to emulate the behavior in some correlators (e.g. HERA).
        allow_drift : bool
            Option to allow resampling of drift mode data. If this is False,
            drift mode data will be phased before resampling and then unphased
            after resampling. Phasing and unphasing can introduce small errors,
            but resampling in drift mode may result in unexpected behavior.

        Returns
        -------
        None
        """
        # figure out integration times relative to target time
        min_int_time = np.amin(self.integration_time)
        max_int_time = np.amax(self.integration_time)

        if int(np.floor(target_time / min_int_time)) >= 2 and not only_upsample:
            downsample = True
        else:
            downsample = False

        if int(np.floor(max_int_time / target_time)) >= 2 and not only_downsample:
            upsample = True
        else:
            upsample = False

        if not downsample and not upsample:
            warnings.warn(
                "No resampling will be done because target time is not "
                "a factor of 2 or more off from integration_time. To "
                "force resampling set only_upsample or only_downsample "
                "keywords or call upsample_in_time or downsample_in_time."
            )
            return

        if downsample:
            self.downsample_in_time(
                target_time,
                blt_order=blt_order,
                minor_order=minor_order,
                keep_ragged=keep_ragged,
                summing_correlator_mode=summing_correlator_mode,
                allow_drift=allow_drift,
            )
        if upsample:
            self.upsample_in_time(
                target_time,
                blt_order=blt_order,
                minor_order=minor_order,
                summing_correlator_mode=summing_correlator_mode,
                allow_drift=allow_drift,
            )

        return

    def frequency_average(
        self, n_chan_to_avg, summing_correlator_mode=False, propagate_flags=False
    ):
        """
        Average in frequency.

        Does a simple average over an integer number of input channels, leaving
        flagged samples out of the average.

        In the future, this method will support non-equally spaced channels
        and varying channel widths. It will also support setting the frequency
        to the true mean of the averaged non-flagged frequencies rather than
        the simple mean of the input channel frequencies. For now it does not.

        Parameters
        ----------
        n_chan_to_avg : int
            Number of channels to average together. If Nfreqs does not divide
            evenly by this number, the frequencies at the end of the freq_array
            will be dropped to make it evenly divisable. To control which
            frequencies are removed, use select before calling this method.
        summing_correlator_mode : bool
            Option to integrate or split the flux from the original samples
            rather than average or duplicate the flux from the original samples
            to emulate the behavior in some correlators (e.g. HERA).
        propagate_flags: bool
            Option to flag an averaged entry even if some of its contributors
            are not flagged. The averaged result will still leave the flagged
            samples out of the average, except when all contributors are
            flagged.
        """
        if self.flex_spw:
            raise NotImplementedError(
                "Frequency averaging not (yet) available for flexible spectral windows"
            )
        self._check_freq_spacing()

        n_final_chan = int(np.floor(self.Nfreqs / n_chan_to_avg))
        nfreq_mod_navg = self.Nfreqs % n_chan_to_avg
        if nfreq_mod_navg != 0:
            # not an even number of final channels
            warnings.warn(
                "Nfreqs does not divide by `n_chan_to_avg` evenly. "
                "The final {} frequencies will be excluded, to "
                "control which frequencies to exclude, use a "
                "select to control.".format(nfreq_mod_navg)
            )
            chan_to_keep = np.arange(n_final_chan * n_chan_to_avg)
            self.select(freq_chans=chan_to_keep)

        self.freq_array = self.freq_array.reshape(
            (1, n_final_chan, n_chan_to_avg)
        ).mean(axis=2)
        self.channel_width = self.channel_width * n_chan_to_avg
        self.Nfreqs = n_final_chan

        if self.eq_coeffs is not None:
            eq_coeff_diff = np.diff(self.eq_coeffs, axis=1)
            if np.abs(np.max(eq_coeff_diff)) > 0:
                warnings.warn(
                    "eq_coeffs vary by frequency. They should be "
                    "applied to the data using `remove_eq_coeffs` "
                    "before frequency averaging."
                )
            self.eq_coeffs = self.eq_coeffs.reshape(
                (self.Nants_telescope, n_final_chan, n_chan_to_avg)
            ).mean(axis=2)

        if not self.metadata_only:
            # TODO: Spw axis to be collapsed in future release
            shape_tuple = (
                self.Nblts,
                1,
                n_final_chan,
                n_chan_to_avg,
                self.Npols,
            )

            mask = self.flag_array.reshape(shape_tuple)

            if propagate_flags:
                # if any contributors are flagged, the result should be flagged
                self.flag_array = np.any(self.flag_array.reshape(shape_tuple), axis=3)
            else:
                # if all inputs are flagged, the flag array should be True,
                # otherwise it should be False.
                # The sum below will be zero if it's all flagged and
                # greater than zero otherwise
                # Then we use a test against 0 to turn it into a Boolean
                self.flag_array = (
                    np.sum(~self.flag_array.reshape(shape_tuple), axis=3) == 0
                )

            # need to update mask if a downsampled visibility will be flagged
            # so that we don't set it to zero
            for n_chan in np.arange(n_final_chan):
                if (self.flag_array[:, :, n_chan]).any():
                    ax0_inds, ax1_inds, ax3_inds = np.nonzero(
                        self.flag_array[:, :, n_chan, :]
                    )
                    # Only if all entries are masked
                    # May not happen due to propagate_flags keyword
                    # mask should be left alone otherwise
                    if np.all(mask[ax0_inds, ax1_inds, n_chan, :, ax3_inds]):
                        mask[ax0_inds, ax1_inds, n_chan, :, ax3_inds] = False

            masked_data = np.ma.masked_array(
                self.data_array.reshape(shape_tuple), mask=mask
            )

            self.nsample_array = self.nsample_array.reshape(shape_tuple)
            # promote nsample dtype if half-precision
            nsample_dtype = self.nsample_array.dtype.type
            if nsample_dtype is np.float16:
                masked_nsample_dtype = np.float32
            else:
                masked_nsample_dtype = nsample_dtype
            masked_nsample = np.ma.masked_array(
                self.nsample_array, mask=mask, dtype=masked_nsample_dtype
            )

            if summing_correlator_mode:
                self.data_array = np.sum(masked_data, axis=3).data
            else:
                # need to weight by the nsample_array
                self.data_array = (
                    np.sum(masked_data * masked_nsample, axis=3)
                    / np.sum(masked_nsample, axis=3)
                ).data

            # nsample array is the fraction of data that we actually kept,
            # relative to the amount that went into the sum or average.
            # Need to take care to return precision back to original value.
            self.nsample_array = (
                np.sum(masked_nsample, axis=3) / float(n_chan_to_avg)
            ).data.astype(nsample_dtype)

    def get_redundancies(
        self,
        tol=1.0,
        use_antpos=False,
        include_conjugates=False,
        include_autos=True,
        conjugate_bls=False,
    ):
        """
        Get redundant baselines to a given tolerance.

        This can be used to identify redundant baselines present in the data,
        or find all possible redundant baselines given the antenna positions.

        Parameters
        ----------
        tol : float
            Redundancy tolerance in meters (default 1m).
        use_antpos : bool
            Use antenna positions to find all possible redundant groups for this
            telescope (default False).
            The returned baselines are in the 'u>0' convention.
        include_conjugates : bool
            Option to include baselines that are redundant under conjugation.
            Only used if use_antpos is False.
        include_autos : bool
            Option to include autocorrelations in the full redundancy list.
            Only used if use_antpos is True.
        conjugate_bls : bool
            If using antenna positions, this will conjugate baselines on this
            object to correspond with those in the returned groups.

        Returns
        -------
        baseline_groups : list of lists of int
            List of lists of redundant baseline numbers
        vec_bin_centers : list of ndarray of float
            List of vectors describing redundant group uvw centers
        lengths : list of float
            List of redundant group baseline lengths in meters
        conjugates : list of int, or None, optional
            List of indices for baselines that must be conjugated to fit into their
            redundant groups.
            Will return None if use_antpos is True and include_conjugates is True
            Only returned if include_conjugates is True

        Notes
        -----
        If use_antpos is set, then this function will find all redundant baseline groups
        for this telescope, under the u>0 antenna ordering convention.
        If use_antpos is not set, this function will look for redundant groups
        in the data.

        """
        if use_antpos:
            antpos, numbers = self.get_ENU_antpos(center=False)
            result = uvutils.get_antenna_redundancies(
                numbers, antpos, tol=tol, include_autos=include_autos
            )
            if conjugate_bls:
                self.conjugate_bls(convention="u>0", uvw_tol=tol)

            if include_conjugates:
                result = result + (None,)
            return result

        _, unique_inds = np.unique(self.baseline_array, return_index=True)
        unique_inds.sort()
        baseline_vecs = np.take(self.uvw_array, unique_inds, axis=0)
        baselines = np.take(self.baseline_array, unique_inds)

        return uvutils.get_baseline_redundancies(
            baselines, baseline_vecs, tol=tol, with_conjugates=include_conjugates
        )

    def compress_by_redundancy(
        self, method="select", tol=1.0, inplace=True, keep_all_metadata=True
    ):
        """
        Downselect or average to only have one baseline per redundant group.

        Either select the first baseline in the redundant group or average over
        the baselines in the redundant group.

        Uses utility functions to find redundant baselines to the given tolerance,
        then select on those.

        Parameters
        ----------
        tol : float
            Redundancy tolerance in meters, default is 1.0 corresponding to 1 meter.
        method : str
            Options are "select", which just keeps the first baseline in each
            redundant group or "average" which averages over the baselines in each
            redundant group and assigns the average to the first baseline in the group.
        inplace : bool
            Option to do selection on current object.
        keep_all_metadata : bool
            Option to keep all the metadata associated with antennas,
            even those that do not remain after the select option.

        Returns
        -------
        UVData object or None
            if inplace is False, return the compressed UVData object

        """
        allowed_methods = ["select", "average"]
        if method not in allowed_methods:
            raise ValueError(f"method must be one of {allowed_methods}")

        red_gps, centers, lengths, conjugates = self.get_redundancies(
            tol, include_conjugates=True
        )
        bl_ants = [self.baseline_to_antnums(gp[0]) for gp in red_gps]

        if method == "average":
            # do a metadata only select to get all the metadata right
            new_obj = self.copy(metadata_only=True)
            new_obj.select(bls=bl_ants, keep_all_metadata=keep_all_metadata)

            temp_time_array = np.zeros_like(new_obj.time_array)
            if not self.metadata_only:
                # initalize the data like arrays
                # TODO: Spw axis to be collapsed in future release
                temp_data_array = np.zeros(
                    (new_obj.Nblts, 1, new_obj.Nfreqs, new_obj.Npols),
                    dtype=self.data_array.dtype,
                )
                # TODO: Spw axis to be collapsed in future release
                temp_nsample_array = np.zeros(
                    (new_obj.Nblts, 1, new_obj.Nfreqs, new_obj.Npols),
                    dtype=self.nsample_array.dtype,
                )
                # TODO: Spw axis to be collapsed in future release
                temp_flag_array = np.zeros(
                    (new_obj.Nblts, 1, new_obj.Nfreqs, new_obj.Npols),
                    dtype=self.flag_array.dtype,
                )
            for grp_ind, group in enumerate(red_gps):
                if len(conjugates) > 0:
                    conj_group = set(group).intersection(conjugates)
                    reg_group = list(set(group) - conj_group)
                    conj_group = list(conj_group)
                else:
                    reg_group = group
                    conj_group = []
                group_times = []
                group_inds = []
                conj_group_inds = []
                conj_group_times = []
                for bl in reg_group:
                    bl_inds = np.where(self.baseline_array == bl)[0]
                    group_inds.extend(bl_inds)
                    group_times.extend(self.time_array[bl_inds])
                for bl in conj_group:
                    bl_inds = np.where(self.baseline_array == bl)[0]
                    conj_group_inds.extend(bl_inds)
                    conj_group_times.extend(self.time_array[bl_inds])

                group_inds = np.array(group_inds, dtype=np.int)
                conj_group_inds = np.array(conj_group_inds, dtype=np.int)
                # now we have to figure out which times are the same to a tolerance
                # so we can average over them.
                time_inds = np.arange(len(group_times + conj_group_times))
                time_gps = uvutils.find_clusters(
                    time_inds,
                    np.array(group_times + conj_group_times),
                    self._time_array.tols[1],
                )

                # average over the same times
                obj_bl = bl_ants[grp_ind]
                obj_inds = new_obj._key2inds(obj_bl)[0]
                obj_times = new_obj.time_array[obj_inds]

                for gp in time_gps:
                    avg_time = np.average(np.array(group_times + conj_group_times)[gp])

                    obj_time_ind = np.where(
                        np.abs(obj_times - avg_time) < self._time_array.tols[1]
                    )[0]

                    if obj_time_ind.size == 1:
                        this_obj_ind = obj_inds[obj_time_ind[0]]
                        temp_time_array[this_obj_ind] = avg_time
                    else:
                        warnings.warn(
                            "Index baseline in the redundant group does not "
                            "have all the times, compressed object will be "
                            "missing those times."
                        )
                        continue

                    # time_ind contains indices for both regular and conjugated bls
                    # because we needed to group them together in time.
                    # The regular ones are first and extend the length of group_times,
                    # so we use that to split them back up.
                    regular_orientation = np.array(
                        [time_ind for time_ind in gp if time_ind < len(group_times)],
                        dtype=np.int,
                    )
                    regular_inds = group_inds[np.array(regular_orientation)]
                    conj_orientation = np.array(
                        [
                            time_ind - len(group_times)
                            for time_ind in gp
                            if time_ind >= len(group_times)
                        ],
                        dtype=np.int,
                    )
                    conj_inds = conj_group_inds[np.array(conj_orientation)]
                    # check that the integration times are all the same
                    int_times = np.concatenate(
                        (
                            self.integration_time[regular_inds],
                            self.integration_time[conj_inds],
                        )
                    )
                    if not np.all(
                        np.abs(int_times - new_obj.integration_time[obj_time_ind])
                        < new_obj._integration_time.tols[1]
                    ):
                        warnings.warn(
                            "Integrations times are not identical in a redundant "
                            "group. Averaging anyway but this may cause unexpected "
                            "behavior."
                        )

                    if not self.metadata_only:
                        vis_to_avg = np.concatenate(
                            (
                                self.data_array[regular_inds, :, :, :],
                                np.conj(self.data_array[conj_inds, :, :, :]),
                            )
                        )
                        nsample_to_avg = np.concatenate(
                            (
                                self.nsample_array[regular_inds, :, :, :],
                                self.nsample_array[conj_inds, :, :, :],
                            )
                        )
                        flags_to_avg = np.concatenate(
                            (
                                self.flag_array[regular_inds, :, :, :],
                                self.flag_array[conj_inds, :, :, :],
                            )
                        )
                        # if all data is flagged, average it all as if it were not
                        if np.all(flags_to_avg):
                            mask = np.zeros_like(flags_to_avg)
                        else:
                            mask = flags_to_avg

                        vis_to_avg = np.ma.masked_array(vis_to_avg, mask=mask)

                        nsample_to_avg = np.ma.masked_array(nsample_to_avg, mask=mask)

                        avg_vis = np.ma.average(
                            vis_to_avg, weights=nsample_to_avg, axis=0
                        )
                        avg_nsample = np.sum(nsample_to_avg, axis=0)
                        avg_flag = np.all(flags_to_avg, axis=0)

                        temp_data_array[this_obj_ind, :, :, :] = avg_vis
                        temp_nsample_array[this_obj_ind, :, :, :] = avg_nsample
                        temp_flag_array[this_obj_ind, :, :, :] = avg_flag

            temp_ntimes = np.unique(temp_time_array).size

            if inplace:
                self.select(bls=bl_ants, keep_all_metadata=keep_all_metadata)
                self.time_array = temp_time_array
                self.Ntimes = temp_ntimes
                self.set_lsts_from_time_array()
                if not self.metadata_only:
                    self.data_array = temp_data_array
                    self.nsample_array = temp_nsample_array
                    self.flag_array = temp_flag_array
                return
            else:
                new_obj.time_array = temp_time_array
                new_obj.Ntimes = temp_ntimes
                new_obj.set_lsts_from_time_array()
                if not self.metadata_only:
                    new_obj.data_array = temp_data_array
                    new_obj.nsample_array = temp_nsample_array
                    new_obj.flag_array = temp_flag_array
                return new_obj
        else:
            return self.select(
                bls=bl_ants, inplace=inplace, keep_all_metadata=keep_all_metadata
            )

    def inflate_by_redundancy(self, tol=1.0, blt_order="time", blt_minor_order=None):
        """
        Expand data to full size, copying data among redundant baselines.

        Note that this method conjugates baselines to the 'u>0' convention in order
        to inflate the redundancies.

        Parameters
        ----------
        tol : float
            Redundancy tolerance in meters, default is 1.0 corresponding to 1 meter.
        blt_order : str
            string specifying primary order along the blt axis (see `reorder_blts`)
        blt_minor_order : str
            string specifying minor order along the blt axis (see `reorder_blts`)

        """
        self.conjugate_bls(convention="u>0")
        red_gps, centers, lengths = self.get_redundancies(
            tol=tol, use_antpos=True, conjugate_bls=True
        )

        # Stack redundant groups into one array.
        group_index, bl_array_full = zip(
            *[(i, bl) for i, gp in enumerate(red_gps) for bl in gp]
        )

        # TODO should be an assert that each baseline only ends up in one group

        # Map group index to blt indices in the compressed array.
        bl_array_comp = self.baseline_array
        uniq_bl = np.unique(bl_array_comp)

        group_blti = {}
        Nblts_full = 0
        for i, gp in enumerate(red_gps):
            for bl in gp:
                # First baseline in the group that is also in the compressed
                # baseline array.
                if bl in uniq_bl:
                    group_blti[i] = np.where(bl == bl_array_comp)[0]
                    # add number of blts for this group
                    Nblts_full += group_blti[i].size * len(gp)
                    break

        blt_map = np.zeros(Nblts_full, dtype=int)
        full_baselines = np.zeros(Nblts_full, dtype=int)
        missing = []
        counter = 0
        for bl, gi in zip(bl_array_full, group_index):
            try:
                # this makes the time the fastest axis
                blt_map[counter : counter + group_blti[gi].size] = group_blti[gi]
                full_baselines[counter : counter + group_blti[gi].size] = bl
                counter += group_blti[gi].size
            except KeyError:
                missing.append(bl)
                pass

        if np.any(missing):
            warnings.warn("Missing some redundant groups. Filling in available data.")

        # blt_map is an index array mapping compressed blti indices to uncompressed
        self.data_array = self.data_array[blt_map, ...]
        self.nsample_array = self.nsample_array[blt_map, ...]
        self.flag_array = self.flag_array[blt_map, ...]
        self.time_array = self.time_array[blt_map]
        self.lst_array = self.lst_array[blt_map]
        self.integration_time = self.integration_time[blt_map]
        self.uvw_array = self.uvw_array[blt_map, ...]

        self.baseline_array = full_baselines
        self.ant_1_array, self.ant_2_array = self.baseline_to_antnums(
            self.baseline_array
        )
        self.Nants_data = self._calc_nants_data()
        self.Nbls = np.unique(self.baseline_array).size
        self.Nblts = Nblts_full

        if self.phase_center_app_ra is not None:
            self.phase_center_app_ra = self.phase_center_app_ra[blt_map]
        if self.phase_center_app_dec is not None:
            self.phase_center_app_dec = self.phase_center_app_dec[blt_map]
        if self.multi_object:
            self.object_id_array = self.object_id_array[blt_map]

        self.reorder_blts(order=blt_order, minor_order=blt_minor_order)

        self.check()

    def _convert_from_filetype(self, other):
        """
        Convert from a file-type specific object to a UVData object.

        Used in reads.

        Parameters
        ----------
        other : object that inherits from UVData
            File type specific object to convert to UVData
        """
        for p in other:
            param = getattr(other, p)
            setattr(self, p, param)

    def _convert_to_filetype(self, filetype):
        """
        Convert from a UVData object to a file-type specific object.

        Used in writes.

        Parameters
        ----------
        filetype : str
            Specifies what file type object to convert to. Options are: 'uvfits',
            'fhd', 'miriad', 'uvh5', 'mir'

        Raises
        ------
        ValueError
            if filetype is not a known type
        """
        if filetype == "uvfits":
            from . import uvfits

            other_obj = uvfits.UVFITS()
        elif filetype == "fhd":
            from . import fhd

            other_obj = fhd.FHD()
        elif filetype == "miriad":
            from . import miriad

            other_obj = miriad.Miriad()
        elif filetype == "uvh5":
            from . import uvh5

            other_obj = uvh5.UVH5()
        elif filetype == "mir":
            from . import mir

            other_obj = mir.Mir()
        else:
            raise ValueError("filetype must be uvfits, mir, miriad, fhd, or uvh5")
        for p in self:
            param = getattr(self, p)
            setattr(other_obj, p, param)
        return other_obj

    def read_fhd(
        self,
        filelist,
        use_model=False,
        axis=None,
        read_data=True,
        background_lsts=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
    ):
        """
        Read in data from a list of FHD files.

        Parameters
        ----------
        filelist : array_like of str
            The list/array of FHD save files to read from. Must include at
            least one polarization file, a params file, a layout file and a flag file.
            An obs file is also required if `read_data` is False.
        use_model : bool
            Option to read in the model visibilities rather than the dirty
            visibilities (the default is False, meaning the dirty visibilities
            will be read).
        axis : str
            Axis to concatenate files along. This enables fast concatenation
            along the specified axis without the normal checking that all other
            metadata agrees. This method does not guarantee correct resulting
            objects. Please see the docstring for fast_concat for details.
            Allowed values are: 'blt', 'freq', 'polarization'. Only used if
            multiple data sets are passed.
        read_data : bool
            Read in the visibility, nsample and flag data. If set to False, only
            the metadata will be read in. Setting read_data to False results in
            a metadata only object. If read_data is False, an obs file must be
            included in the filelist.
        background_lsts : bool
            When set to True, the lst_array is calculated in a background thread.
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
            If required files are missing or multiple files for any polarization
            are included in filelist.
            If there is no recognized key for visibility weights in the flags_file.

        """
        from . import fhd

        if isinstance(filelist[0], (list, tuple, np.ndarray)):
            raise ValueError(
                "Reading multiple files from class specific "
                "read functions is no longer supported. "
                "Use the generic `uvdata.read` function instead."
            )

        fhd_obj = fhd.FHD()
        fhd_obj.read_fhd(
            filelist,
            use_model=use_model,
            background_lsts=background_lsts,
            read_data=read_data,
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
            strict_uvw_antpos_check=strict_uvw_antpos_check,
        )
        self._convert_from_filetype(fhd_obj)
        del fhd_obj

    def read_mir(
        self,
        filepath,
        isource=None,
        irec=None,
        isb=None,
        corrchunk=None,
        pseudo_cont=False,
    ):
        """
        Read in data from an SMA MIR file.

        Note that with the exception of filepath, the reset of the parameters are
        used to sub-select a range of data that matches the limitations of the current
        instantiation of pyuvdata  -- namely 1 spectral window, 1 source. These could
        be dropped in the future, as pyuvdata capabilities grow.

        Parameters
        ----------
        filepath : str
             The file path to the MIR folder to read from.
        isource : int
            Source code for MIR dataset
        irec : int
            Receiver code for MIR dataset
        isb : int
            Sideband code for MIR dataset
        corrchunk : int
            Correlator chunk code for MIR dataset
        pseudo_cont : boolean
            Read in only pseudo-continuuum values. Default is false.
        """
        from . import mir

        mir_obj = mir.Mir()
        mir_obj.read_mir(
            filepath,
            isource=isource,
            irec=irec,
            isb=isb,
            corrchunk=corrchunk,
            pseudo_cont=pseudo_cont,
        )
        self._convert_from_filetype(mir_obj)
        del mir_obj

    def read_miriad(
        self,
        filepath,
        axis=None,
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
        skip_extra_sources=False,
    ):
        """
        Read in data from a miriad file.

        Parameters
        ----------
        filepath : str
            The miriad root directory to read from.
        axis : str
            Axis to concatenate files along. This enables fast concatenation
            along the specified axis without the normal checking that all other
            metadata agrees. This method does not guarantee correct resulting
            objects. Please see the docstring for fast_concat for details.
            Allowed values are: 'blt', 'freq', 'polarization'. Only used if
            multiple files are passed.
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
            Read in the visibility and flag data. If set to false,
            only the metadata will be read in. Setting read_data to False
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
        skip_extra_sources : bool
            As multiple sources are not (yet) supported, this will simply allow
            the reader to read just the first source in the dataset, rather than
            throwing an exception. Default is false.

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
        from . import miriad

        if isinstance(filepath, (list, tuple, np.ndarray)):
            raise ValueError(
                "Reading multiple files from class specific "
                "read functions is no longer supported. "
                "Use the generic `uvdata.read` function instead."
            )

        miriad_obj = miriad.Miriad()
        miriad_obj.read_miriad(
            filepath,
            correct_lat_lon=correct_lat_lon,
            read_data=read_data,
            phase_type=phase_type,
            antenna_nums=antenna_nums,
            ant_str=ant_str,
            bls=bls,
            polarizations=polarizations,
            time_range=time_range,
            background_lsts=background_lsts,
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
            strict_uvw_antpos_check=strict_uvw_antpos_check,
            skip_extra_sources=skip_extra_sources,
        )
        self._convert_from_filetype(miriad_obj)
        del miriad_obj

    def read_ms(
        self,
        filepath,
        axis=None,
        data_column="DATA",
        pol_order="AIPS",
        background_lsts=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
    ):
        """
        Read in data from a measurement set.

        Parameters
        ----------
        filepath : str
            The measurement set root directory to read from.
        axis : str
            Axis to concatenate files along. This enables fast concatenation
            along the specified axis without the normal checking that all other
            metadata agrees. This method does not guarantee correct resulting
            objects. Please see the docstring for fast_concat for details.
            Allowed values are: 'blt', 'freq', 'polarization'. Only used if
            multiple files are passed.
        data_column : str
            name of CASA data column to read into data_array. Options are:
            'DATA', 'MODEL', or 'CORRECTED_DATA'
        pol_order : str
            Option to specify polarizations order convention, options are
            'CASA' or 'AIPS'.
        background_lsts : bool
            When set to True, the lst_array is calculated in a background thread.
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
        IOError
            If root file directory doesn't exist.
        ValueError
            If the `data_column` is not set to an allowed value.
            If the data are have multiple subarrays or are multi source or have
            multiple spectral windows.
            If the data have multiple data description ID values.

        """
        if isinstance(filepath, (list, tuple, np.ndarray)):
            raise ValueError(
                "Reading multiple files from class specific "
                "read functions is no longer supported. "
                "Use the generic `uvdata.read` function instead."
            )

        from . import ms

        ms_obj = ms.MS()
        ms_obj.read_ms(
            filepath,
            data_column=data_column,
            pol_order=pol_order,
            background_lsts=background_lsts,
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
            strict_uvw_antpos_check=strict_uvw_antpos_check,
        )
        self._convert_from_filetype(ms_obj)
        del ms_obj

    def read_mwa_corr_fits(
        self,
        filelist,
        axis=None,
        use_cotter_flags=None,
        remove_dig_gains=True,
        remove_coarse_band=True,
        correct_cable_len=False,
        flag_init=True,
        edge_width=80e3,
        start_flag=2.0,
        end_flag=2.0,
        flag_dc_offset=True,
        phase_to_pointing_center=False,
        read_data=True,
        data_array_dtype=np.complex64,
        nsample_array_dtype=np.float32,
        background_lsts=True,
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
        axis : str
            Axis to concatenate files along. This enables fast concatenation
            along the specified axis without the normal checking that all other
            metadata agrees. This method does not guarantee correct resulting
            objects. Please see the docstring for fast_concat for details.
            Allowed values are: 'blt', 'freq', 'polarization'. Only used if
            multiple files are passed.
        use_cotter_flags : bool
            Option to use cotter output mwaf flag files. Defaults to true if cotter
            flag files are submitted.
        remove_dig_gains : bool
            Option to divide out digital gains.
        remove_coarse_band : bool
            Option to divide out coarse band shape.
        correct_cable_len : bool
            Option to apply a cable delay correction.
        flag_init: bool
            Set to True in order to do routine flagging of coarse channel edges,
            start or end integrations, or the center fine channel of each coarse
            channel. See associated keywords.
        edge_width: float
            Only used if flag_init is True. Set to the width to flag on the edge
            of each coarse channel, in hz. Errors if not equal to integer
            multiple of channel_width. Set to 0 for no edge flagging.
        start_flag: float
            Only used if flag_init is True. Set to the number of seconds to flag
            at the beginning of the observation. Set to 0 for no flagging.
            Errors if not an integer multiple of the integration time.
        end_flag: floats
            Only used if flag_init is True. Set to the number of seconds to flag
            at the end of the observation. Set to 0 for no flagging. Errors if
            not an integer multiple of the integration time.
        flag_dc_offset: bool
            Only used if flag_init is True. Set to True to flag the center fine
            channel of each coarse channel. Only used if file_type is
            'mwa_corr_fits'.
        phase_to_pointing_center : bool
            Option to phase to the observation pointing center.
        read_data : bool
            Read in the visibility and flag data. If set to false, only the
            basic header info and metadata read in. Setting read_data to False
            results in a metdata only object.
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
        background_lsts : bool
            When set to True, the lst_array is calculated in a background thread.
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
        from . import mwa_corr_fits

        if isinstance(filelist[0], (list, tuple, np.ndarray)):
            raise ValueError(
                "Reading multiple files from class specific "
                "read functions is no longer supported. "
                "Use the generic `uvdata.read` function instead."
            )

        corr_obj = mwa_corr_fits.MWACorrFITS()
        corr_obj.read_mwa_corr_fits(
            filelist,
            use_cotter_flags=use_cotter_flags,
            remove_dig_gains=remove_dig_gains,
            remove_coarse_band=remove_coarse_band,
            correct_cable_len=correct_cable_len,
            flag_init=flag_init,
            edge_width=edge_width,
            start_flag=start_flag,
            end_flag=end_flag,
            flag_dc_offset=flag_dc_offset,
            phase_to_pointing_center=phase_to_pointing_center,
            read_data=read_data,
            data_array_dtype=data_array_dtype,
            nsample_array_dtype=nsample_array_dtype,
            background_lsts=background_lsts,
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
            strict_uvw_antpos_check=strict_uvw_antpos_check,
        )
        self._convert_from_filetype(corr_obj)
        del corr_obj

    def read_uvfits(
        self,
        filename,
        axis=None,
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
    ):
        """
        Read in header, metadata and data from a single uvfits file.

        Parameters
        ----------
        filename : str
            The uvfits file to read from.
        axis : str
            Axis to concatenate files along. This enables fast concatenation
            along the specified axis without the normal checking that all other
            metadata agrees. This method does not guarantee correct resulting
            objects. Please see the docstring for fast_concat for details.
            Allowed values are: 'blt', 'freq', 'polarization'. Only used if
            multiple files are passed.
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
            passed here should exist in the time_array in the file.
            Cannot be used with `time_range`.
        time_range : array_like of float, optional
            The time range in Julian Date to include when reading data into
            the object, must be length 2. Some of the times in the file should
            fall between the first and last elements.
            Cannot be used with `times`.
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
            Read in the visibility and flag data. If set to false, only the
            basic header info and metadata read in. Setting read_data to False
            results in a metdata only object.
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
            If the data are multi source or have multiple
            spectral windows.
            If the metadata are not internally consistent or missing.

        """
        from . import uvfits

        if isinstance(filename, (list, tuple, np.ndarray)):
            raise ValueError(
                "Reading multiple files from class specific "
                "read functions is no longer supported. "
                "Use the generic `uvdata.read` function instead."
            )

        uvfits_obj = uvfits.UVFITS()
        uvfits_obj.read_uvfits(
            filename,
            antenna_nums=antenna_nums,
            antenna_names=antenna_names,
            ant_str=ant_str,
            bls=bls,
            frequencies=frequencies,
            freq_chans=freq_chans,
            times=times,
            time_range=time_range,
            polarizations=polarizations,
            blt_inds=blt_inds,
            keep_all_metadata=keep_all_metadata,
            read_data=read_data,
            background_lsts=background_lsts,
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
            strict_uvw_antpos_check=strict_uvw_antpos_check,
        )
        self._convert_from_filetype(uvfits_obj)
        del uvfits_obj

    def read_uvh5(
        self,
        filename,
        axis=None,
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
        Read a UVH5 file.

        Parameters
        ----------
        filename : str
             The UVH5 file to read from.
        axis : str
            Axis to concatenate files along. This enables fast concatenation
            along the specified axis without the normal checking that all other
            metadata agrees. This method does not guarantee correct resulting
            objects. Please see the docstring for fast_concat for details.
            Allowed values are: 'blt', 'freq', 'polarization'. Only used if
            multiple files are passed.
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
            passed here should exist in the time_array in the file.
            Cannot be used with `time_range`.
        time_range : array_like of float, optional
            The time range in Julian Date to include when reading data into
            the object, must be length 2. Some of the times in the file should
            fall between the first and last elements.
            Cannot be used with `times`.
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
            Read in the visibility and flag data. If set to false, only the
            basic header info and metadata will be read in. Setting read_data to
            False results in an incompletely defined object (check will not pass).
        data_array_dtype : numpy dtype
            Datatype to store the output data_array as. Must be either
            np.complex64 (single-precision real and imaginary) or np.complex128 (double-
            precision real and imaginary). Only used if the datatype of the visibility
            data on-disk is not 'c8' or 'c16'.
        multidim_index : bool
            [Only for HDF5] If True, attempt to index the HDF5 dataset
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
        from . import uvh5

        if isinstance(filename, (list, tuple, np.ndarray)):
            raise ValueError(
                "Reading multiple files from class specific "
                "read functions is no longer supported. "
                "Use the generic `uvdata.read` function instead."
            )

        uvh5_obj = uvh5.UVH5()
        uvh5_obj.read_uvh5(
            filename,
            antenna_nums=antenna_nums,
            antenna_names=antenna_names,
            ant_str=ant_str,
            bls=bls,
            frequencies=frequencies,
            freq_chans=freq_chans,
            times=times,
            time_range=time_range,
            polarizations=polarizations,
            blt_inds=blt_inds,
            data_array_dtype=data_array_dtype,
            keep_all_metadata=keep_all_metadata,
            read_data=read_data,
            multidim_index=multidim_index,
            background_lsts=background_lsts,
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
            strict_uvw_antpos_check=strict_uvw_antpos_check,
        )
        self._convert_from_filetype(uvh5_obj)
        del uvh5_obj

    def read(
        self,
        filename,
        axis=None,
        file_type=None,
        allow_rephase=True,
        phase_center_radec=None,
        unphase_to_drift=False,
        phase_frame="icrs",
        orig_phase_frame=None,
        phase_use_ant_pos=False,
        antenna_nums=None,
        antenna_names=None,
        ant_str=None,
        bls=None,
        frequencies=None,
        freq_chans=None,
        times=None,
        polarizations=None,
        blt_inds=None,
        time_range=None,
        keep_all_metadata=True,
        read_data=True,
        phase_type=None,
        correct_lat_lon=True,
        use_model=False,
        data_column="DATA",
        pol_order="AIPS",
        data_array_dtype=np.complex128,
        nsample_array_dtype=np.float32,
        use_cotter_flags=None,
        remove_dig_gains=True,
        remove_coarse_band=True,
        correct_cable_len=False,
        flag_init=True,
        edge_width=80e3,
        start_flag=2.0,
        end_flag=2.0,
        flag_dc_offset=True,
        phase_to_pointing_center=False,
        skip_bad_files=False,
        multidim_index=False,
        background_lsts=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
        skip_extra_sources=False,
        isource=None,
        irec=None,
        isb=None,
        corrchunk=None,
        pseudo_cont=False,
    ):
        """
        Read a generic file into a UVData object.

        Parameters
        ----------
        filename : str or array_like of str
            The file(s) or list(s) (or array(s)) of files to read from.
        file_type : str
            One of ['uvfits', 'miriad', 'fhd', 'ms', 'uvh5'] or None.
            If None, the code attempts to guess what the file type is.
            For miriad and ms types, this is based on the standard directory
            structure. For FHD, uvfits and uvh5 files it's based on file
            extensions (FHD: .sav, .txt; uvfits: .uvfits; uvh5: .uvh5).
            Note that if a list of datasets is passed, the file type is
            determined from the first dataset.
        axis : str
            Axis to concatenate files along. This enables fast concatenation
            along the specified axis without the normal checking that all other
            metadata agrees. This method does not guarantee correct resulting
            objects. Please see the docstring for fast_concat for details.
            Allowed values are: 'blt', 'freq', 'polarization'. Only used if
            multiple files are passed.
        allow_rephase :  bool
            Allow rephasing of phased file data so that data from files with
            different phasing can be combined.
        phase_center_radec : array_like of float
            The phase center to phase the files to before adding the objects in
            radians (in the ICRS frame). If set to None and multiple files are
            read with different phase centers, the phase center of the first
            file will be used.
        unphase_to_drift : bool
            Unphase the data from the files before combining them.
        phase_frame : str
            The astropy frame to phase to. Either 'icrs' or 'gcrs'.
            'gcrs' accounts for precession & nutation,
            'icrs' accounts for precession, nutation & abberation.
            Only used if `phase_center_radec` is set.
        orig_phase_frame : str
            The original phase frame of the data (if it is already phased). Used
            for unphasing, only if `unphase_to_drift` or `phase_center_radec`
            are set. Defaults to using the 'phase_center_frame' attribute or
            'icrs' if that attribute is None.
        phase_use_ant_pos : bool
            If True, calculate the phased or unphased uvws directly from the
            antenna positions rather than from the existing uvws.
            Only used if `unphase_to_drift` or `phase_center_radec` are set.
        antenna_nums : array_like of int, optional
            The antennas numbers to include when reading data into the object
            (antenna positions and names for the removed antennas will be retained
            unless `keep_all_metadata` is False). This cannot be provided if
            `antenna_names` is also provided.
        antenna_names : array_like of str, optional
            The antennas names to include when reading data into the object
            (antenna positions and names for the removed antennas will be retained
            unless `keep_all_metadata` is False). This cannot be provided if
            `antenna_nums` is also provided.
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
            `antenna_names`, `bls` args or the `polarizations` parameters,
            if it is a ValueError will be raised.
        frequencies : array_like of float, optional
            The frequencies to include when reading data into the object, each
            value passed here should exist in the freq_array.
        freq_chans : array_like of int, optional
            The frequency channel numbers to include when reading data into the
            object. Ignored if read_data is False.
        times : array_like of float, optional
            The times to include when reading data into the object, each value
            passed here should exist in the time_array in the file.
            Cannot be used with `time_range`.
        time_range : array_like of float, optional
            The time range in Julian Date to include when reading data into
            the object, must be length 2. Some of the times in the file should
            fall between the first and last elements.
            Cannot be used with `times`.
        polarizations : array_like of int, optional
            The polarizations numbers to include when reading data into the
            object, each value passed here should exist in the polarization_array.
        blt_inds : array_like of int, optional
            The baseline-time indices to include when reading data into the
            object. This is not commonly used.
        keep_all_metadata : bool
            Option to keep all the metadata associated with antennas, even those
            that do not have data associated with them after the select option.
        read_data : bool
            Read in the data. Only used if file_type is 'uvfits',
            'miriad' or 'uvh5'. If set to False, only the metadata will be
            read in. Setting read_data to False results in a metdata only
            object.
        phase_type : str, optional
            Option to specify the phasing status of the data. Only used if
            file_type is 'miriad'. Options are 'drift', 'phased' or None.
            'drift' means the data are zenith drift data, 'phased' means the
            data are phased to a single RA/Dec. Default is None
            meaning it will be guessed at based on the file contents.
        correct_lat_lon : bool
            Option to update the latitude and longitude from the known_telescopes
            list if the altitude is missing. Only used if file_type is 'miriad'.
        use_model : bool
            Option to read in the model visibilities rather than the dirty
            visibilities (the default is False, meaning the dirty visibilities
            will be read). Only used if file_type is 'fhd'.
        data_column : str
            name of CASA data column to read into data_array. Options are:
            'DATA', 'MODEL', or 'CORRECTED_DATA'. Only used if file_type is 'ms'.
        pol_order : str
            Option to specify polarizations order convention, options are
            'CASA' or 'AIPS'. Only used if file_type is 'ms'.
        data_array_dtype : numpy dtype
            Datatype to store the output data_array as. Must be either
            np.complex64 (single-precision real and imaginary) or np.complex128 (double-
            precision real and imaginary). Only used if the datatype of the visibility
            data on-disk is not 'c8' or 'c16'. Only used if file_type is 'uvh5' or
            'mwa_corr_fits'.
        nsample_array_dtype : numpy dtype
            Datatype to store the output nsample_array as. Must be either
            np.float64 (double-precision), np.float32 (single-precision), or
            np.float16 (half-precision). Half-precision is only recommended for
            cases where no sampling or averaging of baselines will occur,
            because round-off errors can be quite large (~1e-3). Only used if
            file_type is 'mwa_corr_fits'.
        use_cotter_flags : bool
            Only used if file_type is 'mwa_corr_fits'. Option to use cotter output
            mwaf flag files. Defaults to true if cotter flag files are submitted.
        remove_dig_gains : bool
            Only used if file_type is 'mwa_corr_fits'. Option to divide out digital
            gains.
        remove_coarse_band : bool
            Only used if file_type is 'mwa_corr_fits'. Option to divide out coarse
            band shape.
        correct_cable_len : bool
            Flag to apply cable length correction. Only used if file_type is
            'mwa_corr_fits'.
        flag_init: bool
            Only used if file_type is 'mwa_corr_fits'. Set to True in order to
            do routine flagging of coarse channel edges, start or end
            integrations, or the center fine channel of each coarse
            channel. See associated keywords.
        edge_width: float
            Only used if file_type is 'mwa_corr_fits' and flag_init is True. Set
            to the width to flag on the edge of each coarse channel, in hz.
            Errors if not equal to integer multiple of channel_width. Set to 0
            for no edge flagging.
        start_flag: float
            Only used if file_type is 'mwa_corr_fits' and flag_init is True. Set
            to the number of seconds to flag at the beginning of the observation.
            Set to 0 for no flagging. Errors if not an integer multiple of the
            integration time.
        end_flag: floats
            Only used if file_type is 'mwa_corr_fits' and flag_init is True. Set
            to the number of seconds to flag at the end of the observation. Set
            to 0 for no flagging. Errors if not an integer multiple of the
            integration time.
        flag_dc_offset: bool
            Only used if file_type is 'mwa_corr_fits' and flag_init is True. Set
            to True to flag the center fine channel of each coarse channel. Only
            used if file_type is 'mwa_corr_fits'.
        phase_to_pointing_center : bool
            Flag to phase to the pointing center. Only used if file_type is
            'mwa_corr_fits'. Cannot be set if phase_center_radec is not None.
        skip_bad_files : bool
            Option when reading multiple files to catch read errors such that
            the read continues even if one or more files are corrupted. Files
            that produce errors will be printed. Default is False (files will
            not be skipped).
        multidim_index : bool
            [Only for HDF5] If True, attempt to index the HDF5 dataset
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
        skip_extra_sources : bool
            As multiple sources are not (yet) supported, this will simply allow
            the reader to read just the first source in the dataset, rather than
            throwing an exception. Default is false.
        isource : int
            Source code for MIR dataset
        irec : int
            Receiver code for MIR dataset
        isb : int
            Sideband code for MIR dataset
        corrchunk : int
            Correlator chunk code for MIR dataset
        pseudo_cont : boolean
            Read in only pseudo-continuuum values in MIR dataset. Default is false.

        Raises
        ------
        ValueError
            If the file_type is not set and cannot be determined from the file name.
            If incompatible select keywords are set (e.g. `ant_str` with other
            antenna selectors, `times` and `time_range`) or select keywords
            exclude all data or if keywords are set to the wrong type.
            If the data are multi source or have multiple
            spectral windows.
            If phase_center_radec is not None and is not length 2.

        """
        if isinstance(filename, (list, tuple, np.ndarray)):
            # this is either a list of separate files to read or a list of
            # FHD files or MWA correlator FITS files
            if isinstance(filename[0], (list, tuple, np.ndarray)):
                if file_type is None:
                    # this must be a list of lists of FHD or MWA correlator FITS
                    basename, extension = os.path.splitext(filename[0][0])
                    if extension == ".sav" or extension == ".txt":
                        file_type = "fhd"
                    elif (
                        extension == ".fits"
                        or extension == ".metafits"
                        or extension == ".mwaf"
                    ):
                        file_type = "mwa_corr_fits"
                multi = True
            else:
                if file_type is None:
                    basename, extension = os.path.splitext(filename[0])
                    if extension == ".sav" or extension == ".txt":
                        file_type = "fhd"
                    elif (
                        extension == ".fits"
                        or extension == ".metafits"
                        or extension == ".mwaf"
                    ):
                        file_type = "mwa_corr_fits"

                if file_type == "fhd" or file_type == "mwa_corr_fits":
                    multi = False
                else:
                    multi = True
        else:
            multi = False

        if file_type is None:
            if multi:
                file_test = filename[0]
            else:
                file_test = filename

            if os.path.isdir(file_test):
                # it's a directory, so it's either miriad, mir, or ms file type
                if os.path.exists(os.path.join(file_test, "vartable")):
                    # It's miriad.
                    file_type = "miriad"
                elif os.path.exists(os.path.join(file_test, "OBSERVATION")):
                    # It's a measurement set.
                    file_type = "ms"
                elif os.path.exists(os.path.join(file_test, "sch_read")):
                    # It's Submillimeter Array mir format.
                    file_type = "mir"

            else:
                basename, extension = os.path.splitext(file_test)
                if extension == ".uvfits":
                    file_type = "uvfits"
                elif extension == ".uvh5":
                    file_type = "uvh5"

        if file_type is None:
            raise ValueError(
                "File type could not be determined, use the "
                "file_type keyword to specify the type."
            )

        if time_range is not None:
            if times is not None:
                raise ValueError("Only one of times and time_range can be provided.")

        if antenna_names is not None and antenna_nums is not None:
            raise ValueError(
                "Only one of antenna_nums and antenna_names can " "be provided."
            )

        if multi:

            file_num = 0
            file_warnings = ""
            unread = True
            f = filename[file_num]
            while unread and file_num < len(filename):
                try:
                    self.read(
                        filename[file_num],
                        file_type=file_type,
                        antenna_nums=antenna_nums,
                        antenna_names=antenna_names,
                        ant_str=ant_str,
                        bls=bls,
                        frequencies=frequencies,
                        freq_chans=freq_chans,
                        times=times,
                        polarizations=polarizations,
                        blt_inds=blt_inds,
                        time_range=time_range,
                        keep_all_metadata=keep_all_metadata,
                        read_data=read_data,
                        phase_type=phase_type,
                        correct_lat_lon=correct_lat_lon,
                        use_model=use_model,
                        data_column=data_column,
                        pol_order=pol_order,
                        data_array_dtype=data_array_dtype,
                        nsample_array_dtype=nsample_array_dtype,
                        skip_bad_files=skip_bad_files,
                        background_lsts=background_lsts,
                        run_check=run_check,
                        check_extra=check_extra,
                        run_check_acceptability=run_check_acceptability,
                        strict_uvw_antpos_check=strict_uvw_antpos_check,
                    )
                    unread = False
                except KeyError as err:
                    file_warnings = (
                        file_warnings + f"Failed to read {f} due to KeyError: {err}\n"
                    )
                    file_num += 1
                    if skip_bad_files is False:
                        raise
                except ValueError as err:
                    file_warnings = (
                        file_warnings + f"Failed to read {f} due to ValueError: {err}\n"
                    )
                    file_num += 1
                    if skip_bad_files is False:
                        raise
                except OSError as err:  # pragma: nocover
                    file_warnings = (
                        file_warnings + f"Failed to read {f} due to OSError: {err}\n"
                    )
                    file_num += 1
                    if skip_bad_files is False:
                        raise
            if (
                allow_rephase
                and phase_center_radec is None
                and not unphase_to_drift
                and self.phase_type == "phased"
            ):
                # set the phase center to be the phase center of the first file
                phase_center_radec = [self.phase_center_ra, self.phase_center_dec]

            uv_list = []
            if len(filename) > file_num + 1:
                for f in filename[file_num + 1 :]:
                    uv2 = UVData()
                    try:
                        uv2.read(
                            f,
                            file_type=file_type,
                            phase_center_radec=phase_center_radec,
                            antenna_nums=antenna_nums,
                            antenna_names=antenna_names,
                            ant_str=ant_str,
                            bls=bls,
                            frequencies=frequencies,
                            freq_chans=freq_chans,
                            times=times,
                            polarizations=polarizations,
                            blt_inds=blt_inds,
                            time_range=time_range,
                            keep_all_metadata=keep_all_metadata,
                            read_data=read_data,
                            phase_type=phase_type,
                            correct_lat_lon=correct_lat_lon,
                            use_model=use_model,
                            data_column=data_column,
                            pol_order=pol_order,
                            data_array_dtype=data_array_dtype,
                            nsample_array_dtype=nsample_array_dtype,
                            skip_bad_files=skip_bad_files,
                            background_lsts=background_lsts,
                            run_check=run_check,
                            check_extra=check_extra,
                            run_check_acceptability=run_check_acceptability,
                            strict_uvw_antpos_check=strict_uvw_antpos_check,
                        )
                        uv_list.append(uv2)
                    except KeyError as err:
                        file_warnings = (
                            file_warnings
                            + f"Failed to read {f} due to KeyError: {err}\n"
                        )
                        if skip_bad_files:
                            continue
                        else:
                            raise
                    except ValueError as err:
                        file_warnings = (
                            file_warnings
                            + f"Failed to read {f} due to ValueError: {err}\n"
                        )
                        if skip_bad_files:
                            continue
                        else:
                            raise
                    except OSError as err:  # pragma: nocover
                        file_warnings = (
                            file_warnings
                            + f"Failed to read {f} due to OSError: {err}\n"
                        )
                        if skip_bad_files:
                            continue
                        else:
                            raise
            if unread is True:
                warnings.warn(
                    "########################################################\n"
                    "ALL FILES FAILED ON READ - NO READABLE FILES IN FILENAME\n"
                    "########################################################"
                )
            elif len(file_warnings) > 0:
                warnings.warn(file_warnings)

            # Concatenate once at end
            if axis is not None:
                # Rewrote fast_concat to operate on lists
                self.fast_concat(
                    uv_list,
                    axis,
                    phase_center_radec=phase_center_radec,
                    unphase_to_drift=unphase_to_drift,
                    phase_frame=phase_frame,
                    orig_phase_frame=orig_phase_frame,
                    use_ant_pos=phase_use_ant_pos,
                    run_check=run_check,
                    check_extra=check_extra,
                    run_check_acceptability=run_check_acceptability,
                    inplace=True,
                )
            else:
                # Too much work to rewrite __add__ to operate on lists
                # of files, so instead doing a binary tree merge
                uv_list = [self] + uv_list
                while len(uv_list) > 1:
                    for uv1, uv2 in zip(uv_list[0::2], uv_list[1::2]):
                        uv1.__iadd__(
                            uv2,
                            phase_center_radec=phase_center_radec,
                            unphase_to_drift=unphase_to_drift,
                            phase_frame=phase_frame,
                            orig_phase_frame=orig_phase_frame,
                            use_ant_pos=phase_use_ant_pos,
                            run_check=run_check,
                            check_extra=check_extra,
                            run_check_acceptability=run_check_acceptability,
                        )
                    uv_list = uv_list[0::2]
                # Because self was at the beginning of the list,
                # everything is merged into it at the end of this loop

        else:
            if file_type in ["fhd", "ms", "mwa_corr_fits"]:
                if (
                    antenna_nums is not None
                    or antenna_names is not None
                    or ant_str is not None
                    or bls is not None
                    or frequencies is not None
                    or freq_chans is not None
                    or times is not None
                    or time_range is not None
                    or polarizations is not None
                    or blt_inds is not None
                ):
                    select = True
                    warnings.warn(
                        "Warning: select on read keyword set, but "
                        'file_type is "{ftype}" which does not support select '
                        "on read. Entire file will be read and then select "
                        "will be performed".format(ftype=file_type)
                    )
                    # these file types do not have select on read, so set all
                    # select parameters
                    select_antenna_nums = antenna_nums
                    select_antenna_names = antenna_names
                    select_ant_str = ant_str
                    select_bls = bls
                    select_frequencies = frequencies
                    select_freq_chans = freq_chans
                    select_times = times
                    select_time_range = time_range
                    select_polarizations = polarizations
                    select_blt_inds = blt_inds
                else:
                    select = False
            elif file_type in ["uvfits", "uvh5"]:
                select = False
            elif file_type in ["miriad"]:
                if (
                    antenna_names is not None
                    or frequencies is not None
                    or freq_chans is not None
                    or times is not None
                    or blt_inds is not None
                ):

                    if blt_inds is not None:
                        if (
                            antenna_nums is not None
                            or ant_str is not None
                            or bls is not None
                            or time_range is not None
                        ):
                            warnings.warn(
                                "Warning: blt_inds is set along with select "
                                "on read keywords that are supported by "
                                "read_miriad and may downselect blts. "
                                "This may result in incorrect results "
                                "because the select on read will happen "
                                "before the blt_inds selection so the indices "
                                "may not match the expected locations."
                            )
                    else:
                        warnings.warn(
                            "Warning: a select on read keyword is set that is "
                            "not supported by read_miriad. This select will "
                            "be done after reading the file."
                        )
                    select = True
                    # these are all done by partial read, so set to None
                    select_antenna_nums = None
                    select_ant_str = None
                    select_bls = None
                    select_time_range = None
                    select_polarizations = None

                    # these aren't supported by partial read, so do it in select
                    select_antenna_names = antenna_names
                    select_frequencies = frequencies
                    select_freq_chans = freq_chans
                    select_times = times
                    select_blt_inds = blt_inds
                else:
                    select = False

            # reading a single "file". Call the appropriate file-type read
            if file_type == "uvfits":
                self.read_uvfits(
                    filename,
                    antenna_nums=antenna_nums,
                    antenna_names=antenna_names,
                    ant_str=ant_str,
                    bls=bls,
                    frequencies=frequencies,
                    freq_chans=freq_chans,
                    times=times,
                    time_range=time_range,
                    polarizations=polarizations,
                    blt_inds=blt_inds,
                    read_data=read_data,
                    keep_all_metadata=keep_all_metadata,
                    background_lsts=background_lsts,
                    run_check=run_check,
                    check_extra=check_extra,
                    run_check_acceptability=run_check_acceptability,
                    strict_uvw_antpos_check=strict_uvw_antpos_check,
                )

            elif file_type == "mir":
                self.read_mir(
                    filename,
                    isource=isource,
                    irec=irec,
                    isb=isb,
                    corrchunk=corrchunk,
                    pseudo_cont=pseudo_cont,
                )
                select = False

            elif file_type == "miriad":
                self.read_miriad(
                    filename,
                    antenna_nums=antenna_nums,
                    ant_str=ant_str,
                    bls=bls,
                    polarizations=polarizations,
                    time_range=time_range,
                    read_data=read_data,
                    phase_type=phase_type,
                    correct_lat_lon=correct_lat_lon,
                    background_lsts=background_lsts,
                    run_check=run_check,
                    check_extra=check_extra,
                    run_check_acceptability=run_check_acceptability,
                    strict_uvw_antpos_check=strict_uvw_antpos_check,
                    skip_extra_sources=skip_extra_sources,
                )

            elif file_type == "mwa_corr_fits":
                self.read_mwa_corr_fits(
                    filename,
                    use_cotter_flags=use_cotter_flags,
                    remove_dig_gains=remove_dig_gains,
                    remove_coarse_band=remove_coarse_band,
                    correct_cable_len=correct_cable_len,
                    flag_init=flag_init,
                    edge_width=edge_width,
                    start_flag=start_flag,
                    end_flag=end_flag,
                    flag_dc_offset=True,
                    phase_to_pointing_center=phase_to_pointing_center,
                    read_data=read_data,
                    data_array_dtype=data_array_dtype,
                    nsample_array_dtype=nsample_array_dtype,
                    background_lsts=background_lsts,
                    run_check=run_check,
                    check_extra=check_extra,
                    run_check_acceptability=run_check_acceptability,
                    strict_uvw_antpos_check=strict_uvw_antpos_check,
                )

            elif file_type == "fhd":
                self.read_fhd(
                    filename,
                    use_model=use_model,
                    background_lsts=background_lsts,
                    read_data=read_data,
                    run_check=run_check,
                    check_extra=check_extra,
                    run_check_acceptability=run_check_acceptability,
                    strict_uvw_antpos_check=strict_uvw_antpos_check,
                )

            elif file_type == "ms":
                self.read_ms(
                    filename,
                    data_column=data_column,
                    pol_order=pol_order,
                    background_lsts=background_lsts,
                    run_check=run_check,
                    check_extra=check_extra,
                    run_check_acceptability=run_check_acceptability,
                    strict_uvw_antpos_check=strict_uvw_antpos_check,
                )

            elif file_type == "uvh5":
                self.read_uvh5(
                    filename,
                    antenna_nums=antenna_nums,
                    antenna_names=antenna_names,
                    ant_str=ant_str,
                    bls=bls,
                    frequencies=frequencies,
                    freq_chans=freq_chans,
                    times=times,
                    time_range=time_range,
                    polarizations=polarizations,
                    blt_inds=blt_inds,
                    read_data=read_data,
                    data_array_dtype=data_array_dtype,
                    keep_all_metadata=keep_all_metadata,
                    multidim_index=multidim_index,
                    background_lsts=background_lsts,
                    run_check=run_check,
                    check_extra=check_extra,
                    run_check_acceptability=run_check_acceptability,
                    strict_uvw_antpos_check=strict_uvw_antpos_check,
                )
                select = False

            if select:
                self.select(
                    antenna_nums=select_antenna_nums,
                    antenna_names=select_antenna_names,
                    ant_str=select_ant_str,
                    bls=select_bls,
                    frequencies=select_frequencies,
                    freq_chans=select_freq_chans,
                    times=select_times,
                    time_range=select_time_range,
                    polarizations=select_polarizations,
                    blt_inds=select_blt_inds,
                    keep_all_metadata=keep_all_metadata,
                    run_check=run_check,
                    check_extra=check_extra,
                    run_check_acceptability=run_check_acceptability,
                    strict_uvw_antpos_check=strict_uvw_antpos_check,
                )

            if unphase_to_drift:
                if self.phase_type != "drift":
                    warnings.warn("Unphasing this UVData object to drift")
                    self.unphase_to_drift(
                        phase_frame=orig_phase_frame, use_ant_pos=phase_use_ant_pos
                    )

            if phase_center_radec is not None:
                if np.array(phase_center_radec).size != 2:
                    raise ValueError("phase_center_radec should have length 2.")

                # If this object is not phased or is not phased close to
                # phase_center_radec, (re)phase it.
                # Close is defined using the phase_center_ra/dec tolerances.
                if self.phase_type == "drift" or (
                    not np.isclose(
                        self.phase_center_ra,
                        phase_center_radec[0],
                        rtol=self._phase_center_ra.tols[0],
                        atol=self._phase_center_ra.tols[1],
                    )
                    or not np.isclose(
                        self.phase_center_dec,
                        phase_center_radec[1],
                        rtol=self._phase_center_dec.tols[0],
                        atol=self._phase_center_dec.tols[1],
                    )
                ):
                    warnings.warn("Phasing this UVData object to phase_center_radec")
                    self.phase(
                        phase_center_radec[0],
                        phase_center_radec[1],
                        phase_frame=phase_frame,
                        orig_phase_frame=orig_phase_frame,
                        use_ant_pos=phase_use_ant_pos,
                        allow_rephase=True,
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
        miriad_obj = self._convert_to_filetype("miriad")
        miriad_obj.write_miriad(
            filepath,
            clobber=clobber,
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
            strict_uvw_antpos_check=strict_uvw_antpos_check,
            no_antnums=no_antnums,
        )
        del miriad_obj

    def write_mir(
        self, filepath,
    ):
        """
        Write the data to a mir file.

        Parameters
        ----------
        filename : str
            The mir root directory to write to.

        Raises
        ------
        NotImplementedError
            Method is not fully implemented yet, and thus will raise an error

        """
        mir_obj = self._convert_to_filetype("mir")
        mir_obj.write_mir(filepath,)
        del mir_obj

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
        force_phase:  : bool
            Option to automatically phase drift scan data to zenith of the first
            timestamp.
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
        uvfits_obj = self._convert_to_filetype("uvfits")
        uvfits_obj.write_uvfits(
            filename,
            spoof_nonessential=spoof_nonessential,
            write_lst=write_lst,
            force_phase=force_phase,
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
            strict_uvw_antpos_check=strict_uvw_antpos_check,
        )
        del uvfits_obj

    def write_uvh5(
        self,
        filename,
        clobber=False,
        chunks=True,
        data_compression=None,
        flags_compression="lzf",
        nsample_compression="lzf",
        data_write_dtype=None,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
    ):
        """
        Write a completely in-memory UVData object to a UVH5 file.

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
            HDF5 filter to apply when writing the data_array. Default is
            None meaning no filter or compression. Dataset must be chunked.
        flags_compression : str
            HDF5 filter to apply when writing the flags_array. Default is "lzf"
            for the LZF filter. Dataset must be chunked.
        nsample_compression : str
            HDF5 filter to apply when writing the nsample_array. Default is "lzf"
            for the LZF filter. Dataset must be chunked.
        data_write_dtype : numpy dtype
            datatype of output visibility data. If 'None', then the same datatype
            as data_array will be used. Otherwise, a numpy dtype object must be
            specified with an 'r' field and an 'i' field for real and imaginary
            parts, respectively. See uvh5.py for an example of defining such a datatype.
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

        """
        uvh5_obj = self._convert_to_filetype("uvh5")
        uvh5_obj.write_uvh5(
            filename,
            clobber=clobber,
            chunks=chunks,
            data_compression=data_compression,
            flags_compression=flags_compression,
            nsample_compression=nsample_compression,
            data_write_dtype=data_write_dtype,
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
            strict_uvw_antpos_check=strict_uvw_antpos_check,
        )
        del uvh5_obj

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
        Initialize a UVH5 file on disk with the header metadata and empty data arrays.

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
            HDF5 filter to apply when writing the data_array. Default is
            None meaning no filter or compression. Dataset must be chunked.
        flags_compression : str
            HDF5 filter to apply when writing the flags_array. Default is "lzf"
            for the LZF filter. Dataset must be chunked.
        nsample_compression : str
            HDF5 filter to apply when writing the nsample_array. Default is "lzf"
            for the LZF filter. Dataset must be chunked.
        data_write_dtype : numpy dtype
            datatype of output visibility data. If 'None', then the same datatype
            as data_array will be used. Otherwise, a numpy dtype object must be
            specified with an 'r' field and an 'i' field for real and imaginary
            parts, respectively. See uvh5.py for an example of defining such a datatype.

        Notes
        -----
        When partially writing out data, this function should be called first
        to initialize the file on disk. The data is then actually written by
        calling the write_uvh5_part method, with the same filename as the one
        specified in this function. See the tutorial for a worked example.

        """
        uvh5_obj = self._convert_to_filetype("uvh5")
        uvh5_obj.initialize_uvh5_file(
            filename,
            clobber=clobber,
            chunks=chunks,
            data_compression=data_compression,
            flags_compression=flags_compression,
            nsample_compression=nsample_compression,
            data_write_dtype=data_write_dtype,
        )
        del uvh5_obj

    def write_uvh5_part(
        self,
        filename,
        data_array,
        flags_array,
        nsample_array,
        check_header=True,
        antenna_nums=None,
        antenna_names=None,
        ant_str=None,
        bls=None,
        frequencies=None,
        freq_chans=None,
        times=None,
        polarizations=None,
        blt_inds=None,
        add_to_history=None,
        run_check_acceptability=True,
    ):
        """
        Write data to a UVH5 file that has already been initialized.

        Parameters
        ----------
        filename : str
            The UVH5 file to write to. It must already exist, and is assumed to
            have been initialized with initialize_uvh5_file.
        data_array : ndarray
            The data to write to disk. A check is done to ensure that the
            dimensions of the data passed in conform to the ones specified by
            the "selection" arguments.
        flags_array : ndarray
            The flags array to write to disk. A check is done to ensure that the
            dimensions of the data passed in conform to the ones specified by
            the "selection" arguments.
        nsample_array : ndarray
            The nsample array to write to disk. A check is done to ensure that the
            dimensions of the data passed in conform to the ones specified by
            the "selection" arguments.
        check_header : bool
            Option to check that the metadata present in the header on disk
            matches that in the object.
        antenna_nums : array_like of int, optional
            The antennas numbers to include when writing data into the file
            (antenna positions and names for the removed antennas will be retained).
            This cannot be provided if `antenna_names` is also provided.
        antenna_names : array_like of str, optional
            The antennas names to include when writing data into the file
            (antenna positions and names for the removed antennas will be retained).
            This cannot be provided if `antenna_nums` is also provided.
        bls : list of tuple, optional
            A list of antenna number tuples (e.g. [(0, 1), (3, 2)]) or a list of
            baseline 3-tuples (e.g. [(0, 1, 'xx'), (2, 3, 'yy')]) specifying baselines
            to include when writing data into the file. For length-2 tuples,
            the ordering of the numbers within the tuple does not matter. For
            length-3 tuples, the polarization string is in the order of the two
            antennas. If length-3 tuples are provided, `polarizations` must be
            None.
        ant_str : str, optional
            A string containing information about what antenna numbers
            and polarizations to include writing data into the file.
            Can be 'auto', 'cross', 'all', or combinations of antenna numbers
            and polarizations (e.g. '1', '1_2', '1x_2y').  See tutorial for more
            examples of valid strings and the behavior of different forms for ant_str.
            If '1x_2y,2y_3y' is passed, both polarizations 'xy' and 'yy' will
            be kept for both baselines (1, 2) and (2, 3) to return a valid
            pyuvdata object.
            An ant_str cannot be passed in addition to any of `antenna_nums`,
            `antenna_names`, `bls` args or the `polarizations` parameters,
            if it is a ValueError will be raised.
        frequencies : array_like of float, optional
            The frequencies to include when writing data into the file, each
            value passed here should exist in the freq_array.
        freq_chans : array_like of int, optional
            The frequency channel numbers to include writing data into the file.
        times : array_like of float, optional
            The times to include when writing data into the file, each value
            passed here should exist in the time_array.
        polarizations : array_like of int, optional
            The polarizations numbers to include when writing data into the file,
            each value passed here should exist in the polarization_array.
        blt_inds : array_like of int, optional
            The baseline-time indices to include when writing data into the file.
            This is not commonly used.
        add_to_history : str
            String to append to history before write out. Default is no appending.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters before
            writing the file (the default is True, meaning the acceptable
            range check will be done).
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.

        """
        uvh5_obj = self._convert_to_filetype("uvh5")
        uvh5_obj.write_uvh5_part(
            filename,
            data_array,
            flags_array,
            nsample_array,
            check_header=check_header,
            antenna_nums=antenna_nums,
            antenna_names=antenna_names,
            bls=bls,
            ant_str=ant_str,
            frequencies=frequencies,
            freq_chans=freq_chans,
            times=times,
            polarizations=polarizations,
            blt_inds=blt_inds,
            add_to_history=add_to_history,
            run_check_acceptability=run_check_acceptability,
        )
        del uvh5_obj
