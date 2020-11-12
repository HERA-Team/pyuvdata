# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Class for reading FHD save files."""
import numpy as np
import warnings
from scipy.io.idl import readsav
from astropy import constants as const

from .uvdata import UVData
from .. import utils as uvutils
from .. import telescopes as uvtel

__all__ = ["get_fhd_history", "get_fhd_layout_info", "FHD"]


def get_fhd_history(settings_file, return_user=False):
    """
    Small function to get the important history from an FHD settings text file.

    Includes information about the command line call, the user, machine name and date

    Parameters
    ----------
    settings_file : str
        FHD settings file name
    return_user : bool
        optionally return the username who ran FHD

    Returns
    -------
    history : str
        string of history extracted from the settings file
    user : str
        Only returned if return_user is True

    """
    with open(settings_file, "r") as f:
        settings_lines = f.readlines()
    main_loc = None
    command_loc = None
    obs_loc = None
    user_line = None
    for ind, line in enumerate(settings_lines):
        if line.startswith("##MAIN"):
            main_loc = ind
        if line.startswith("##COMMAND_LINE"):
            command_loc = ind
        if line.startswith("##OBS"):
            obs_loc = ind
        if line.startswith("User"):
            user_line = ind
        if (
            main_loc is not None
            and command_loc is not None
            and obs_loc is not None
            and user_line is not None
        ):
            break

    main_lines = settings_lines[main_loc + 1 : command_loc]
    command_lines = settings_lines[command_loc + 1 : obs_loc]
    history_lines = ["FHD history\n"] + main_lines + command_lines
    for ind, line in enumerate(history_lines):
        history_lines[ind] = line.rstrip().replace("\t", " ")
    history = "\n".join(history_lines)
    user = settings_lines[user_line].split()[1]

    if return_user:
        return history, user
    else:
        return history


def _xyz_close(xyz1, xyz2, loc_tols):
    return np.allclose(xyz1, xyz2, rtol=loc_tols[0], atol=loc_tols[1])


def _latlonalt_close(latlonalt1, latlonalt2, radian_tols, loc_tols):
    latlon_close = np.allclose(
        np.array(latlonalt1[0:2]),
        np.array(latlonalt2[0:2]),
        rtol=radian_tols[0],
        atol=radian_tols[1],
    )
    alt_close = np.isclose(
        latlonalt1[2], latlonalt2[2], rtol=loc_tols[0], atol=loc_tols[1]
    )
    if latlon_close and alt_close:
        return True
    else:
        return False


def get_fhd_layout_info(
    layout_file,
    telescope_name,
    latitude,
    longitude,
    altitude,
    radian_tols,
    loc_tols,
    obs_tile_names,
    run_check_acceptability=True,
):
    """
    Get the telescope and antenna positions from an FHD layout file.

    Parameters
    ----------
    layout_file : str
        FHD layout file name
    telescope_name : str
        Telescope name
    latitude : float
        telescope latitude in radians
    longitude : float
        telescope longitude in radians
    altitude : float
        telescope altitude in meters
    loc_tols : float
        telescope_location tolerance in meters.
    radian_tols : float
        lat/lon tolerance in radians.
    obs_tile_names : array-like of str
        Tile names from the bl_info structure inside the obs structure.
        Only used if telescope_name is "mwa".
    run_check_acceptability : bool
        Option to check acceptable range of the telescope locations.

    Returns
    -------
    dict
        A dictionary of parameters from the layout file to assign to the object. The
        keys are:

        * telescope_xyz : Telescope location in ECEF, shape (3,) (float)
        * Nants_telescope : Number of antennas in the telescope (int)
        * antenna_postions : Antenna positions in relative ECEF,
            shape (Nants_telescope, 3) (float)
        * antenna_names : Antenna names, length Nants_telescope (list of str)
        * antenna_numbers : Antenna numbers, shape (Nants_telescope,) (array of int)
        * gst0 : Greenwich sidereal time at midnight on reference date. (float)
        * earth_omega : Earth's rotation rate in degrees per day. (float)
        * dut1 : DUT1 (google it) AIPS 117 calls it UT1UTC. (float)
        * timesys : Time system (should only ever be UTC). (str)
        * diameters : Antenna diameters in meters. shape (Nants_telescope,) (float)
        * extra_keywords : Dictionary of extra keywords to preserve on the object.

    """
    layout_dict = readsav(layout_file, python_dict=True)
    layout = layout_dict["layout"]

    layout_fields = [name.lower() for name in layout.dtype.names]
    # Try to get the telescope location from the layout file &
    # compare it to the position from the obs structure.
    arr_center = layout["array_center"][0]
    layout_fields.remove("array_center")

    xyz_telescope_frame = layout["coordinate_frame"][0].decode("utf8").lower()
    layout_fields.remove("coordinate_frame")

    if xyz_telescope_frame == "itrf":
        # compare to lat/lon/alt
        location_latlonalt = uvutils.XYZ_from_LatLonAlt(latitude, longitude, altitude)
        latlonalt_arr_center = uvutils.LatLonAlt_from_XYZ(
            arr_center, check_acceptability=run_check_acceptability
        )

        # check both lat/lon/alt and xyz because of subtle differences
        # in tolerances
        if _xyz_close(location_latlonalt, arr_center, loc_tols) or _latlonalt_close(
            (latitude, longitude, altitude),
            latlonalt_arr_center,
            radian_tols,
            loc_tols,
        ):
            telescope_location = arr_center
        else:
            # values do not agree with each other to within the tolerances.
            # this is a known issue with FHD runs on cotter uvfits
            # files for the MWA
            # compare with the known_telescopes values
            telescope_obj = uvtel.get_telescope(telescope_name)
            # start warning message
            message = (
                "Telescope location derived from obs lat/lon/alt "
                "values does not match the location in the layout file."
            )

            if telescope_obj is not False:
                message += " Using the value from known_telescopes."
                telescope_location = telescope_obj.telescope_location
            else:
                message += (
                    " Telescope is not in known_telescopes. "
                    "Defaulting to using the obs derived values."
                )
                telescope_location = location_latlonalt
            # issue warning
            warnings.warn(message)
    else:
        telescope_location = uvutils.XYZ_from_LatLonAlt(latitude, longitude, altitude)

    # The FHD positions derive directly from uvfits, so they are in the rotated
    # ECEF frame and must be converted to ECEF
    rot_ecef_positions = layout["antenna_coords"][0]
    layout_fields.remove("antenna_coords")
    # use the longitude from the layout file because that's how the antenna
    # positions were calculated
    latitude, longitude, altitude = uvutils.LatLonAlt_from_XYZ(
        arr_center, check_acceptability=run_check_acceptability,
    )
    antenna_positions = uvutils.ECEF_from_rotECEF(rot_ecef_positions, longitude)

    antenna_names = [
        ant.decode("utf8").strip() for ant in layout["antenna_names"][0].tolist()
    ]
    layout_fields.remove("antenna_names")

    # make these 0-indexed (rather than one indexed)
    antenna_numbers = layout["antenna_numbers"][0] - 1
    layout_fields.remove("antenna_numbers")

    Nants_telescope = int(layout["n_antenna"][0])
    layout_fields.remove("n_antenna")

    if telescope_name.lower() == "mwa":
        # check that obs.baseline_info.tile_names match the antenna names
        # this only applies for MWA because the tile_names come from
        # metafits files

        # tile_names are assumed to be ordered: so their index gives
        # the antenna number
        # make an comparison array from antenna_names ordered this way.
        ant_names = np.zeros((np.max(antenna_numbers) + 1), str).tolist()
        for index, number in enumerate(antenna_numbers):
            ant_names[number] = antenna_names[index]
        if obs_tile_names != ant_names:
            warnings.warn(
                "tile_names from obs structure does not match "
                "antenna_names from layout"
            )

    gst0 = float(layout["gst0"][0])
    layout_fields.remove("gst0")

    if layout["ref_date"][0] != "":
        rdate = layout["ref_date"][0].decode("utf8").lower()
    else:
        rdate = None
    layout_fields.remove("ref_date")

    earth_omega = float(layout["earth_degpd"][0])
    layout_fields.remove("earth_degpd")

    dut1 = float(layout["dut1"][0])
    layout_fields.remove("dut1")

    timesys = layout["time_system"][0].decode("utf8").upper().strip()
    layout_fields.remove("time_system")

    if "diameters" in layout_fields:
        diameters = np.asarray(layout["diameters"])
        layout_fields.remove("diameters")
    else:
        diameters = None

    extra_keywords = {}
    # ignore some fields, put everything else in extra_keywords
    layout_fields_ignore = [
        "diff_utc",
        "pol_type",
        "n_pol_cal_params",
        "mount_type",
        "axis_offset",
        "pola",
        "pola_orientation",
        "pola_cal_params",
        "polb",
        "polb_orientation",
        "polb_cal_params",
        "beam_fwhm",
    ]
    for field in layout_fields_ignore:
        if field in layout_fields:
            layout_fields.remove(field)
    for field in layout_fields:
        keyword = field
        if len(keyword) > 8:
            keyword = field.replace("_", "")

        value = layout[field][0]
        if isinstance(value, bytes):
            value = value.decode("utf8")

        extra_keywords[keyword.upper()] = value

    layout_param_dict = {
        "telescope_location": telescope_location,
        "Nants_telescope": Nants_telescope,
        "antenna_positions": antenna_positions,
        "antenna_names": antenna_names,
        "antenna_numbers": antenna_numbers,
        "gst0": gst0,
        "rdate": rdate,
        "earth_omega": earth_omega,
        "dut1": dut1,
        "timesys": timesys,
        "diameters": diameters,
        "extra_keywords": extra_keywords,
    }

    return layout_param_dict


class FHD(UVData):
    """
    Defines a FHD-specific subclass of UVData for reading FHD save files.

    This class should not be interacted with directly, instead use the read_fhd
    method on the UVData class.
    """

    def read_fhd(
        self,
        filelist,
        use_model=False,
        background_lsts=True,
        read_data=True,
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
        background_lsts : bool
            When set to True, the lst_array is calculated in a background thread.
        read_data : bool
            Read in the visibility, nsample and flag data. If set to False, only
            the metadata will be read in. Setting read_data to False results in
            a metadata only object. If read_data is False, an obs file must be
            included in the filelist. Note that if read_data is False, Npols is
            derived from the obs file and reflects the number of polarizations
            used in the FHD run. If read_data is True, Npols is given by the
            number of visibility data files provided in `filelist`.
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
            If required files are missing or multiple files for any polarization
            are included in filelist.
            If there is no recognized key for visibility weights in the flags_file.

        """
        datafiles = {}
        params_file = None
        obs_file = None
        flags_file = None
        layout_file = None
        settings_file = None
        if use_model:
            data_name = "_vis_model_"
        else:
            data_name = "_vis_"
        for file in filelist:
            if file.lower().endswith(data_name + "xx.sav"):
                if "xx" in list(datafiles.keys()):
                    raise ValueError("multiple xx datafiles in filelist")
                datafiles["xx"] = file
            elif file.lower().endswith(data_name + "yy.sav"):
                if "yy" in list(datafiles.keys()):
                    raise ValueError("multiple yy datafiles in filelist")
                datafiles["yy"] = file
            elif file.lower().endswith(data_name + "xy.sav"):
                if "xy" in list(datafiles.keys()):
                    raise ValueError("multiple xy datafiles in filelist")
                datafiles["xy"] = file
            elif file.lower().endswith(data_name + "yx.sav"):
                if "yx" in list(datafiles.keys()):
                    raise ValueError("multiple yx datafiles in filelist")
                datafiles["yx"] = file
            elif file.lower().endswith("_params.sav"):
                if params_file is not None:
                    raise ValueError("multiple params files in filelist")
                params_file = file
            elif file.lower().endswith("_obs.sav"):
                if obs_file is not None:
                    raise ValueError("multiple obs files in filelist")
                obs_file = file
            elif file.lower().endswith("_flags.sav"):
                if flags_file is not None:
                    raise ValueError("multiple flags files in filelist")
                flags_file = file
            elif file.lower().endswith("_layout.sav"):
                if layout_file is not None:
                    raise ValueError("multiple layout files in filelist")
                layout_file = file
            elif file.lower().endswith("_settings.txt"):
                if settings_file is not None:
                    raise ValueError("multiple settings files in filelist")
                settings_file = file
            else:
                # this is reached in tests but marked as uncovered because
                # CPython's peephole optimizer replaces a jump to a continue
                # with a jump to the top of the loop
                continue  # pragma: no cover

        if len(datafiles) < 1 and read_data is True:
            raise ValueError(
                "No data files included in file list and read_data is True."
            )
        if obs_file is None and read_data is False:
            raise ValueError(
                "No obs file included in file list and read_data is False."
            )
        if params_file is None:
            raise ValueError("No params file included in file list")
        if flags_file is None:
            raise ValueError("No flags file included in file list")
        if layout_file is None:
            warnings.warn(
                "No layout file included in file list, "
                "antenna_postions will not be defined."
            )
        if settings_file is None:
            warnings.warn("No settings file included in file list")

        if not read_data:
            obs_dict = readsav(obs_file, python_dict=True)
            this_obs = obs_dict["obs"]
            self.Npols = int(this_obs[0]["N_POL"])
        else:
            # TODO: add checking to make sure params, flags and datafiles are
            # consistent with each other
            vis_data = {}
            for pol, file in datafiles.items():
                this_dict = readsav(file, python_dict=True)
                if use_model:
                    vis_data[pol] = this_dict["vis_model_ptr"]
                else:
                    vis_data[pol] = this_dict["vis_ptr"]
                this_obs = this_dict["obs"]
            self.Npols = len(list(vis_data.keys()))

        obs = this_obs
        bl_info = obs["BASELINE_INFO"][0]
        astrometry = obs["ASTR"][0]
        fhd_pol_list = []
        for pol in obs["POL_NAMES"][0]:
            fhd_pol_list.append(pol.decode("utf8").lower())

        params_dict = readsav(params_file, python_dict=True)
        params = params_dict["params"]

        if read_data:
            flag_file_dict = readsav(flags_file, python_dict=True)
            # The name for this variable changed recently (July 2016). Test for both.
            vis_weights_data = {}
            if "flag_arr" in flag_file_dict:
                weights_key = "flag_arr"
            elif "vis_weights" in flag_file_dict:
                weights_key = "vis_weights"
            else:
                raise ValueError(
                    "No recognized key for visibility weights in flags_file."
                )
            for index, w in enumerate(flag_file_dict[weights_key]):
                vis_weights_data[fhd_pol_list[index]] = w

        self.Ntimes = int(obs["N_TIME"][0])
        self.Nbls = int(obs["NBASELINES"][0])
        self.Nblts = params["UU"][0].size
        self.Nfreqs = int(obs["N_FREQ"][0])
        self.Nspws = 1
        self.spw_array = np.array([0])
        self.vis_units = "JY"

        # bl_info.JDATE (a vector of length Ntimes) is the only safe date/time
        # to use in FHD files.
        # (obs.JD0 (float) and params.TIME (vector of length Nblts) are
        #   context dependent and are not safe
        #   because they depend on the phasing of the visibilities)
        # the values in bl_info.JDATE are the JD for each integration.
        # We need to expand up to Nblts.
        int_times = list(uvutils._get_iterable(bl_info["JDATE"][0]))
        bin_offset = bl_info["BIN_OFFSET"][0]
        if self.Ntimes != len(int_times):
            warnings.warn(
                "Ntimes does not match the number of unique times in the data"
            )
        self.time_array = np.zeros(self.Nblts)
        if self.Ntimes == 1:
            self.time_array.fill(int_times[0])
        else:
            for ii in range(0, len(int_times)):
                if ii < (len(int_times) - 1):
                    self.time_array[bin_offset[ii] : bin_offset[ii + 1]] = int_times[ii]
                else:
                    self.time_array[bin_offset[ii] :] = int_times[ii]

        # this is generated in FHD by subtracting the JD of neighboring
        # integrations. This can have limited accuracy, so it can be slightly
        # off the actual value.
        # (e.g. 1.999426... rather than 2)
        time_res = obs["TIME_RES"]
        # time_res is constrained to be a scalar currently
        self.integration_time = (
            np.ones_like(self.time_array, dtype=np.float64) * time_res[0]
        )
        # # --- observation information ---
        self.telescope_name = obs["INSTRUMENT"][0].decode("utf8")

        # This is a bit of a kludge because nothing like object_name exists
        # in FHD files.
        # At least for the MWA, obs.ORIG_PHASERA and obs.ORIG_PHASEDEC specify
        # the field the telescope was nominally pointing at
        # (May need to be revisited, but probably isn't too important)
        self.object_name = (
            "Field RA(deg): "
            + str(obs["ORIG_PHASERA"][0])
            + ", Dec:"
            + str(obs["ORIG_PHASEDEC"][0])
        )
        # For the MWA, this can sometimes be converted to EoR fields
        if self.telescope_name.lower() == "mwa":
            if np.isclose(obs["ORIG_PHASERA"][0], 0) and np.isclose(
                obs["ORIG_PHASEDEC"][0], -27
            ):
                self.object_name = "EoR 0 Field"

        self.instrument = self.telescope_name
        latitude = np.deg2rad(float(obs["LAT"][0]))
        longitude = np.deg2rad(float(obs["LON"][0]))
        altitude = float(obs["ALT"][0])

        # get the stuff FHD read from the antenna table (in layout file)
        if layout_file is not None:
            obs_tile_names = [
                ant.decode("utf8").strip() for ant in bl_info["TILE_NAMES"][0].tolist()
            ]
            obs_tile_names = [
                "Tile" + "0" * (3 - len(ant)) + ant for ant in obs_tile_names
            ]

            layout_param_dict = get_fhd_layout_info(
                layout_file,
                self.telescope_name,
                latitude,
                longitude,
                altitude,
                self._phase_center_ra.tols,
                self._telescope_location.tols,
                obs_tile_names,
                run_check_acceptability=True,
            )

            for key, value in layout_param_dict.items():
                setattr(self, key, value)

        else:
            self.telescope_location_lat_lon_alt = (latitude, longitude, altitude)
            self.antenna_names = [
                ant.decode("utf8").strip() for ant in bl_info["TILE_NAMES"][0].tolist()
            ]
            if self.telescope_name.lower() == "mwa":
                self.antenna_names = [
                    "Tile" + "0" * (3 - len(ant)) + ant for ant in self.antenna_names
                ]
            self.Nants_telescope = len(self.antenna_names)
            self.antenna_numbers = np.arange(self.Nants_telescope)

        try:
            self.set_telescope_params()
        except ValueError as ve:
            warnings.warn(str(ve))

        # need to make sure telescope location is defined properly before this call
        proc = self.set_lsts_from_time_array(background=background_lsts)

        if not np.isclose(obs["OBSRA"][0], obs["PHASERA"][0]) or not np.isclose(
            obs["OBSDEC"][0], obs["PHASEDEC"][0]
        ):
            warnings.warn(
                "These visibilities may have been phased "
                "improperly -- without changing the uvw locations"
            )

        self._set_phased()
        self.phase_center_ra_degrees = np.float(obs["OBSRA"][0])
        self.phase_center_dec_degrees = np.float(obs["OBSDEC"][0])

        self.phase_center_epoch = astrometry["EQUINOX"][0]
        self._set_app_coords_helper()
        # Note that FHD antenna arrays are 1-indexed so we subtract 1
        # to get 0-indexed arrays
        self.ant_1_array = bl_info["TILE_A"][0] - 1
        self.ant_2_array = bl_info["TILE_B"][0] - 1

        self.Nants_data = int(np.union1d(self.ant_1_array, self.ant_2_array).size)

        self.baseline_array = self.antnums_to_baseline(
            self.ant_1_array, self.ant_2_array
        )
        if self.Nbls != len(np.unique(self.baseline_array)):
            warnings.warn(
                "Nbls does not match the number of unique baselines in the data"
            )

        # TODO: Spw axis to be collapsed in future release
        self.freq_array = np.zeros((1, len(bl_info["FREQ"][0])), dtype=np.float_)
        self.freq_array[0, :] = bl_info["FREQ"][0]

        self.channel_width = float(obs["FREQ_RES"][0])

        # In FHD, uvws are in seconds not meters.
        # FHD follows the FITS uvw direction convention, which is opposite
        # ours and Miriad's.
        # So conjugate the visibilities and flip the uvws:
        self.uvw_array = np.zeros((self.Nblts, 3))
        self.uvw_array[:, 0] = (-1) * params["UU"][0] * const.c.to("m/s").value
        self.uvw_array[:, 1] = (-1) * params["VV"][0] * const.c.to("m/s").value
        self.uvw_array[:, 2] = (-1) * params["WW"][0] * const.c.to("m/s").value

        lin_pol_order = ["xx", "yy", "xy", "yx"]
        linear_pol_dict = dict(zip(lin_pol_order, np.arange(5, 9) * -1))
        pol_list = []
        if read_data:
            for pol in lin_pol_order:
                if pol in vis_data:
                    pol_list.append(linear_pol_dict[pol])
            self.polarization_array = np.asarray(pol_list)
        else:
            # Use Npols because for FHD, npol fully specifies which pols to use
            pol_strings = lin_pol_order[: self.Npols]
            self.polarization_array = np.asarray(
                [linear_pol_dict[pol] for pol in pol_strings]
            )

        # history: add the first few lines from the settings file
        if settings_file is not None:
            self.history = get_fhd_history(settings_file)
        else:
            self.history = ""

        if not uvutils._check_history_version(self.history, self.pyuvdata_version_str):
            self.history += self.pyuvdata_version_str

        if read_data:
            # TODO: Spw axis to be collapsed in future release
            self.data_array = np.zeros(
                (self.Nblts, 1, self.Nfreqs, self.Npols), dtype=np.complex_
            )
            # TODO: Spw axis to be collapsed in future release
            self.nsample_array = np.zeros(
                (self.Nblts, 1, self.Nfreqs, self.Npols), dtype=np.float_
            )
            # TODO: Spw axis to be collapsed in future release
            self.flag_array = np.zeros(
                (self.Nblts, 1, self.Nfreqs, self.Npols), dtype=np.bool_
            )
            for pol, vis in vis_data.items():
                pol_i = pol_list.index(linear_pol_dict[pol])
                # FHD follows the FITS uvw direction convention, which is opposite
                # ours and Miriad's.
                # So conjugate the visibilities and flip the uvws:
                self.data_array[:, 0, :, pol_i] = np.conj(vis)
                self.flag_array[:, 0, :, pol_i] = vis_weights_data[pol] <= 0
                self.nsample_array[:, 0, :, pol_i] = np.abs(vis_weights_data[pol])

        # wait for LSTs if set in background
        if proc is not None:
            proc.join()

        # check if object has all required uv_properties set
        if run_check:
            self.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
            )
