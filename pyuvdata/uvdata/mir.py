# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2020 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Class for reading and writing Mir files."""
import numpy as np
from itertools import compress

from .uvdata import UVData
from . import mir_parser
from .. import utils as uvutils
from .. import get_telescope

__all__ = ["Mir"]


class Mir(UVData):
    """
    A class for Mir file objects.

    This class defines an Mir-specific subclass of UVData for reading and
    writing Mir files. This class should not be interacted with directly,
    instead use the read_mir and write_mir methods on the UVData class.
    """

    def read_mir(
        self,
        filepath,
        isource=None,
        irec=None,
        isb=None,
        corrchunk=None,
        pseudo_cont=False,
        flex_spw=True,
    ):
        """
        Read in data from an SMA MIR file, and map to the UVData model.

        Note that with the exception of filename, the rest of the parameters are
        used to sub-select a range of data that matches the limitations of the current
        instantiation of pyuvdata  -- namely 1 source. This could be dropped in the
        future, as pyuvdata capabilities grow.

        Parameters
        ----------
        filepath : str
            The file path to the MIR folder to read from.
        isource : list of int
            Source code(s) for MIR dataset
        irec : int
            Receiver code for MIR dataset
        isb : list of int
            Sideband codes for MIR dataset (0 = LSB, 1 = USB). Default is both sb.
        corrchunk : list of int
            Correlator chunk codes for MIR dataset
        pseudo_cont : boolean
            Read in only pseudo-continuuum values. Default is false.
        flex_spw : boolean
            Allow for support of multiple spectral windows. Default is true.
        """
        # Use the mir_parser to read in metadata, which can be used to select data.
        mir_data = mir_parser.MirParser(filepath)

        # By default, we will want to assume that MIR datasets are phased, multi-spw,
        # and multi-object. At present, there is no advantage to allowing these not to
        # be true on read-in, particularly as in the long-term, these settings will
        # hopefully become the default for all data sets.
        self._set_phased()
        self._set_multi_object()
        self._set_flex_spw()

        isource_full_list = np.unique(mir_data.in_read["isource"]).tolist()
        if isource is None:
            isource = isource_full_list.copy()

        # Grab the list of sources we want to select on
        isource_dict = {key: key in isource for key in isource_full_list}
        if len(isource_dict) == 0:
            raise IndexError("No valid sources selected!")

        # Select out the receiver that we want to deal with, since we can only
        # currently handle one of each
        if irec is None:
            irec = mir_data.bl_read["irec"][0]

        # Begin sorting out the spectral windows, starting with which sidebands to
        # include in the read
        isb_full_list = np.unique(mir_data.bl_read["isb"]).tolist()
        if isb is None:
            isb = isb_full_list.copy()

        isb_dict = {key: key in isb for key in isb_full_list}
        if len(isb) == 0:
            raise IndexError("No valid sidebands selected!")
        elif not (set(isb).issubset(set(isb_full_list))):
            raise IndexError("isb values contain invalid entries")

        # Remember whether or not we're dealing with DSB (2 windows per corrchunk)
        dsb_spws = True if len(isb) == 2 else False

        corrchunk_full_list = np.unique(mir_data.sp_read["corrchunk"]).tolist()
        if corrchunk is None:
            if pseudo_cont:
                corrchunk = [0]
            else:
                corrchunk = corrchunk_full_list.copy()
                corrchunk.remove(0) if 0 in corrchunk else None
        corrchunk_dict = {key: key in corrchunk for key in corrchunk_full_list}

        mir_data.use_in = np.array(
            [isource_dict[key] for key in mir_data.in_read["isource"]]
        )
        mir_data.use_bl = np.logical_and(
            np.logical_and(
                mir_data.bl_read["irec"] == irec, mir_data.bl_read["ipol"] == 0
            ),
            np.array([isb_dict[key] for key in mir_data.bl_read["isb"]]),
        )

        mir_data.use_sp = np.array(
            [corrchunk_dict[key] for key in mir_data.sp_read["corrchunk"]]
        )

        # Update the filters, and will make sure we're looking at the right metadata.
        mir_data._update_filter()
        if len(mir_data.in_data) == 0:
            raise IndexError("No valid records matching those selections!")

        # Create a simple list for broadcasting values stored on a
        # per-intergration basis in MIR into the (tasty) per-blt records in UVDATA.
        bl_in_maparr = [
            mir_data.inhid_dict[idx]
            for idx in mir_data.bl_data["inhid"][mir_data.bl_data["isb"] == isb[0]]
        ]

        # Create a simple array/list for broadcasting values stored on a
        # per-blt basis into per-spw records, and per-time into per-blt records
        sp_bl_maparr = np.array(
            [mir_data.blhid_dict[idx] for idx in mir_data.sp_data["blhid"]]
        )
        bl_in_maparr = np.array(
            [mir_data.inhid_dict[idx] for idx in mir_data.bl_data["inhid"]]
        )
        # Different sidebands in MIR are (strangely enough) recorded as being
        # different baseline records. To be compatible w/ UVData, we just splice
        # the sidebands together.
        corrchunk_sb = [idx for jdx in sorted(isb) for idx in [jdx] * len(corrchunk)]
        corrchunk *= 1 + dsb_spws
        sb_screen = mir_data.bl_data["isb"] == isb[0]

        # Here we'll want to construct a simple dictionary, that'll basically help us
        # to construct the frequency axis, and map the UVData spectral window ID number
        # to our weird mapping system in MIR.
        Nfreqs = 0  # Set to zero to starts for flex_spw

        # Initialize some arrays that we'll be appending to
        flex_spw_id_array = np.array([], dtype=np.int)
        channel_width = np.array([], dtype=np.float)
        freq_array = np.array([], dtype=np.float)
        for idx in range(len(corrchunk)):
            data_mask = np.logical_and(
                mir_data.sp_data["corrchunk"] == corrchunk[idx],
                mir_data.bl_data["isb"][sp_bl_maparr] == corrchunk_sb[idx],
            )

            # Grab values, get them into appropriate types
            spw_fsky = np.unique(mir_data.sp_data["fsky"][data_mask])
            spw_fres = np.unique(mir_data.sp_data["fres"][data_mask])
            spw_nchan = np.unique(mir_data.sp_data["nch"][data_mask])

            # Make sure that something weird hasn't happend with the metadata (this
            # really should never happen)
            assert len(spw_fsky) == 1
            assert len(spw_fres) == 1
            assert len(spw_nchan) == 1

            #  Get the data in the right units and dtype
            spw_fsky = np.float(spw_fsky * 1e9)  # GHz -> Hz
            spw_fres = np.float(spw_fres * 1e6)  # MHz -> Hz
            spw_nchan = np.int(spw_nchan)

            # Tally up the number of channels
            Nfreqs += spw_nchan

            # Populate the channel width array
            channel_width = np.append(
                channel_width, abs(spw_fres) + np.zeros(spw_nchan, dtype=np.float)
            )

            # Populate the the spw_id_array
            flex_spw_id_array = np.append(
                flex_spw_id_array, idx + np.zeros(spw_nchan, dtype=np.int)
            )

            # So the freq array here is a little weird, because the current fsky
            # refers to the point between the nch/2 and nch/2 + 1 channel in the
            # raw (unaveraged) spectrum. This was done for the sake of some
            # convenience, at the cost of clarity. In some future format of the
            # data, we expect to be able to drop seemingly random offset here.
            freq_array = np.append(
                freq_array,
                spw_fsky
                - (np.sign(spw_fres) * 139648437.5)
                + (spw_fres * (np.arange(spw_nchan) + 0.5 - (spw_nchan / 2))),
            )

        # Now assign our flexible arrays to the object itself
        # TODO: Spw axis to be collapsed in future release
        self.freq_array = freq_array[np.newaxis, :]
        self.Nfreqs = Nfreqs
        self.channel_width = channel_width
        self.flex_spw_id_array = flex_spw_id_array

        # Load up the visibilities into the MirParser object.
        mir_data.load_data(load_vis=True, load_raw=True)

        # Derive Nants_data from baselines.
        self.Nants_data = len(
            np.unique(
                np.concatenate((mir_data.bl_data["iant1"], mir_data.bl_data["iant2"]))
            )
        )
        self.Nants_telescope = 8
        self.Nbls = int(self.Nants_data * (self.Nants_data - 1) / 2)
        self.Nblts = len(mir_data.bl_data) // (1 + dsb_spws)
        self.Npols = 1  # todo: We will need to go back and expand this.
        self.Nspws = len(corrchunk)
        self.Ntimes = len(mir_data.in_data)
        self.ant_1_array = mir_data.bl_data["iant1"][sb_screen] - 1
        self.ant_2_array = mir_data.bl_data["iant2"][sb_screen] - 1
        self.antenna_names = [
            "Ant 1",
            "Ant 2",
            "Ant 3",
            "Ant 4",
            "Ant 5",
            "Ant 6",
            "Ant 7",
            "Ant 8",
        ]
        self.antenna_numbers = np.arange(8)

        # Prepare the XYZ coordinates of the antenna positions.
        antXYZ = np.zeros([self.Nants_telescope, 3])
        for idx in range(self.Nants_telescope):
            if (idx + 1) in mir_data.antpos_data["antenna"]:
                antXYZ[idx] = mir_data.antpos_data["xyz_pos"][
                    mir_data.antpos_data["antenna"] == (idx + 1)
                ]

        # Get the coordinates from the entry in telescope.py
        lat, lon, alt = get_telescope("SMA")._telescope_location.lat_lon_alt()
        self.telescope_location_lat_lon_alt = (lat, lon, alt)

        # Calculate antenna postions in EFEF frame. Note that since both
        # coordinate systems are in relative units, no subtraction from
        # telescope geocentric position is required , i.e we are going from
        # relRotECEF -> relECEF
        self.antenna_positions = uvutils.ECEF_from_rotECEF(antXYZ, lon)
        self.baseline_array = self.antnums_to_baseline(
            self.ant_1_array, self.ant_2_array, attempt256=False
        )

        self.history = "Raw Data"
        self.instrument = "SWARM"

        # We can just skip an appropriate number of records
        self.integration_time = mir_data.sp_data["integ"][:: len(corrchunk)]

        # TODO: Using MIR V3 convention, will need to be V2 compatible eventually.
        self.lst_array = (
            mir_data.in_data["lst"][bl_in_maparr[:: 1 + dsb_spws]].astype(float)
        ) * (np.pi / 12.0)

        # TODO: We change between xx yy and rr ll, so we will need to update this.
        self.polarization_array = np.asarray([-5])

        self.spw_array = np.arange(len(corrchunk))

        self.telescope_name = "SMA"
        time_array_mjd = mir_data.in_read["mjd"][bl_in_maparr[sb_screen]]
        self.time_array = time_array_mjd + 2400000.5

        # Need to flip the sign convention here on uvw, since we use a1-a2 versus the
        # standard a2-a1 that uvdata expects
        self.uvw_array = (-1.0) * np.transpose(
            np.vstack(
                (
                    mir_data.bl_data["u"][sb_screen],
                    mir_data.bl_data["v"][sb_screen],
                    mir_data.bl_data["w"][sb_screen],
                )
            )
        )

        # todo: Raw data is in correlation coefficients, we may want to convert to Jy.
        self.vis_units = "uncalib"

        sou_list = mir_data.codes_data[mir_data.codes_data["v_name"] == b"source"]

        self.object_name = [
            sou_list[sou_list["icode"] == idx]["code"][0].decode("utf-8")
            for idx in isource
        ]
        object_dict = {}
        for idx in range(len(isource)):
            source_mask = mir_data.in_data["isource"] == isource[idx]
            object_name = self.object_name[idx]
            object_ra = np.mean(mir_data.in_data["rar"][source_mask])
            object_dec = np.mean(mir_data.in_data["decr"][source_mask])
            coord_epoch = np.mean(mir_data.in_data["epoch"][source_mask])
            object_dict[object_name] = {
                "object_type": "sidereal",
                "object_name": object_name,
                "object_ra": object_ra,
                "object_dec": object_dec,
                "coord_frame": "icrs",  # default for SMA datasets (verify)
                "coord_epoch": coord_epoch,
            }

        self.object_dict = object_dict

        # Regenerate the sou_id_array thats native to MIR into a zero-indexed per-blt
        # entry for UVData, then grab ra/dec/position data.
        object_id_array = mir_data.in_data["isource"][bl_in_maparr[sb_screen]]
        object_id_dict = {isource[idx]: idx for idx in range(len(isource))}
        object_id_array = np.array(
            [object_id_dict[key] for key in object_id_array], dtype=np.int
        )
        self.object_id_array = object_id_array.astype(np.int)

        self.Nobjects = len(self.object_name)
        self.phase_center_ra = 0.0  # This are ignored w/ multi-obj data sets
        self.phase_center_dec = 0.0  # This are ignored w/ multi-obj data sets
        self.phase_center_epoch = 2000.0  # This are ignored w/ multi-obj data sets
        self.phase_center_frame = "icrs"

        # Fill in the apparent coord calculations
        self.phase_center_app_ra = mir_data.in_data["rar"][bl_in_maparr[sb_screen]]
        self.phase_center_app_dec = mir_data.in_data["decr"][bl_in_maparr[sb_screen]]

        self.antenna_diameters = np.zeros(self.Nants_telescope) + 6
        self.blt_order = ("time", "baseline")

        # TODO: Spw axis to be collapsed in future release
        if dsb_spws:
            # Gotta do a little bit of sleight-of-hand here, on account of
            # the fact that the ordering of the sidebands makes absolutely
            # no sense.
            data_array = np.concatenate(
                (
                    np.reshape(
                        np.concatenate(
                            list(compress(mir_data.vis_data, sb_screen[sp_bl_maparr])),
                        ),
                        (self.Nblts, 1, self.Nfreqs // 2, self.Npols),
                    ),
                    np.reshape(
                        np.concatenate(
                            list(compress(mir_data.vis_data, ~sb_screen[sp_bl_maparr])),
                        ),
                        (self.Nblts, 1, self.Nfreqs // 2, self.Npols),
                    ),
                ),
                axis=2,
            )
        else:
            # If single sideband, then this is a pretty simple operation
            data_array = np.reshape(
                np.concatenate(mir_data.vis_data),
                (self.Nblts, 1, self.Nfreqs, self.Npols),
            )

        # Don't need the data anymore, so drop it
        mir_data.unload_data()

        self.data_array = data_array
        self.flag_array = np.zeros(self.data_array.shape, dtype=bool)
        self.nsample_array = np.ones(self.data_array.shape, dtype=np.float32)

    def write_mir(self, filename):
        """
        Write out the SMA MIR files.

        Parameters
        ----------
        filename : str
            The path to the folder on disk to write data to.
        """
        raise NotImplementedError
