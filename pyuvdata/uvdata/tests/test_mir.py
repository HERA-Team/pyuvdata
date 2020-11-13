# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2020 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for Mir class.

Performs a series of test for the Mir class, which inherits from UVData. Note that
there is a separate test module for the MirParser class (mir_parser.py), which is
what is used to read the raw binary data into something that the Mir class can
manipulate into a UVData object.
"""
import os

import pytest
import numpy as np

from ... import UVData
from ...data import DATA_PATH
from ...uvdata.mir import mir_parser


@pytest.fixture
def mir_data_object():
    testfile = os.path.join(DATA_PATH, "sma_test.mir")
    mir_data = mir_parser.MirParser(
        testfile, load_vis=True, load_raw=True, load_auto=True,
    )

    yield mir_data

    # cleanup
    del mir_data


@pytest.fixture
def uv_in_uvfits(tmp_path):
    uv_in = UVData()
    testfile = os.path.join(DATA_PATH, "sma_test.mir")
    write_file = os.path.join(tmp_path, "outtest_mir.uvfits")

    # Currently only one source is supported.
    uv_in.read(testfile, pseudo_cont=True)
    uv_out = UVData()

    yield uv_in, uv_out, write_file

    # cleanup
    del uv_in, uv_out


@pytest.fixture
def uv_in_uvh5(tmp_path):
    uv_in = UVData()
    testfile = os.path.join(DATA_PATH, "sma_test.mir")
    write_file = os.path.join(tmp_path, "outtest_mir.uvh5")

    # Currently only one source is supported.
    uv_in.read(testfile)
    uv_out = UVData()

    yield uv_in, uv_out, write_file

    # cleanup
    del uv_in, uv_out


@pytest.mark.filterwarnings("ignore:LST values stored in this file are not ")
def test_read_mir_write_uvfits(uv_in_uvfits):
    """
    Mir to uvfits loopback test.

    Read in Mir files, write out as uvfits, read back in and check for
    object equality.
    """
    mir_uv, uvfits_uv, testfile = uv_in_uvfits

    mir_uv.write_uvfits(testfile, spoof_nonessential=True)
    uvfits_uv.read_uvfits(testfile)

    # Check the history first via find
    assert 0 == uvfits_uv.history.find(
        mir_uv.history + "  Read/written with pyuvdata version:"
    )
    mir_uv.history = uvfits_uv.history

    # We have to do a bit of special handling for he object_dict, because _very_ small
    # errors (like last bit in the mantissa) creep in when passing through the util
    # function translate_sidereal_to_sidereal (for multi-obj datasets). Verify the
    # two match up in terms of their coordinates
    for object_name in mir_uv.object_name:
        assert np.isclose(
            mir_uv.object_dict[object_name]["object_lat"],
            uvfits_uv.object_dict[object_name]["object_lat"],
        )
        assert np.isclose(
            mir_uv.object_dict[object_name]["object_lon"],
            uvfits_uv.object_dict[object_name]["object_lon"],
        )
    uvfits_uv.object_dict = mir_uv.object_dict

    # There's a minor difference between what SMA calculates online for app coords
    # and what NOVAS calculates, to the tune of ~10 mas. Check those values here,
    # then set them equal to one another.
    assert np.all(
        np.abs(mir_uv.phase_center_app_ra - uvfits_uv.phase_center_app_ra) < 1e-6
    )

    assert np.all(
        np.abs(mir_uv.phase_center_app_dec - uvfits_uv.phase_center_app_dec) < 1e-6
    )

    mir_uv._set_app_coords_helper()
    uvfits_uv._set_app_coords_helper()

    assert mir_uv == uvfits_uv


@pytest.mark.filterwarnings("ignore:LST values stored ")
def test_read_mir_write_uvh5(uv_in_uvh5):
    """
    Mir to uvfits loopback test.

    Read in Mir files, write out as uvfits, read back in and check for
    object equality.
    """
    mir_uv, uvh5_uv, testfile = uv_in_uvh5

    mir_uv.write_uvh5(testfile)
    uvh5_uv.read_uvh5(testfile)

    # Check the history first via find
    assert 0 == uvh5_uv.history.find(
        mir_uv.history + "  Read/written with pyuvdata version:"
    )

    # test fails because of updated history, so this is our workaround for now.
    mir_uv.history = uvh5_uv.history

    assert mir_uv == uvh5_uv


def test_write_mir(uv_in_uvfits, err_type=NotImplementedError):
    """
    Mir writer test

    Check and make sure that attempts to use the writer return a
    'not implemented; error.
    """
    mir_uv, uvfits_uv, testfile = uv_in_uvfits

    # Check and see if the correct error is raised
    with pytest.raises(err_type):
        mir_uv.write_mir("dummy.mir")


def test_flex_spw_read(tmp_path):
    """
    Mir test for flexible spws.

    Read in Mir files using flexible spectral windows, all of the same nchan
    """
    testfile = os.path.join(DATA_PATH, "sma_test.mir")
    uv_in = UVData()
    uv_in.read_mir(testfile)
    dummyfile = os.path.join(tmp_path, "dummy.mirtest.uvfits")

    uv_in2 = uv_in.copy()
    with pytest.raises(NotImplementedError):
        uv_in2.frequency_average(2)

    uv_in2.flex_spw_id_array[0] = 1
    with pytest.raises(ValueError):
        uv_in2._check_flex_spw_contiguous()

    uv_in2 = uv_in.copy()
    uv_in2.channel_width[0] = 0
    with pytest.raises(ValueError, match="The frequencies are not evenly spaced"):
        uv_in2._check_freq_spacing()

    uv_in2 = uv_in.copy()
    uv_in2.channel_width[:] = 0
    with pytest.raises(ValueError, match="The frequencies are separated by more"):
        uv_in2._check_freq_spacing()

    uv_in2 = uv_in.copy()
    uv_in2.freq_array *= -1
    with pytest.raises(ValueError, match="Frequency values must be > 0 for UVFITS!"):
        uv_in2.write_uvfits(dummyfile, spoof_nonessential=True)

    uv_in2 = uv_in.copy()
    uv_in2.freq_array[:] = 1
    uv_in2.channel_width *= 0
    with pytest.raises(ValueError, match="Something is wrong, frequency values not"):
        uv_in2.write_uvfits(dummyfile, spoof_nonessential=True)


def test_multi_nchan_spw_read(tmp_path):
    """
    Mir to uvfits error test for spws of different sizes.

    Read in Mir files, write out as uvfits, read back in and check for
    object equality.
    """
    testfile = os.path.join(DATA_PATH, "sma_test.mir")
    uv_in = UVData()
    uv_in.read_mir(testfile, corrchunk=[0, 1, 2, 3, 4])

    dummyfile = os.path.join(tmp_path, "dummy.mirtest.uvfits")
    with pytest.raises(IndexError):
        uv_in.write_uvfits(dummyfile, spoof_nonessential=True)


def test_read_mir_no_records():
    """
    Mir no-records check

    Make sure that mir correctly handles the case where no matching records are found
    """
    testfile = os.path.join(DATA_PATH, "sma_test.mir")
    uv_in = UVData()
    with pytest.raises(IndexError, match="No valid records matching those selections!"):
        uv_in.read_mir(testfile, isource=-1)

    with pytest.raises(IndexError, match="No valid sidebands selected!"):
        uv_in.read_mir(testfile, isb=[])

    with pytest.raises(IndexError, match="isb values contain invalid entries"):
        uv_in.read_mir(testfile, isb=[-156])


def test_mir_auto_read(
    err_type=IndexError, err_msg="Could not determine auto-correlation record size!"
):
    """
    Mir read tester

    Make sure that Mir autocorrelations are read correctly
    """
    testfile = os.path.join(DATA_PATH, "sma_test.mir")
    mir_data = mir_parser.MirParser(testfile)
    with pytest.raises(err_type, match=err_msg):
        ac_data = mir_data.scan_auto_data(testfile, nchunks=999)

    ac_data = mir_data.scan_auto_data(testfile)
    assert np.all(ac_data["nchunks"] == 8)

    mir_data.load_data(load_vis=False, load_auto=True)

    # Select the relevant auto records, which should be for spwin 0-3
    auto_data = mir_data.read_auto_data(testfile, ac_data)[:, 0:4, :, :]
    assert np.all(
        np.logical_or(
            auto_data == mir_data.auto_data,
            np.logical_and(np.isnan(auto_data), np.isnan(mir_data.auto_data)),
        )
    )
    mir_data.unload_data()


# Below are a series of checks that are designed to check to make sure that the
# MirParser class is able to produce consistent values from an engineering data
# set (originally stored in /data/engineering/mir_data/200724_16:35:14), to make
# sure that we haven't broken the ability of the reader to handle the data. Since
# this file is the basis for the above checks, we've put this here rather than in
# test_mir_parser.py


def test_mir_remember_me_record_lengths(mir_data_object):
    """
    Mir record length checker

    Make sure the test file containts the right number of records
    """
    mir_data = mir_data_object

    # Check to make sure we've got the right number of records everywhere
    assert len(mir_data.ac_read) == 2

    assert len(mir_data.bl_read) == 4

    assert len(mir_data.codes_read) == 99

    assert len(mir_data.eng_read) == 2

    assert len(mir_data.in_read) == 1

    assert len(mir_data.raw_data) == 20

    assert len(mir_data.raw_scale_fac) == 20

    assert len(mir_data.sp_read) == 20

    assert len(mir_data.vis_data) == 20

    assert len(mir_data.we_read) == 1


def test_mir_remember_me_codes_read(mir_data_object):
    """
    Mir codes_read checker.

    Make sure that certain values in the codes_read file of the test data set match
    whatwe know to be 'true' at the time of observations.
    """
    mir_data = mir_data_object

    assert mir_data.codes_read[0][0] == b"filever"

    assert mir_data.codes_read[0][2] == b"3"

    assert mir_data.codes_read[90][0] == b"ref_time"

    assert mir_data.codes_read[90][1] == 0

    assert mir_data.codes_read[90][2] == b"Jul 24, 2020"

    assert mir_data.codes_read[90][3] == 0

    assert mir_data.codes_read[91][0] == b"ut"

    assert mir_data.codes_read[91][1] == 1

    assert mir_data.codes_read[91][2] == b"Jul 24 2020  4:34:39.00PM"

    assert mir_data.codes_read[91][3] == 0

    assert mir_data.codes_read[93][0] == b"source"

    assert mir_data.codes_read[93][2] == b"3c84"

    assert mir_data.codes_read[97][0] == b"ra"

    assert mir_data.codes_read[97][2] == b"03:19:48.15"

    assert mir_data.codes_read[98][0] == b"dec"

    assert mir_data.codes_read[98][2] == b"+41:30:42.1"


def test_mir_remember_me_in_read(mir_data_object):
    """
    Mir in_read checker.

    Make sure that certain values in the in_read file of the test data set match what
    we know to be 'true' at the time of observations, including that spare values are
    stored as zero.
    """
    mir_data = mir_data_object

    # Check to make sure that things seem right in in_read
    assert np.all(mir_data.in_read["traid"] == 484)

    assert np.all(mir_data.in_read["proid"] == 484)

    assert np.all(mir_data.in_read["inhid"] == 1)

    assert np.all(mir_data.in_read["ints"] == 1)

    assert np.all(mir_data.in_read["souid"] == 1)

    assert np.all(mir_data.in_read["isource"] == 1)

    assert np.all(mir_data.in_read["ivrad"] == 1)

    assert np.all(mir_data.in_read["ira"] == 1)

    assert np.all(mir_data.in_read["idec"] == 1)

    assert np.all(mir_data.in_read["epoch"] == 2000.0)

    assert np.all(mir_data.in_read["tile"] == 0)

    assert np.all(mir_data.in_read["obsflag"] == 0)

    assert np.all(mir_data.in_read["obsmode"] == 0)

    assert np.all(np.round(mir_data.in_read["mjd"]) == 59055)

    assert np.all(mir_data.in_read["spareshort"] == 0)

    assert np.all(mir_data.in_read["spareint6"] == 0)


def test_mir_remember_me_bl_read(mir_data_object):
    """
    Mir bl_read checker.

    Make sure that certain values in the bl_read file of the test data set match what
    we know to be 'true' at the time of observations, including that spare values are
    stored as zero.
    """
    mir_data = mir_data_object

    # Now check bl_read
    assert np.all(mir_data.bl_read["blhid"] == np.arange(1, 5))

    assert np.all(mir_data.bl_read["isb"] == [0, 0, 1, 1])

    assert np.all(mir_data.bl_read["ipol"] == [0, 0, 0, 0])

    assert np.all(mir_data.bl_read["ant1rx"] == [0, 1, 0, 1])

    assert np.all(mir_data.bl_read["ant2rx"] == [0, 1, 0, 1])

    assert np.all(mir_data.bl_read["pointing"] == 0)

    assert np.all(mir_data.bl_read["irec"] == [0, 3, 0, 3])

    assert np.all(mir_data.bl_read["iant1"] == 1)

    assert np.all(mir_data.bl_read["iant2"] == 4)

    assert np.all(mir_data.bl_read["iblcd"] == 2)

    assert np.all(mir_data.bl_read["spareint1"] == 0)

    assert np.all(mir_data.bl_read["spareint2"] == 0)

    assert np.all(mir_data.bl_read["spareint3"] == 0)

    assert np.all(mir_data.bl_read["spareint4"] == 0)

    assert np.all(mir_data.bl_read["spareint5"] == 0)

    assert np.all(mir_data.bl_read["spareint6"] == 0)

    assert np.all(mir_data.bl_read["sparedbl3"] == 0.0)

    assert np.all(mir_data.bl_read["sparedbl4"] == 0.0)

    assert np.all(mir_data.bl_read["sparedbl5"] == 0.0)

    assert np.all(mir_data.bl_read["sparedbl6"] == 0.0)


def test_mir_remember_me_eng_read(mir_data_object):
    """
    Mir bl_read checker.

    Make sure that certain values in the eng_read file of the test data set match what
    we know to be 'true' at the time of observations.
    """
    mir_data = mir_data_object

    # Now check eng_read
    assert np.all(mir_data.eng_read["antennaNumber"] == [1, 4])

    assert np.all(mir_data.eng_read["padNumber"] == [5, 8])

    assert np.all(mir_data.eng_read["trackStatus"] == 1)

    assert np.all(mir_data.eng_read["commStatus"] == 1)

    assert np.all(mir_data.eng_read["inhid"] == 1)


def test_mir_remember_me_ac_read(mir_data_object):
    """
    Mir bl_read checker.

    Make sure that certain values in the autoCorrelations file of the test data set
    match what we know to be 'true' at the time of observations.
    """
    mir_data = mir_data_object

    # Now check ac_read
    assert np.all(mir_data.ac_read["inhid"] == 1)

    assert np.all(mir_data.ac_read["achid"] == np.arange(1, 3))

    assert np.all(mir_data.ac_read["antenna"] == [1, 4])

    assert np.all(mir_data.ac_read["nchunks"] == 8)

    assert np.all(mir_data.ac_read["datasize"] == 1048596)

    assert np.all(mir_data.we_read["scanNumber"] == 1)

    assert np.all(mir_data.we_read["flags"] == 0)


def test_mir_remember_me_sp_read(mir_data_object):
    """
    Mir sp_read checker.

    Make sure that certain values in the sp_read file of the test data set match what
    we know to be 'true' at the time of observations, including that spare values are
    stored as zero.
    """
    mir_data = mir_data_object

    # Now check sp_read
    assert np.all(mir_data.sp_read["sphid"] == np.arange(1, 21))

    assert np.all(mir_data.sp_read["sphid"] == np.arange(1, 21))

    assert np.all(mir_data.sp_read["igq"] == 0)

    assert np.all(mir_data.sp_read["ipq"] == 1)

    assert np.all(mir_data.sp_read["igq"] == 0)

    assert np.all(mir_data.sp_read["iband"] == [0, 1, 2, 3, 4] * 4)

    assert np.all(mir_data.sp_read["ipstate"] == 0)

    assert np.all(mir_data.sp_read["tau0"] == 0.0)

    assert np.all(mir_data.sp_read["cabinLO"] == 0.0)

    assert np.all(mir_data.sp_read["corrLO1"] == 0.0)

    assert np.all(mir_data.sp_read["vradcat"] == 0.0)

    assert np.all(mir_data.sp_read["nch"] == [4, 16384, 16384, 16384, 16384] * 4)

    assert np.all(mir_data.sp_read["corrblock"] == [0, 1, 1, 1, 1] * 4)

    assert np.all(mir_data.sp_read["corrchunk"] == [0, 1, 2, 3, 4] * 4)

    assert np.all(mir_data.sp_read["correlator"] == 1)

    assert np.all(mir_data.sp_read["spareint2"] == 0)

    assert np.all(mir_data.sp_read["spareint3"] == 0)

    assert np.all(mir_data.sp_read["spareint4"] == 0)

    assert np.all(mir_data.sp_read["spareint5"] == 0)

    assert np.all(mir_data.sp_read["spareint6"] == 0)

    assert np.all(mir_data.sp_read["sparedbl1"] == 0.0)

    assert np.all(mir_data.sp_read["sparedbl2"] == 0.0)

    assert np.all(mir_data.sp_read["sparedbl3"] == 0.0)

    assert np.all(mir_data.sp_read["sparedbl4"] == 0.0)

    assert np.all(mir_data.sp_read["sparedbl5"] == 0.0)

    assert np.all(mir_data.sp_read["sparedbl6"] == 0.0)


def test_mir_remember_me_sch_read(mir_data_object):
    """
    Mir sch_read checker.

    Make sure that certain values in the sch_read file of the test data set match what
    we know to be 'true' at the time of observations.
    """
    mir_data = mir_data_object

    # Now check sch_read related values. Thanks to a glitch in the data recorder,
    # all of the pseudo-cont values are the same
    assert np.all(mir_data.raw_scale_fac[0::5] == [-26] * 4)

    assert (
        np.array(mir_data.raw_data[0::5]).flatten().tolist()
        == [-4302, -20291, -5261, -21128, -4192, -19634, -4999, -16346] * 4
    )
