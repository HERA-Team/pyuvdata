# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for uvdata object."""
import pytest
import os
import copy
import itertools
import h5py

import numpy as np
from astropy.time import Time
from astropy.coordinates import Angle, EarthLocation
from astropy.utils import iers

from pyuvdata import UVData, UVCal
import pyuvdata.utils as uvutils
import pyuvdata.tests as uvtest
from pyuvdata.data import DATA_PATH

# needed for multifile read error test
from pyuvdata.uvdata.tests.test_mwa_corr_fits import filelist as mwa_corr_files
from pyuvdata.uvdata.tests.test_fhd import testfiles as fhd_files


from collections import Counter


@pytest.fixture(scope="function")
def uvdata_props():
    required_parameters = [
        "_data_array",
        "_nsample_array",
        "_flag_array",
        "_Ntimes",
        "_Nbls",
        "_Nblts",
        "_Nfreqs",
        "_Npols",
        "_Nspws",
        "_uvw_array",
        "_time_array",
        "_ant_1_array",
        "_ant_2_array",
        "_lst_array",
        "_baseline_array",
        "_freq_array",
        "_polarization_array",
        "_spw_array",
        "_integration_time",
        "_channel_width",
        "_object_name",
        "_telescope_name",
        "_instrument",
        "_telescope_location",
        "_history",
        "_vis_units",
        "_Nants_data",
        "_Nants_telescope",
        "_antenna_names",
        "_antenna_numbers",
        "_antenna_positions",
        "_phase_type",
        "_flex_spw",
        "_multi_object",
    ]

    required_properties = [
        "data_array",
        "nsample_array",
        "flag_array",
        "Ntimes",
        "Nbls",
        "Nblts",
        "Nfreqs",
        "Npols",
        "Nspws",
        "uvw_array",
        "time_array",
        "ant_1_array",
        "ant_2_array",
        "lst_array",
        "baseline_array",
        "freq_array",
        "polarization_array",
        "spw_array",
        "integration_time",
        "channel_width",
        "object_name",
        "telescope_name",
        "instrument",
        "telescope_location",
        "history",
        "vis_units",
        "Nants_data",
        "Nants_telescope",
        "antenna_names",
        "antenna_numbers",
        "antenna_positions",
        "phase_type",
        "flex_spw",
        "multi_object",
    ]

    extra_parameters = [
        "_extra_keywords",
        "_x_orientation",
        "_antenna_diameters",
        "_blt_order",
        "_gst0",
        "_rdate",
        "_earth_omega",
        "_dut1",
        "_timesys",
        "_uvplane_reference_time",
        "_phase_center_ra",
        "_phase_center_dec",
        "_phase_center_app_ra",
        "_phase_center_app_dec",
        "_phase_center_epoch",
        "_phase_center_frame",
        "_Nobjects",
        "_object_dict",
        "_object_id_array",
        "_eq_coeffs",
        "_eq_coeffs_convention",
        "_flex_spw_id_array",
    ]

    extra_properties = [
        "extra_keywords",
        "x_orientation",
        "antenna_diameters",
        "blt_order",
        "gst0",
        "rdate",
        "earth_omega",
        "dut1",
        "timesys",
        "uvplane_reference_time",
        "phase_center_ra",
        "phase_center_dec",
        "phase_center_app_ra",
        "phase_center_app_dec",
        "phase_center_epoch",
        "phase_center_frame",
        "Nobjects",
        "object_dict",
        "object_id_array",
        "eq_coeffs",
        "eq_coeffs_convention",
        "flex_spw_id_array",
    ]

    other_properties = [
        "telescope_location_lat_lon_alt",
        "telescope_location_lat_lon_alt_degrees",
        "phase_center_ra_degrees",
        "phase_center_dec_degrees",
        "pyuvdata_version_str",
    ]

    uv_object = UVData()

    class DataHolder:
        def __init__(
            self,
            uv_object,
            required_parameters,
            required_properties,
            extra_parameters,
            extra_properties,
            other_properties,
        ):
            self.uv_object = uv_object
            self.required_parameters = required_parameters
            self.required_properties = required_properties
            self.extra_parameters = extra_parameters
            self.extra_properties = extra_properties
            self.other_properties = other_properties

    uvdata_props = DataHolder(
        uv_object,
        required_parameters,
        required_properties,
        extra_parameters,
        extra_properties,
        other_properties,
    )
    # yields the data we need but will continue to the del call after tests
    yield uvdata_props

    # some post-test object cleanup
    del uvdata_props

    return


@pytest.fixture(scope="session")
def hera_uvh5_main():
    # read in test file for the resampling in time functions
    uv_object = UVData()
    testfile = os.path.join(DATA_PATH, "zen.2458661.23480.HH.uvh5")
    uv_object.read(testfile)

    # During some investigating, I learned that the uvw_array here appears to be
    # flipped. For the sake of not having to fold in this correction into a dozen
    # or so tests, correcting it here.
    uv_object.uvw_array *= -1.0

    yield uv_object

    # cleanup
    del uv_object

    return


@pytest.fixture(scope="function")
def hera_uvh5(hera_uvh5_main):
    # read in test file for the resampling in time functions
    uv_object = hera_uvh5_main.copy()

    yield uv_object

    # cleanup
    del uv_object

    return


@pytest.fixture(scope="session")
def paper_uvh5_main():
    # read in test file for the resampling in time functions
    uv_object = UVData()
    uvh5_file = os.path.join(DATA_PATH, "zen.2456865.60537.xy.uvcRREAA.uvh5")
    uv_object.read_uvh5(uvh5_file)

    yield uv_object

    # cleanup
    del uv_object

    return


@pytest.fixture(scope="function")
def paper_uvh5(paper_uvh5_main):
    # read in test file for the resampling in time functions
    uv_object = paper_uvh5_main.copy()

    yield uv_object

    # cleanup
    del uv_object

    return


@pytest.fixture(scope="session")
def bda_test_file_main():
    # read in test file for BDA-like data
    uv_object = UVData()
    testfile = os.path.join(DATA_PATH, "simulated_bda_file.uvh5")
    uv_object.read(testfile)

    yield uv_object

    # cleanup
    del uv_object

    return


@pytest.fixture(scope="function")
def bda_test_file(bda_test_file_main):
    # read in test file for BDA-like data
    uv_object = bda_test_file_main.copy()

    yield uv_object

    # cleanup
    del uv_object

    return


@pytest.fixture(scope="function")
def uvdata_data(casa_uvfits):
    uv_object = casa_uvfits

    class DataHolder:
        def __init__(self, uv_object):
            self.uv_object = uv_object
            self.uv_object2 = uv_object.copy()

    uvdata_data = DataHolder(uv_object)
    # yields the data we need but will continue to the del call after tests
    yield uvdata_data

    # some post-test object cleanup
    del uvdata_data

    return


@pytest.fixture(scope="function")
def uvdata_baseline():
    uv_object = UVData()
    uv_object.Nants_telescope = 128
    uv_object2 = UVData()
    uv_object2.Nants_telescope = 2049

    class DataHolder:
        def __init__(self, uv_object, uv_object2):
            self.uv_object = uv_object
            self.uv_object2 = uv_object2

    uvdata_baseline = DataHolder(uv_object, uv_object2)

    # yields the data we need but will continue to the del call after tests
    yield uvdata_baseline

    # Post test clean-up
    del uvdata_baseline
    return


@pytest.fixture(scope="session")
def set_uvws_main(hera_uvh5_main):
    uv1 = hera_uvh5_main.copy()
    # uvws in the file are wrong. reset them.
    uv1.set_uvws_from_antenna_positions()

    yield uv1

    del uv1

    return


@pytest.fixture
def uv1_2_set_uvws(set_uvws_main):
    uv1 = set_uvws_main.copy()
    uv2 = set_uvws_main.copy()

    yield uv1, uv2

    del uv1, uv2

    return


@pytest.fixture()
def uv_phase_time_split(uv1_2_set_uvws):
    uv_phase, uv_raw = uv1_2_set_uvws

    uv_phase.reorder_blts(order="time", minor_order="baseline")
    uv_raw.reorder_blts(order="time", minor_order="baseline")

    uv_phase.phase(ra=0, dec=0, epoch="J2000", use_ant_pos=True)
    times = np.unique(uv_phase.time_array)
    time_set_1, time_set_2 = times[::2], times[1::2]

    uv_phase_1 = uv_phase.select(times=time_set_1, inplace=False)
    uv_phase_2 = uv_phase.select(times=time_set_2, inplace=False)

    uv_raw_1 = uv_raw.select(times=time_set_1, inplace=False)
    uv_raw_2 = uv_raw.select(times=time_set_2, inplace=False)

    yield uv_phase_1, uv_phase_2, uv_phase, uv_raw_1, uv_raw_2, uv_raw

    del uv_phase_1, uv_phase_2, uv_raw_1, uv_raw_2, uv_phase, uv_raw


def test_parameter_iter(uvdata_props):
    """Test expected parameters."""
    all_params = []
    for prop in uvdata_props.uv_object:
        all_params.append(prop)
    for a in uvdata_props.required_parameters + uvdata_props.extra_parameters:
        assert a in all_params, (
            "expected attribute " + a + " not returned in object iterator"
        )


def test_required_parameter_iter(uvdata_props):
    """Test expected required parameters."""
    # at first it's a metadata_only object, so need to modify required_parameters
    required = []
    for prop in uvdata_props.uv_object.required():
        required.append(prop)
    expected_required = copy.copy(uvdata_props.required_parameters)
    expected_required.remove("_data_array")
    expected_required.remove("_nsample_array")
    expected_required.remove("_flag_array")
    for a in expected_required:
        assert a in required, (
            "expected attribute " + a + " not returned in required iterator"
        )

    uvdata_props.uv_object.data_array = 1
    uvdata_props.uv_object.nsample_array = 1
    uvdata_props.uv_object.flag_array = 1
    required = []
    for prop in uvdata_props.uv_object.required():
        required.append(prop)
    for a in uvdata_props.required_parameters:
        assert a in required, (
            "expected attribute " + a + " not returned in required iterator"
        )


def test_extra_parameter_iter(uvdata_props):
    """Test expected optional parameters."""
    extra = []
    for prop in uvdata_props.uv_object.extra():
        extra.append(prop)
    for a in uvdata_props.extra_parameters:
        assert a in extra, "expected attribute " + a + " not returned in extra iterator"


def test_unexpected_parameters(uvdata_props):
    """Test for extra parameters."""
    expected_parameters = (
        uvdata_props.required_parameters + uvdata_props.extra_parameters
    )
    attributes = [i for i in uvdata_props.uv_object.__dict__.keys() if i[0] == "_"]
    for a in attributes:
        assert a in expected_parameters, (
            "unexpected parameter " + a + " found in UVData"
        )


def test_unexpected_attributes(uvdata_props):
    """Test for extra attributes."""
    expected_attributes = (
        uvdata_props.required_properties
        + uvdata_props.extra_properties
        + uvdata_props.other_properties
    )
    attributes = [i for i in uvdata_props.uv_object.__dict__.keys() if i[0] != "_"]
    for a in attributes:
        assert a in expected_attributes, (
            "unexpected attribute " + a + " found in UVData"
        )


def test_properties(uvdata_props):
    """Test that properties can be get and set properly."""
    prop_dict = dict(
        list(
            zip(
                uvdata_props.required_properties + uvdata_props.extra_properties,
                uvdata_props.required_parameters + uvdata_props.extra_parameters,
            )
        )
    )
    for k, v in prop_dict.items():
        rand_num = np.random.rand()
        setattr(uvdata_props.uv_object, k, rand_num)
        this_param = getattr(uvdata_props.uv_object, v)
        try:
            assert rand_num == this_param.value
        except AssertionError:
            print("setting {prop_name} to a random number failed".format(prop_name=k))
            raise


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_metadata_only_property(uvdata_data):
    uvdata_data.uv_object.data_array = None
    assert uvdata_data.uv_object.metadata_only is False
    pytest.raises(ValueError, uvdata_data.uv_object.check)
    uvdata_data.uv_object.flag_array = None
    assert uvdata_data.uv_object.metadata_only is False
    pytest.raises(ValueError, uvdata_data.uv_object.check)
    uvdata_data.uv_object.nsample_array = None
    assert uvdata_data.uv_object.metadata_only is True


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_equality(uvdata_data):
    """Basic equality test."""
    assert uvdata_data.uv_object == uvdata_data.uv_object


@pytest.mark.filterwarnings("ignore:Telescope location derived from obs")
@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_check(uvdata_data):
    """Test simple check function."""
    assert uvdata_data.uv_object.check()
    # Check variety of special cases
    uvdata_data.uv_object.Nants_data += 1
    with pytest.raises(
        ValueError,
        match=(
            "Nants_data must be equal to the number of unique values in "
            "ant_1_array and ant_2_array"
        ),
    ):
        uvdata_data.uv_object.check()
    uvdata_data.uv_object.Nants_data -= 1
    uvdata_data.uv_object.Nbls += 1
    with pytest.raises(
        ValueError,
        match=(
            "Nbls must be equal to the number of unique baselines in the data_array"
        ),
    ):
        uvdata_data.uv_object.check()
    uvdata_data.uv_object.Nbls -= 1
    uvdata_data.uv_object.Ntimes += 1
    with pytest.raises(
        ValueError,
        match=("Ntimes must be equal to the number of unique times in the time_array"),
    ):
        uvdata_data.uv_object.check()
    uvdata_data.uv_object.Ntimes -= 1

    with pytest.raises(
        ValueError,
        match=(
            "The uvw_array does not match the expected values given the antenna "
            "positions."
        ),
    ):
        uvdata_data.uv_object.check(strict_uvw_antpos_check=True)

    # Check case where all data is autocorrelations
    # Currently only test files that have autos are fhd files
    testdir = os.path.join(DATA_PATH, "fhd_vis_data/")
    file_list = [
        testdir + "1061316296_flags.sav",
        testdir + "1061316296_vis_XX.sav",
        testdir + "1061316296_params.sav",
        testdir + "1061316296_layout.sav",
        testdir + "1061316296_settings.txt",
    ]

    uvdata_data.uv_object.read_fhd(file_list)

    uvdata_data.uv_object.select(
        blt_inds=np.where(
            uvdata_data.uv_object.ant_1_array == uvdata_data.uv_object.ant_2_array
        )[0]
    )
    assert uvdata_data.uv_object.check()

    # test auto and cross corr uvw_array
    uvd = UVData()
    uvd.read_uvh5(os.path.join(DATA_PATH, "zen.2457698.40355.xx.HH.uvcA.uvh5"))
    autos = np.isclose(uvd.ant_1_array - uvd.ant_2_array, 0.0)
    auto_inds = np.where(autos)[0]
    cross_inds = np.where(~autos)[0]

    # make auto have non-zero uvw coords, assert ValueError
    uvd.uvw_array[auto_inds[0], 0] = 0.1
    with pytest.raises(
        ValueError,
        match=("Some auto-correlations have non-zero uvw_array coordinates."),
    ):
        uvd.check()

    # make cross have |uvw| zero, assert ValueError
    uvd.read_uvh5(os.path.join(DATA_PATH, "zen.2457698.40355.xx.HH.uvcA.uvh5"))
    uvd.uvw_array[cross_inds[0]][:] = 0.0
    with pytest.raises(
        ValueError,
        match=("Some cross-correlations have near-zero uvw_array magnitudes."),
    ):
        uvd.check()


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_nants_data_telescope_larger(uvdata_data):
    # make sure it's okay for Nants_telescope to be strictly greater than Nants_data
    uvdata_data.uv_object.Nants_telescope += 1
    # add dummy information for "new antenna" to pass object check
    uvdata_data.uv_object.antenna_names = np.concatenate(
        (uvdata_data.uv_object.antenna_names, ["dummy_ant"])
    )
    uvdata_data.uv_object.antenna_numbers = np.concatenate(
        (uvdata_data.uv_object.antenna_numbers, [20])
    )
    uvdata_data.uv_object.antenna_positions = np.concatenate(
        (uvdata_data.uv_object.antenna_positions, np.zeros((1, 3))), axis=0
    )
    assert uvdata_data.uv_object.check()


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_ant1_array_not_in_antnums(uvdata_data):
    # make sure an error is raised if antennas in ant_1_array not in antenna_numbers
    # remove antennas from antenna_names & antenna_numbers by hand
    uvdata_data.uv_object.antenna_names = uvdata_data.uv_object.antenna_names[1:]
    uvdata_data.uv_object.antenna_numbers = uvdata_data.uv_object.antenna_numbers[1:]
    uvdata_data.uv_object.antenna_positions = uvdata_data.uv_object.antenna_positions[
        1:, :
    ]
    uvdata_data.uv_object.Nants_telescope = uvdata_data.uv_object.antenna_numbers.size
    with pytest.raises(ValueError) as cm:
        uvdata_data.uv_object.check()
    assert str(cm.value).startswith(
        "All antennas in ant_1_array must be in antenna_numbers"
    )


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_ant2_array_not_in_antnums(uvdata_data):
    # make sure an error is raised if antennas in ant_2_array not in antenna_numbers
    # remove antennas from antenna_names & antenna_numbers by hand
    uvobj = uvdata_data.uv_object
    uvobj.antenna_names = uvobj.antenna_names[:-1]
    uvobj.antenna_numbers = uvobj.antenna_numbers[:-1]
    uvobj.antenna_positions = uvobj.antenna_positions[:-1]
    uvobj.Nants_telescope = uvobj.antenna_numbers.size
    with pytest.raises(ValueError) as cm:
        uvobj.check()
    assert str(cm.value).startswith(
        "All antennas in ant_2_array must be in antenna_numbers"
    )


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_converttofiletype(uvdata_data):
    fhd_obj = uvdata_data.uv_object._convert_to_filetype("fhd")
    uvdata_data.uv_object._convert_from_filetype(fhd_obj)
    assert uvdata_data.uv_object == uvdata_data.uv_object2

    with pytest.raises(ValueError) as cm:
        uvdata_data.uv_object._convert_to_filetype("foo")
    assert str(cm.value).startswith(
        "filetype must be uvfits, mir, miriad, fhd, or uvh5"
    )


def test_baseline_to_antnums(uvdata_baseline):
    """Test baseline to antnum conversion for 256 & larger conventions."""
    assert uvdata_baseline.uv_object.baseline_to_antnums(67585) == (0, 0)
    with pytest.raises(Exception) as cm:
        uvdata_baseline.uv_object2.baseline_to_antnums(67585)
    assert str(cm.value).startswith(
        "error Nants={Nants}>2048"
        " not supported".format(Nants=uvdata_baseline.uv_object2.Nants_telescope)
    )

    ant_pairs = [(10, 20), (280, 310)]
    for pair in ant_pairs:
        if np.max(np.array(pair)) < 255:
            bl = uvdata_baseline.uv_object.antnums_to_baseline(
                pair[0], pair[1], attempt256=True
            )
            ant_pair_out = uvdata_baseline.uv_object.baseline_to_antnums(bl)
            assert pair == ant_pair_out

        bl = uvdata_baseline.uv_object.antnums_to_baseline(
            pair[0], pair[1], attempt256=False
        )
        ant_pair_out = uvdata_baseline.uv_object.baseline_to_antnums(bl)
        assert pair == ant_pair_out


def test_baseline_to_antnums_vectorized(uvdata_baseline):
    """Test vectorized antnum to baseline conversion."""
    ant_1 = [10, 280]
    ant_2 = [20, 310]
    baseline_array = uvdata_baseline.uv_object.antnums_to_baseline(ant_1, ant_2)
    assert np.array_equal(baseline_array, [88085, 641335])
    ant_1_out, ant_2_out = uvdata_baseline.uv_object.baseline_to_antnums(
        baseline_array.tolist()
    )
    assert np.array_equal(ant_1, ant_1_out)
    assert np.array_equal(ant_2, ant_2_out)


def test_antnums_to_baselines(uvdata_baseline):
    """Test antums to baseline conversion for 256 & larger conventions."""
    assert uvdata_baseline.uv_object.antnums_to_baseline(0, 0) == 67585
    assert uvdata_baseline.uv_object.antnums_to_baseline(257, 256) == 594177
    assert uvdata_baseline.uv_object.baseline_to_antnums(594177) == (257, 256)
    # Check attempt256
    assert uvdata_baseline.uv_object.antnums_to_baseline(0, 0, attempt256=True) == 257
    assert uvdata_baseline.uv_object.antnums_to_baseline(257, 256) == 594177
    with uvtest.check_warnings(UserWarning, "found > 256 antennas"):
        uvdata_baseline.uv_object.antnums_to_baseline(257, 256, attempt256=True)
    pytest.raises(Exception, uvdata_baseline.uv_object2.antnums_to_baseline, 0, 0)
    # check a len-1 array returns as an array
    ant1 = np.array([1])
    ant2 = np.array([2])
    assert isinstance(
        uvdata_baseline.uv_object.antnums_to_baseline(ant1, ant2), np.ndarray
    )


def test_known_telescopes():
    """Test known_telescopes method returns expected results."""
    uv_object = UVData()
    astropy_sites = EarthLocation.get_site_names()
    while "" in astropy_sites:
        astropy_sites.remove("")

    # Using set to drop duplicate entries
    known_telescopes = list(set(astropy_sites + ["PAPER", "HERA", "SMA"]))
    # calling np.sort().tolist() because [].sort() acts inplace and returns None
    # Before test had None == None
    assert (
        np.sort(known_telescopes).tolist()
        == np.sort(uv_object.known_telescopes()).tolist()
    )


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_hera_diameters(paper_uvh5):
    uv_in = paper_uvh5

    uv_in.telescope_name = "HERA"
    with uvtest.check_warnings(
        UserWarning, "antenna_diameters is not set. Using known values for HERA."
    ):
        uv_in.set_telescope_params()

    assert uv_in.telescope_name == "HERA"
    assert uv_in.antenna_diameters is not None

    uv_in.check()


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_generic_read():
    uv_in = UVData()
    uvfits_file = os.path.join(DATA_PATH, "day2_TDEM0003_10s_norx_1src_1spw.uvfits")
    uv_in.read(uvfits_file, read_data=False)
    unique_times = np.unique(uv_in.time_array)

    pytest.raises(
        ValueError,
        uv_in.read,
        uvfits_file,
        times=unique_times[0:2],
        time_range=[unique_times[0], unique_times[1]],
    )

    pytest.raises(
        ValueError,
        uv_in.read,
        uvfits_file,
        antenna_nums=uv_in.antenna_numbers[0],
        antenna_names=uv_in.antenna_names[1],
    )

    pytest.raises(ValueError, uv_in.read, "foo")


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "phase_kwargs",
    [
        {"ra": 0.0, "dec": 0.0, "epoch": "J2000"},
        {"ra": Angle("5d").rad, "dec": Angle("30d").rad, "phase_frame": "gcrs"},
        {
            "ra": Angle("180d").rad,
            "dec": Angle("90d").rad,
            "epoch": Time("2010-01-01T00:00:00", format="isot", scale="utc"),
        },
    ],
)
def test_phase_unphase_hera(uv1_2_set_uvws, phase_kwargs):
    """
    Read in drift data, phase to an RA/DEC, unphase and check for object equality.
    """
    uv1, uv_raw = uv1_2_set_uvws
    uv1.phase(**phase_kwargs)
    uv1.unphase_to_drift()
    # check that phase + unphase gets back to raw
    assert uv_raw == uv1


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_phase_unphase_hera_one_bl(uv1_2_set_uvws):
    uv_phase, uv_raw = uv1_2_set_uvws
    # check that phase + unphase work with one baseline
    uv_raw_small = uv_raw.select(blt_inds=[0], inplace=False)
    uv_phase_small = uv_raw_small.copy()
    uv_phase_small.phase(Angle("23h").rad, Angle("15d").rad)
    uv_phase_small.unphase_to_drift()
    assert uv_raw_small == uv_phase_small


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_phase_unphase_hera_antpos(uv1_2_set_uvws):
    uv_phase, uv_raw = uv1_2_set_uvws
    # check that they match if you phase & unphase using antenna locations
    # first replace the uvws with the right values
    antenna_enu = uvutils.ENU_from_ECEF(
        (uv_raw.antenna_positions + uv_raw.telescope_location),
        *uv_raw.telescope_location_lat_lon_alt,
    )
    uvw_calc = np.zeros_like(uv_raw.uvw_array)
    unique_times, unique_inds = np.unique(uv_raw.time_array, return_index=True)
    for ind, jd in enumerate(unique_times):
        inds = np.where(uv_raw.time_array == jd)[0]
        for bl_ind in inds:
            wh_ant1 = np.where(uv_raw.antenna_numbers == uv_raw.ant_1_array[bl_ind])
            ant1_index = wh_ant1[0][0]
            wh_ant2 = np.where(uv_raw.antenna_numbers == uv_raw.ant_2_array[bl_ind])
            ant2_index = wh_ant2[0][0]
            uvw_calc[bl_ind, :] = (
                antenna_enu[ant2_index, :] - antenna_enu[ant1_index, :]
            )

    uv_raw_new = uv_raw.copy()
    uv_raw_new.uvw_array = uvw_calc
    uv_phase.phase(0.0, 0.0, epoch="J2000", use_ant_pos=True)
    uv_phase2 = uv_raw_new.copy()
    uv_phase2.phase(0.0, 0.0, epoch="J2000")

    # The uvw's only agree to ~1mm. should they be better?
    assert np.allclose(uv_phase2.uvw_array, uv_phase.uvw_array, atol=1e-3)
    # the data array are just multiplied by the w's for phasing, so a difference
    # at the 1e-3 level makes the data array different at that level too.
    # -> change the tolerance on data_array for this test
    uv_phase2._data_array.tols = (0, 1e-3 * np.amax(np.abs(uv_phase2.data_array)))
    assert uv_phase2 == uv_phase

    # check that phase + unphase gets back to raw using antpos
    uv_phase.unphase_to_drift(use_ant_pos=True)
    assert uv_raw_new == uv_phase


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_phase_hera_zenith_timestamp_minimal_changes(uv1_2_set_uvws):
    uv_phase, uv_raw = uv1_2_set_uvws
    # check that phasing to zenith with one timestamp has small changes
    # (it won't be identical because of precession/nutation changing the
    # coordinate axes)
    # use gcrs rather than icrs to reduce differences (don't include abberation)
    uv_raw_small = uv_raw.select(times=uv_raw.time_array[0], inplace=False)
    uv_phase_simple_small = uv_raw_small.copy()
    uv_phase_simple_small.phase_to_time(
        time=Time(uv_raw.time_array[0], format="jd"), phase_frame="gcrs"
    )

    # it's unclear to me how close this should be...
    assert np.allclose(
        uv_phase_simple_small.uvw_array, uv_raw_small.uvw_array, atol=1e-1
    )


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_phase_to_time_jd_input(uv1_2_set_uvws):
    uv_phase, uv_raw = uv1_2_set_uvws
    uv_phase.phase_to_time(uv_raw.time_array[0])
    uv_phase.unphase_to_drift()
    assert uv_phase == uv_raw


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_phase_to_time_error(uv1_2_set_uvws):
    uv_phase, uv_raw = uv1_2_set_uvws
    # check error if not passing a Time object to phase_to_time
    with pytest.raises(TypeError) as cm:
        uv_phase.phase_to_time("foo")
    assert str(cm.value).startswith("time must be an astropy.time.Time object")


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_unphase_drift_data_error(uv1_2_set_uvws):
    uv_phase, uv_raw = uv1_2_set_uvws
    # check error if not passing a Time object to phase_to_time
    with pytest.raises(ValueError) as cm:
        uv_phase.unphase_to_drift()
    assert str(cm.value).startswith("The data is already drift scanning;")


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "phase_func,phase_kwargs,err_msg",
    [
        (
            "unphase_to_drift",
            {},
            "The phasing type of the data is unknown. Set the phase_type",
        ),
        (
            "phase",
            {"ra": 0, "dec": 0, "epoch": "J2000", "allow_rephase": False},
            "The phasing type of the data is unknown. Set the phase_type",
        ),
        (
            "phase_to_time",
            {"time": 0, "allow_rephase": False},
            "The phasing type of the data is unknown. Set the phase_type",
        ),
    ],
)
def test_unknown_phase_unphase_hera_errors(
    uv1_2_set_uvws, phase_func, phase_kwargs, err_msg
):
    uv_phase, uv_raw = uv1_2_set_uvws

    # Set phase type to unkown on some tests, ignore on others.
    uv_phase._set_unknown_phase_type()
    # if this is phase_to_time, use this index set in the dictionary and
    # assign the value of the time_array associated with that index
    # this is a little hacky, but we cannot acces uv_phase.time_array in the
    # parametrize
    if phase_func == "phase_to_time":
        phase_kwargs["time"] = uv_phase.time_array[phase_kwargs["time"]]

    with pytest.raises(ValueError) as cm:
        getattr(uv_phase, phase_func)(**phase_kwargs)
    assert str(cm.value).startswith(err_msg)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "phase_func,phase_kwargs,err_msg",
    [
        (
            "phase",
            {"ra": 0, "dec": 0, "epoch": "J2000", "allow_rephase": False},
            "The data is already phased;",
        ),
        (
            "phase_to_time",
            {"time": 0, "allow_rephase": False},
            "The data is already phased;",
        ),
    ],
)
def test_phase_rephase_hera_errors(uv1_2_set_uvws, phase_func, phase_kwargs, err_msg):
    uv_phase, uv_raw = uv1_2_set_uvws

    uv_phase.phase(0.0, 0.0, epoch="J2000")
    # if this is phase_to_time, use this index set in the dictionary and
    # assign the value of the time_array associated with that index
    # this is a little hacky, but we cannot acces uv_phase.time_array in the
    # parametrize
    if phase_func == "phase_to_time":
        phase_kwargs["time"] = uv_phase.time_array[int(phase_kwargs["time"])]

    with pytest.raises(ValueError) as cm:
        getattr(uv_phase, phase_func)(**phase_kwargs)
    assert str(cm.value).startswith(err_msg)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_phase_unphase_hera_bad_frame(uv1_2_set_uvws):
    uv_phase, uv_raw = uv1_2_set_uvws
    # check errors when trying to phase to an unsupported frame
    with pytest.raises(ValueError) as cm:
        uv_phase.phase(0.0, 0.0, epoch="J2000", phase_frame="cirs", use_old_phase=True)
    assert str(cm.value).startswith("phase_frame can only be set to icrs or gcrs.")


# We're using the old phase method here since these values were all derived using that
# method, so we'll just filter out those warnings now.
@pytest.mark.filterwarnings("ignore:The original `phase` method will be deprecated in")
def test_phasing():
    """Use MWA files phased to 2 different places to test phasing."""
    file1 = os.path.join(DATA_PATH, "1133866760.uvfits")
    file2 = os.path.join(DATA_PATH, "1133866760_rephase.uvfits")
    uvd1 = UVData()
    uvd2 = UVData()
    uvd1.read_uvfits(file1)
    uvd2.read_uvfits(file2)

    uvd1_drift = uvd1.copy()
    uvd1_drift.unphase_to_drift(phase_frame="gcrs", use_old_phase=True)
    uvd1_drift_antpos = uvd1.copy()
    uvd1_drift_antpos.unphase_to_drift(
        phase_frame="gcrs", use_ant_pos=True, use_old_phase=True,
    )

    uvd2_drift = uvd2.copy()
    uvd2_drift.unphase_to_drift(phase_frame="gcrs", use_old_phase=True)
    uvd2_drift_antpos = uvd2.copy()
    uvd2_drift_antpos.unphase_to_drift(
        phase_frame="gcrs", use_ant_pos=True, use_old_phase=True,
    )

    # the tolerances here are empirical -- based on what was seen in the
    # external phasing test. See the phasing memo in docs/references for
    # details.
    assert np.allclose(uvd1_drift.uvw_array, uvd2_drift.uvw_array, atol=2e-2)
    assert np.allclose(uvd1_drift_antpos.uvw_array, uvd2_drift_antpos.uvw_array)

    uvd2_rephase = uvd2.copy()
    uvd2_rephase.phase(
        uvd1.phase_center_ra,
        uvd1.phase_center_dec,
        uvd1.phase_center_epoch,
        orig_phase_frame="gcrs",
        phase_frame="gcrs",
        use_old_phase=True,
    )
    uvd2_rephase_antpos = uvd2.copy()
    uvd2_rephase_antpos.phase(
        uvd1.phase_center_ra,
        uvd1.phase_center_dec,
        uvd1.phase_center_epoch,
        orig_phase_frame="gcrs",
        phase_frame="gcrs",
        use_ant_pos=True,
        use_old_phase=True,
    )

    # the tolerances here are empirical -- based on what was seen in the
    # external phasing test. See the phasing memo in docs/references for
    # details.
    assert np.allclose(uvd1.uvw_array, uvd2_rephase.uvw_array, atol=2e-2)
    assert np.allclose(uvd1.uvw_array, uvd2_rephase_antpos.uvw_array, atol=5e-3)

    # rephase the drift objects to the original pointing and verify that they
    # match
    uvd1_drift.phase(
        uvd1.phase_center_ra,
        uvd1.phase_center_dec,
        uvd1.phase_center_epoch,
        phase_frame="gcrs",
        use_old_phase=True,
    )
    uvd1_drift_antpos.phase(
        uvd1.phase_center_ra,
        uvd1.phase_center_dec,
        uvd1.phase_center_epoch,
        phase_frame="gcrs",
        use_ant_pos=True,
        use_old_phase=True,
    )

    # the tolerances here are empirical -- caused by one unphase/phase cycle.
    # the antpos-based phasing differences are based on what was seen in the
    # external phasing test. See the phasing memo in docs/references for
    # details.
    assert np.allclose(uvd1.uvw_array, uvd1_drift.uvw_array, atol=1e-4)
    assert np.allclose(uvd1.uvw_array, uvd1_drift_antpos.uvw_array, atol=5e-3)

    uvd2_drift.phase(
        uvd2.phase_center_ra,
        uvd2.phase_center_dec,
        uvd2.phase_center_epoch,
        phase_frame="gcrs",
        use_old_phase=True,
    )
    uvd2_drift_antpos.phase(
        uvd2.phase_center_ra,
        uvd2.phase_center_dec,
        uvd2.phase_center_epoch,
        phase_frame="gcrs",
        use_ant_pos=True,
        use_old_phase=True,
    )

    # the tolerances here are empirical -- caused by one unphase/phase cycle.
    # the antpos-based phasing differences are based on what was seen in the
    # external phasing test. See the phasing memo in docs/references for
    # details.
    assert np.allclose(uvd2.uvw_array, uvd2_drift.uvw_array, atol=1e-4)
    assert np.allclose(uvd2.uvw_array, uvd2_drift_antpos.uvw_array, atol=2e-2)


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_set_phase_unknown(casa_uvfits):
    uv_object = casa_uvfits

    uv_object._set_unknown_phase_type()
    assert uv_object.phase_type == "unknown"
    assert not uv_object._phase_center_epoch.required
    assert not uv_object._phase_center_ra.required
    assert not uv_object._phase_center_dec.required
    assert uv_object.check()


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_select_blts(paper_uvh5):
    uv_object = paper_uvh5
    old_history = uv_object.history
    # fmt: off
    blt_inds = np.array([172, 182, 132, 227, 144, 44, 16, 104, 385, 134, 326, 140, 116,
                         218, 178, 391, 111, 276, 274, 308, 38, 64, 317, 76, 239, 246,
                         34, 39, 83, 184, 208, 60, 374, 295, 118, 337, 261, 21, 375,
                         396, 355, 187, 95, 122, 186, 113, 260, 264, 156, 13, 228, 291,
                         302, 72, 137, 216, 299, 341, 207, 256, 223, 250, 268, 147, 73,
                         32, 142, 383, 221, 203, 258, 286, 324, 265, 170, 236, 8, 275,
                         304, 117, 29, 167, 15, 388, 171, 82, 322, 248, 160, 85, 66,
                         46, 272, 328, 323, 152, 200, 119, 359, 23, 363, 56, 219, 257,
                         11, 307, 336, 289, 136, 98, 37, 163, 158, 80, 125, 40, 298,
                         75, 320, 74, 57, 346, 121, 129, 332, 238, 93, 18, 330, 339,
                         381, 234, 176, 22, 379, 199, 266, 100, 90, 292, 205, 58, 222,
                         350, 109, 273, 191, 368, 88, 101, 65, 155, 2, 296, 306, 398,
                         369, 378, 254, 67, 249, 102, 348, 392, 20, 28, 169, 262, 269,
                         287, 86, 300, 143, 177, 42, 290, 284, 123, 189, 175, 97, 340,
                         242, 342, 331, 282, 235, 344, 63, 115, 78, 30, 226, 157, 133,
                         71, 35, 212, 333])
    # fmt: on
    selected_data = uv_object.data_array[np.sort(blt_inds), :, :, :]

    uv_object2 = uv_object.copy()
    uv_object2.select(blt_inds=blt_inds)
    assert len(blt_inds) == uv_object2.Nblts

    # verify that histories are different
    assert not uvutils._check_histories(old_history, uv_object2.history)

    assert uvutils._check_histories(
        old_history + "  Downselected to " "specific baseline-times using pyuvdata.",
        uv_object2.history,
    )

    assert np.all(selected_data == uv_object2.data_array)

    # check that it also works with higher dimension array
    uv_object2 = uv_object.copy()
    uv_object2.select(blt_inds=blt_inds[np.newaxis, :])
    assert len(blt_inds) == uv_object2.Nblts

    assert uvutils._check_histories(
        old_history + "  Downselected to " "specific baseline-times using pyuvdata.",
        uv_object2.history,
    )
    assert np.all(selected_data == uv_object2.data_array)

    # check that just doing the metadata works properly
    uv_object3 = uv_object.copy()
    uv_object3.data_array = None
    uv_object3.flag_array = None
    uv_object3.nsample_array = None
    assert uv_object3.metadata_only is True
    uv_object4 = uv_object3.select(blt_inds=blt_inds, inplace=False)
    for param in uv_object4:
        param_name = getattr(uv_object4, param).name
        if param_name not in ["data_array", "flag_array", "nsample_array"]:
            assert getattr(uv_object4, param) == getattr(uv_object2, param)
        else:
            assert getattr(uv_object4, param_name) is None

    # also check with inplace=True
    uv_object3.select(blt_inds=blt_inds)
    assert uv_object3 == uv_object4

    # check for errors associated with out of bounds indices
    pytest.raises(ValueError, uv_object.select, blt_inds=np.arange(-10, -5))
    pytest.raises(
        ValueError,
        uv_object.select,
        blt_inds=np.arange(uv_object.Nblts + 1, uv_object.Nblts + 10),
    )


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_select_antennas(casa_uvfits):
    uv_object = casa_uvfits
    old_history = uv_object.history
    unique_ants = np.unique(
        uv_object.ant_1_array.tolist() + uv_object.ant_2_array.tolist()
    )
    ants_to_keep = np.array([0, 19, 11, 24, 3, 23, 1, 20, 21])

    blts_select = [
        (a1 in ants_to_keep) & (a2 in ants_to_keep)
        for (a1, a2) in zip(uv_object.ant_1_array, uv_object.ant_2_array)
    ]
    Nblts_selected = np.sum(blts_select)

    uv_object2 = uv_object.copy()
    uv_object2.select(antenna_nums=ants_to_keep)

    assert len(ants_to_keep) == uv_object2.Nants_data
    assert Nblts_selected == uv_object2.Nblts
    for ant in ants_to_keep:
        assert ant in uv_object2.ant_1_array or ant in uv_object2.ant_2_array
    for ant in np.unique(
        uv_object2.ant_1_array.tolist() + uv_object2.ant_2_array.tolist()
    ):
        assert ant in ants_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to " "specific antennas using pyuvdata.",
        uv_object2.history,
    )

    # check that it also works with higher dimension array
    uv_object2 = uv_object.copy()
    uv_object2.select(antenna_nums=ants_to_keep[np.newaxis, :])

    assert len(ants_to_keep) == uv_object2.Nants_data
    assert Nblts_selected == uv_object2.Nblts
    for ant in ants_to_keep:
        assert ant in uv_object2.ant_1_array or ant in uv_object2.ant_2_array
    for ant in np.unique(
        uv_object2.ant_1_array.tolist() + uv_object2.ant_2_array.tolist()
    ):
        assert ant in ants_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to " "specific antennas using pyuvdata.",
        uv_object2.history,
    )

    # now test using antenna_names to specify antennas to keep
    uv_object3 = uv_object.copy()
    ants_to_keep = np.array(sorted(ants_to_keep))
    ant_names = []
    for a in ants_to_keep:
        ind = np.where(uv_object3.antenna_numbers == a)[0][0]
        ant_names.append(uv_object3.antenna_names[ind])

    uv_object3.select(antenna_names=ant_names)

    assert uv_object2 == uv_object3

    # check that it also works with higher dimension array
    uv_object3 = uv_object.copy()
    ants_to_keep = np.array(sorted(ants_to_keep))
    ant_names = []
    for a in ants_to_keep:
        ind = np.where(uv_object3.antenna_numbers == a)[0][0]
        ant_names.append(uv_object3.antenna_names[ind])

    uv_object3.select(antenna_names=[ant_names])

    assert uv_object2 == uv_object3

    # test removing metadata associated with antennas that are no longer present
    # also add (different) antenna_diameters to test downselection
    uv_object.antenna_diameters = 1.0 * np.ones(
        (uv_object.Nants_telescope,), dtype=np.float
    )
    for i in range(uv_object.Nants_telescope):
        uv_object.antenna_diameters += i
    uv_object4 = uv_object.copy()
    uv_object4.select(antenna_nums=ants_to_keep, keep_all_metadata=False)
    assert uv_object4.Nants_telescope == 9
    assert set(uv_object4.antenna_numbers) == set(ants_to_keep)
    for a in ants_to_keep:
        idx1 = uv_object.antenna_numbers.tolist().index(a)
        idx2 = uv_object4.antenna_numbers.tolist().index(a)
        assert uv_object.antenna_names[idx1] == uv_object4.antenna_names[idx2]
        assert np.allclose(
            uv_object.antenna_positions[idx1, :], uv_object4.antenna_positions[idx2, :]
        )
        assert uv_object.antenna_diameters[idx1], uv_object4.antenna_diameters[idx2]

    # remove antenna_diameters from object
    uv_object.antenna_diameters = None

    # check for errors associated with antennas not included in data, bad names
    # or providing numbers and names
    pytest.raises(
        ValueError, uv_object.select, antenna_nums=np.max(unique_ants) + np.arange(1, 3)
    )
    pytest.raises(ValueError, uv_object.select, antenna_names="test1")
    pytest.raises(
        ValueError, uv_object.select, antenna_nums=ants_to_keep, antenna_names=ant_names
    )


def sort_bl(p):
    """Sort a tuple that starts with a pair of antennas, and may have stuff after."""
    if p[1] >= p[0]:
        return p
    return (p[1], p[0]) + p[2:]


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_select_bls(casa_uvfits):
    uv_object = casa_uvfits
    old_history = uv_object.history
    first_ants = [6, 2, 7, 2, 21, 27, 8]
    second_ants = [0, 20, 8, 1, 2, 3, 22]
    new_unique_ants = np.unique(first_ants + second_ants)
    ant_pairs_to_keep = list(zip(first_ants, second_ants))
    sorted_pairs_to_keep = [sort_bl(p) for p in ant_pairs_to_keep]

    blts_select = [
        sort_bl((a1, a2)) in sorted_pairs_to_keep
        for (a1, a2) in zip(uv_object.ant_1_array, uv_object.ant_2_array)
    ]
    Nblts_selected = np.sum(blts_select)

    uv_object2 = uv_object.copy()
    uv_object2.select(bls=ant_pairs_to_keep)
    sorted_pairs_object2 = [
        sort_bl(p) for p in zip(uv_object2.ant_1_array, uv_object2.ant_2_array)
    ]

    assert len(new_unique_ants) == uv_object2.Nants_data
    assert Nblts_selected == uv_object2.Nblts
    for ant in new_unique_ants:
        assert ant in uv_object2.ant_1_array or ant in uv_object2.ant_2_array
    for ant in np.unique(
        uv_object2.ant_1_array.tolist() + uv_object2.ant_2_array.tolist()
    ):
        assert ant in new_unique_ants
    for pair in sorted_pairs_to_keep:
        assert pair in sorted_pairs_object2
    for pair in sorted_pairs_object2:
        assert pair in sorted_pairs_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to " "specific baselines using pyuvdata.",
        uv_object2.history,
    )

    # check using baseline number parameter
    uv_object3 = uv_object.copy()
    bls_nums_to_keep = [
        uv_object.antnums_to_baseline(ant1, ant2) for ant1, ant2 in sorted_pairs_to_keep
    ]

    uv_object3.select(bls=bls_nums_to_keep)
    sorted_pairs_object3 = [
        sort_bl(p) for p in zip(uv_object3.ant_1_array, uv_object3.ant_2_array)
    ]

    assert len(new_unique_ants) == uv_object3.Nants_data
    assert Nblts_selected == uv_object3.Nblts
    for ant in new_unique_ants:
        assert ant in uv_object3.ant_1_array or ant in uv_object3.ant_2_array
    for ant in np.unique(
        uv_object3.ant_1_array.tolist() + uv_object3.ant_2_array.tolist()
    ):
        assert ant in new_unique_ants
    for pair in sorted_pairs_to_keep:
        assert pair in sorted_pairs_object3
    for pair in sorted_pairs_object3:
        assert pair in sorted_pairs_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to " "specific baselines using pyuvdata.",
        uv_object3.history,
    )

    # check select with polarizations
    first_ants = [6, 2, 7, 2, 21, 27, 8]
    second_ants = [0, 20, 8, 1, 2, 3, 22]
    pols = ["RR", "RR", "RR", "RR", "RR", "RR", "RR"]
    new_unique_ants = np.unique(first_ants + second_ants)
    bls_to_keep = list(zip(first_ants, second_ants, pols))
    sorted_bls_to_keep = [sort_bl(p) for p in bls_to_keep]

    blts_select = [
        sort_bl((a1, a2, "RR")) in sorted_bls_to_keep
        for (a1, a2) in zip(uv_object.ant_1_array, uv_object.ant_2_array)
    ]
    Nblts_selected = np.sum(blts_select)

    uv_object2 = uv_object.copy()
    uv_object2.select(bls=bls_to_keep)
    sorted_pairs_object2 = [
        sort_bl(p) + ("RR",)
        for p in zip(uv_object2.ant_1_array, uv_object2.ant_2_array)
    ]

    assert len(new_unique_ants) == uv_object2.Nants_data
    assert Nblts_selected == uv_object2.Nblts
    for ant in new_unique_ants:
        assert ant in uv_object2.ant_1_array or ant in uv_object2.ant_2_array
    for ant in np.unique(
        uv_object2.ant_1_array.tolist() + uv_object2.ant_2_array.tolist()
    ):
        assert ant in new_unique_ants
    for bl in sorted_bls_to_keep:
        assert bl in sorted_pairs_object2
    for bl in sorted_pairs_object2:
        assert bl in sorted_bls_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to "
        "specific baselines, polarizations using pyuvdata.",
        uv_object2.history,
    )

    # check that you can use numpy integers with out errors:
    first_ants = list(map(np.int32, [6, 2, 7, 2, 21, 27, 8]))
    second_ants = list(map(np.int32, [0, 20, 8, 1, 2, 3, 22]))
    ant_pairs_to_keep = list(zip(first_ants, second_ants))

    uv_object2 = uv_object.select(bls=ant_pairs_to_keep, inplace=False)
    sorted_pairs_object2 = [
        sort_bl(p) for p in zip(uv_object2.ant_1_array, uv_object2.ant_2_array)
    ]

    assert len(new_unique_ants) == uv_object2.Nants_data
    assert Nblts_selected == uv_object2.Nblts
    for ant in new_unique_ants:
        assert ant in uv_object2.ant_1_array or ant in uv_object2.ant_2_array
    for ant in np.unique(
        uv_object2.ant_1_array.tolist() + uv_object2.ant_2_array.tolist()
    ):
        assert ant in new_unique_ants
    for pair in sorted_pairs_to_keep:
        assert pair in sorted_pairs_object2
    for pair in sorted_pairs_object2:
        assert pair in sorted_pairs_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to " "specific baselines using pyuvdata.",
        uv_object2.history,
    )

    # check that you can specify a single pair without errors
    uv_object2.select(bls=(0, 6))
    sorted_pairs_object2 = [
        sort_bl(p) for p in zip(uv_object2.ant_1_array, uv_object2.ant_2_array)
    ]
    assert list(set(sorted_pairs_object2)) == [(0, 6)]

    # check for errors associated with antenna pairs not included in data and bad inputs
    with pytest.raises(ValueError) as cm:
        uv_object.select(bls=list(zip(first_ants, second_ants)) + [0, 6])
    assert str(cm.value).startswith("bls must be a list of tuples of antenna numbers")

    with pytest.raises(ValueError) as cm:
        uv_object.select(bls=[(uv_object.antenna_names[0], uv_object.antenna_names[1])])
    assert str(cm.value).startswith("bls must be a list of tuples of antenna numbers")

    with pytest.raises(ValueError) as cm:
        uv_object.select(bls=(5, 1))
    assert str(cm.value).startswith(
        "Antenna number 5 is not present in the " "ant_1_array or ant_2_array"
    )
    with pytest.raises(ValueError) as cm:
        uv_object.select(bls=(0, 5))
    assert str(cm.value).startswith(
        "Antenna number 5 is not present in the " "ant_1_array or ant_2_array"
    )
    with pytest.raises(ValueError) as cm:
        uv_object.select(bls=(27, 27))
    assert str(cm.value).startswith("Antenna pair (27, 27) does not have any data")
    with pytest.raises(ValueError) as cm:
        uv_object.select(bls=(6, 0, "RR"), polarizations="RR")
    assert str(cm.value).startswith(
        "Cannot provide length-3 tuples and also " "specify polarizations."
    )
    with pytest.raises(ValueError) as cm:
        uv_object.select(bls=(6, 0, 8))
    assert str(cm.value).startswith(
        "The third element in each bl must be a " "polarization string"
    )
    with pytest.raises(ValueError) as cm:
        uv_object.select(bls=[])
    assert str(cm.value).startswith("bls must be a list of tuples of antenna numbers")
    with pytest.raises(ValueError) as cm:
        uv_object.select(bls=[100])
    assert str(cm.value).startswith("Baseline number 100 is not present in the")


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_select_times(casa_uvfits):
    uv_object = casa_uvfits
    old_history = uv_object.history
    unique_times = np.unique(uv_object.time_array)
    times_to_keep = unique_times[[0, 3, 5, 6, 7, 10, 14]]

    Nblts_selected = np.sum([t in times_to_keep for t in uv_object.time_array])

    uv_object2 = uv_object.copy()
    uv_object2.select(times=times_to_keep)

    assert len(times_to_keep) == uv_object2.Ntimes
    assert Nblts_selected == uv_object2.Nblts
    for t in times_to_keep:
        assert t in uv_object2.time_array
    for t in np.unique(uv_object2.time_array):
        assert t in times_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to " "specific times using pyuvdata.",
        uv_object2.history,
    )
    # check that it also works with higher dimension array
    uv_object2 = uv_object.copy()
    uv_object2.select(times=times_to_keep[np.newaxis, :])

    assert len(times_to_keep) == uv_object2.Ntimes
    assert Nblts_selected == uv_object2.Nblts
    for t in times_to_keep:
        assert t in uv_object2.time_array
    for t in np.unique(uv_object2.time_array):
        assert t in times_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to " "specific times using pyuvdata.",
        uv_object2.history,
    )

    # check for errors associated with times not included in data
    pytest.raises(
        ValueError,
        uv_object.select,
        times=[np.min(unique_times) - uv_object.integration_time[0]],
    )


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_select_time_range(casa_uvfits):
    uv_object = casa_uvfits
    old_history = uv_object.history
    unique_times = np.unique(uv_object.time_array)
    mean_time = np.mean(unique_times)
    time_range = [np.min(unique_times), mean_time]
    times_to_keep = unique_times[
        np.nonzero((unique_times <= time_range[1]) & (unique_times >= time_range[0]))
    ]

    Nblts_selected = np.nonzero(
        (uv_object.time_array <= time_range[1])
        & (uv_object.time_array >= time_range[0])
    )[0].size

    uv_object2 = uv_object.copy()
    uv_object2.select(time_range=time_range)

    assert times_to_keep.size == uv_object2.Ntimes
    assert Nblts_selected == uv_object2.Nblts
    for t in times_to_keep:
        assert t in uv_object2.time_array
    for t in np.unique(uv_object2.time_array):
        assert t in times_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to " "specific times using pyuvdata.",
        uv_object2.history,
    )


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_select_time_range_no_data(casa_uvfits):
    """Check for error associated with times not included in data."""
    uv_object = casa_uvfits
    unique_times = np.unique(uv_object.time_array)
    with pytest.raises(ValueError) as cm:
        uv_object.select(
            time_range=[
                np.min(unique_times) - uv_object.integration_time[0] * 2,
                np.min(unique_times) - uv_object.integration_time[0],
            ]
        )
    assert str(cm.value).startswith("No elements in time range")


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_select_time_and_time_range(casa_uvfits):
    """Check for error setting times and time_range."""
    uv_object = casa_uvfits
    unique_times = np.unique(uv_object.time_array)
    mean_time = np.mean(unique_times)
    time_range = [np.min(unique_times), mean_time]
    times_to_keep = unique_times[[0, 3, 5, 6, 7, 10, 14]]
    with pytest.raises(ValueError) as cm:
        uv_object.select(time_range=time_range, times=times_to_keep)
    assert str(cm.value).startswith('Only one of "times" and "time_range" can be set')


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_select_time_range_one_elem(casa_uvfits):
    """Check for error if time_range not length 2."""
    uv_object = casa_uvfits
    unique_times = np.unique(uv_object.time_array)
    mean_time = np.mean(unique_times)
    time_range = [np.min(unique_times), mean_time]
    with pytest.raises(ValueError) as cm:
        uv_object.select(time_range=time_range[0])
    assert str(cm.value).startswith("time_range must be length 2")


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_select_frequencies_uvfits(casa_uvfits, tmp_path):
    uv_object = casa_uvfits
    old_history = uv_object.history
    freqs_to_keep = uv_object.freq_array[0, np.arange(12, 22)]

    uv_object2 = uv_object.copy()
    uv_object2.select(frequencies=freqs_to_keep)

    assert len(freqs_to_keep) == uv_object2.Nfreqs
    for f in freqs_to_keep:
        assert f in uv_object2.freq_array
    for f in np.unique(uv_object2.freq_array):
        assert f in freqs_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to " "specific frequencies using pyuvdata.",
        uv_object2.history,
    )

    # check that it also works with higher dimension array
    uv_object2 = uv_object.copy()
    uv_object2.select(frequencies=freqs_to_keep[np.newaxis, :])

    assert len(freqs_to_keep) == uv_object2.Nfreqs
    for f in freqs_to_keep:
        assert f in uv_object2.freq_array
    for f in np.unique(uv_object2.freq_array):
        assert f in freqs_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to " "specific frequencies using pyuvdata.",
        uv_object2.history,
    )

    # check that selecting one frequency works
    uv_object2 = uv_object.copy()
    uv_object2.select(frequencies=freqs_to_keep[0])
    assert 1 == uv_object2.Nfreqs
    assert freqs_to_keep[0] in uv_object2.freq_array
    for f in uv_object2.freq_array:
        assert f in [freqs_to_keep[0]]

    assert uvutils._check_histories(
        old_history + "  Downselected to " "specific frequencies using pyuvdata.",
        uv_object2.history,
    )

    # check for errors associated with frequencies not included in data
    pytest.raises(
        ValueError,
        uv_object.select,
        frequencies=[np.max(uv_object.freq_array) + uv_object.channel_width],
    )

    # check for warnings and errors associated with unevenly spaced or
    # non-contiguous frequencies
    uv_object2 = uv_object.copy()
    with uvtest.check_warnings(
        UserWarning,
        [
            "Selected frequencies are not evenly spaced",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv_object2.select(frequencies=uv_object2.freq_array[0, [0, 5, 6]])
    write_file_uvfits = str(tmp_path / "select_test.uvfits")
    pytest.raises(ValueError, uv_object2.write_uvfits, write_file_uvfits)

    uv_object2 = uv_object.copy()
    with uvtest.check_warnings(
        UserWarning,
        [
            "Selected frequencies are not contiguous",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv_object2.select(frequencies=uv_object2.freq_array[0, [0, 2, 4]])

    pytest.raises(ValueError, uv_object2.write_uvfits, write_file_uvfits)


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_select_frequencies_miriad(casa_uvfits, tmp_path):
    pytest.importorskip("pyuvdata._miriad")
    uv_object = casa_uvfits
    old_history = uv_object.history
    freqs_to_keep = uv_object.freq_array[0, np.arange(12, 22)]

    uv_object2 = uv_object.copy()
    uv_object2.select(frequencies=freqs_to_keep)

    assert len(freqs_to_keep) == uv_object2.Nfreqs
    for f in freqs_to_keep:
        assert f in uv_object2.freq_array
    for f in np.unique(uv_object2.freq_array):
        assert f in freqs_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to " "specific frequencies using pyuvdata.",
        uv_object2.history,
    )

    # check that it also works with higher dimension array
    uv_object2 = uv_object.copy()
    uv_object2.select(frequencies=freqs_to_keep[np.newaxis, :])

    assert len(freqs_to_keep) == uv_object2.Nfreqs
    for f in freqs_to_keep:
        assert f in uv_object2.freq_array
    for f in np.unique(uv_object2.freq_array):
        assert f in freqs_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to " "specific frequencies using pyuvdata.",
        uv_object2.history,
    )

    # check that selecting one frequency works
    uv_object2 = uv_object.copy()
    uv_object2.select(frequencies=freqs_to_keep[0])
    assert 1 == uv_object2.Nfreqs
    assert freqs_to_keep[0] in uv_object2.freq_array
    for f in uv_object2.freq_array:
        assert f in [freqs_to_keep[0]]

    assert uvutils._check_histories(
        old_history + "  Downselected to specific frequencies using pyuvdata.",
        uv_object2.history,
    )

    # check for errors associated with frequencies not included in data
    pytest.raises(
        ValueError,
        uv_object.select,
        frequencies=[np.max(uv_object.freq_array) + uv_object.channel_width],
    )

    # check for warnings and errors associated with unevenly spaced or
    # non-contiguous frequencies
    uv_object2 = uv_object.copy()
    with uvtest.check_warnings(
        UserWarning,
        [
            "Selected frequencies are not evenly spaced",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv_object2.select(frequencies=uv_object2.freq_array[0, [0, 5, 6]])
    write_file_miriad = str(tmp_path / "select_test.uvfits")
    pytest.raises(ValueError, uv_object2.write_miriad, write_file_miriad)

    uv_object2 = uv_object.copy()
    with uvtest.check_warnings(
        UserWarning,
        [
            "Selected frequencies are not contiguous",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv_object2.select(frequencies=uv_object2.freq_array[0, [0, 2, 4]])
    pytest.raises(ValueError, uv_object2.write_miriad, write_file_miriad)


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_select_freq_chans(casa_uvfits):
    uv_object = casa_uvfits
    old_history = uv_object.history
    chans_to_keep = np.arange(12, 22)

    uv_object2 = uv_object.copy()
    uv_object2.select(freq_chans=chans_to_keep)

    assert len(chans_to_keep) == uv_object2.Nfreqs
    for chan in chans_to_keep:
        assert uv_object.freq_array[0, chan] in uv_object2.freq_array
    for f in np.unique(uv_object2.freq_array):
        assert f in uv_object.freq_array[0, chans_to_keep]

    assert uvutils._check_histories(
        old_history + "  Downselected to " "specific frequencies using pyuvdata.",
        uv_object2.history,
    )

    # check that it also works with higher dimension array
    uv_object2 = uv_object.copy()
    uv_object2.select(freq_chans=chans_to_keep[np.newaxis, :])

    assert len(chans_to_keep) == uv_object2.Nfreqs
    for chan in chans_to_keep:
        assert uv_object.freq_array[0, chan] in uv_object2.freq_array
    for f in np.unique(uv_object2.freq_array):
        assert f in uv_object.freq_array[0, chans_to_keep]

    assert uvutils._check_histories(
        old_history + "  Downselected to " "specific frequencies using pyuvdata.",
        uv_object2.history,
    )

    # Test selecting both channels and frequencies
    freqs_to_keep = uv_object.freq_array[0, np.arange(20, 30)]  # Overlaps with chans
    all_chans_to_keep = np.arange(12, 30)

    uv_object2 = uv_object.copy()
    uv_object2.select(frequencies=freqs_to_keep, freq_chans=chans_to_keep)

    assert len(all_chans_to_keep) == uv_object2.Nfreqs
    for chan in all_chans_to_keep:
        assert uv_object.freq_array[0, chan] in uv_object2.freq_array
    for f in np.unique(uv_object2.freq_array):
        assert f in uv_object.freq_array[0, all_chans_to_keep]


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_select_polarizations(casa_uvfits, tmp_path):
    uv_object = casa_uvfits
    old_history = uv_object.history
    pols_to_keep = [-1, -2]

    uv_object2 = uv_object.copy()
    uv_object2.select(polarizations=pols_to_keep)

    assert len(pols_to_keep) == uv_object2.Npols
    for p in pols_to_keep:
        assert p in uv_object2.polarization_array
    for p in np.unique(uv_object2.polarization_array):
        assert p in pols_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to " "specific polarizations using pyuvdata.",
        uv_object2.history,
    )

    # check that it also works with higher dimension array
    uv_object2 = uv_object.copy()
    uv_object2.select(polarizations=[pols_to_keep])

    assert len(pols_to_keep) == uv_object2.Npols
    for p in pols_to_keep:
        assert p in uv_object2.polarization_array
    for p in np.unique(uv_object2.polarization_array):
        assert p in pols_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to " "specific polarizations using pyuvdata.",
        uv_object2.history,
    )

    # check for errors associated with polarizations not included in data
    pytest.raises(ValueError, uv_object2.select, polarizations=[-3, -4])

    # check for warnings and errors associated with unevenly spaced polarizations
    with uvtest.check_warnings(
        UserWarning,
        [
            "Selected polarization values are not evenly spaced",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv_object.select(polarizations=uv_object.polarization_array[[0, 1, 3]])
    write_file_uvfits = str(tmp_path / "select_test.uvfits")
    pytest.raises(ValueError, uv_object.write_uvfits, write_file_uvfits)


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_select(casa_uvfits):
    # now test selecting along all axes at once
    uv_object = casa_uvfits
    old_history = uv_object.history
    # fmt: off
    blt_inds = np.array([1057, 461, 1090, 354, 528, 654, 882, 775, 369, 906, 748,
                         875, 296, 773, 554, 395, 1003, 476, 762, 976, 1285, 874,
                         717, 383, 1281, 924, 264, 1163, 297, 857, 1258, 1000, 180,
                         1303, 1139, 393, 42, 135, 789, 713, 527, 1218, 576, 100,
                         1311, 4, 653, 724, 591, 889, 36, 1033, 113, 479, 322,
                         118, 898, 1263, 477, 96, 935, 238, 195, 531, 124, 198,
                         992, 1131, 305, 154, 961, 6, 1175, 76, 663, 82, 637,
                         288, 1152, 845, 1290, 379, 1225, 1240, 733, 1172, 937, 1325,
                         817, 416, 261, 1316, 957, 723, 215, 237, 270, 1309, 208,
                         17, 1028, 895, 574, 166, 784, 834, 732, 1022, 1068, 1207,
                         356, 474, 313, 137, 172, 181, 925, 201, 190, 1277, 1044,
                         1242, 702, 567, 557, 1032, 1352, 504, 545, 422, 179, 780,
                         280, 890, 774, 884])
    # fmt: on
    ants_to_keep = np.array([11, 6, 20, 26, 2, 27, 7, 14])

    ant_pairs_to_keep = [(2, 11), (20, 26), (6, 7), (3, 27), (14, 6)]
    sorted_pairs_to_keep = [sort_bl(p) for p in ant_pairs_to_keep]

    freqs_to_keep = uv_object.freq_array[0, np.arange(31, 39)]

    unique_times = np.unique(uv_object.time_array)
    times_to_keep = unique_times[[0, 2, 6, 8, 10, 13, 14]]

    pols_to_keep = [-1, -3]

    # Independently count blts that should be selected
    blts_blt_select = [i in blt_inds for i in np.arange(uv_object.Nblts)]
    blts_ant_select = [
        (a1 in ants_to_keep) & (a2 in ants_to_keep)
        for (a1, a2) in zip(uv_object.ant_1_array, uv_object.ant_2_array)
    ]
    blts_pair_select = [
        sort_bl((a1, a2)) in sorted_pairs_to_keep
        for (a1, a2) in zip(uv_object.ant_1_array, uv_object.ant_2_array)
    ]
    blts_time_select = [t in times_to_keep for t in uv_object.time_array]
    Nblts_select = np.sum(
        [
            bi & (ai & pi) & ti
            for (bi, ai, pi, ti) in zip(
                blts_blt_select, blts_ant_select, blts_pair_select, blts_time_select
            )
        ]
    )

    uv_object2 = uv_object.copy()
    uv_object2.select(
        blt_inds=blt_inds,
        antenna_nums=ants_to_keep,
        bls=ant_pairs_to_keep,
        frequencies=freqs_to_keep,
        times=times_to_keep,
        polarizations=pols_to_keep,
    )

    assert Nblts_select == uv_object2.Nblts
    for ant in np.unique(
        uv_object2.ant_1_array.tolist() + uv_object2.ant_2_array.tolist()
    ):
        assert ant in ants_to_keep

    assert len(freqs_to_keep) == uv_object2.Nfreqs
    for f in freqs_to_keep:
        assert f in uv_object2.freq_array
    for f in np.unique(uv_object2.freq_array):
        assert f in freqs_to_keep

    for t in np.unique(uv_object2.time_array):
        assert t in times_to_keep

    assert len(pols_to_keep) == uv_object2.Npols
    for p in pols_to_keep:
        assert p in uv_object2.polarization_array
    for p in np.unique(uv_object2.polarization_array):
        assert p in pols_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to "
        "specific baseline-times, antennas, "
        "baselines, times, frequencies, "
        "polarizations using pyuvdata.",
        uv_object2.history,
    )

    # test that a ValueError is raised if the selection eliminates all blts
    pytest.raises(ValueError, uv_object.select, times=unique_times[0], antenna_nums=1)


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_select_not_inplace(casa_uvfits):
    # Test non-inplace select
    uv_object = casa_uvfits
    old_history = uv_object.history
    uv1 = uv_object.select(freq_chans=np.arange(32), inplace=False)
    uv1 += uv_object.select(freq_chans=np.arange(32, 64), inplace=False)
    assert uvutils._check_histories(
        old_history + "  Downselected to "
        "specific frequencies using pyuvdata. "
        "Combined data along frequency axis "
        "using pyuvdata.",
        uv1.history,
    )

    uv1.history = old_history
    assert uv1 == uv_object


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("metadata_only", [True, False])
def test_conjugate_bls(casa_uvfits, metadata_only):
    testfile = os.path.join(DATA_PATH, "day2_TDEM0003_10s_norx_1src_1spw.uvfits")

    if not metadata_only:
        uv1 = casa_uvfits
    else:
        uv1 = UVData()
        uv1.read_uvfits(testfile, read_data=False)
    if metadata_only:
        assert uv1.metadata_only
    # file comes in with ant1<ant2
    assert np.min(uv1.ant_2_array - uv1.ant_1_array) >= 0

    # check everything swapped & conjugated when go to ant2<ant1
    uv2 = uv1.copy()
    uv2.conjugate_bls(convention="ant2<ant1")
    assert np.min(uv2.ant_1_array - uv2.ant_2_array) >= 0

    assert np.allclose(uv1.ant_1_array, uv2.ant_2_array)
    assert np.allclose(uv1.ant_2_array, uv2.ant_1_array)
    assert np.allclose(
        uv1.uvw_array,
        -1 * uv2.uvw_array,
        rtol=uv1._uvw_array.tols[0],
        atol=uv1._uvw_array.tols[1],
    )

    if not metadata_only:
        # complicated because of the polarization swaps
        # polarization_array = [-1 -2 -3 -4]
        assert np.allclose(
            uv1.data_array[:, :, :, :2],
            np.conj(uv2.data_array[:, :, :, :2]),
            rtol=uv1._data_array.tols[0],
            atol=uv1._data_array.tols[1],
        )

        assert np.allclose(
            uv1.data_array[:, :, :, 2],
            np.conj(uv2.data_array[:, :, :, 3]),
            rtol=uv1._data_array.tols[0],
            atol=uv1._data_array.tols[1],
        )

        assert np.allclose(
            uv1.data_array[:, :, :, 3],
            np.conj(uv2.data_array[:, :, :, 2]),
            rtol=uv1._data_array.tols[0],
            atol=uv1._data_array.tols[1],
        )

    # check everything returned to original values with original convention
    uv2.conjugate_bls(convention="ant1<ant2")
    assert uv1 == uv2

    # conjugate a particular set of blts
    blts_to_conjugate = np.arange(uv2.Nblts // 2)
    blts_not_conjugated = np.arange(uv2.Nblts // 2, uv2.Nblts)
    uv2.conjugate_bls(convention=blts_to_conjugate)

    assert np.allclose(
        uv1.ant_1_array[blts_to_conjugate], uv2.ant_2_array[blts_to_conjugate]
    )
    assert np.allclose(
        uv1.ant_2_array[blts_to_conjugate], uv2.ant_1_array[blts_to_conjugate]
    )
    assert np.allclose(
        uv1.ant_1_array[blts_not_conjugated], uv2.ant_1_array[blts_not_conjugated]
    )
    assert np.allclose(
        uv1.ant_2_array[blts_not_conjugated], uv2.ant_2_array[blts_not_conjugated]
    )

    assert np.allclose(
        uv1.uvw_array[blts_to_conjugate],
        -1 * uv2.uvw_array[blts_to_conjugate],
        rtol=uv1._uvw_array.tols[0],
        atol=uv1._uvw_array.tols[1],
    )
    assert np.allclose(
        uv1.uvw_array[blts_not_conjugated],
        uv2.uvw_array[blts_not_conjugated],
        rtol=uv1._uvw_array.tols[0],
        atol=uv1._uvw_array.tols[1],
    )
    if not metadata_only:
        # complicated because of the polarization swaps
        # polarization_array = [-1 -2 -3 -4]
        assert np.allclose(
            uv1.data_array[blts_to_conjugate, :, :, :2],
            np.conj(uv2.data_array[blts_to_conjugate, :, :, :2]),
            rtol=uv1._data_array.tols[0],
            atol=uv1._data_array.tols[1],
        )
        assert np.allclose(
            uv1.data_array[blts_not_conjugated, :, :, :2],
            uv2.data_array[blts_not_conjugated, :, :, :2],
            rtol=uv1._data_array.tols[0],
            atol=uv1._data_array.tols[1],
        )

        assert np.allclose(
            uv1.data_array[blts_to_conjugate, :, :, 2],
            np.conj(uv2.data_array[blts_to_conjugate, :, :, 3]),
            rtol=uv1._data_array.tols[0],
            atol=uv1._data_array.tols[1],
        )
        assert np.allclose(
            uv1.data_array[blts_not_conjugated, :, :, 2],
            uv2.data_array[blts_not_conjugated, :, :, 2],
            rtol=uv1._data_array.tols[0],
            atol=uv1._data_array.tols[1],
        )

        assert np.allclose(
            uv1.data_array[blts_to_conjugate, :, :, 3],
            np.conj(uv2.data_array[blts_to_conjugate, :, :, 2]),
            rtol=uv1._data_array.tols[0],
            atol=uv1._data_array.tols[1],
        )
        assert np.allclose(
            uv1.data_array[blts_not_conjugated, :, :, 3],
            uv2.data_array[blts_not_conjugated, :, :, 3],
            rtol=uv1._data_array.tols[0],
            atol=uv1._data_array.tols[1],
        )

    # check uv half plane conventions
    uv2.conjugate_bls(convention="u<0", use_enu=False)
    assert np.max(uv2.uvw_array[:, 0]) <= 0

    uv2.conjugate_bls(convention="u>0", use_enu=False)
    assert np.min(uv2.uvw_array[:, 0]) >= 0

    uv2.conjugate_bls(convention="v<0", use_enu=False)
    assert np.max(uv2.uvw_array[:, 1]) <= 0

    uv2.conjugate_bls(convention="v>0", use_enu=False)
    assert np.min(uv2.uvw_array[:, 1]) >= 0

    # unphase to drift to test using ENU positions
    uv2.unphase_to_drift(use_ant_pos=True)
    uv2.conjugate_bls(convention="u<0")
    assert np.max(uv2.uvw_array[:, 0]) <= 0

    uv2.conjugate_bls(convention="u>0")
    assert np.min(uv2.uvw_array[:, 0]) >= 0

    uv2.conjugate_bls(convention="v<0")
    assert np.max(uv2.uvw_array[:, 1]) <= 0

    uv2.conjugate_bls(convention="v>0")
    assert np.min(uv2.uvw_array[:, 1]) >= 0

    # test errors
    with pytest.raises(ValueError) as cm:
        uv2.conjugate_bls(convention="foo")
    assert str(cm.value).startswith("convention must be one of")

    with pytest.raises(ValueError) as cm:
        uv2.conjugate_bls(convention=np.arange(5) - 1)
    assert str(cm.value).startswith("If convention is an index array")

    with pytest.raises(ValueError) as cm:
        uv2.conjugate_bls(convention=[uv2.Nblts])

    assert str(cm.value).startswith("If convention is an index array")


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_reorder_pols(casa_uvfits):
    # Test function to fix polarization order
    uv1 = casa_uvfits
    uv2 = uv1.copy()
    uv3 = uv1.copy()
    # reorder uv2 manually
    order = [1, 3, 2, 0]
    uv2.polarization_array = uv2.polarization_array[order]
    uv2.data_array = uv2.data_array[:, :, :, order]
    uv2.nsample_array = uv2.nsample_array[:, :, :, order]
    uv2.flag_array = uv2.flag_array[:, :, :, order]
    uv1.reorder_pols(order=order)
    assert uv1 == uv2

    # Restore original order
    uv1 = uv3.copy()
    uv2.reorder_pols()
    assert uv1 == uv2

    uv1.reorder_pols(order="AIPS")
    # check that we have aips ordering
    aips_pols = np.array([-1, -2, -3, -4]).astype(int)
    assert np.all(uv1.polarization_array == aips_pols)

    uv2 = uv1.copy()
    uv2.reorder_pols(order="CASA")
    # check that we have casa ordering
    casa_pols = np.array([-1, -3, -4, -2]).astype(int)
    assert np.all(uv2.polarization_array == casa_pols)
    order = np.array([0, 2, 3, 1])
    assert np.all(uv2.data_array == uv1.data_array[:, :, :, order])
    assert np.all(uv2.flag_array == uv1.flag_array[:, :, :, order])

    uv2.reorder_pols(order="AIPS")
    # check that we have aips ordering again
    assert uv1 == uv2

    # check error on unknown order
    pytest.raises(ValueError, uv2.reorder_pols, {"order": "foo"})

    # check error if order is an array of the wrong length
    with pytest.raises(ValueError) as cm:
        uv2.reorder_pols(order=[3, 2, 1])
    assert str(cm.value).startswith("If order is an index array, it must")


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_reorder_blts(casa_uvfits):
    uv1 = casa_uvfits

    # test default reordering in detail
    uv2 = uv1.copy()
    uv2.reorder_blts()
    assert uv2.blt_order == ("time", "baseline")
    assert np.min(np.diff(uv2.time_array)) >= 0
    for this_time in np.unique(uv2.time_array):
        bls_2 = uv2.baseline_array[np.where(uv2.time_array == this_time)]
        bls_1 = uv1.baseline_array[np.where(uv2.time_array == this_time)]
        assert bls_1.shape == bls_2.shape
        assert np.min(np.diff(bls_2)) >= 0
        bl_inds = [np.where(bls_1 == bl)[0][0] for bl in bls_2]
        assert np.allclose(bls_1[bl_inds], bls_2)

        uvw_1 = uv1.uvw_array[np.where(uv2.time_array == this_time)[0], :]
        uvw_2 = uv2.uvw_array[np.where(uv2.time_array == this_time)[0], :]
        assert uvw_1.shape == uvw_2.shape
        assert np.allclose(uvw_1[bl_inds, :], uvw_2)

        data_1 = uv1.data_array[np.where(uv2.time_array == this_time)[0], :, :, :]
        data_2 = uv2.data_array[np.where(uv2.time_array == this_time)[0], :, :, :]
        assert data_1.shape == data_2.shape
        assert np.allclose(data_1[bl_inds, :, :, :], data_2)

    # check that ordering by time, ant1 is identical to time, baseline
    uv3 = uv1.copy()
    uv3.reorder_blts(order="time", minor_order="ant1")
    assert uv3.blt_order == ("time", "ant1")
    assert np.min(np.diff(uv3.time_array)) >= 0
    uv3.blt_order = uv2.blt_order
    assert uv2 == uv3

    uv3.reorder_blts(order="time", minor_order="ant2")
    assert uv3.blt_order == ("time", "ant2")
    assert np.min(np.diff(uv3.time_array)) >= 0

    # check that loopback works
    uv3.reorder_blts()
    assert uv2 == uv3

    # sort with a specified index array
    new_order = np.lexsort((uv3.baseline_array, uv3.time_array))
    uv3.reorder_blts(order=new_order)
    assert uv3.blt_order is None
    assert np.min(np.diff(uv3.time_array)) >= 0
    uv3.blt_order = ("time", "baseline")
    assert uv2 == uv3

    # test sensible defaulting if minor order = major order
    uv3.reorder_blts(order="time", minor_order="time")
    assert uv2 == uv3

    # test all combinations of major, minor order
    uv3.reorder_blts(order="baseline")
    assert uv3.blt_order == ("baseline", "time")
    assert np.min(np.diff(uv3.baseline_array)) >= 0

    uv3.reorder_blts(order="ant1")
    assert uv3.blt_order == ("ant1", "ant2")
    assert np.min(np.diff(uv3.ant_1_array)) >= 0

    uv3.reorder_blts(order="ant1", minor_order="time")
    assert uv3.blt_order == ("ant1", "time")
    assert np.min(np.diff(uv3.ant_1_array)) >= 0

    uv3.reorder_blts(order="ant1", minor_order="baseline")
    assert uv3.blt_order == ("ant1", "baseline")
    assert np.min(np.diff(uv3.ant_1_array)) >= 0

    uv3.reorder_blts(order="ant2")
    assert uv3.blt_order == ("ant2", "ant1")
    assert np.min(np.diff(uv3.ant_2_array)) >= 0

    uv3.reorder_blts(order="ant2", minor_order="time")
    assert uv3.blt_order == ("ant2", "time")
    assert np.min(np.diff(uv3.ant_2_array)) >= 0

    uv3.reorder_blts(order="ant2", minor_order="baseline")
    assert uv3.blt_order == ("ant2", "baseline")
    assert np.min(np.diff(uv3.ant_2_array)) >= 0

    uv3.reorder_blts(order="bda")
    assert uv3.blt_order == ("bda",)
    assert np.min(np.diff(uv3.integration_time)) >= 0
    assert np.min(np.diff(uv3.baseline_array)) >= 0

    # test doing conjugation along with a reorder
    # the file is already conjugated this way, so should be equal
    uv3.reorder_blts(order="time", conj_convention="ant1<ant2")
    assert uv2 == uv3

    # test errors
    with pytest.raises(ValueError) as cm:
        uv3.reorder_blts(order="foo")
    assert str(cm.value).startswith("order must be one of")

    with pytest.raises(ValueError) as cm:
        uv3.reorder_blts(order=np.arange(5))
    assert str(cm.value).startswith("If order is an index array, it must")

    with pytest.raises(ValueError) as cm:
        uv3.reorder_blts(order=np.arange(5, dtype=np.float))
    assert str(cm.value).startswith("If order is an index array, it must")

    with pytest.raises(ValueError) as cm:
        uv3.reorder_blts(order=np.arange(uv3.Nblts), minor_order="time")
    assert str(cm.value).startswith(
        "Minor order cannot be set if order is an index array"
    )

    with pytest.raises(ValueError) as cm:
        uv3.reorder_blts(order="bda", minor_order="time")
    assert str(cm.value).startswith("minor_order cannot be specified if order is")

    with pytest.raises(ValueError) as cm:
        uv3.reorder_blts(order="baseline", minor_order="ant1")
    assert str(cm.value).startswith("minor_order conflicts with order")

    with pytest.raises(ValueError) as cm:
        uv3.reorder_blts(order="time", minor_order="foo")
    assert str(cm.value).startswith("minor_order can only be one of")


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_sum_vis(casa_uvfits):
    # check sum_vis
    uv_full = casa_uvfits

    uv_half = uv_full.copy()
    uv_half.data_array = uv_full.data_array / 2
    uv_summed = uv_half.sum_vis(uv_half)

    assert np.array_equal(uv_summed.data_array, uv_full.data_array)
    assert uvutils._check_histories(
        uv_half.history + " Visibilities summed " "using pyuvdata.", uv_summed.history
    )

    # check diff_vis
    uv_diffed = uv_full.diff_vis(uv_half)

    assert np.array_equal(uv_diffed.data_array, uv_half.data_array)
    assert uvutils._check_histories(
        uv_full.history + " Visibilities " "differenced using pyuvdata.",
        uv_diffed.history,
    )

    # check in place
    uv_summed.diff_vis(uv_half, inplace=True)
    assert np.array_equal(uv_summed.data_array, uv_half.data_array)

    # check object_name merge
    uv_zenith = uv_full.copy()
    uv_zenith.object_name = "zenith"
    uv_merged = uv_zenith.sum_vis(uv_full)
    assert uv_merged.object_name == "zenith-J1008+0730"

    # check extra_keywords handling
    uv_keys = uv_full.copy()
    uv_keys.extra_keywords["test_key"] = "test_value"
    uv_keys.extra_keywords["SPECSYS"] = "altered_value"
    with uvtest.check_warnings(
        UserWarning,
        [
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "Keyword SPECSYS in _extra_keywords is different in the two objects. "
            "Taking the first object's entry.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv_merged_keys = uv_keys.sum_vis(uv_full)
    assert uv_merged_keys.extra_keywords["test_key"] == "test_value"
    assert uv_merged_keys.extra_keywords["SPECSYS"] == "altered_value"

    # check override_params
    uv_overrides = uv_full.copy()
    uv_overrides.instrument = "test_telescope"
    uv_overrides.telescope_location = [
        -1601183.15377712,
        -5042003.74810822,
        3554841.17192104,
    ]
    uv_overrides_2 = uv_overrides.sum_vis(
        uv_full, override_params=["instrument", "telescope_location"]
    )

    assert uv_overrides_2.instrument == "test_telescope"
    assert uv_overrides_2.telescope_location == [
        -1601183.15377712,
        -5042003.74810822,
        3554841.17192104,
    ]

    # check error messages
    with pytest.raises(ValueError) as cm:
        uv_overrides = uv_overrides.sum_vis(uv_full, override_params=["fake"])
    assert str(cm.value).startswith("Provided parameter fake is not a recognizable")

    with pytest.raises(ValueError) as cm:
        uv_full.sum_vis("foo")
    assert str(cm.value).startswith("Only UVData (or subclass) objects can be")

    uv_full.instrument = "foo"
    with pytest.raises(ValueError) as cm:
        uv_full.sum_vis(uv_half, inplace=True)
    assert str(cm.value).startswith("UVParameter instrument " "does not match")


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_add(casa_uvfits):
    uv_full = casa_uvfits

    # Add frequencies
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(32, 64))
    uv1 += uv2
    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific frequencies using pyuvdata. "
        "Combined data along frequency axis "
        "using pyuvdata.",
        uv1.history,
    )

    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add frequencies - out of order
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(32, 64))
    uv2 += uv1
    uv2.history = uv_full.history
    assert uv2 == uv_full

    # Add polarizations
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(polarizations=uv1.polarization_array[0:2])
    uv2.select(polarizations=uv2.polarization_array[2:4])
    uv1 += uv2
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific polarizations using pyuvdata. "
        "Combined data along polarization axis "
        "using pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add polarizations - out of order
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(polarizations=uv1.polarization_array[0:2])
    uv2.select(polarizations=uv2.polarization_array[2:4])
    uv2 += uv1
    uv2.history = uv_full.history
    assert uv2 == uv_full

    # Add times
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0 : len(times) // 2])
    uv2.select(times=times[len(times) // 2 :])
    uv1 += uv2
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific times using pyuvdata. "
        "Combined data along baseline-time axis "
        "using pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add baselines
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    ant_list = list(range(15))  # Roughly half the antennas in the data
    # All blts where ant_1 is in list
    ind1 = [i for i in range(uv1.Nblts) if uv1.ant_1_array[i] in ant_list]
    ind2 = [i for i in range(uv1.Nblts) if uv1.ant_1_array[i] not in ant_list]
    uv1.select(blt_inds=ind1)
    uv2.select(blt_inds=ind2)
    uv1 += uv2
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific baseline-times using pyuvdata. "
        "Combined data along baseline-time axis "
        "using pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add baselines - out of order
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv3 = uv_full.copy()
    ants = uv_full.get_ants()
    ants1 = ants[0:6]
    ants2 = ants[6:12]
    ants3 = ants[12:]

    # All blts where ant_1 is in list
    ind1 = [i for i in range(uv1.Nblts) if uv1.ant_1_array[i] in ants1]
    ind2 = [i for i in range(uv2.Nblts) if uv2.ant_1_array[i] in ants2]
    ind3 = [i for i in range(uv3.Nblts) if uv3.ant_1_array[i] in ants3]
    uv1.select(blt_inds=ind1)
    uv2.select(blt_inds=ind2)
    uv3.select(blt_inds=ind3)
    uv3.data_array = uv3.data_array[-1::-1, :, :, :]
    uv3.nsample_array = uv3.nsample_array[-1::-1, :, :, :]
    uv3.flag_array = uv3.flag_array[-1::-1, :, :, :]
    uv3.uvw_array = uv3.uvw_array[-1::-1, :]
    uv3.time_array = uv3.time_array[-1::-1]
    uv3.lst_array = uv3.lst_array[-1::-1]
    uv3.ant_1_array = uv3.ant_1_array[-1::-1]
    uv3.ant_2_array = uv3.ant_2_array[-1::-1]
    uv3.baseline_array = uv3.baseline_array[-1::-1]
    uv1 += uv3
    uv1 += uv2
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific baseline-times using pyuvdata. "
        "Combined data along baseline-time axis "
        "using pyuvdata. Combined data along "
        "baseline-time axis using pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add multiple axes
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv_ref = uv_full.copy()
    times = np.unique(uv_full.time_array)
    uv1.select(
        times=times[0 : len(times) // 2], polarizations=uv1.polarization_array[0:2]
    )
    uv2.select(
        times=times[len(times) // 2 :], polarizations=uv2.polarization_array[2:4]
    )
    uv1 += uv2
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific times, polarizations using "
        "pyuvdata. Combined data along "
        "baseline-time, polarization axis "
        "using pyuvdata.",
        uv1.history,
    )
    blt_ind1 = np.array(
        [
            ind
            for ind in range(uv_full.Nblts)
            if uv_full.time_array[ind] in times[0 : len(times) // 2]
        ]
    )
    blt_ind2 = np.array(
        [
            ind
            for ind in range(uv_full.Nblts)
            if uv_full.time_array[ind] in times[len(times) // 2 :]
        ]
    )
    # Zero out missing data in reference object
    uv_ref.data_array[blt_ind1, :, :, 2:] = 0.0
    uv_ref.nsample_array[blt_ind1, :, :, 2:] = 0.0
    uv_ref.flag_array[blt_ind1, :, :, 2:] = True
    uv_ref.data_array[blt_ind2, :, :, 0:2] = 0.0
    uv_ref.nsample_array[blt_ind2, :, :, 0:2] = 0.0
    uv_ref.flag_array[blt_ind2, :, :, 0:2] = True
    uv1.history = uv_full.history
    assert uv1 == uv_ref

    # Another combo
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv_ref = uv_full.copy()
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0 : len(times) // 2], freq_chans=np.arange(0, 32))
    uv2.select(times=times[len(times) // 2 :], freq_chans=np.arange(32, 64))
    uv1 += uv2
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific times, frequencies using "
        "pyuvdata. Combined data along "
        "baseline-time, frequency axis using "
        "pyuvdata.",
        uv1.history,
    )
    blt_ind1 = np.array(
        [
            ind
            for ind in range(uv_full.Nblts)
            if uv_full.time_array[ind] in times[0 : len(times) // 2]
        ]
    )
    blt_ind2 = np.array(
        [
            ind
            for ind in range(uv_full.Nblts)
            if uv_full.time_array[ind] in times[len(times) // 2 :]
        ]
    )
    # Zero out missing data in reference object
    uv_ref.data_array[blt_ind1, :, 32:, :] = 0.0
    uv_ref.nsample_array[blt_ind1, :, 32:, :] = 0.0
    uv_ref.flag_array[blt_ind1, :, 32:, :] = True
    uv_ref.data_array[blt_ind2, :, 0:32, :] = 0.0
    uv_ref.nsample_array[blt_ind2, :, 0:32, :] = 0.0
    uv_ref.flag_array[blt_ind2, :, 0:32, :] = True
    uv1.history = uv_full.history
    assert uv1 == uv_ref

    # Add without inplace
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0 : len(times) // 2])
    uv2.select(times=times[len(times) // 2 :])
    uv1 = uv1 + uv2
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific times using pyuvdata. "
        "Combined data along baseline-time "
        "axis using pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Check warnings
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(33, 64))
    with uvtest.check_warnings(
        UserWarning,
        [
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "Combined frequencies are not evenly spaced",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv1.__add__(uv2)

    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(freq_chans=[0])
    uv2.select(freq_chans=[3])
    with uvtest.check_warnings(
        UserWarning,
        [
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "Combined frequencies are not contiguous",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv1.__iadd__(uv2)

    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(freq_chans=[0])
    uv2.select(freq_chans=[1])
    uv2.freq_array += uv2._channel_width.tols[1] / 2.0
    with uvtest.check_warnings(
        UserWarning,
        [
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv1.__iadd__(uv2)

    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(polarizations=uv1.polarization_array[0:2])
    uv2.select(polarizations=uv2.polarization_array[3])
    with uvtest.check_warnings(
        UserWarning,
        [
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "Combined polarizations are not evenly spaced",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv1.__iadd__(uv2)

    # Combining histories
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(polarizations=uv1.polarization_array[0:2])
    uv2.select(polarizations=uv2.polarization_array[2:4])
    uv2.history += " testing the history. AIPS WTSCAL = 1.0"
    uv1 += uv2
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific polarizations using pyuvdata. "
        "Combined data along polarization "
        "axis using pyuvdata. testing the history.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # test add of autocorr-only and crosscorr-only objects
    uv_full = UVData()
    uv_full.read_uvh5(os.path.join(DATA_PATH, "zen.2457698.40355.xx.HH.uvcA.uvh5"))
    bls = uv_full.get_antpairs()
    autos = [bl for bl in bls if bl[0] == bl[1]]
    cross = sorted(set(bls) - set(autos))
    uv_auto = uv_full.select(bls=autos, inplace=False)
    uv_cross = uv_full.select(bls=cross, inplace=False)
    uv1 = uv_auto + uv_cross
    assert uv1.Nbls == uv_auto.Nbls + uv_cross.Nbls
    uv2 = uv_cross + uv_auto
    assert uv2.Nbls == uv_auto.Nbls + uv_cross.Nbls


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_add_drift(casa_uvfits):
    uv_full = casa_uvfits
    uv_full.unphase_to_drift()

    # Add frequencies
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(32, 64))
    uv1 += uv2
    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific frequencies using pyuvdata. "
        "Combined data along frequency "
        "axis using pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add polarizations
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(polarizations=uv1.polarization_array[0:2])
    uv2.select(polarizations=uv2.polarization_array[2:4])
    uv1 += uv2
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific polarizations using pyuvdata. "
        "Combined data along polarization "
        "axis using pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add times
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0 : len(times) // 2])
    uv2.select(times=times[len(times) // 2 :])
    uv1 += uv2
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific times using pyuvdata. "
        "Combined data along baseline-time "
        "axis using pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add baselines
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    ant_list = list(range(15))  # Roughly half the antennas in the data
    # All blts where ant_1 is in list
    ind1 = [i for i in range(uv1.Nblts) if uv1.ant_1_array[i] in ant_list]
    ind2 = [i for i in range(uv1.Nblts) if uv1.ant_1_array[i] not in ant_list]
    uv1.select(blt_inds=ind1)
    uv2.select(blt_inds=ind2)
    uv1 += uv2
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific baseline-times using pyuvdata. "
        "Combined data along baseline-time "
        "axis using pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add multiple axes
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv_ref = uv_full.copy()
    times = np.unique(uv_full.time_array)
    uv1.select(
        times=times[0 : len(times) // 2], polarizations=uv1.polarization_array[0:2]
    )
    uv2.select(
        times=times[len(times) // 2 :], polarizations=uv2.polarization_array[2:4]
    )
    uv1 += uv2
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific times, polarizations using "
        "pyuvdata. Combined data along "
        "baseline-time, polarization "
        "axis using pyuvdata.",
        uv1.history,
    )
    blt_ind1 = np.array(
        [
            ind
            for ind in range(uv_full.Nblts)
            if uv_full.time_array[ind] in times[0 : len(times) // 2]
        ]
    )
    blt_ind2 = np.array(
        [
            ind
            for ind in range(uv_full.Nblts)
            if uv_full.time_array[ind] in times[len(times) // 2 :]
        ]
    )
    # Zero out missing data in reference object
    uv_ref.data_array[blt_ind1, :, :, 2:] = 0.0
    uv_ref.nsample_array[blt_ind1, :, :, 2:] = 0.0
    uv_ref.flag_array[blt_ind1, :, :, 2:] = True
    uv_ref.data_array[blt_ind2, :, :, 0:2] = 0.0
    uv_ref.nsample_array[blt_ind2, :, :, 0:2] = 0.0
    uv_ref.flag_array[blt_ind2, :, :, 0:2] = True
    uv1.history = uv_full.history
    assert uv1 == uv_ref

    # Another combo
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv_ref = uv_full.copy()
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0 : len(times) // 2], freq_chans=np.arange(0, 32))
    uv2.select(times=times[len(times) // 2 :], freq_chans=np.arange(32, 64))
    uv1 += uv2
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific times, frequencies using "
        "pyuvdata. Combined data along "
        "baseline-time, frequency "
        "axis using pyuvdata.",
        uv1.history,
    )
    blt_ind1 = np.array(
        [
            ind
            for ind in range(uv_full.Nblts)
            if uv_full.time_array[ind] in times[0 : len(times) // 2]
        ]
    )
    blt_ind2 = np.array(
        [
            ind
            for ind in range(uv_full.Nblts)
            if uv_full.time_array[ind] in times[len(times) // 2 :]
        ]
    )
    # Zero out missing data in reference object
    uv_ref.data_array[blt_ind1, :, 32:, :] = 0.0
    uv_ref.nsample_array[blt_ind1, :, 32:, :] = 0.0
    uv_ref.flag_array[blt_ind1, :, 32:, :] = True
    uv_ref.data_array[blt_ind2, :, 0:32, :] = 0.0
    uv_ref.nsample_array[blt_ind2, :, 0:32, :] = 0.0
    uv_ref.flag_array[blt_ind2, :, 0:32, :] = True
    uv1.history = uv_full.history
    assert uv1 == uv_ref

    # Add without inplace
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0 : len(times) // 2])
    uv2.select(times=times[len(times) // 2 :])
    uv1 = uv1 + uv2
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific times using pyuvdata. "
        "Combined data along baseline-time "
        "axis using pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Check warnings
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(33, 64))
    with uvtest.check_warnings(
        UserWarning,
        [
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "Combined frequencies are not evenly spaced",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv1.__add__(uv2)

    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(freq_chans=[0])
    uv2.select(freq_chans=[3])
    with uvtest.check_warnings(
        UserWarning,
        [
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "Combined frequencies are not contiguous",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv1.__iadd__(uv2)

    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(polarizations=uv1.polarization_array[0:2])
    uv2.select(polarizations=uv2.polarization_array[3])
    with uvtest.check_warnings(
        UserWarning,
        [
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "Combined polarizations are not evenly spaced",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv1.__iadd__(uv2)

    # Combining histories
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(polarizations=uv1.polarization_array[0:2])
    uv2.select(polarizations=uv2.polarization_array[2:4])
    uv2.history += " testing the history. AIPS WTSCAL = 1.0"
    uv1 += uv2
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific polarizations using pyuvdata. "
        "Combined data along polarization "
        "axis using pyuvdata. testing the history.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1 == uv_full


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_break_add(casa_uvfits):
    # Test failure modes of add function
    uv_full = casa_uvfits

    # Wrong class
    uv1 = uv_full.copy()
    uv1.select(freq_chans=np.arange(0, 32))
    pytest.raises(ValueError, uv1.__iadd__, np.zeros(5))

    # One phased, one not
    uv2 = uv_full.copy()
    uv2.unphase_to_drift()

    pytest.raises(ValueError, uv1.__iadd__, uv2)

    # Different units
    uv2 = uv_full.copy()
    uv2.select(freq_chans=np.arange(32, 64))
    uv2.vis_units = "Jy"
    pytest.raises(ValueError, uv1.__iadd__, uv2)

    # Overlapping data
    uv2 = uv_full.copy()
    pytest.raises(ValueError, uv1.__iadd__, uv2)

    # Different integration_time
    uv2 = uv_full.copy()
    uv2.select(freq_chans=np.arange(32, 64))
    uv2.integration_time *= 2
    pytest.raises(ValueError, uv1.__iadd__, uv2)


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "test_func,extra_kwargs", [("__add__", {}), ("fast_concat", {"axis": "blt"})]
)
def test_add_error_drift_and_rephase(casa_uvfits, test_func, extra_kwargs):
    uv_full = casa_uvfits

    with pytest.raises(ValueError) as cm:
        getattr(uv_full, test_func)(
            uv_full, phase_center_radec=(0, 45), unphase_to_drift=True, **extra_kwargs
        )
    assert str(cm.value).startswith(
        "phase_center_radec cannot be set if " "unphase_to_drift is True."
    )


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "test_func,extra_kwargs", [("__add__", {}), ("fast_concat", {"axis": "blt"})]
)
def test_add_this_phased_unphase_to_drift(uv_phase_time_split, test_func, extra_kwargs):
    (uv_phase_1, uv_phase_2, uv_phase, uv_raw_1, uv_raw_2, uv_raw) = uv_phase_time_split
    func_kwargs = {
        "unphase_to_drift": True,
        "inplace": False,
    }
    func_kwargs.update(extra_kwargs)
    with uvtest.check_warnings(UserWarning, "Unphasing this UVData object to drift"):
        uv_out = getattr(uv_phase_1, test_func)(uv_raw_2, **func_kwargs)
    # the histories will be different here
    # but everything else should match.
    uv_out.history = copy.deepcopy(uv_raw.history)
    # ensure baseline time order is the same
    # because fast_concat will not order for us
    uv_out.reorder_blts(order="time", minor_order="baseline")
    assert uv_out.phase_type == "drift"
    assert uv_out == uv_raw


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "test_func,extra_kwargs", [("__add__", {}), ("fast_concat", {"axis": "blt"})]
)
def test_add_other_phased_unphase_to_drift(
    uv_phase_time_split, test_func, extra_kwargs
):
    (uv_phase_1, uv_phase_2, uv_phase, uv_raw_1, uv_raw_2, uv_raw) = uv_phase_time_split

    func_kwargs = {
        "unphase_to_drift": True,
        "inplace": False,
    }
    func_kwargs.update(extra_kwargs)
    with uvtest.check_warnings(UserWarning, "Unphasing other UVData object to drift"):
        uv_out = getattr(uv_raw_1, test_func)(uv_phase_2, **func_kwargs)

    # the histories will be different here
    # but everything else should match.
    uv_out.history = copy.deepcopy(uv_raw.history)
    # ensure baseline time order is the same
    # because fast_concat will not order for us
    uv_out.reorder_blts(order="time", minor_order="baseline")
    assert uv_out.phase_type == "drift"
    assert uv_out == uv_raw


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "test_func,extra_kwargs", [("__add__", {}), ("fast_concat", {"axis": "blt"})]
)
def test_add_this_rephase_new_phase_center(
    uv_phase_time_split, test_func, extra_kwargs
):
    (uv_phase_1, uv_phase_2, uv_phase, uv_raw_1, uv_raw_2, uv_raw) = uv_phase_time_split

    phase_center_radec = (Angle("0d").rad, Angle("-30d").rad)

    # phase each half to different spots
    uv_raw_1.phase(
        ra=0, dec=0, use_ant_pos=True,
    )
    uv_raw_2.phase(
        ra=phase_center_radec[0], dec=phase_center_radec[1], use_ant_pos=True
    )
    # phase original to phase_center_radec
    uv_raw.phase(ra=phase_center_radec[0], dec=phase_center_radec[1], use_ant_pos=True)

    func_kwargs = {
        "inplace": False,
        "phase_center_radec": phase_center_radec,
        "use_ant_pos": True,
    }
    func_kwargs.update(extra_kwargs)
    with uvtest.check_warnings(
        UserWarning, "Phasing this UVData object to phase_center_radec",
    ):
        uv_out = getattr(uv_raw_1, test_func)(uv_raw_2, **func_kwargs)

    # the histories will be different here
    # but everything else should match.
    uv_out.history = copy.deepcopy(uv_raw.history)
    # ensure baseline time order is the same
    # because fast_concat will not order for us
    uv_out.reorder_blts(order="time", minor_order="baseline")
    assert (uv_out.phase_center_ra, uv_out.phase_center_dec) == phase_center_radec
    assert uv_out == uv_raw


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "test_func,extra_kwargs", [("__add__", {}), ("fast_concat", {"axis": "blt"})]
)
def test_add_other_rephase_new_phase_center(
    uv_phase_time_split, test_func, extra_kwargs
):
    (uv_phase_1, uv_phase_2, uv_phase, uv_raw_1, uv_raw_2, uv_raw) = uv_phase_time_split

    phase_center_radec = (Angle("0d").rad, Angle("-30d").rad)

    # phase each half to different spots
    uv_raw_1.phase(
        ra=phase_center_radec[0], dec=phase_center_radec[1], use_ant_pos=True,
    )
    uv_raw_2.phase(
        ra=0, dec=0, use_ant_pos=True,
    )
    # phase original to phase_center_radec
    uv_raw.phase(
        ra=phase_center_radec[0], dec=phase_center_radec[1], use_ant_pos=True,
    )

    func_kwargs = {
        "inplace": False,
        "phase_center_radec": phase_center_radec,
        "use_ant_pos": True,
    }
    func_kwargs.update(extra_kwargs)
    with uvtest.check_warnings(
        UserWarning, "Phasing other UVData object to phase_center_radec"
    ):
        uv_out = getattr(uv_raw_1, test_func)(uv_raw_2, **func_kwargs)

    # the histories will be different here
    # but everything else should match.
    uv_out.history = copy.deepcopy(uv_raw.history)

    # ensure baseline time order is the same
    # because fast_concat will not order for us
    uv_out.reorder_blts(order="time", minor_order="baseline")
    assert uv_out.phase_type == "phased"
    assert (uv_out.phase_center_ra, uv_out.phase_center_dec) == phase_center_radec
    assert uv_out == uv_raw


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "test_func,extra_kwargs", [("__add__", {}), ("fast_concat", {"axis": "blt"})]
)
def test_add_error_too_long_phase_center(uv_phase_time_split, test_func, extra_kwargs):
    (uv_phase_1, uv_phase_2, uv_phase, uv_raw_1, uv_raw_2, uv_raw) = uv_phase_time_split
    phase_center_radec = (Angle("0d").rad, Angle("-30d").rad, 7)
    func_kwargs = {
        "inplace": False,
        "phase_center_radec": phase_center_radec,
    }
    func_kwargs.update(extra_kwargs)
    with pytest.raises(ValueError) as cm:
        getattr(uv_phase_1, test_func)(uv_phase_2, **func_kwargs)
    assert str(cm.value).startswith("phase_center_radec should have length 2.")


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_fast_concat(casa_uvfits):
    uv_full = casa_uvfits

    # Add frequencies
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv3 = uv_full.copy()
    uv1.select(freq_chans=np.arange(0, 20))
    uv2.select(freq_chans=np.arange(20, 40))
    uv3.select(freq_chans=np.arange(40, 64))
    uv1.fast_concat([uv2, uv3], "freq", inplace=True)
    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific frequencies using pyuvdata. "
        "Combined data along frequency axis "
        "using pyuvdata.",
        uv1.history,
    )

    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add frequencies - out of order
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv3 = uv_full.copy()
    uv1.select(freq_chans=np.arange(0, 20))
    uv2.select(freq_chans=np.arange(20, 40))
    uv3.select(freq_chans=np.arange(40, 64))
    with uvtest.check_warnings(
        UserWarning,
        [
            "The uvw_array does not match the expected values given the antenna "
            "positions."
        ]
        * 4
        + ["Combined frequencies are not evenly spaced"],
    ):
        uv2.fast_concat([uv1, uv3], "freq", inplace=True)

    assert uv2.Nfreqs == uv_full.Nfreqs
    assert uv2._freq_array != uv_full._freq_array
    assert uv2._data_array != uv_full._data_array

    # reorder frequencies and test that they are equal
    index_array = np.argsort(uv2.freq_array[0, :])
    uv2.freq_array = uv2.freq_array[:, index_array]
    uv2.data_array = uv2.data_array[:, :, index_array, :]
    uv2.nsample_array = uv2.nsample_array[:, :, index_array, :]
    uv2.flag_array = uv2.flag_array[:, :, index_array, :]
    uv2.history = uv_full.history
    assert uv2._freq_array == uv_full._freq_array
    assert uv2 == uv_full

    # Add polarizations
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv3 = uv_full.copy()
    uv1.select(polarizations=uv1.polarization_array[0:1])
    uv2.select(polarizations=uv2.polarization_array[1:3])
    uv3.select(polarizations=uv3.polarization_array[3:4])
    uv1.fast_concat([uv2, uv3], "polarization", inplace=True)
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific polarizations using pyuvdata. "
        "Combined data along polarization axis "
        "using pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add polarizations - out of order
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv3 = uv_full.copy()
    uv1.select(polarizations=uv1.polarization_array[0:1])
    uv2.select(polarizations=uv2.polarization_array[1:3])
    uv3.select(polarizations=uv3.polarization_array[3:4])
    with uvtest.check_warnings(
        UserWarning,
        [
            "The uvw_array does not match the expected values given the antenna "
            "positions."
        ]
        * 4
        + ["Combined polarizations are not evenly spaced"],
    ):
        uv2.fast_concat([uv1, uv3], "polarization", inplace=True)

    assert uv2._polarization_array != uv_full._polarization_array
    assert uv2._data_array != uv_full._data_array

    # reorder pols
    uv2.reorder_pols()
    uv2.history = uv_full.history
    assert uv2 == uv_full

    # Add times
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv3 = uv_full.copy()
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0 : len(times) // 3])
    uv2.select(times=times[len(times) // 3 : (len(times) // 3) * 2])
    uv3.select(times=times[(len(times) // 3) * 2 :])
    uv1.fast_concat([uv2, uv3], "blt", inplace=True)
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific times using pyuvdata. "
        "Combined data along baseline-time axis "
        "using pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add baselines
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    # divide in half to keep in order
    ind1 = np.arange(uv1.Nblts // 2)
    ind2 = np.arange(uv1.Nblts // 2, uv1.Nblts)
    uv1.select(blt_inds=ind1)
    uv2.select(blt_inds=ind2)
    uv1.fast_concat(uv2, "blt", inplace=True)
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific baseline-times using pyuvdata. "
        "Combined data along baseline-time axis "
        "using pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1, uv_full

    # Add baselines out of order
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(blt_inds=ind1)
    uv2.select(blt_inds=ind2)
    uv2.fast_concat(uv1, "blt", inplace=True)
    # test freq & pol arrays equal
    assert uv2._freq_array == uv_full._freq_array
    assert uv2._polarization_array == uv_full._polarization_array

    # test Nblt length arrays not equal but same shape
    assert uv2._ant_1_array != uv_full._ant_1_array
    assert uv2.ant_1_array.shape == uv_full.ant_1_array.shape
    assert uv2._ant_2_array != uv_full._ant_2_array
    assert uv2.ant_2_array.shape == uv_full.ant_2_array.shape
    assert uv2._uvw_array != uv_full._uvw_array
    assert uv2.uvw_array.shape == uv_full.uvw_array.shape
    assert uv2._time_array != uv_full._time_array
    assert uv2.time_array.shape == uv_full.time_array.shape
    assert uv2._baseline_array != uv_full._baseline_array
    assert uv2.baseline_array.shape == uv_full.baseline_array.shape
    assert uv2._data_array != uv_full._data_array
    assert uv2.data_array.shape == uv_full.data_array.shape

    # reorder blts to enable comparison
    uv2.reorder_blts()
    assert uv2.blt_order == ("time", "baseline")
    uv2.blt_order = None
    uv2.history = uv_full.history
    assert uv2 == uv_full

    # add baselines such that Nants_data needs to change
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    ant_list = list(range(15))  # Roughly half the antennas in the data
    # All blts where ant_1 is in list
    ind1 = [i for i in range(uv1.Nblts) if uv1.ant_1_array[i] in ant_list]
    ind2 = [i for i in range(uv1.Nblts) if uv1.ant_1_array[i] not in ant_list]
    uv1.select(blt_inds=ind1)
    uv2.select(blt_inds=ind2)
    uv2.fast_concat(uv1, "blt", inplace=True)

    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific baseline-times using pyuvdata. "
        "Combined data along baseline-time "
        "axis using pyuvdata.",
        uv2.history,
    )

    # test freq & pol arrays equal
    assert uv2._freq_array == uv_full._freq_array
    assert uv2._polarization_array == uv_full._polarization_array

    # test Nblt length arrays not equal but same shape
    assert uv2._ant_1_array != uv_full._ant_1_array
    assert uv2.ant_1_array.shape == uv_full.ant_1_array.shape
    assert uv2._ant_2_array != uv_full._ant_2_array
    assert uv2.ant_2_array.shape == uv_full.ant_2_array.shape
    assert uv2._uvw_array != uv_full._uvw_array
    assert uv2.uvw_array.shape == uv_full.uvw_array.shape
    assert uv2._time_array != uv_full._time_array
    assert uv2.time_array.shape == uv_full.time_array.shape
    assert uv2._baseline_array != uv_full._baseline_array
    assert uv2.baseline_array.shape == uv_full.baseline_array.shape
    assert uv2._data_array != uv_full._data_array
    assert uv2.data_array.shape == uv_full.data_array.shape

    # reorder blts to enable comparison
    uv2.reorder_blts()
    assert uv2.blt_order == ("time", "baseline")
    uv2.blt_order = None
    uv2.history = uv_full.history
    assert uv2 == uv_full

    # Add multiple axes
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    times = np.unique(uv_full.time_array)
    uv1.select(
        times=times[0 : len(times) // 2], polarizations=uv1.polarization_array[0:2]
    )
    uv2.select(
        times=times[len(times) // 2 :], polarizations=uv2.polarization_array[2:4]
    )
    pytest.raises(ValueError, uv1.fast_concat, uv2, "blt", inplace=True)

    # Another combo
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0 : len(times) // 2], freq_chans=np.arange(0, 32))
    uv2.select(times=times[len(times) // 2 :], freq_chans=np.arange(32, 64))
    pytest.raises(ValueError, uv1.fast_concat, uv2, "blt", inplace=True)

    # Add without inplace
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0 : len(times) // 2])
    uv2.select(times=times[len(times) // 2 :])
    uv1 = uv1.fast_concat(uv2, "blt", inplace=False)
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific times using pyuvdata. "
        "Combined data along baseline-time "
        "axis using pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Check warnings
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(33, 64))
    with uvtest.check_warnings(
        UserWarning,
        [
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "Combined frequencies are not evenly spaced",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv1.fast_concat(uv1, "freq", inplace=True)

    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(freq_chans=[0])
    uv2.select(freq_chans=[3])
    with uvtest.check_warnings(
        UserWarning,
        [
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "Combined frequencies are not contiguous",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv1.fast_concat(uv2, "freq")

    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(freq_chans=[0])
    uv2.select(freq_chans=[1])
    uv2.freq_array += uv2._channel_width.tols[1] / 2.0
    with uvtest.check_warnings(
        UserWarning,
        "The uvw_array does not match the expected values given the antenna "
        "positions.",
        nwarnings=3,
    ):
        uv1.fast_concat(uv2, "freq")

    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(polarizations=uv1.polarization_array[0:2])
    uv2.select(polarizations=uv2.polarization_array[3])
    with uvtest.check_warnings(
        UserWarning,
        [
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "Combined polarizations are not evenly spaced",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv1.fast_concat(uv2, "polarization")

    # Combining histories
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(polarizations=uv1.polarization_array[0:2])
    uv2.select(polarizations=uv2.polarization_array[2:4])
    uv2.history += " testing the history. AIPS WTSCAL = 1.0"
    uv1.fast_concat(uv2, "polarization", inplace=True)
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific polarizations using pyuvdata. "
        "Combined data along polarization "
        "axis using pyuvdata. testing the history.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # test add of autocorr-only and crosscorr-only objects
    uv_full = UVData()
    uv_full.read_uvh5(os.path.join(DATA_PATH, "zen.2457698.40355.xx.HH.uvcA.uvh5"))
    bls = uv_full.get_antpairs()
    autos = [bl for bl in bls if bl[0] == bl[1]]
    cross = sorted(set(bls) - set(autos))
    uv_auto = uv_full.select(bls=autos, inplace=False)
    uv_cross = uv_full.select(bls=cross, inplace=False)
    uv1 = uv_auto.fast_concat(uv_cross, "blt")
    assert uv1.Nbls == uv_auto.Nbls + uv_cross.Nbls
    uv2 = uv_cross.fast_concat(uv_auto, "blt")
    assert uv2.Nbls == uv_auto.Nbls + uv_cross.Nbls


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_fast_concat_errors(casa_uvfits):
    uv_full = casa_uvfits

    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(32, 64))
    with pytest.raises(ValueError, match="If axis is specifed it must be one of"):
        uv1.fast_concat(uv2, "foo", inplace=True)

    cal = UVCal()
    with pytest.raises(
        ValueError, match="Only UVData \\(or subclass\\) objects can be added"
    ):
        uv1.fast_concat(cal, "freq", inplace=True)


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_key2inds(casa_uvfits):
    # Test function to interpret key as antpair, pol
    uv = casa_uvfits

    # Get an antpair/pol combo
    ant1 = uv.ant_1_array[0]
    ant2 = uv.ant_2_array[0]
    pol = uv.polarization_array[0]
    bltind = np.where((uv.ant_1_array == ant1) & (uv.ant_2_array == ant2))[0]
    ind1, ind2, indp = uv._key2inds((ant1, ant2, pol))
    assert np.array_equal(bltind, ind1)
    assert np.array_equal(np.array([]), ind2)
    assert np.array_equal([0], indp[0])
    # Any of these inputs can also be a tuple of a tuple, so need to be checked twice.
    ind1, ind2, indp = uv._key2inds(((ant1, ant2, pol),))
    assert np.array_equal(bltind, ind1)
    assert np.array_equal(np.array([]), ind2)
    assert np.array_equal([0], indp[0])

    # Combo with pol as string
    ind1, ind2, indp = uv._key2inds((ant1, ant2, uvutils.polnum2str(pol)))
    assert np.array_equal([0], indp[0])
    ind1, ind2, indp = uv._key2inds(((ant1, ant2, uvutils.polnum2str(pol)),))
    assert np.array_equal([0], indp[0])

    # Check conjugation
    ind1, ind2, indp = uv._key2inds((ant2, ant1, pol))
    assert np.array_equal(bltind, ind2)
    assert np.array_equal(np.array([]), ind1)
    assert np.array_equal([0], indp[1])
    # Conjugation with pol as string
    ind1, ind2, indp = uv._key2inds((ant2, ant1, uvutils.polnum2str(pol)))
    assert np.array_equal(bltind, ind2)
    assert np.array_equal(np.array([]), ind1)
    assert np.array_equal([0], indp[1])
    assert np.array_equal([], indp[0])

    # Antpair only
    ind1, ind2, indp = uv._key2inds((ant1, ant2))
    assert np.array_equal(bltind, ind1)
    assert np.array_equal(np.array([]), ind2)
    assert np.array_equal(np.arange(uv.Npols), indp[0])
    ind1, ind2, indp = uv._key2inds(((ant1, ant2)))
    assert np.array_equal(bltind, ind1)
    assert np.array_equal(np.array([]), ind2)
    assert np.array_equal(np.arange(uv.Npols), indp[0])

    # Baseline number only
    ind1, ind2, indp = uv._key2inds(uv.antnums_to_baseline(ant1, ant2))
    assert np.array_equal(bltind, ind1)
    assert np.array_equal(np.array([]), ind2)
    assert np.array_equal(np.arange(uv.Npols), indp[0])
    ind1, ind2, indp = uv._key2inds((uv.antnums_to_baseline(ant1, ant2),))
    assert np.array_equal(bltind, ind1)
    assert np.array_equal(np.array([]), ind2)
    assert np.array_equal(np.arange(uv.Npols), indp[0])

    # Pol number only
    ind1, ind2, indp = uv._key2inds(pol)
    assert np.array_equal(np.arange(uv.Nblts), ind1)
    assert np.array_equal(np.array([]), ind2)
    assert np.array_equal(np.array([0]), indp[0])
    ind1, ind2, indp = uv._key2inds((pol))
    assert np.array_equal(np.arange(uv.Nblts), ind1)
    assert np.array_equal(np.array([]), ind2)
    assert np.array_equal(np.array([0]), indp[0])

    # Pol string only
    ind1, ind2, indp = uv._key2inds("LL")
    assert np.array_equal(np.arange(uv.Nblts), ind1)
    assert np.array_equal(np.array([]), ind2)
    assert np.array_equal(np.array([1]), indp[0])
    ind1, ind2, indp = uv._key2inds(("LL"))
    assert np.array_equal(np.arange(uv.Nblts), ind1)
    assert np.array_equal(np.array([]), ind2)
    assert np.array_equal(np.array([1]), indp[0])

    # Test invalid keys
    pytest.raises(KeyError, uv._key2inds, "I")  # pol str not in data
    pytest.raises(KeyError, uv._key2inds, -8)  # pol num not in data
    pytest.raises(KeyError, uv._key2inds, 6)  # bl num not in data
    pytest.raises(KeyError, uv._key2inds, (1, 1))  # ant pair not in data
    pytest.raises(KeyError, uv._key2inds, (1, 1, "rr"))  # ant pair not in data
    pytest.raises(KeyError, uv._key2inds, (0, 1, "xx"))  # pol not in data

    # Test autos are handled correctly
    uv.ant_2_array[0] = uv.ant_1_array[0]
    ind1, ind2, indp = uv._key2inds((ant1, ant1, pol))
    assert np.array_equal(ind1, [0])
    assert np.array_equal(ind2, [])


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_key2inds_conj_all_pols(casa_uvfits):
    uv = casa_uvfits

    ant1 = uv.ant_1_array[0]
    ant2 = uv.ant_2_array[0]
    bltind = np.where((uv.ant_1_array == ant1) & (uv.ant_2_array == ant2))[0]
    ind1, ind2, indp = uv._key2inds((ant2, ant1))

    # Pols in data are 'rr', 'll', 'rl', 'lr'
    # So conjugated order should be [0, 1, 3, 2]
    assert np.array_equal(bltind, ind2)
    assert np.array_equal(np.array([]), ind1)
    assert np.array_equal(np.array([]), indp[0])
    assert np.array_equal([0, 1, 3, 2], indp[1])


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_key2inds_conj_all_pols_fringe(casa_uvfits):
    uv = casa_uvfits

    uv.select(polarizations=["rl"])
    ant1 = uv.ant_1_array[0]
    ant2 = uv.ant_2_array[0]
    # Mix one instance of this baseline.
    uv.ant_1_array[0] = ant2
    uv.ant_2_array[0] = ant1
    bltind = np.where((uv.ant_1_array == ant1) & (uv.ant_2_array == ant2))[0]
    ind1, ind2, indp = uv._key2inds((ant1, ant2))

    assert np.array_equal(bltind, ind1)
    assert np.array_equal(np.array([]), ind2)
    assert np.array_equal(np.array([0]), indp[0])
    assert np.array_equal(np.array([]), indp[1])


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_key2inds_conj_all_pols_bl_fringe(casa_uvfits):
    uv = casa_uvfits

    uv.select(polarizations=["rl"])
    ant1 = uv.ant_1_array[0]
    ant2 = uv.ant_2_array[0]
    # Mix one instance of this baseline.
    uv.ant_1_array[0] = ant2
    uv.ant_2_array[0] = ant1
    uv.baseline_array[0] = uvutils.antnums_to_baseline(ant2, ant1, uv.Nants_telescope)
    bl = uvutils.antnums_to_baseline(ant1, ant2, uv.Nants_telescope)
    bltind = np.where((uv.ant_1_array == ant1) & (uv.ant_2_array == ant2))[0]
    ind1, ind2, indp = uv._key2inds(bl)

    assert np.array_equal(bltind, ind1)
    assert np.array_equal(np.array([]), ind2)
    assert np.array_equal(np.array([0]), indp[0])
    assert np.array_equal(np.array([]), indp[1])


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_key2inds_conj_all_pols_missing_data(casa_uvfits):
    uv = casa_uvfits

    uv.select(polarizations=["rl"])
    ant1 = uv.ant_1_array[0]
    ant2 = uv.ant_2_array[0]

    pytest.raises(KeyError, uv._key2inds, (ant2, ant1))


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_key2inds_conj_all_pols_bls(casa_uvfits):
    uv = casa_uvfits

    ant1 = uv.ant_1_array[0]
    ant2 = uv.ant_2_array[0]
    bl = uvutils.antnums_to_baseline(ant2, ant1, uv.Nants_telescope)
    bltind = np.where((uv.ant_1_array == ant1) & (uv.ant_2_array == ant2))[0]
    ind1, ind2, indp = uv._key2inds(bl)

    # Pols in data are 'rr', 'll', 'rl', 'lr'
    # So conjugated order should be [0, 1, 3, 2]
    assert np.array_equal(bltind, ind2)
    assert np.array_equal(np.array([]), ind1)
    assert np.array_equal(np.array([]), indp[0])
    assert np.array_equal([0, 1, 3, 2], indp[1])


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_key2inds_conj_all_pols_missing_data_bls(casa_uvfits):
    uv = casa_uvfits
    uv.select(polarizations=["rl"])
    ant1 = uv.ant_1_array[0]
    ant2 = uv.ant_2_array[0]
    bl = uvutils.antnums_to_baseline(ant2, ant1, uv.Nants_telescope)

    pytest.raises(KeyError, uv._key2inds, bl)


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_smart_slicing(casa_uvfits):
    # Test function to slice data
    uv = casa_uvfits

    # ind1 reg, ind2 empty, pol reg
    ind1 = 10 * np.arange(9)
    ind2 = []
    indp = [0, 1]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, (indp, []))
    dcheck = uv.data_array[ind1, :, :, :]
    dcheck = np.squeeze(dcheck[:, :, :, indp])
    assert np.all(d == dcheck)
    assert not d.flags.writeable
    # Ensure a view was returned
    uv.data_array[ind1[1], 0, 0, indp[0]] = 5.43
    assert d[1, 0, 0] == uv.data_array[ind1[1], 0, 0, indp[0]]

    # force copy
    d = uv._smart_slicing(uv.data_array, ind1, ind2, (indp, []), force_copy=True)
    dcheck = uv.data_array[ind1, :, :, :]
    dcheck = np.squeeze(dcheck[:, :, :, indp])
    assert np.all(d == dcheck)
    assert d.flags.writeable
    # Ensure a copy was returned
    uv.data_array[ind1[1], 0, 0, indp[0]] = 4.3
    assert d[1, 0, 0] != uv.data_array[ind1[1], 0, 0, indp[0]]

    # ind1 reg, ind2 empty, pol not reg
    ind1 = 10 * np.arange(9)
    ind2 = []
    indp = [0, 1, 3]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, (indp, []))
    dcheck = uv.data_array[ind1, :, :, :]
    dcheck = np.squeeze(dcheck[:, :, :, indp])
    assert np.all(d == dcheck)
    assert not d.flags.writeable
    # Ensure a copy was returned
    uv.data_array[ind1[1], 0, 0, indp[0]] = 1.2
    assert d[1, 0, 0] != uv.data_array[ind1[1], 0, 0, indp[0]]

    # ind1 not reg, ind2 empty, pol reg
    ind1 = [0, 4, 5]
    ind2 = []
    indp = [0, 1]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, (indp, []))
    dcheck = uv.data_array[ind1, :, :, :]
    dcheck = np.squeeze(dcheck[:, :, :, indp])
    assert np.all(d == dcheck)
    assert not d.flags.writeable
    # Ensure a copy was returned
    uv.data_array[ind1[1], 0, 0, indp[0]] = 8.2
    assert d[1, 0, 0] != uv.data_array[ind1[1], 0, 0, indp[0]]

    # ind1 not reg, ind2 empty, pol not reg
    ind1 = [0, 4, 5]
    ind2 = []
    indp = [0, 1, 3]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, (indp, []))
    dcheck = uv.data_array[ind1, :, :, :]
    dcheck = np.squeeze(dcheck[:, :, :, indp])
    assert np.all(d == dcheck)
    assert not d.flags.writeable
    # Ensure a copy was returned
    uv.data_array[ind1[1], 0, 0, indp[0]] = 3.4
    assert d[1, 0, 0] != uv.data_array[ind1[1], 0, 0, indp[0]]

    # ind1 empty, ind2 reg, pol reg
    # Note conjugation test ensures the result is a copy, not a view.
    ind1 = []
    ind2 = 10 * np.arange(9)
    indp = [0, 1]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, ([], indp))
    dcheck = uv.data_array[ind2, :, :, :]
    dcheck = np.squeeze(np.conj(dcheck[:, :, :, indp]))
    assert np.all(d == dcheck)

    # ind1 empty, ind2 reg, pol not reg
    ind1 = []
    ind2 = 10 * np.arange(9)
    indp = [0, 1, 3]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, ([], indp))
    dcheck = uv.data_array[ind2, :, :, :]
    dcheck = np.squeeze(np.conj(dcheck[:, :, :, indp]))
    assert np.all(d == dcheck)

    # ind1 empty, ind2 not reg, pol reg
    ind1 = []
    ind2 = [1, 4, 5, 10]
    indp = [0, 1]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, ([], indp))
    dcheck = uv.data_array[ind2, :, :, :]
    dcheck = np.squeeze(np.conj(dcheck[:, :, :, indp]))
    assert np.all(d == dcheck)

    # ind1 empty, ind2 not reg, pol not reg
    ind1 = []
    ind2 = [1, 4, 5, 10]
    indp = [0, 1, 3]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, ([], indp))
    dcheck = uv.data_array[ind2, :, :, :]
    dcheck = np.squeeze(np.conj(dcheck[:, :, :, indp]))
    assert np.all(d == dcheck)

    # ind1, ind2 not empty, pol reg
    ind1 = np.arange(20)
    ind2 = np.arange(30, 40)
    indp = [0, 1]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, (indp, indp))
    dcheck = np.append(
        uv.data_array[ind1, :, :, :], np.conj(uv.data_array[ind2, :, :, :]), axis=0
    )
    dcheck = np.squeeze(dcheck[:, :, :, indp])
    assert np.all(d == dcheck)

    # ind1, ind2 not empty, pol not reg
    ind1 = np.arange(20)
    ind2 = np.arange(30, 40)
    indp = [0, 1, 3]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, (indp, indp))
    dcheck = np.append(
        uv.data_array[ind1, :, :, :], np.conj(uv.data_array[ind2, :, :, :]), axis=0
    )
    dcheck = np.squeeze(dcheck[:, :, :, indp])
    assert np.all(d == dcheck)

    # test single element
    ind1 = [45]
    ind2 = []
    indp = [0, 1]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, (indp, []))
    dcheck = uv.data_array[ind1, :, :, :]
    dcheck = np.squeeze(dcheck[:, :, :, indp], axis=1)
    assert np.all(d == dcheck)

    # test single element
    ind1 = []
    ind2 = [45]
    indp = [0, 1]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, ([], indp))
    assert np.all(d == np.conj(dcheck))

    # Full squeeze
    ind1 = [45]
    ind2 = []
    indp = [0, 1]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, (indp, []), squeeze="full")
    dcheck = uv.data_array[ind1, :, :, :]
    dcheck = np.squeeze(dcheck[:, :, :, indp])
    assert np.all(d == dcheck)

    # Test invalid squeeze
    pytest.raises(
        ValueError,
        uv._smart_slicing,
        uv.data_array,
        ind1,
        ind2,
        (indp, []),
        squeeze="notasqueeze",
    )


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_get_data(casa_uvfits):
    # Test get_data function for easy access to data
    uv = casa_uvfits

    # Get an antpair/pol combo
    ant1 = uv.ant_1_array[0]
    ant2 = uv.ant_2_array[0]
    pol = uv.polarization_array[0]
    bltind = np.where((uv.ant_1_array == ant1) & (uv.ant_2_array == ant2))[0]
    dcheck = np.squeeze(uv.data_array[bltind, :, :, 0])
    d = uv.get_data(ant1, ant2, pol)
    assert np.all(dcheck == d)

    d = uv.get_data(ant1, ant2, uvutils.polnum2str(pol))
    assert np.all(dcheck == d)

    d = uv.get_data((ant1, ant2, pol))
    assert np.all(dcheck == d)

    with pytest.raises(ValueError) as cm:
        uv.get_data((ant1, ant2, pol), (ant1, ant2, pol))
    assert str(cm.value).startswith("no more than 3 key values can be passed")

    # Check conjugation
    d = uv.get_data(ant2, ant1, pol)
    assert np.all(dcheck == np.conj(d))

    # Check cross pol conjugation
    d = uv.get_data(ant2, ant1, uv.polarization_array[2])
    d1 = uv.get_data(ant1, ant2, uv.polarization_array[3])
    assert np.all(d == np.conj(d1))

    # Antpair only
    dcheck = np.squeeze(uv.data_array[bltind, :, :, :])
    d = uv.get_data(ant1, ant2)
    assert np.all(dcheck == d)

    # Pol number only
    dcheck = np.squeeze(uv.data_array[:, :, :, 0])
    d = uv.get_data(pol)
    assert np.all(dcheck == d)


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_get_flags(casa_uvfits):
    # Test function for easy access to flags
    uv = casa_uvfits

    # Get an antpair/pol combo
    ant1 = uv.ant_1_array[0]
    ant2 = uv.ant_2_array[0]
    pol = uv.polarization_array[0]
    bltind = np.where((uv.ant_1_array == ant1) & (uv.ant_2_array == ant2))[0]
    dcheck = np.squeeze(uv.flag_array[bltind, :, :, 0])
    d = uv.get_flags(ant1, ant2, pol)
    assert np.all(dcheck == d)

    d = uv.get_flags(ant1, ant2, uvutils.polnum2str(pol))
    assert np.all(dcheck == d)

    d = uv.get_flags((ant1, ant2, pol))
    assert np.all(dcheck == d)

    with pytest.raises(ValueError) as cm:
        uv.get_flags((ant1, ant2, pol), (ant1, ant2, pol))
    assert str(cm.value).startswith("no more than 3 key values can be passed")

    # Check conjugation
    d = uv.get_flags(ant2, ant1, pol)
    assert np.all(dcheck == d)
    assert d.dtype == np.bool

    # Antpair only
    dcheck = np.squeeze(uv.flag_array[bltind, :, :, :])
    d = uv.get_flags(ant1, ant2)
    assert np.all(dcheck == d)

    # Pol number only
    dcheck = np.squeeze(uv.flag_array[:, :, :, 0])
    d = uv.get_flags(pol)
    assert np.all(dcheck == d)


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_get_nsamples(casa_uvfits):
    # Test function for easy access to nsample array
    uv = casa_uvfits

    # Get an antpair/pol combo
    ant1 = uv.ant_1_array[0]
    ant2 = uv.ant_2_array[0]
    pol = uv.polarization_array[0]
    bltind = np.where((uv.ant_1_array == ant1) & (uv.ant_2_array == ant2))[0]
    dcheck = np.squeeze(uv.nsample_array[bltind, :, :, 0])
    d = uv.get_nsamples(ant1, ant2, pol)
    assert np.all(dcheck == d)

    d = uv.get_nsamples(ant1, ant2, uvutils.polnum2str(pol))
    assert np.all(dcheck == d)

    d = uv.get_nsamples((ant1, ant2, pol))
    assert np.all(dcheck == d)

    with pytest.raises(ValueError) as cm:
        uv.get_nsamples((ant1, ant2, pol), (ant1, ant2, pol))
    assert str(cm.value).startswith("no more than 3 key values can be passed")

    # Check conjugation
    d = uv.get_nsamples(ant2, ant1, pol)
    assert np.all(dcheck == d)

    # Antpair only
    dcheck = np.squeeze(uv.nsample_array[bltind, :, :, :])
    d = uv.get_nsamples(ant1, ant2)
    assert np.all(dcheck == d)

    # Pol number only
    dcheck = np.squeeze(uv.nsample_array[:, :, :, 0])
    d = uv.get_nsamples(pol)
    assert np.all(dcheck == d)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_antpair2ind(paper_uvh5):
    # Test for baseline-time axis indexer
    uv = paper_uvh5

    # get indices
    inds = uv.antpair2ind(0, 1, ordered=False)
    # fmt: off
    np.testing.assert_array_equal(
        inds,
        np.array(
            [
                1, 22, 43, 64, 85, 106, 127, 148, 169,
                190, 211, 232, 253, 274, 295, 316, 337,
                358, 379
            ]
        )
    )
    # fmt: on
    assert np.issubdtype(inds.dtype, np.integer)

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_antpair2ind_conj(paper_uvh5):
    # conjugate (and use key rather than arg expansion)
    uv = paper_uvh5
    inds = uv.antpair2ind(0, 1, ordered=False)
    inds2 = uv.antpair2ind((1, 0), ordered=False)
    np.testing.assert_array_equal(inds, inds2)
    assert np.issubdtype(inds2.dtype, np.integer)

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_antpair2ind_ordered(paper_uvh5):
    # test ordered
    uv = paper_uvh5
    inds = uv.antpair2ind(0, 1, ordered=False)

    # make sure conjugated baseline returns nothing
    inds2 = uv.antpair2ind(1, 0, ordered=True)
    assert inds2.size == 0

    # now use baseline actually in data
    inds2 = uv.antpair2ind(0, 1, ordered=True)
    np.testing.assert_array_equal(inds, inds2)
    assert np.issubdtype(inds2.dtype, np.integer)

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_antpair2ind_autos(paper_uvh5):
    # test autos w/ and w/o ordered
    uv = paper_uvh5

    inds = uv.antpair2ind(0, 0, ordered=True)
    inds2 = uv.antpair2ind(0, 0, ordered=False)
    np.testing.assert_array_equal(inds, inds2)
    assert np.issubdtype(inds.dtype, np.integer)
    assert np.issubdtype(inds2.dtype, np.integer)

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_antpair2ind_exceptions(paper_uvh5):
    # test exceptions
    uv = paper_uvh5

    with pytest.raises(ValueError, match="antpair2ind must be fed an antpair tuple"):
        uv.antpair2ind(1)
    with pytest.raises(ValueError, match="antpair2ind must be fed an antpair tuple"):
        uv.antpair2ind("bar", "foo")
    with pytest.raises(ValueError, match="ordered must be a boolean"):
        uv.antpair2ind(0, 1, "foo")

    return


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_get_times(casa_uvfits):
    # Test function for easy access to times, to work in conjunction with get_data
    uv = casa_uvfits
    # Get an antpair/pol combo (pol shouldn't actually effect result)
    ant1 = uv.ant_1_array[0]
    ant2 = uv.ant_2_array[0]
    pol = uv.polarization_array[0]
    bltind = np.where((uv.ant_1_array == ant1) & (uv.ant_2_array == ant2))[0]
    dcheck = uv.time_array[bltind]
    d = uv.get_times(ant1, ant2, pol)
    assert np.all(dcheck == d)

    d = uv.get_times(ant1, ant2, uvutils.polnum2str(pol))
    assert np.all(dcheck == d)

    d = uv.get_times((ant1, ant2, pol))
    assert np.all(dcheck == d)

    with pytest.raises(ValueError) as cm:
        uv.get_times((ant1, ant2, pol), (ant1, ant2, pol))
    assert str(cm.value).startswith("no more than 3 key values can be passed")

    # Check conjugation
    d = uv.get_times(ant2, ant1, pol)
    assert np.all(dcheck == d)

    # Antpair only
    d = uv.get_times(ant1, ant2)
    assert np.all(dcheck == d)

    # Pol number only
    d = uv.get_times(pol)
    assert np.all(d == uv.time_array)


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_antpairpol_iter(casa_uvfits):
    # Test generator
    uv = casa_uvfits
    pol_dict = {
        uvutils.polnum2str(uv.polarization_array[i]): i for i in range(uv.Npols)
    }
    keys = []
    pols = set()
    bls = set()
    for key, d in uv.antpairpol_iter():
        keys += key
        bl = uv.antnums_to_baseline(key[0], key[1])
        blind = np.where(uv.baseline_array == bl)[0]
        bls.add(bl)
        pols.add(key[2])
        dcheck = np.squeeze(uv.data_array[blind, :, :, pol_dict[key[2]]])
        assert np.all(dcheck == d)
    assert len(bls) == len(uv.get_baseline_nums())
    assert len(pols) == uv.Npols


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_get_ants(casa_uvfits):
    # Test function to get unique antennas in data
    uv = casa_uvfits

    ants = uv.get_ants()
    for ant in ants:
        assert (ant in uv.ant_1_array) or (ant in uv.ant_2_array)
    for ant in uv.ant_1_array:
        assert ant in ants
    for ant in uv.ant_2_array:
        assert ant in ants


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_get_enu_antpos():
    uvd = UVData()
    uvd.read_uvh5(os.path.join(DATA_PATH, "zen.2457698.40355.xx.HH.uvcA.uvh5"))
    # no center, no pick data ants
    antpos, ants = uvd.get_ENU_antpos(center=False, pick_data_ants=False)
    assert len(ants) == 113
    assert np.isclose(antpos[0, 0], 19.340211050751535)
    assert ants[0] == 0
    # test default behavior
    antpos2, ants = uvd.get_ENU_antpos()

    assert np.all(antpos == antpos2)
    # center
    antpos, ants = uvd.get_ENU_antpos(center=True, pick_data_ants=False)
    assert np.isclose(antpos[0, 0], 22.472442651767714)
    # pick data ants
    antpos, ants = uvd.get_ENU_antpos(center=True, pick_data_ants=True)
    assert ants[0] == 9
    assert np.isclose(antpos[0, 0], -0.0026981323386223721)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_telescope_loc_xyz_check(paper_uvh5, tmp_path):
    # test that improper telescope locations can still be read
    uv = paper_uvh5
    uv.telescope_location = uvutils.XYZ_from_LatLonAlt(*uv.telescope_location)
    # fix LST values
    uv.set_lsts_from_time_array()
    fname = str(tmp_path / "test.uvh5")
    uv.write_uvh5(fname, run_check=False, check_extra=False, clobber=True)

    # try to read file without checks (passing is implicit)
    uv.read(fname, run_check=False)

    # try to read without checks: assert it fails
    pytest.raises(ValueError, uv.read, fname)


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_get_pols(casa_uvfits):
    # Test function to get unique polarizations in string format
    uv = casa_uvfits
    pols = uv.get_pols()
    pols_data = ["rr", "ll", "lr", "rl"]
    assert sorted(pols) == sorted(pols_data)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_get_pols_x_orientation(paper_uvh5):
    uv_in = paper_uvh5

    uv_in.x_orientation = "east"

    pols = uv_in.get_pols()
    pols_data = ["en"]
    assert pols == pols_data

    uv_in.x_orientation = "north"

    pols = uv_in.get_pols()
    pols_data = ["ne"]
    assert pols == pols_data


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_get_feedpols(casa_uvfits):
    # Test function to get unique antenna feed polarizations in data. String format.
    uv = casa_uvfits
    pols = uv.get_feedpols()
    pols_data = ["r", "l"]
    assert sorted(pols) == sorted(pols_data)

    # Test break when pseudo-Stokes visibilities are present
    uv.polarization_array[0] = 1  # pseudo-Stokes I
    pytest.raises(ValueError, uv.get_feedpols)


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_parse_ants(casa_uvfits):
    # Test function to get correct antenna pairs and polarizations
    uv = casa_uvfits

    # All baselines
    ant_str = "all"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    assert isinstance(ant_pairs_nums, type(None))
    assert isinstance(polarizations, type(None))

    # Auto correlations
    ant_str = "auto"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    assert Counter(ant_pairs_nums) == Counter([])
    assert isinstance(polarizations, type(None))

    # Cross correlations
    ant_str = "cross"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    assert Counter(uv.get_antpairs()) == Counter(ant_pairs_nums)
    assert isinstance(polarizations, type(None))

    # pseudo-Stokes params
    ant_str = "pI,pq,pU,pv"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    pols_expected = [4, 3, 2, 1]
    assert isinstance(ant_pairs_nums, type(None))
    assert Counter(polarizations) == Counter(pols_expected)

    # Unparsible string
    ant_str = "none"
    pytest.raises(ValueError, uv.parse_ants, ant_str)

    # Single antenna number
    ant_str = "0"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    # fmt: off
    ant_pairs_expected = [(0, 1), (0, 2), (0, 3), (0, 6), (0, 7), (0, 8),
                          (0, 11), (0, 14), (0, 18), (0, 19), (0, 20),
                          (0, 21), (0, 22), (0, 23), (0, 24), (0, 26),
                          (0, 27)]
    # fmt: on
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Single antenna number not in the data
    ant_str = "10"
    with uvtest.check_warnings(
        UserWarning,
        "Warning: Antenna number 10 passed, but not present in the ant_1_array "
        "or ant_2_array",
    ):
        ant_pairs_nums, polarizations = uv.parse_ants(ant_str)

    assert isinstance(ant_pairs_nums, type(None))
    assert isinstance(polarizations, type(None))

    # Single antenna number with polarization, both not in the data
    ant_str = "10x"
    with uvtest.check_warnings(
        UserWarning,
        [
            "Warning: Antenna number 10 passed, but not present in the ant_1_array or "
            "ant_2_array",
            "Warning: Polarization XX,XY is not present in the polarization_array",
        ],
    ):
        ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    assert isinstance(ant_pairs_nums, type(None))
    assert isinstance(polarizations, type(None))

    # Multiple antenna numbers as list
    ant_str = "22,26"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    # fmt: off
    ant_pairs_expected = [(0, 22), (0, 26), (1, 22), (1, 26), (2, 22), (2, 26),
                          (3, 22), (3, 26), (6, 22), (6, 26), (7, 22),
                          (7, 26), (8, 22), (8, 26), (11, 22), (11, 26),
                          (14, 22), (14, 26), (18, 22), (18, 26),
                          (19, 22), (19, 26), (20, 22), (20, 26),
                          (21, 22), (21, 26), (22, 23), (22, 24),
                          (22, 26), (22, 27), (23, 26), (24, 26),
                          (26, 27)]
    # fmt: on
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Single baseline
    ant_str = "1_3"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 3)]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Single baseline with polarization
    ant_str = "1l_3r"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 3)]
    pols_expected = [-4]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert Counter(polarizations) == Counter(pols_expected)

    # Single baseline with single polarization in first entry
    ant_str = "1l_3,2x_3"
    with uvtest.check_warnings(
        UserWarning,
        "Warning: Polarization XX,XY is not present in the polarization_array",
    ):
        ant_pairs_nums, polarizations = uv.parse_ants(ant_str)

    ant_pairs_expected = [(1, 3), (2, 3)]
    pols_expected = [-2, -4]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert Counter(polarizations) == Counter(pols_expected)

    # Single baseline with single polarization in last entry
    ant_str = "1_3l,2_3x"
    with uvtest.check_warnings(
        UserWarning,
        "Warning: Polarization XX,YX is not present in the polarization_array",
    ):
        ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 3), (2, 3)]
    pols_expected = [-2, -3]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert Counter(polarizations) == Counter(pols_expected)

    # Multiple baselines as list
    ant_str = "1_2,1_3,1_11"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 2), (1, 3), (1, 11)]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Multiples baselines with polarizations as list
    ant_str = "1r_2l,1l_3l,1r_11r"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 2), (1, 3), (1, 11)]
    pols_expected = [-1, -2, -3]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert Counter(polarizations) == Counter(pols_expected)

    # Specific baselines with parenthesis
    ant_str = "(1,3)_11"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 11), (3, 11)]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Specific baselines with parenthesis
    ant_str = "1_(3,11)"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 3), (1, 11)]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Antenna numbers with polarizations
    ant_str = "(1l,2r)_(3l,6r)"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 3), (1, 6), (2, 3), (2, 6)]
    pols_expected = [-1, -2, -3, -4]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert Counter(polarizations) == Counter(pols_expected)

    # Antenna numbers with - for avoidance
    ant_str = "1_(-3,11)"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 11)]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Remove specific antenna number
    ant_str = "1,-3"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [
        (0, 1),
        (1, 2),
        (1, 6),
        (1, 7),
        (1, 8),
        (1, 11),
        (1, 14),
        (1, 18),
        (1, 19),
        (1, 20),
        (1, 21),
        (1, 22),
        (1, 23),
        (1, 24),
        (1, 26),
        (1, 27),
    ]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Remove specific baseline (same expected antenna pairs as above example)
    ant_str = "1,-1_3"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Antenna numbers with polarizations and - for avoidance
    ant_str = "1l_(-3r,11l)"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 11)]
    pols_expected = [-2]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert Counter(polarizations) == Counter(pols_expected)

    # Antenna numbers and pseudo-Stokes parameters
    ant_str = "(1l,2r)_(3l,6r),pI,pq"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 3), (1, 6), (2, 3), (2, 6)]
    pols_expected = [2, 1, -1, -2, -3, -4]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert Counter(polarizations) == Counter(pols_expected)

    # Multiple baselines with multiple polarizations, one pol to be removed
    ant_str = "1l_2,1l_3,-1l_3r"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 2), (1, 3)]
    pols_expected = [-2]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert Counter(polarizations) == Counter(pols_expected)

    # Multiple baselines with multiple polarizations, one pol (not in data)
    # to be removed
    ant_str = "1l_2,1l_3,-1x_3y"
    with uvtest.check_warnings(
        UserWarning, "Warning: Polarization XY is not present in the polarization_array"
    ):
        ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 2), (1, 3)]
    pols_expected = [-2, -4]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert Counter(polarizations) == Counter(pols_expected)

    # Test print toggle on single baseline with polarization
    ant_str = "1l_2l"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str, print_toggle=True)
    ant_pairs_expected = [(1, 2)]
    pols_expected = [-2]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert Counter(polarizations) == Counter(pols_expected)

    # Test ant_str='auto' on file with auto correlations
    uv = UVData()
    testfile = os.path.join(DATA_PATH, "zen.2457698.40355.xx.HH.uvcA.uvh5")
    uv.read(testfile)

    ant_str = "auto"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_nums = [
        9,
        10,
        20,
        22,
        31,
        43,
        53,
        64,
        65,
        72,
        80,
        81,
        88,
        89,
        96,
        97,
        104,
        105,
        112,
    ]
    ant_pairs_autos = [(ant_i, ant_i) for ant_i in ant_nums]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_autos)
    assert isinstance(polarizations, type(None))

    # Test cross correlation extraction on data with auto + cross
    ant_str = "cross"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_cross = list(itertools.combinations(ant_nums, 2))
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_cross)
    assert isinstance(polarizations, type(None))

    # Remove only polarization of single baseline
    ant_str = "all,-9x_10x"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = ant_pairs_autos + ant_pairs_cross
    ant_pairs_expected.remove((9, 10))
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Test appending all to beginning of strings that start with -
    ant_str = "-9"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = ant_pairs_autos + ant_pairs_cross
    for ant_i in ant_nums:
        ant_pairs_expected.remove((9, ant_i))
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_select_with_ant_str(casa_uvfits):
    # Test select function with ant_str argument
    uv = casa_uvfits
    inplace = False

    # All baselines
    ant_str = "all"
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(uv.get_antpairs())
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Cross correlations
    ant_str = "cross"
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(uv.get_antpairs())
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())
    # All baselines in data are cross correlations

    # Single antenna number
    ant_str = "0"
    ant_pairs = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 6),
        (0, 7),
        (0, 8),
        (0, 11),
        (0, 14),
        (0, 18),
        (0, 19),
        (0, 20),
        (0, 21),
        (0, 22),
        (0, 23),
        (0, 24),
        (0, 26),
        (0, 27),
    ]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Single antenna number not present in data
    ant_str = "10"
    with uvtest.check_warnings(
        UserWarning,
        [
            "Warning: Antenna number 10 passed, but not present in the "
            "ant_1_array or ant_2_array",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv.select(ant_str=ant_str, inplace=inplace)

    # Multiple antenna numbers as list
    ant_str = "22,26"
    ant_pairs = [
        (0, 22),
        (0, 26),
        (1, 22),
        (1, 26),
        (2, 22),
        (2, 26),
        (3, 22),
        (3, 26),
        (6, 22),
        (6, 26),
        (7, 22),
        (7, 26),
        (8, 22),
        (8, 26),
        (11, 22),
        (11, 26),
        (14, 22),
        (14, 26),
        (18, 22),
        (18, 26),
        (19, 22),
        (19, 26),
        (20, 22),
        (20, 26),
        (21, 22),
        (21, 26),
        (22, 23),
        (22, 24),
        (22, 26),
        (22, 27),
        (23, 26),
        (24, 26),
        (26, 27),
    ]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Single baseline
    ant_str = "1_3"
    ant_pairs = [(1, 3)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Single baseline with polarization
    ant_str = "1l_3r"
    ant_pairs = [(1, 3)]
    pols = ["lr"]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(pols)

    # Single baseline with single polarization in first entry
    ant_str = "1l_3,2x_3"
    # x,y pols not present in data
    with uvtest.check_warnings(
        UserWarning,
        [
            "Warning: Polarization XX,XY is not present in the polarization_array",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv.select(ant_str=ant_str, inplace=inplace)
    # with polarizations in data
    ant_str = "1l_3,2_3"
    ant_pairs = [(1, 3), (2, 3)]
    pols = ["ll", "lr"]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(pols)

    # Single baseline with single polarization in last entry
    ant_str = "1_3l,2_3x"
    # x,y pols not present in data
    with uvtest.check_warnings(
        UserWarning,
        [
            "Warning: Polarization XX,YX is not present in the polarization_array",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    # with polarizations in data
    ant_str = "1_3l,2_3"
    ant_pairs = [(1, 3), (2, 3)]
    pols = ["ll", "rl"]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(pols)

    # Multiple baselines as list
    ant_str = "1_2,1_3,1_10"
    # Antenna number 10 not in data
    with uvtest.check_warnings(
        UserWarning,
        [
            "Warning: Antenna number 10 passed, but not present in the "
            "ant_1_array or ant_2_array",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv2 = uv.select(ant_str=ant_str, inplace=inplace)

    ant_pairs = [(1, 2), (1, 3)]
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Multiples baselines with polarizations as list
    ant_str = "1r_2l,1l_3l,1r_11r"
    ant_pairs = [(1, 2), (1, 3), (1, 11)]
    pols = ["rr", "ll", "rl"]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(pols)

    # Specific baselines with parenthesis
    ant_str = "(1,3)_11"
    ant_pairs = [(1, 11), (3, 11)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Specific baselines with parenthesis
    ant_str = "1_(3,11)"
    ant_pairs = [(1, 3), (1, 11)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Antenna numbers with polarizations
    ant_str = "(1l,2r)_(3l,6r)"
    ant_pairs = [(1, 3), (1, 6), (2, 3), (2, 6)]
    pols = ["rr", "ll", "rl", "lr"]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(pols)

    # Antenna numbers with - for avoidance
    ant_str = "1_(-3,11)"
    ant_pairs = [(1, 11)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    ant_str = "(-1,3)_11"
    ant_pairs = [(3, 11)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Remove specific antenna number
    ant_str = "1,-3"
    ant_pairs = [
        (0, 1),
        (1, 2),
        (1, 6),
        (1, 7),
        (1, 8),
        (1, 11),
        (1, 14),
        (1, 18),
        (1, 19),
        (1, 20),
        (1, 21),
        (1, 22),
        (1, 23),
        (1, 24),
        (1, 26),
        (1, 27),
    ]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Remove specific baseline
    ant_str = "1,-1_3"
    ant_pairs = [
        (0, 1),
        (1, 2),
        (1, 6),
        (1, 7),
        (1, 8),
        (1, 11),
        (1, 14),
        (1, 18),
        (1, 19),
        (1, 20),
        (1, 21),
        (1, 22),
        (1, 23),
        (1, 24),
        (1, 26),
        (1, 27),
    ]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Antenna numbers with polarizations and - for avoidance
    ant_str = "1l_(-3r,11l)"
    ant_pairs = [(1, 11)]
    pols = ["ll"]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(pols)

    # Test pseudo-Stokes params with select
    ant_str = "pi,pQ"
    pols = ["pQ", "pI"]
    uv.polarization_array = np.array([4, 3, 2, 1])
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(uv.get_antpairs())
    assert Counter(uv2.get_pols()) == Counter(pols)

    # Test ant_str = 'auto' on file with auto correlations
    uv = UVData()
    testfile = os.path.join(DATA_PATH, "zen.2457698.40355.xx.HH.uvcA.uvh5")
    uv.read(testfile)

    ant_str = "auto"
    ant_nums = [
        9,
        10,
        20,
        22,
        31,
        43,
        53,
        64,
        65,
        72,
        80,
        81,
        88,
        89,
        96,
        97,
        104,
        105,
        112,
    ]
    ant_pairs_autos = [(ant_i, ant_i) for ant_i in ant_nums]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs_autos)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Test cross correlation extraction on data with auto + cross
    ant_str = "cross"
    ant_pairs_cross = list(itertools.combinations(ant_nums, 2))
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs_cross)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Remove only polarization of single baseline
    ant_str = "all,-9x_10x"
    ant_pairs = ant_pairs_autos + ant_pairs_cross
    ant_pairs.remove((9, 10))
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Test appending all to beginning of strings that start with -
    ant_str = "-9"
    ant_pairs = ant_pairs_autos + ant_pairs_cross
    for ant_i in ant_nums:
        ant_pairs.remove((9, ant_i))
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "kwargs,message",
    [
        (
            {"ant_str": "", "antenna_nums": []},
            "Cannot provide ant_str with antenna_nums, antenna_names, bls, or "
            "polarizations.",
        ),
        (
            {"ant_str": "", "antenna_names": []},
            "Cannot provide ant_str with antenna_nums, antenna_names, bls, or "
            "polarizations.",
        ),
        (
            {"ant_str": "", "bls": []},
            "Cannot provide ant_str with antenna_nums, antenna_names, bls, or "
            "polarizations.",
        ),
        (
            {"ant_str": "", "polarizations": []},
            "Cannot provide ant_str with antenna_nums, antenna_names, bls, or "
            "polarizations.",
        ),
        ({"ant_str": "auto"}, "There is no data matching ant_str=auto in this object."),
        (
            {"ant_str": "pI,pq,pU,pv"},
            "Polarization 4 is not present in the polarization_array",
        ),
        ({"ant_str": "none"}, "Unparsible argument none"),
    ],
)
def test_select_with_ant_str_errors(casa_uvfits, kwargs, message):
    uv = casa_uvfits

    with pytest.raises(ValueError, match=message):
        uv.select(**kwargs)


def test_set_uvws_from_antenna_pos():
    # Test set_uvws_from_antenna_positions function with phased data
    uv_object = UVData()
    testfile = os.path.join(DATA_PATH, "1133866760.uvfits")
    uv_object.read_uvfits(testfile)
    orig_uvw_array = np.copy(uv_object.uvw_array)

    with pytest.raises(ValueError) as cm:
        uv_object.set_uvws_from_antenna_positions(use_old_phase=True)
    assert str(cm.value).startswith("UVW calculation requires unphased data.")

    with pytest.raises(ValueError) as cm:
        with uvtest.check_warnings(
            [UserWarning, DeprecationWarning],
            [
                "Data will be unphased",
                "The original `phase` method will be deprecated in a future release.",
            ],
        ):
            uv_object.set_uvws_from_antenna_positions(
                allow_phasing=True, orig_phase_frame="xyz", use_old_phase=True
            )
    assert str(cm.value).startswith("Invalid parameter orig_phase_frame.")

    with pytest.raises(ValueError) as cm:
        with uvtest.check_warnings(
            [UserWarning, DeprecationWarning],
            [
                "Data will be unphased",
                "The original `phase` method will be deprecated in a future release.",
            ],
        ):
            uv_object.set_uvws_from_antenna_positions(
                allow_phasing=True,
                orig_phase_frame="gcrs",
                output_phase_frame="xyz",
                use_old_phase=True,
            )
    assert str(cm.value).startswith("Invalid parameter output_phase_frame.")

    with uvtest.check_warnings(
        [UserWarning, DeprecationWarning, DeprecationWarning],
        [
            "Data will be unphased",
            "The original `phase` method will be deprecated in",
            "The original `phase` method will be deprecated in",
        ],
    ):
        uv_object.set_uvws_from_antenna_positions(
            allow_phasing=True,
            orig_phase_frame="gcrs",
            output_phase_frame="gcrs",
            use_old_phase=True,
        )

    with uvtest.check_warnings(None):
        uv_object.set_uvws_from_antenna_positions(
            allow_phasing=True,
            orig_phase_frame="gcrs",
            output_phase_frame="gcrs",
            use_old_phase=True,
        )

    max_diff = np.amax(np.absolute(np.subtract(orig_uvw_array, uv_object.uvw_array)))
    assert np.isclose(max_diff, 0.0, atol=2)


def test_get_antenna_redundancies():
    uv0 = UVData()
    uv0.read_uvfits(
        os.path.join(DATA_PATH, "fewant_randsrc_airybeam_Nsrc100_10MHz.uvfits")
    )

    old_bl_array = np.copy(uv0.baseline_array)
    red_gps, centers, lengths = uv0.get_redundancies(
        use_antpos=True, include_autos=False, conjugate_bls=True
    )
    # new and old baseline Numbers are not the same (different conjugation)
    assert not np.allclose(uv0.baseline_array, old_bl_array)

    # assert all baselines are in the data (because it's conjugated to match)
    for i, gp in enumerate(red_gps):
        for bl in gp:
            assert bl in uv0.baseline_array

    # conjugate data differently
    uv0.conjugate_bls(convention="ant1<ant2")
    new_red_gps, new_centers, new_lengths, conjs = uv0.get_redundancies(
        use_antpos=True, include_autos=False, include_conjugates=True
    )

    assert conjs is None

    apos, anums = uv0.get_ENU_antpos()
    new_red_gps, new_centers, new_lengths = uvutils.get_antenna_redundancies(
        anums, apos, include_autos=False
    )

    # all redundancy info is the same
    assert red_gps == new_red_gps
    assert np.allclose(centers, new_centers)
    assert np.allclose(lengths, new_lengths)


@pytest.mark.parametrize("method", ("select", "average"))
@pytest.mark.parametrize("reconjugate", (True, False))
@pytest.mark.parametrize("flagging_level", ("none", "some", "all"))
def test_redundancy_contract_expand(method, reconjugate, flagging_level):
    # Test that a UVData object can be reduced to one baseline from each redundant group
    # and restored to its original form.

    uv0 = UVData()
    uv0.read_uvfits(
        os.path.join(DATA_PATH, "fewant_randsrc_airybeam_Nsrc100_10MHz.uvfits")
    )

    # Fails at lower precision because some baselines fall into multiple
    # redundant groups
    tol = 0.02

    if reconjugate:
        # the test file has groups that are either all not conjugated or all conjugated.
        # need to conjugate some so we have mixed groups to properly test the average
        # method.
        (
            orig_red_gps,
            orig_centers,
            orig_lengths,
            orig_conjugates,
        ) = uv0.get_redundancies(tol, include_conjugates=True)
        blt_inds_to_conj = []
        for gp_ind, gp in enumerate(orig_red_gps):
            if len(gp) > 1:
                blt_inds_to_conj.extend(
                    list(np.nonzero(uv0.baseline_array == gp[0])[0])
                )
        uv0.conjugate_bls(convention=np.array(blt_inds_to_conj))

    # Assign identical data to each redundant group, set up flagging.
    # This must be done after reconjugation because reconjugation can alter the index
    # baseline
    red_gps, centers, lengths, conjugates = uv0.get_redundancies(
        tol, include_conjugates=True
    )
    index_bls = []
    for gp_ind, gp in enumerate(red_gps):
        for bl in gp:
            inds = np.where(bl == uv0.baseline_array)
            uv0.data_array[inds] *= 0
            uv0.data_array[inds] += complex(gp_ind)
        index_bls.append(gp[0])

    if flagging_level == "none":
        assert np.all(~uv0.flag_array)
    elif flagging_level == "some":
        # flag all the index baselines in a redundant group
        for bl in index_bls:
            bl_locs = np.where(uv0.baseline_array == bl)
            uv0.flag_array[bl_locs, :, :, :] = True
    elif flagging_level == "all":
        uv0.flag_array[:] = True
        uv0.check()
        assert np.all(uv0.flag_array)

    uv3 = uv0.copy()
    if reconjugate:
        # undo the conjugations to make uv3 have different conjugations than uv0 to test
        # that we still get the same answer
        uv3.conjugate_bls(convention=np.array(blt_inds_to_conj))

    uv2 = uv0.compress_by_redundancy(method=method, tol=tol, inplace=False)

    if method == "average":
        gp_bl_use = []
        nbls_group = []
        for gp_ind, gp in enumerate(red_gps):
            bls_init = [bl for bl in gp if bl in uv0.baseline_array]
            nbls_group.append(len(bls_init))
            bl_use = [bl for bl in gp if bl in uv2.baseline_array]
            assert len(bl_use) == 1
            gp_bl_use.append(bl_use[0])

        for gp_ind, bl in enumerate(gp_bl_use):
            if flagging_level == "none" or flagging_level == "all":
                assert np.all(uv2.get_nsamples(bl) == nbls_group[gp_ind])
            else:
                assert np.all(uv2.get_nsamples(bl) == max((nbls_group[gp_ind] - 1), 1))
        if flagging_level == "all":
            assert np.all(uv2.flag_array)
        else:
            for gp_ind, bl in enumerate(gp_bl_use):
                if nbls_group[gp_ind] > 1:
                    assert np.all(~uv2.get_flags(bl))
    else:
        assert np.all(uv2.nsample_array == 1)
        if flagging_level == "some" or flagging_level == "all":
            assert np.all(uv2.flag_array)
        else:
            assert np.all(~uv2.flag_array)

    # Compare in-place to separated compression without the conjugation.
    uv3.compress_by_redundancy(method=method, tol=tol)
    if reconjugate:
        assert len(orig_red_gps) == len(red_gps)
        match_ind_list = []
        for gp_ind, gp in enumerate(red_gps):
            for bl in gp:
                match_ind = [
                    ind for ind, orig_gp in enumerate(orig_red_gps) if bl in orig_gp
                ]
                if len(match_ind) > 0:
                    break
            assert len(match_ind) == 1
            match_ind_list.append(match_ind[0])

        # the reconjugation of select baselines causes the set of baselines on the
        # two objects to differ. Need to match up baselines again
        unique_bls_2 = np.unique(uv2.baseline_array)
        unique_bls_3 = np.unique(uv3.baseline_array)
        unmatched_bls = list(
            set(unique_bls_2) - set(unique_bls_2).intersection(unique_bls_3)
        )

        # first find the ones that will be fixed by simple conjugation
        ant1, ant2 = uv2.baseline_to_antnums(unmatched_bls)
        conj_bls = uv2.antnums_to_baseline(ant2, ant1)
        bls_to_conj = list(set(unique_bls_3).intersection(conj_bls))
        if len(bls_to_conj) > 0:
            blts_to_conj = []
            for bl in bls_to_conj:
                blts_to_conj.extend(list(np.nonzero(uv3.baseline_array == bl)[0]))
            uv3.conjugate_bls(convention=blts_to_conj)

        # now check for ones that are still not matching
        unique_bls_3 = np.unique(uv3.baseline_array)
        unmatched_bls = list(
            set(unique_bls_2) - set(unique_bls_2).intersection(unique_bls_3)
        )
        for bl in unmatched_bls:
            assert bl in uv0.baseline_array
            for gp_ind, gp in enumerate(red_gps):
                if bl in gp:
                    bl_match = [
                        bl3
                        for bl3 in orig_red_gps[match_ind_list[gp_ind]]
                        if bl3 in unique_bls_3
                    ]
                    assert len(bl_match) == 1
                    blts = np.nonzero(uv3.baseline_array == bl_match[0])[0]
                    uv3.baseline_array[blts] = bl
                    # use the uvw values from the original for this baseline
                    orig_blts = np.nonzero(uv0.baseline_array == bl)[0]
                    uv3.uvw_array[blts, :] = uv0.uvw_array[orig_blts, :]
                    if method == "select":
                        # use the data values from the original for this baseline
                        # TODO: Spw axis to be collapsed in future release
                        uv2_blts = np.nonzero(uv2.baseline_array == bl)[0]
                        assert np.allclose(
                            uv2.data_array[uv2_blts, :, :, :],
                            uv0.data_array[orig_blts, :, :, :],
                        )
                        uv3.data_array[blts, :, :, :] = uv2.data_array[
                            uv2_blts, :, :, :
                        ]
                        if flagging_level == "some":
                            uv3.flag_array[:] = True

        uv3.ant_1_array, uv3.ant_2_array = uv3.baseline_to_antnums(uv3.baseline_array)
        uv3.Nants_data = uv3._calc_nants_data()
        unique_bls_3 = np.unique(uv3.baseline_array)
        unmatched_bls = list(
            set(unique_bls_2) - set(unique_bls_2).intersection(unique_bls_3)
        )
        assert set(unique_bls_2) == set(unique_bls_3)

    uv4 = uv2.copy()
    uv4.reorder_blts()
    uv3.reorder_blts()
    assert uv4 == uv3

    # check inflating gets back to the original
    with uvtest.check_warnings(
        UserWarning, match="Missing some redundant groups. Filling in available data."
    ):
        uv2.inflate_by_redundancy(tol=tol)

    # Confirm that we get the same result looping inflate -> compress -> inflate.
    uv3 = uv2.compress_by_redundancy(method=method, tol=tol, inplace=False)
    with uvtest.check_warnings(
        UserWarning, match="Missing some redundant groups. Filling in available data."
    ):
        uv3.inflate_by_redundancy(tol=tol)

    if method == "average":
        # with average, the nsample_array goes up by the number of baselines
        # averaged together.
        assert not np.allclose(uv3.nsample_array, uv2.nsample_array)
        # reset it to test other parameters
        uv3.nsample_array = uv2.nsample_array
    uv3.history = uv2.history
    assert uv2 == uv3

    uv2.history = uv0.history
    # Inflation changes the baseline ordering into the order of the redundant groups.
    # reorder bls for comparison
    uv0.reorder_blts(conj_convention="u>0")
    uv2.reorder_blts(conj_convention="u>0")
    uv2._uvw_array.tols = [0, tol]

    if method == "average":
        # with average, the nsample_array goes up by the number of baselines
        # averaged together.
        assert not np.allclose(uv2.nsample_array, uv0.nsample_array)
        # reset it to test other parameters
        uv2.nsample_array = uv0.nsample_array
    if flagging_level == "some":
        if method == "select":
            # inflated array will be entirely flagged
            assert np.all(uv2.flag_array)
            assert not np.allclose(uv0.flag_array, uv2.flag_array)
            uv2.flag_array = uv0.flag_array
        else:
            # flag arrays will not match -- inflated array will mostly be unflagged
            # it will only be flagged if only one in group
            assert not np.allclose(uv0.flag_array, uv2.flag_array)
            uv2.flag_array = uv0.flag_array

    assert uv2 == uv0


@pytest.mark.parametrize("method", ("select", "average"))
@pytest.mark.parametrize("flagging_level", ("none", "some", "all"))
def test_redundancy_contract_expand_variable_data(method, flagging_level):
    # Test that a UVData object can be reduced to one baseline from each redundant group
    # and restored to its original form.

    uv0 = UVData()
    uv0.read_uvfits(
        os.path.join(DATA_PATH, "fewant_randsrc_airybeam_Nsrc100_10MHz.uvfits")
    )

    # Fails at lower precision because some baselines fall into multiple
    # redundant groups
    tol = 0.02
    # Assign identical data to each redundant group in comparison object
    # Assign data to the index baseline and zeros elsewhere in the one to compress
    red_gps, centers, lengths = uv0.get_redundancies(
        tol=tol, use_antpos=True, conjugate_bls=True
    )
    index_bls = [gp[0] for gp in red_gps]
    uv0.data_array *= 0
    uv1 = uv0.copy()
    for gp_ind, gp in enumerate(red_gps):
        for bl in gp:
            inds = np.where(bl == uv0.baseline_array)
            uv1.data_array[inds] += complex(gp_ind)
            if bl in index_bls:
                uv0.data_array[inds] += complex(gp_ind)

    if flagging_level == "none":
        assert np.all(~uv0.flag_array)
    elif flagging_level == "some":
        # flag all the non index baselines in a redundant group
        uv0.flag_array[:, :, :, :] = True
        for bl in index_bls:
            bl_locs = np.where(uv0.baseline_array == bl)
            uv0.flag_array[bl_locs, :, :, :] = False
    elif flagging_level == "all":
        uv0.flag_array[:] = True
        uv0.check()
        assert np.all(uv0.flag_array)

    uv2 = uv0.compress_by_redundancy(method=method, tol=tol, inplace=False)

    # inflate to get back to the original size
    with uvtest.check_warnings(
        UserWarning, match="Missing some redundant groups. Filling in available data."
    ):
        uv2.inflate_by_redundancy(tol=tol)

    uv2.history = uv1.history
    # Inflation changes the baseline ordering into the order of the redundant groups.
    # reorder bls for comparison
    uv1.reorder_blts(conj_convention="u>0")
    uv2.reorder_blts(conj_convention="u>0")
    uv2._uvw_array.tols = [0, tol]

    if method == "select":
        if flagging_level == "all":
            assert uv2._flag_array != uv1._flag_array
            uv2.flag_array = uv1.flag_array
        assert uv2 == uv1
    else:
        if flagging_level == "some":
            for gp in red_gps:
                bls_init = [bl for bl in gp if bl in uv1.baseline_array]
                for bl in bls_init:
                    assert np.all(uv2.get_data(bl) == uv1.get_data(bl))
                    assert np.all(uv2.get_nsamples(bl) == uv1.get_nsamples(bl))
        else:
            assert uv2.data_array.min() < uv1.data_array.min()
            assert np.all(uv2.data_array <= uv1.data_array)
            for gp in red_gps:
                bls_init = [bl for bl in gp if bl in uv1.baseline_array]
                for bl in bls_init:
                    assert np.all(
                        uv2.get_data(bl) == (uv1.get_data(bl) / len(bls_init))
                    )
                    assert np.all(uv2.get_nsamples(bl) == len(bls_init))


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("method", ("select", "average"))
def test_redundancy_contract_expand_nblts_not_nbls_times_ntimes(method, casa_uvfits):
    uv0 = casa_uvfits

    # check that Nblts != Nbls * Ntimes
    assert uv0.Nblts != uv0.Nbls * uv0.Ntimes

    tol = 1.0

    # Assign identical data to each redundant group:
    red_gps, centers, lengths = uv0.get_redundancies(
        tol=tol, use_antpos=True, conjugate_bls=True
    )
    for i, gp in enumerate(red_gps):
        for bl in gp:
            inds = np.where(bl == uv0.baseline_array)
            uv0.data_array[inds, ...] *= 0
            uv0.data_array[inds, ...] += complex(i)

    if method == "average":
        with uvtest.check_warnings(
            UserWarning,
            "Index baseline in the redundant group does not have all the "
            "times, compressed object will be missing those times.",
            nwarnings=4,
        ):
            uv2 = uv0.compress_by_redundancy(method=method, tol=tol, inplace=False)
    else:
        uv2 = uv0.compress_by_redundancy(method=method, tol=tol, inplace=False)

    # check inflating gets back to the original
    with uvtest.check_warnings(
        UserWarning,
        [
            "Missing some redundant groups. Filling in available data.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv2.inflate_by_redundancy(tol=tol)

    uv2.history = uv0.history
    # Inflation changes the baseline ordering into the order of the redundant groups.
    # reorder bls for comparison
    uv0.reorder_blts()
    uv2.reorder_blts()
    uv2._uvw_array.tols = [0, tol]

    blt_inds = []
    missing_inds = []
    for bl, t in zip(uv0.baseline_array, uv0.time_array):
        if (bl, t) in zip(uv2.baseline_array, uv2.time_array):
            this_ind = np.where((uv2.baseline_array == bl) & (uv2.time_array == t))[0]
            blt_inds.append(this_ind[0])
        else:
            # this is missing because of the compress_by_redundancy step
            missing_inds.append(
                np.where((uv0.baseline_array == bl) & (uv0.time_array == t))[0]
            )

    uv3 = uv2.select(blt_inds=blt_inds, inplace=False)

    orig_inds_keep = list(np.arange(uv0.Nblts))
    for ind in missing_inds:
        orig_inds_keep.remove(ind)
    uv1 = uv0.select(blt_inds=orig_inds_keep, inplace=False)

    if method == "average":
        # the nsample array in the original object varies, so they
        # don't come out the same
        assert not np.allclose(uv3.nsample_array, uv1.nsample_array)
        uv3.nsample_array = uv1.nsample_array

    assert uv3 == uv1


def test_compress_redundancy_variable_inttime():
    uv0 = UVData()
    uv0.read_uvfits(
        os.path.join(DATA_PATH, "fewant_randsrc_airybeam_Nsrc100_10MHz.uvfits")
    )
    tol = 0.05
    ntimes_in = uv0.Ntimes

    # Assign identical data to each redundant group:
    red_gps, centers, lengths = uv0.get_redundancies(
        tol=tol, use_antpos=True, conjugate_bls=True
    )
    index_bls = [gp[0] for gp in red_gps]
    uv0.data_array *= 0
    # set different int time for index baseline in object to compress
    uv1 = uv0.copy()
    ave_int_time = np.average(uv0.integration_time)
    nbls_group = np.zeros(len(red_gps))
    for gp_ind, gp in enumerate(red_gps):
        for bl in gp:
            inds = np.where(bl == uv0.baseline_array)
            if inds[0].size > 0:
                nbls_group[gp_ind] += 1
            uv1.data_array[inds] += complex(gp_ind)
            uv0.data_array[inds] += complex(gp_ind)
            if bl not in index_bls:
                uv0.integration_time[inds] = ave_int_time / 2

    assert uv0._integration_time != uv1._integration_time

    with uvtest.check_warnings(
        UserWarning,
        "Integrations times are not identical in a redundant "
        "group. Averaging anyway but this may cause unexpected "
        "behavior.",
        nwarnings=56,
    ) as warn_record:
        uv0.compress_by_redundancy(method="average", tol=tol)
    assert len(warn_record) == np.sum(nbls_group > 1) * ntimes_in

    uv1.compress_by_redundancy(method="average", tol=tol)

    assert uv0 == uv1


@pytest.mark.parametrize("method", ("select", "average"))
def test_compress_redundancy_metadata_only_lst_update(method):
    uv0 = UVData()
    uv0.read_uvfits(
        os.path.join(DATA_PATH, "fewant_randsrc_airybeam_Nsrc100_10MHz.uvfits")
    )
    tol = 0.05

    # Assign identical data to each redundant group
    # modify times slightly to test lsts:
    red_gps, centers, lengths = uv0.get_redundancies(
        tol=tol, use_antpos=True, conjugate_bls=True
    )
    for i, gp in enumerate(red_gps):
        for bl_ind, bl in enumerate(gp):
            inds = np.where(bl == uv0.baseline_array)
            uv0.data_array[inds] *= 0
            uv0.data_array[inds] += complex(i)
            if bl_ind > 0:
                uv0.time_array[inds] += uv0._time_array.tols[1] * 0.9

    uv0.set_lsts_from_time_array()
    uv2 = uv0.copy(metadata_only=True)
    uv2.compress_by_redundancy(method=method, tol=tol, inplace=True)

    uv0.compress_by_redundancy(method=method, tol=tol)
    uv0.data_array = None
    uv0.flag_array = None
    uv0.nsample_array = None
    assert uv0 == uv2

    uv3 = uv2.copy()
    uv3.set_lsts_from_time_array()
    assert uv3._lst_array == uv2._lst_array


def test_compress_redundancy_wrong_method():
    uv0 = UVData()
    uv0.read_uvfits(
        os.path.join(DATA_PATH, "fewant_randsrc_airybeam_Nsrc100_10MHz.uvfits")
    )
    tol = 0.05
    with pytest.raises(ValueError, match="method must be one of"):
        uv0.compress_by_redundancy(method="foo", tol=tol, inplace=True)


@pytest.mark.parametrize("method", ("select", "average"))
def test_redundancy_missing_groups(method, tmp_path):
    # Check that if I try to inflate a compressed UVData that is missing
    # redundant groups, it will raise the right warnings and fill only what
    # data are available.

    uv0 = UVData()
    uv0.read_uvfits(
        os.path.join(DATA_PATH, "fewant_randsrc_airybeam_Nsrc100_10MHz.uvfits")
    )
    tol = 0.02
    num_select = 19

    uv0.compress_by_redundancy(method=method, tol=tol)
    fname = str(tmp_path / "temp_hera19_missingreds.uvfits")

    bls = np.unique(uv0.baseline_array)[:num_select]  # First twenty baseline groups
    uv0.select(bls=[uv0.baseline_to_antnums(bl) for bl in bls])
    uv0.write_uvfits(fname)
    uv1 = UVData()
    uv1.read_uvfits(fname)

    assert uv0 == uv1  # Check that writing compressed files causes no issues.

    with uvtest.check_warnings(
        UserWarning, match="Missing some redundant groups. Filling in available data."
    ):
        uv1.inflate_by_redundancy(tol=tol)

    uv2 = uv1.compress_by_redundancy(method=method, tol=tol, inplace=False)

    assert np.unique(uv2.baseline_array).size == num_select


def test_quick_redundant_vs_redundant_test_array():
    """Verify the quick redundancy calc returns the same groups as a known array."""
    uv = UVData()
    uv.read_uvfits(
        os.path.join(DATA_PATH, "fewant_randsrc_airybeam_Nsrc100_10MHz.uvfits")
    )
    uv.select(times=uv.time_array[0])
    uv.unphase_to_drift()
    uv.conjugate_bls(convention="u>0", use_enu=True)
    tol = 0.05
    # a quick and dirty redundancy calculation
    unique_bls, baseline_inds = np.unique(uv.baseline_array, return_index=True)
    uvw_vectors = np.take(uv.uvw_array, baseline_inds, axis=0)
    uvw_diffs = np.expand_dims(uvw_vectors, axis=0) - np.expand_dims(
        uvw_vectors, axis=1
    )
    uvw_diffs = np.linalg.norm(uvw_diffs, axis=2)

    reds = np.where(uvw_diffs < tol, unique_bls, 0)
    reds = np.ma.masked_where(reds == 0, reds)
    groups = []
    for bl in reds:
        grp = []
        grp.extend(bl.compressed())
        for other_bls in reds:
            if set(reds.compressed()).issubset(other_bls.compressed()):
                grp.extend(other_bls.compressed())
        grp = np.unique(grp).tolist()
        groups.append(grp)

    pad = len(max(groups, key=len))
    groups = np.array([i + [-1] * (pad - len(i)) for i in groups])
    groups = np.unique(groups, axis=0)
    groups = [[bl for bl in grp if bl != -1] for grp in groups]
    groups.sort(key=len)

    redundant_groups, centers, lengths, conj_inds = uv.get_redundancies(
        tol=tol, include_conjugates=True
    )
    redundant_groups.sort(key=len)
    assert groups == redundant_groups


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_redundancy_finder_when_nblts_not_nbls_times_ntimes(casa_uvfits):
    """Test the redundancy finder functions when Nblts != Nbls * Ntimes."""
    tol = 1  # meter
    uv = casa_uvfits
    uv.conjugate_bls(convention="u>0", use_enu=True)
    # check that Nblts != Nbls * Ntimes
    assert uv.Nblts != uv.Nbls * uv.Ntimes

    # a quick and dirty redundancy calculation
    unique_bls, baseline_inds = np.unique(uv.baseline_array, return_index=True)
    uvw_vectors = np.take(uv.uvw_array, baseline_inds, axis=0)
    uvw_diffs = np.expand_dims(uvw_vectors, axis=0) - np.expand_dims(
        uvw_vectors, axis=1
    )
    uvw_diffs = np.linalg.norm(uvw_diffs, axis=2)

    reds = np.where(uvw_diffs < tol, unique_bls, 0)
    reds = np.ma.masked_where(reds == 0, reds)
    groups = []
    for bl in reds:
        grp = []
        grp.extend(bl.compressed())
        for other_bls in reds:
            if set(reds.compressed()).issubset(other_bls.compressed()):
                grp.extend(other_bls.compressed())
        grp = np.unique(grp).tolist()
        groups.append(grp)

    pad = len(max(groups, key=len))
    groups = np.array([i + [-1] * (pad - len(i)) for i in groups])
    groups = np.unique(groups, axis=0)
    groups = [[bl for bl in grp if bl != -1] for grp in groups]
    groups.sort(key=len)

    redundant_groups, centers, lengths, conj_inds = uv.get_redundancies(
        tol=tol, include_conjugates=True
    )
    redundant_groups.sort(key=len)
    assert groups == redundant_groups


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_overlapping_data_add(casa_uvfits, tmp_path):
    # read in test data
    uv = casa_uvfits

    # slice into four objects
    blts1 = np.arange(500)
    blts2 = np.arange(500, 1360)
    uv1 = uv.select(polarizations=[-1, -2], blt_inds=blts1, inplace=False)
    uv2 = uv.select(polarizations=[-3, -4], blt_inds=blts1, inplace=False)
    uv3 = uv.select(polarizations=[-1, -2], blt_inds=blts2, inplace=False)
    uv4 = uv.select(polarizations=[-3, -4], blt_inds=blts2, inplace=False)

    # combine and check for equality
    uvfull = uv1 + uv2
    uvfull += uv3
    uvfull += uv4
    extra_history = (
        "Downselected to specific baseline-times, polarizations using pyuvdata. "
        "Combined data along polarization axis using pyuvdata. Combined data along "
        "baseline-time axis using pyuvdata. Overwrote invalid data using pyuvdata."
    )
    assert uvutils._check_histories(uvfull.history, uv.history + extra_history)
    uvfull.history = uv.history  # make histories match
    assert uv == uvfull

    # combine in a different order and check for equality
    uvfull = uv1.copy()
    uvfull += uv2
    uvfull2 = uv3.copy()
    uvfull2 += uv4
    uvfull += uvfull2
    extra_history2 = (
        "Downselected to specific baseline-times, polarizations using pyuvdata. "
        "Combined data along polarization axis using pyuvdata. Combined data along "
        "baseline-time axis using pyuvdata."
    )
    assert uvutils._check_histories(uvfull.history, uv.history + extra_history2)
    uvfull.history = uv.history  # make histories match
    assert uv == uvfull

    # check combination not-in-place
    uvfull = uv1 + uv2
    uvfull += uv3
    uvfull = uvfull + uv4
    uvfull.history = uv.history  # make histories match
    assert uv == uvfull

    # test raising error for adding objects incorrectly (i.e., having the object
    # with data to be overwritten come second)
    uvfull = uv1 + uv2
    uvfull += uv3
    pytest.raises(ValueError, uv4.__iadd__, uvfull)
    pytest.raises(ValueError, uv4.__add__, uv4, uvfull)

    # write individual objects out, and make sure that we can read in the list
    uv1_out = str(tmp_path / "uv1.uvfits")
    uv1.write_uvfits(uv1_out)
    uv2_out = str(tmp_path / "uv2.uvfits")
    uv2.write_uvfits(uv2_out)
    uv3_out = str(tmp_path / "uv3.uvfits")
    uv3.write_uvfits(uv3_out)
    uv4_out = str(tmp_path / "uv4.uvfits")
    uv4.write_uvfits(uv4_out)

    uvfull = UVData()
    uvfull.read(np.array([uv1_out, uv2_out, uv3_out, uv4_out]))
    uvfull.reorder_blts()
    uv.reorder_blts()
    assert uvutils._check_histories(uvfull.history, uv.history + extra_history2)
    uvfull.history = uv.history  # make histories match
    assert uvfull == uv

    # clean up after ourselves
    os.remove(uv1_out)
    os.remove(uv2_out)
    os.remove(uv3_out)
    os.remove(uv4_out)

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_lsts_from_time_with_only_unique(paper_uvh5):
    """
    Test `set_lsts_from_time_array` with only unique values is identical to full array.
    """
    uv = paper_uvh5
    lat, lon, alt = uv.telescope_location_lat_lon_alt_degrees
    # calculate the lsts for all elements in time array
    full_lsts = uvutils.get_lst_for_time(uv.time_array, lat, lon, alt)
    # use `set_lst_from_time_array` to set the uv.lst_array using only unique values
    uv.set_lsts_from_time_array()
    assert np.array_equal(full_lsts, uv.lst_array)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_lsts_from_time_with_only_unique_background(paper_uvh5):
    """
    Test `set_lsts_from_time_array` with only unique values is identical to full array.
    """
    uv = paper_uvh5
    lat, lon, alt = uv.telescope_location_lat_lon_alt_degrees
    # calculate the lsts for all elements in time array
    full_lsts = uvutils.get_lst_for_time(uv.time_array, lat, lon, alt)
    # use `set_lst_from_time_array` to set the uv.lst_array using only unique values
    proc = uv.set_lsts_from_time_array(background=True)
    proc.join()
    assert np.array_equal(full_lsts, uv.lst_array)


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_copy(casa_uvfits):
    """Test the copy method"""
    uv_object = casa_uvfits

    uv_object_copy = uv_object.copy()
    assert uv_object_copy == uv_object

    uv_object_copy = uv_object.copy(metadata_only=True)
    assert uv_object_copy.metadata_only

    for name in uv_object._data_params:
        setattr(uv_object, name, None)
    assert uv_object_copy == uv_object

    uv_object_copy = uv_object.copy()
    assert uv_object_copy == uv_object

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_upsample_in_time(hera_uvh5):
    """Test the upsample_in_time method"""
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))

    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline")

    # save some values for later
    init_data_size = uv_object.data_array.size
    init_wf = uv_object.get_data(0, 1)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # change the target integration time
    max_integration_time = np.amin(uv_object.integration_time) / 2.0
    uv_object.upsample_in_time(max_integration_time, blt_order="baseline")

    assert np.allclose(uv_object.integration_time, max_integration_time)
    # we should double the size of the data arrays
    assert uv_object.data_array.size == 2 * init_data_size
    # output data should be the same
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(init_wf[0, 0, 0], out_wf[0, 0, 0])

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(init_ns[0, 0, 0], out_ns[0, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_upsample_in_time_with_flags(hera_uvh5):
    """Test the upsample_in_time method with flags"""
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))

    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline")

    # save some values for later
    init_wf = uv_object.get_data(0, 1)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # change the target integration time
    max_integration_time = np.amin(uv_object.integration_time) / 2.0

    # add flags and upsample again
    inds01 = uv_object.antpair2ind(0, 1)
    uv_object.flag_array[inds01[0], 0, 0, 0] = True
    uv_object.upsample_in_time(max_integration_time, blt_order="baseline")

    # data and nsamples should be changed as normal, but flagged
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(init_wf[0, 0, 0], out_wf[0, 0, 0])
    out_flags = uv_object.get_flags(0, 1)
    assert np.all(out_flags[:2, 0, 0])
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(init_ns[0, 0, 0], out_ns[0, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_upsample_in_time_noninteger_resampling(hera_uvh5):
    """Test the upsample_in_time method with a non-integer resampling factor"""
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))

    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline")

    # save some values for later
    init_data_size = uv_object.data_array.size
    init_wf = uv_object.get_data(0, 1)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # change the target integration time
    max_integration_time = np.amin(uv_object.integration_time) * 0.75
    uv_object.upsample_in_time(max_integration_time, blt_order="baseline")

    assert np.allclose(uv_object.integration_time, max_integration_time * 0.5 / 0.75)
    # we should double the size of the data arrays
    assert uv_object.data_array.size == 2 * init_data_size
    # output data should be different by a factor of 2
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(init_wf[0, 0, 0], out_wf[0, 0, 0])

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(init_ns[0, 0, 0], out_ns[0, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_upsample_in_time_errors(hera_uvh5):
    """Test errors and warnings raised by upsample_in_time"""
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))

    # test using a too-small integration time
    max_integration_time = 1e-3 * np.amin(uv_object.integration_time)
    with pytest.raises(ValueError) as cm:
        uv_object.upsample_in_time(max_integration_time)
    assert str(cm.value).startswith("Decreasing the integration time by more than")

    # catch a warning for doing no work
    uv_object2 = uv_object.copy()
    max_integration_time = 2 * np.amax(uv_object.integration_time)
    with uvtest.check_warnings(
        UserWarning, "All values in the integration_time array are already longer"
    ):
        uv_object.upsample_in_time(max_integration_time)
    assert uv_object == uv_object2

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_upsample_in_time_summing_correlator_mode(hera_uvh5):
    """Test the upsample_in_time method with summing correlator mode"""
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))

    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline")

    # save some values for later
    init_data_size = uv_object.data_array.size
    init_wf = uv_object.get_data(0, 1)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # change the target integration time
    max_integration_time = np.amin(uv_object.integration_time) / 2.0
    uv_object.upsample_in_time(
        max_integration_time, blt_order="baseline", summing_correlator_mode=True
    )

    assert np.allclose(uv_object.integration_time, max_integration_time)
    # we should double the size of the data arrays
    assert uv_object.data_array.size == 2 * init_data_size
    # output data should be the half the input
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(init_wf[0, 0, 0] / 2, out_wf[0, 0, 0])

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(init_ns[0, 0, 0], out_ns[0, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_upsample_in_time_summing_correlator_mode_with_flags(hera_uvh5):
    """Test the upsample_in_time method with summing correlator mode and flags"""
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))

    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline")

    # save some values for later
    init_wf = uv_object.get_data(0, 1)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # add flags and upsample again
    inds01 = uv_object.antpair2ind(0, 1)
    uv_object.flag_array[inds01[0], 0, 0, 0] = True
    max_integration_time = np.amin(uv_object.integration_time) / 2.0
    uv_object.upsample_in_time(
        max_integration_time, blt_order="baseline", summing_correlator_mode=True
    )

    # data and nsamples should be changed as normal, but flagged
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(init_wf[0, 0, 0] / 2, out_wf[0, 0, 0])
    out_flags = uv_object.get_flags(0, 1)
    assert np.all(out_flags[:2, 0, 0])
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(init_ns[0, 0, 0], out_ns[0, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_upsample_in_time_summing_correlator_mode_nonint_resampling(hera_uvh5):
    """Test the upsample_in_time method with summing correlator mode
    and non-integer resampling
    """
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))

    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline")

    # save some values for later
    init_data_size = uv_object.data_array.size
    init_wf = uv_object.get_data(0, 1)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # try again with a non-integer resampling factor
    # change the target integration time
    max_integration_time = np.amin(uv_object.integration_time) * 0.75
    uv_object.upsample_in_time(
        max_integration_time, blt_order="baseline", summing_correlator_mode=True
    )

    assert np.allclose(uv_object.integration_time, max_integration_time * 0.5 / 0.75)
    # we should double the size of the data arrays
    assert uv_object.data_array.size == 2 * init_data_size
    # output data should be half the input
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(init_wf[0, 0, 0] / 2, out_wf[0, 0, 0])

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(init_ns[0, 0, 0], out_ns[0, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_partial_upsample_in_time(hera_uvh5):
    """Test the upsample_in_time method with non-uniform upsampling"""
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))

    # change a whole baseline's integration time
    bl_inds = uv_object.antpair2ind(0, 1)
    uv_object.integration_time[bl_inds] = uv_object.integration_time[0] / 2.0

    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline")

    # save some values for later
    init_wf_01 = uv_object.get_data(0, 1)
    init_wf_02 = uv_object.get_data(0, 2)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns_01 = uv_object.get_nsamples(0, 1)
    init_ns_02 = uv_object.get_nsamples(0, 2)

    # change the target integration time
    max_integration_time = np.amin(uv_object.integration_time)
    uv_object.upsample_in_time(max_integration_time, blt_order="baseline")

    assert np.allclose(uv_object.integration_time, max_integration_time)
    # output data should be the same
    out_wf_01 = uv_object.get_data(0, 1)
    out_wf_02 = uv_object.get_data(0, 2)
    assert np.all(init_wf_01 == out_wf_01)
    assert np.isclose(init_wf_02[0, 0, 0], out_wf_02[0, 0, 0])
    assert init_wf_02.size * 2 == out_wf_02.size

    # this should be true because there are no flags
    out_ns_01 = uv_object.get_nsamples(0, 1)
    out_ns_02 = uv_object.get_nsamples(0, 2)
    assert np.allclose(out_ns_01, init_ns_01)
    assert np.isclose(init_ns_02[0, 0, 0], out_ns_02[0, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_upsample_in_time_drift(hera_uvh5):
    """Test the upsample_in_time method on drift mode data"""
    uv_object = hera_uvh5

    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline")

    # save some values for later
    init_data_size = uv_object.data_array.size
    init_wf = uv_object.get_data(0, 1)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # change the target integration time
    max_integration_time = np.amin(uv_object.integration_time) / 2.0
    uv_object.upsample_in_time(
        max_integration_time, blt_order="baseline", allow_drift=True
    )

    assert np.allclose(uv_object.integration_time, max_integration_time)
    # we should double the size of the data arrays
    assert uv_object.data_array.size == 2 * init_data_size
    # output data should be the same
    out_wf = uv_object.get_data(0, 1)
    # we need a "large" tolerance given the "large" data
    new_tol = 1e-2 * np.amax(np.abs(uv_object.data_array))
    assert np.isclose(init_wf[0, 0, 0], out_wf[0, 0, 0], atol=new_tol)

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(init_ns[0, 0, 0], out_ns[0, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_upsample_in_time_drift_no_phasing(hera_uvh5):
    """Test the upsample_in_time method on drift mode data without phasing"""
    uv_object = hera_uvh5

    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline")

    # save some values for later
    init_data_size = uv_object.data_array.size
    init_wf = uv_object.get_data(0, 1)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # change the target integration time
    max_integration_time = np.amin(uv_object.integration_time) / 2.0
    # upsample with allow_drift=False
    uv_object.upsample_in_time(
        max_integration_time, blt_order="baseline", allow_drift=False
    )

    assert np.allclose(uv_object.integration_time, max_integration_time)
    # we should double the size of the data arrays
    assert uv_object.data_array.size == 2 * init_data_size
    # output data should be similar, but somewhat different because of the phasing
    out_wf = uv_object.get_data(0, 1)
    # we need a "large" tolerance given the "large" data
    new_tol = 1e-2 * np.amax(np.abs(uv_object.data_array))
    assert np.isclose(init_wf[0, 0, 0], out_wf[0, 0, 0], atol=new_tol)

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(init_ns[0, 0, 0], out_ns[0, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time(hera_uvh5):
    """Test the downsample_in_time method"""
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")
    uv_object2 = uv_object.copy()

    # save some values for later
    init_data_size = uv_object.data_array.size
    init_wf = uv_object.get_data(0, 1)
    original_int_time = np.amax(uv_object.integration_time)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # change the target integration time
    min_integration_time = original_int_time * 2.0
    uv_object.downsample_in_time(
        min_int_time=min_integration_time, blt_order="baseline", minor_order="time"
    )

    # Should have half the size of the data array and all the new integration time
    # (for this file with 20 integrations and a factor of 2 downsampling)
    assert np.all(np.isclose(uv_object.integration_time, min_integration_time))
    assert uv_object.data_array.size * 2 == init_data_size

    # output data should be the average
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose((init_wf[0, 0, 0] + init_wf[1, 0, 0]) / 2.0, out_wf[0, 0, 0])

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose((init_ns[0, 0, 0] + init_ns[1, 0, 0]) / 2.0, out_ns[0, 0, 0])

    # Compare doing it with n_times_to_avg
    uv_object2.downsample_in_time(
        n_times_to_avg=2, blt_order="baseline", minor_order="time"
    )
    # histories are different when n_times_to_avg is set vs min_int_time
    assert uv_object.history != uv_object2.history
    uv_object2.history = uv_object.history
    assert uv_object == uv_object2

    assert not isinstance(uv_object.data_array, np.ma.MaskedArray)
    assert not isinstance(uv_object.nsample_array, np.ma.MaskedArray)


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_partial_flags(hera_uvh5):
    """Test the downsample_in_time method with partial flagging"""
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")
    # save some values for later
    init_wf = uv_object.get_data(0, 1)
    original_int_time = np.amax(uv_object.integration_time)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # change the target integration time
    min_integration_time = original_int_time * 2.0

    # add flags and try again. With one of the 2 inputs flagged, the data should
    # just be the unflagged value and nsample should be half the unflagged one
    # and the output should not be flagged.
    inds01 = uv_object.antpair2ind(0, 1)
    uv_object.flag_array[inds01[0], 0, 0, 0] = True
    uv_object2 = uv_object.copy()

    uv_object.downsample_in_time(
        min_int_time=min_integration_time, blt_order="baseline", minor_order="time"
    )
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(init_wf[1, 0, 0], out_wf[0, 0, 0])

    # make sure nsamples is correct
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose((init_ns[1, 0, 0]) / 2.0, out_ns[0, 0, 0])

    # check that there are still no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0

    # Compare doing it with n_times_to_avg
    uv_object2.downsample_in_time(
        n_times_to_avg=2, blt_order="baseline", minor_order="time"
    )
    assert uv_object.history != uv_object2.history
    uv_object2.history = uv_object.history
    assert uv_object == uv_object2


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_totally_flagged(hera_uvh5):
    """Test the downsample_in_time method with totally flagged integrations"""
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")
    uv_object2 = uv_object.copy()

    # save some values for later
    init_wf = uv_object.get_data(0, 1)
    original_int_time = np.amax(uv_object.integration_time)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # change the target integration time
    min_integration_time = original_int_time * 2.0

    # add more flags and try again. When all the input points are flagged,
    # data and nsample should have the same results as no flags but the output
    # should be flagged
    inds01 = uv_object.antpair2ind(0, 1)
    uv_object.flag_array[inds01[:2], 0, 0, 0] = True
    uv_object2 = uv_object.copy()

    uv_object.downsample_in_time(
        min_int_time=min_integration_time, blt_order="baseline", minor_order="time"
    )
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose((init_wf[0, 0, 0] + init_wf[1, 0, 0]) / 2.0, out_wf[0, 0, 0])

    # make sure nsamples is correct
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose((init_ns[0, 0, 0] + init_ns[1, 0, 0]) / 2.0, out_ns[0, 0, 0])

    # check that the new sample is flagged
    out_flag = uv_object.get_flags(0, 1)
    assert out_flag[0, 0, 0]

    # Compare doing it with n_times_to_avg
    uv_object2.downsample_in_time(
        n_times_to_avg=2, blt_order="baseline", minor_order="time"
    )
    assert uv_object.history != uv_object2.history
    uv_object2.history = uv_object.history
    assert uv_object == uv_object2


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_uneven_samples(hera_uvh5):
    """Test the downsample_in_time method with uneven downsampling"""
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")
    uv_object2 = uv_object.copy()

    # save some values for later
    init_wf = uv_object.get_data(0, 1)
    original_int_time = np.amax(uv_object.integration_time)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0

    # test again with a downsample factor that doesn't go evenly into the
    # number of samples
    min_integration_time = original_int_time * 3.0
    uv_object.downsample_in_time(
        min_int_time=min_integration_time,
        blt_order="baseline",
        minor_order="time",
        keep_ragged=False,
    )

    # Only some baselines have an even number of times, so the output integration time
    # is not uniformly the same. For the test case, we'll have *either* the original
    # integration time or twice that.
    assert np.all(
        np.logical_or(
            np.isclose(uv_object.integration_time, original_int_time),
            np.isclose(uv_object.integration_time, min_integration_time),
        )
    )

    # make sure integration time is correct
    # in this case, all integration times should be the target one
    assert np.all(np.isclose(uv_object.integration_time, min_integration_time))

    # as usual, the new data should be the average of the input data (3 points now)
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(np.mean(init_wf[0:3, 0, 0]), out_wf[0, 0, 0])

    # Compare doing it with n_times_to_avg
    uv_object2.downsample_in_time(
        n_times_to_avg=3, blt_order="baseline", minor_order="time", keep_ragged=False
    )
    assert uv_object.history != uv_object2.history
    uv_object2.history = uv_object.history
    assert uv_object == uv_object2


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_uneven_samples_keep_ragged(hera_uvh5):
    """Test downsample_in_time with uneven downsampling and keep_ragged=True."""
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")
    uv_object2 = uv_object.copy()

    # save some values for later
    init_wf = uv_object.get_data(0, 1)
    original_int_time = np.amax(uv_object.integration_time)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0

    # test again with a downsample factor that doesn't go evenly into the
    # number of samples
    min_integration_time = original_int_time * 3.0

    # test again with keep_ragged=False
    uv_object.downsample_in_time(
        min_int_time=min_integration_time,
        blt_order="baseline",
        minor_order="time",
        keep_ragged=True,
    )

    # as usual, the new data should be the average of the input data
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(np.mean(init_wf[0:3, 0, 0]), out_wf[0, 0, 0])

    # Compare doing it with n_times_to_avg
    uv_object2.downsample_in_time(
        n_times_to_avg=3, blt_order="baseline", minor_order="time", keep_ragged=True
    )
    assert uv_object.history != uv_object2.history
    uv_object2.history = uv_object.history
    assert uv_object == uv_object2


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_summing_correlator_mode(hera_uvh5):
    """Test the downsample_in_time method with summing correlator mode"""
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")

    # save some values for later
    init_data_size = uv_object.data_array.size
    init_wf = uv_object.get_data(0, 1)
    original_int_time = np.amax(uv_object.integration_time)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # change the target integration time
    min_integration_time = original_int_time * 2.0
    uv_object.downsample_in_time(
        min_int_time=min_integration_time,
        blt_order="baseline",
        minor_order="time",
        summing_correlator_mode=True,
    )

    # Should have half the size of the data array and all the new integration time
    # (for this file with 20 integrations and a factor of 2 downsampling)
    assert np.all(np.isclose(uv_object.integration_time, min_integration_time))
    assert uv_object.data_array.size * 2 == init_data_size

    # output data should be the sum
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose((init_wf[0, 0, 0] + init_wf[1, 0, 0]), out_wf[0, 0, 0])

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose((init_ns[0, 0, 0] + init_ns[1, 0, 0]) / 2.0, out_ns[0, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_summing_correlator_mode_partial_flags(hera_uvh5):
    """Test the downsample_in_time method with summing correlator mode and
    partial flags
    """
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")

    # save some values for later
    init_wf = uv_object.get_data(0, 1)
    original_int_time = np.amax(uv_object.integration_time)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # change the target integration time
    min_integration_time = original_int_time * 2.0

    # add flags and try again. With one of the 2 inputs flagged, the data should
    # just be the unflagged value and nsample should be half the unflagged one
    # and the output should not be flagged.
    inds01 = uv_object.antpair2ind(0, 1)
    uv_object.flag_array[inds01[0], 0, 0, 0] = True
    uv_object.downsample_in_time(
        min_int_time=min_integration_time,
        blt_order="baseline",
        minor_order="time",
        summing_correlator_mode=True,
    )
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(init_wf[1, 0, 0], out_wf[0, 0, 0])

    # make sure nsamples is correct
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose((init_ns[1, 0, 0]) / 2.0, out_ns[0, 0, 0])

    # check that there are still no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_summing_correlator_mode_totally_flagged(hera_uvh5):
    """Test the downsample_in_time method with summing correlator mode and
    totally flagged integrations.
    """
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")

    # save some values for later
    init_wf = uv_object.get_data(0, 1)
    original_int_time = np.amax(uv_object.integration_time)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # change the target integration time
    min_integration_time = original_int_time * 2.0

    # add more flags and try again. When all the input points are flagged,
    # data and nsample should have the same results as no flags but the output
    # should be flagged
    inds01 = uv_object.antpair2ind(0, 1)
    uv_object.flag_array[inds01[:2], 0, 0, 0] = True
    uv_object.downsample_in_time(
        min_int_time=min_integration_time,
        blt_order="baseline",
        minor_order="time",
        summing_correlator_mode=True,
    )
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose((init_wf[0, 0, 0] + init_wf[1, 0, 0]), out_wf[0, 0, 0])

    # make sure nsamples is correct
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose((init_ns[0, 0, 0] + init_ns[1, 0, 0]) / 2.0, out_ns[0, 0, 0])

    # check that the new sample is flagged
    out_flag = uv_object.get_flags(0, 1)
    assert out_flag[0, 0, 0]

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_summing_correlator_mode_uneven_samples(hera_uvh5):
    """Test the downsample_in_time method with summing correlator mode and
    uneven samples.
    """
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")

    # save some values for later
    init_wf = uv_object.get_data(0, 1)
    original_int_time = np.amax(uv_object.integration_time)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # test again with a downsample factor that doesn't go evenly into the
    # number of samples
    min_integration_time = original_int_time * 3.0
    uv_object.downsample_in_time(
        min_int_time=min_integration_time,
        blt_order="baseline",
        minor_order="time",
        keep_ragged=False,
        summing_correlator_mode=True,
    )

    # Only some baselines have an even number of times, so the output integration time
    # is not uniformly the same. For the test case, we'll have *either* the original
    # integration time or twice that.
    assert np.all(
        np.logical_or(
            np.isclose(uv_object.integration_time, original_int_time),
            np.isclose(uv_object.integration_time, min_integration_time),
        )
    )

    # as usual, the new data should be the average of the input data (3 points now)
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(np.sum(init_wf[0:3, 0, 0]), out_wf[0, 0, 0])

    # make sure nsamples is correct
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(np.mean(init_ns[0:3, 0, 0]), out_ns[0, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_summing_correlator_mode_uneven_samples_drop_ragged(
    hera_uvh5,
):
    """Test the downsample_in_time method with summing correlator mode and
    uneven samples, dropping ragged ones.
    """
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")

    # save some values for later
    init_wf = uv_object.get_data(0, 1)
    original_int_time = np.amax(uv_object.integration_time)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # test again with keep_ragged=False
    min_integration_time = original_int_time * 3.0
    uv_object.downsample_in_time(
        min_int_time=min_integration_time,
        blt_order="baseline",
        minor_order="time",
        keep_ragged=False,
        summing_correlator_mode=True,
    )

    # make sure integration time is correct
    # in this case, all integration times should be the target one
    assert np.all(np.isclose(uv_object.integration_time, min_integration_time))

    # as usual, the new data should be the average of the input data
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(np.sum(init_wf[0:3, 0, 0]), out_wf[0, 0, 0])

    # make sure nsamples is correct
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(np.mean(init_ns[0:3, 0, 0]), out_ns[0, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_partial_downsample_in_time(hera_uvh5):
    """Test the downsample_in_time method without uniform downsampling"""
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))

    # change a whole baseline's integration time
    bl_inds = uv_object.antpair2ind(0, 1)
    uv_object.integration_time[bl_inds] = uv_object.integration_time[0] * 2.0

    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline")

    # save some values for later
    init_wf_01 = uv_object.get_data(0, 1)
    init_wf_02 = uv_object.get_data(0, 2)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns_01 = uv_object.get_nsamples(0, 1)
    init_ns_02 = uv_object.get_nsamples(0, 2)

    # change the target integration time
    min_integration_time = np.amax(uv_object.integration_time)
    uv_object.downsample_in_time(
        min_int_time=min_integration_time, blt_order="baseline"
    )

    # Should have all the new integration time
    # (for this file with 20 integrations and a factor of 2 downsampling)
    assert np.all(np.isclose(uv_object.integration_time, min_integration_time))

    # output data should be the same
    out_wf_01 = uv_object.get_data(0, 1)
    out_wf_02 = uv_object.get_data(0, 2)
    assert np.all(init_wf_01 == out_wf_01)
    assert np.isclose(
        (init_wf_02[0, 0, 0] + init_wf_02[1, 0, 0]) / 2.0, out_wf_02[0, 0, 0]
    )

    # this should be true because there are no flags
    out_ns_01 = uv_object.get_nsamples(0, 1)
    out_ns_02 = uv_object.get_nsamples(0, 2)
    assert np.allclose(out_ns_01, init_ns_01)
    assert np.isclose(
        (init_ns_02[0, 0, 0] + init_ns_02[1, 0, 0]) / 2.0, out_ns_02[0, 0, 0]
    )

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_drift(hera_uvh5):
    """Test the downsample_in_time method on drift mode data"""
    uv_object = hera_uvh5

    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")
    uv_object2 = uv_object.copy()

    # save some values for later
    init_data_size = uv_object.data_array.size
    init_wf = uv_object.get_data(0, 1)
    original_int_time = np.amax(uv_object.integration_time)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # change the target integration time
    min_integration_time = original_int_time * 2.0
    uv_object.downsample_in_time(
        min_int_time=min_integration_time, blt_order="baseline", allow_drift=True
    )

    # Should have half the size of the data array and all the new integration time
    # (for this file with 20 integrations and a factor of 2 downsampling)
    assert np.all(np.isclose(uv_object.integration_time, min_integration_time))
    assert uv_object.data_array.size * 2 == init_data_size

    # output data should be the average
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose((init_wf[0, 0, 0] + init_wf[1, 0, 0]) / 2.0, out_wf[0, 0, 0])

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose((init_ns[0, 0, 0] + init_ns[1, 0, 0]) / 2.0, out_ns[0, 0, 0])

    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0

    # Compare doing it with n_times_to_avg
    uv_object2.downsample_in_time(
        n_times_to_avg=2, blt_order="baseline", allow_drift=True
    )
    assert uv_object.history != uv_object2.history
    uv_object2.history = uv_object.history
    assert uv_object == uv_object2


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_drift_no_phasing(hera_uvh5):
    """Test the downsample_in_time method on drift mode data without phasing"""
    uv_object = hera_uvh5

    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")
    uv_object2 = uv_object.copy()

    # save some values for later
    init_data_size = uv_object.data_array.size
    init_wf = uv_object.get_data(0, 1)
    original_int_time = np.amax(uv_object.integration_time)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # change the target integration time
    min_integration_time = original_int_time * 2.0

    # try again with allow_drift=False
    uv_object.downsample_in_time(
        min_int_time=min_integration_time, blt_order="baseline", allow_drift=False,
    )

    # Should have half the size of the data array and all the new integration time
    # (for this file with 20 integrations and a factor of 2 downsampling)
    assert np.all(np.isclose(uv_object.integration_time, min_integration_time))
    assert uv_object.data_array.size * 2 == init_data_size

    # output data should be similar to the average, but somewhat different
    # because of the phasing
    out_wf = uv_object.get_data(0, 1)
    new_tol = 5e-2 * np.amax(np.abs(uv_object.data_array))
    assert np.isclose(
        (init_wf[0, 0, 0] + init_wf[1, 0, 0]) / 2.0, out_wf[0, 0, 0], atol=new_tol
    )

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose((init_ns[0, 0, 0] + init_ns[1, 0, 0]) / 2.0, out_ns[0, 0, 0])

    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0

    # Compare doing it with n_times_to_avg
    uv_object2.downsample_in_time(
        n_times_to_avg=2, blt_order="baseline", minor_order="time"
    )
    assert uv_object.history != uv_object2.history
    uv_object2.history = uv_object.history
    assert uv_object == uv_object2


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_nsample_precision(hera_uvh5):
    """Test the downsample_in_time method with a half-precision nsample_array"""
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")
    # save some values for later
    init_wf = uv_object.get_data(0, 1)
    original_int_time = np.amax(uv_object.integration_time)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # change the target integration time
    min_integration_time = original_int_time * 2.0

    # add flags and try again. With one of the 2 inputs flagged, the data should
    # just be the unflagged value and nsample should be half the unflagged one
    # and the output should not be flagged.
    inds01 = uv_object.antpair2ind(0, 1)
    uv_object.flag_array[inds01[0], 0, 0, 0] = True
    uv_object2 = uv_object.copy()

    # change precision of nsample array
    uv_object.nsample_array = uv_object.nsample_array.astype(np.float16)
    uv_object.downsample_in_time(
        min_int_time=min_integration_time, blt_order="baseline", minor_order="time"
    )
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(init_wf[1, 0, 0], out_wf[0, 0, 0])

    # make sure nsamples is correct
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose((init_ns[1, 0, 0]) / 2.0, out_ns[0, 0, 0])

    # make sure nsamples has the right dtype
    assert uv_object.nsample_array.dtype.type is np.float16

    # check that there are still no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0

    # Compare doing it with n_times_to_avg
    uv_object2.nsample_array = uv_object2.nsample_array.astype(np.float16)
    uv_object2.downsample_in_time(
        n_times_to_avg=2, blt_order="baseline", minor_order="time"
    )
    assert uv_object.history != uv_object2.history
    uv_object2.history = uv_object.history
    assert uv_object == uv_object2

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_errors(hera_uvh5):
    """Test various errors and warnings are raised"""
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")

    # raise an error if set neither min_int_time and n_times_to_avg
    with pytest.raises(
        ValueError, match="Either min_int_time or n_times_to_avg must be set."
    ):
        uv_object.downsample_in_time()

    # raise an error if set both min_int_time and n_times_to_avg
    with pytest.raises(
        ValueError, match="Only one of min_int_time or n_times_to_avg can be set."
    ):
        uv_object.downsample_in_time(
            min_int_time=2 * np.amin(uv_object.integration_time), n_times_to_avg=2
        )
    # raise an error if only one time
    uv_object2 = uv_object.copy()
    uv_object2.select(times=uv_object2.time_array[0])
    with pytest.raises(
        ValueError, match="Only one time in this object, cannot downsample."
    ):
        uv_object2.downsample_in_time(n_times_to_avg=2)

    # raise an error for a too-large integration time
    max_integration_time = 1e3 * np.amax(uv_object.integration_time)
    with pytest.raises(
        ValueError, match="Increasing the integration time by more than"
    ):
        uv_object.downsample_in_time(min_int_time=max_integration_time)

    # catch a warning for doing no work
    uv_object2 = uv_object.copy()
    max_integration_time = 0.5 * np.amin(uv_object.integration_time)
    with uvtest.check_warnings(
        UserWarning, match="All values in the integration_time array are already longer"
    ):
        uv_object.downsample_in_time(min_int_time=max_integration_time)

    assert uv_object == uv_object2
    del uv_object2

    # raise an error if n_times_to_avg is not an integer
    with pytest.raises(ValueError, match="n_times_to_avg must be an integer."):
        uv_object.downsample_in_time(n_times_to_avg=2.5)

    # save some values for later
    init_data_size = uv_object.data_array.size
    init_wf = uv_object.get_data(0, 1)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # make a gap in the times to check a warning about that
    inds01 = uv_object.antpair2ind(0, 1)
    initial_int_time = uv_object.integration_time[inds01[0]]
    # time array is in jd, integration time is in sec
    uv_object.time_array[inds01[-1]] += initial_int_time / (24 * 3600)
    uv_object.Ntimes += 1
    min_integration_time = 2 * np.amin(uv_object.integration_time)
    times_01 = uv_object.get_times(0, 1)
    assert np.unique(np.diff(times_01)).size > 1
    with uvtest.check_warnings(
        UserWarning, "There is a gap in the times of baseline",
    ):
        uv_object.downsample_in_time(min_int_time=min_integration_time)

    # Should have half the size of the data array and all the new integration time
    # (for this file with 20 integrations and a factor of 2 downsampling)
    assert np.all(np.isclose(uv_object.integration_time, min_integration_time))
    assert uv_object.data_array.size * 2 == init_data_size

    # output data should be the average
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose((init_wf[0, 0, 0] + init_wf[1, 0, 0]) / 2.0, out_wf[0, 0, 0])

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose((init_ns[0, 0, 0] + init_ns[1, 0, 0]) / 2.0, out_ns[0, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_int_time_mismatch_warning(hera_uvh5):
    """Test warning in downsample_in_time about mismatch between integration
    times and the time between integrations.
    """
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")

    # save some values for later
    init_data_size = uv_object.data_array.size
    init_wf = uv_object.get_data(0, 1)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # change the integration times to catch a warning about integration times
    # not matching the time delta between integrations
    uv_object.integration_time *= 0.5
    min_integration_time = 2 * np.amin(uv_object.integration_time)
    with uvtest.check_warnings(
        UserWarning,
        match="The time difference between integrations is not the same",
        nwarnings=10,  # One of these warnings was misID'd as uwv mismatch
    ):
        uv_object.downsample_in_time(min_int_time=min_integration_time)

    # Should have half the size of the data array and all the new integration time
    # (for this file with 20 integrations and a factor of 2 downsampling)
    assert np.all(np.isclose(uv_object.integration_time, min_integration_time))
    assert uv_object.data_array.size * 2 == init_data_size

    # output data should be the average
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose((init_wf[0, 0, 0] + init_wf[1, 0, 0]) / 2.0, out_wf[0, 0, 0])

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose((init_ns[0, 0, 0] + init_ns[1, 0, 0]) / 2.0, out_ns[0, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_varying_integration_time(hera_uvh5):
    """Test downsample_in_time handling of file with integration time changing
    within a baseline
    """
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")

    # save some values for later
    init_wf = uv_object.get_data(0, 1)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # test handling (& warnings) with varying integration time in a baseline
    # First, change both integration time & time array to match
    inds01 = uv_object.antpair2ind(0, 1)
    initial_int_time = uv_object.integration_time[inds01[0]]
    # time array is in jd, integration time is in sec
    uv_object.time_array[inds01[-2]] += (initial_int_time / 2) / (24 * 3600)
    uv_object.time_array[inds01[-1]] += (3 * initial_int_time / 2) / (24 * 3600)
    uv_object.integration_time[inds01[-2:]] += initial_int_time
    uv_object.Ntimes = np.unique(uv_object.time_array).size
    min_integration_time = 2 * np.amin(uv_object.integration_time)
    # check that there are no warnings about inconsistencies between
    # integration_time & time_array
    with uvtest.check_warnings(None):
        uv_object.downsample_in_time(min_int_time=min_integration_time)

    # Should have all the new integration time
    # (for this file with 20 integrations and a factor of 2 downsampling)
    assert np.all(np.isclose(uv_object.integration_time, min_integration_time))

    out_wf = uv_object.get_data(0, 1)

    n_times_in = init_wf.shape[0]
    n_times_out = out_wf.shape[0]
    assert n_times_out == (n_times_in - 2) / 2 + 2

    # output data should be the average for the first set
    assert np.isclose((init_wf[0, 0, 0] + init_wf[1, 0, 0]) / 2.0, out_wf[0, 0, 0])
    # last 2 time samples should be identical to initial ones
    assert np.isclose(init_wf[-1, 0, 0], out_wf[-1, 0, 0])
    assert np.isclose(init_wf[-2, 0, 0], out_wf[-2, 0, 0])

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose((init_ns[0, 0, 0] + init_ns[1, 0, 0]) / 2.0, out_ns[0, 0, 0])
    assert np.isclose(init_ns[-1, 0, 0], out_ns[-1, 0, 0])
    assert np.isclose(init_ns[-2, 0, 0], out_ns[2, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_varying_int_time_partial_flags(hera_uvh5):
    """Test downsample_in_time handling of file with integration time changing
    within a baseline and partial flagging.
    """
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")

    # downselect to 14 times and one baseline
    uv_object.select(times=np.unique(uv_object.time_array)[:14])

    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0

    # change last 2 integrations to be twice as long
    # (so 12 normal length, 2 double length)
    # change integration time & time array to match
    inds01 = uv_object.antpair2ind(0, 1)
    initial_int_time = uv_object.integration_time[inds01[0]]
    # time array is in jd, integration time is in sec
    uv_object.time_array[inds01[-2]] += (initial_int_time / 2) / (24 * 3600)
    uv_object.time_array[inds01[-1]] += (3 * initial_int_time / 2) / (24 * 3600)
    uv_object.integration_time[inds01[-2:]] += initial_int_time
    uv_object.Ntimes = np.unique(uv_object.time_array).size

    # add a flag on last time
    uv_object.flag_array[inds01[-1], :, :, :] = True
    # add a flag on thrid to last time
    uv_object.flag_array[inds01[-3], :, :, :] = True

    uv_object2 = uv_object.copy()

    with uvtest.check_warnings(None):
        uv_object.downsample_in_time(min_int_time=4 * initial_int_time)
    with uvtest.check_warnings(None):
        uv_object.downsample_in_time(min_int_time=8 * initial_int_time)
    with uvtest.check_warnings(None):
        uv_object2.downsample_in_time(min_int_time=8 * initial_int_time)

    assert uv_object.history != uv_object2.history
    uv_object2.history = uv_object.history

    assert uv_object == uv_object2
    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_varying_integration_time_warning(hera_uvh5):
    """Test downsample_in_time handling of file with integration time changing
    within a baseline, but without adjusting the time_array so there is a mismatch.
    """
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")

    # save some values for later
    init_wf = uv_object.get_data(0, 1)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # Next, change just integration time, so time array doesn't match
    inds01 = uv_object.antpair2ind(0, 1)
    initial_int_time = uv_object.integration_time[inds01[0]]
    uv_object.integration_time[inds01[-2:]] += initial_int_time
    min_integration_time = 2 * np.amin(uv_object.integration_time)
    with uvtest.check_warnings(
        UserWarning, "The time difference between integrations is different than",
    ):
        uv_object.downsample_in_time(min_int_time=min_integration_time)

    # Should have all the new integration time
    # (for this file with 20 integrations and a factor of 2 downsampling)
    assert np.all(np.isclose(uv_object.integration_time, min_integration_time))

    # output data should be the average
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose((init_wf[0, 0, 0] + init_wf[1, 0, 0]) / 2.0, out_wf[0, 0, 0])

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose((init_ns[0, 0, 0] + init_ns[1, 0, 0]) / 2.0, out_ns[0, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:Data will be unphased and rephased")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_upsample_downsample_in_time(hera_uvh5):
    """Test round trip works"""
    uv_object = hera_uvh5

    # Using astropy here (and elsewhere) to match previously calculated values
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))

    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")
    uv_object2 = uv_object.copy()

    max_integration_time = np.amin(uv_object.integration_time) / 2.0
    uv_object.upsample_in_time(max_integration_time, blt_order="baseline")
    assert np.amax(uv_object.integration_time) <= max_integration_time
    new_Nblts = uv_object.Nblts

    # check that calling upsample again with the same max_integration_time
    # gives warning and does nothing
    with uvtest.check_warnings(
        UserWarning, "All values in the integration_time array are already longer"
    ):
        uv_object.upsample_in_time(max_integration_time, blt_order="baseline")
    assert uv_object.Nblts == new_Nblts

    # check that calling upsample again with the almost the same max_integration_time
    # gives warning and does nothing
    small_number = 0.9 * uv_object._integration_time.tols[1]
    with uvtest.check_warnings(
        UserWarning, "All values in the integration_time array are already longer"
    ):
        uv_object.upsample_in_time(
            max_integration_time - small_number, blt_order="baseline",
        )
    assert uv_object.Nblts == new_Nblts

    uv_object.downsample_in_time(
        min_int_time=np.amin(uv_object2.integration_time), blt_order="baseline",
    )

    # increase tolerance on LST if iers.conf.auto_max_age is set to None, as we
    # do in testing if the iers url is down. See conftest.py for more info.
    if iers.conf.auto_max_age is None:
        uv_object._lst_array.tols = (0, 1e-4)

    # make sure that history is correct
    assert (
        "Upsampled data to 0.939524 second integration time using pyuvdata."
        in uv_object.history
    )
    assert (
        "Downsampled data to 1.879048 second integration time using pyuvdata."
        in uv_object.history
    )

    # overwrite history and check for equality
    uv_object.history = uv_object2.history
    assert uv_object == uv_object2

    # check that calling downsample again with the same min_integration_time
    # gives warning and does nothing
    with uvtest.check_warnings(
        UserWarning, match="All values in the integration_time array are already longer"
    ):
        uv_object.downsample_in_time(
            min_int_time=np.amin(uv_object2.integration_time), blt_order="baseline"
        )
    assert uv_object.Nblts == uv_object2.Nblts

    # check that calling upsample again with the almost the same min_integration_time
    # gives warning and does nothing
    with uvtest.check_warnings(
        UserWarning, match="All values in the integration_time array are already longer"
    ):
        uv_object.upsample_in_time(
            np.amin(uv_object2.integration_time) + small_number, blt_order="baseline"
        )

    assert uv_object.Nblts == uv_object2.Nblts

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:Data will be unphased and rephased")
@pytest.mark.filterwarnings("ignore:There is a gap in the times of baseline")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_upsample_downsample_in_time_odd_resample(hera_uvh5):
    """Test round trip works with odd resampling"""
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))

    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")
    uv_object2 = uv_object.copy()

    # try again with a resampling factor of 3 (test odd numbers)
    max_integration_time = np.amin(uv_object.integration_time) / 3.0
    uv_object.upsample_in_time(max_integration_time, blt_order="baseline")
    assert np.amax(uv_object.integration_time) <= max_integration_time

    uv_object.downsample_in_time(
        np.amin(uv_object2.integration_time), blt_order="baseline",
    )

    # increase tolerance on LST if iers.conf.auto_max_age is set to None, as we
    # do in testing if the iers url is down. See conftest.py for more info.
    if iers.conf.auto_max_age is None:
        uv_object._lst_array.tols = (0, 1e-4)

    # make sure that history is correct
    assert (
        "Upsampled data to 0.626349 second integration time using pyuvdata."
        in uv_object.history
    )
    assert (
        "Downsampled data to 1.879048 second integration time using pyuvdata."
        in uv_object.history
    )

    # overwrite history and check for equality
    uv_object.history = uv_object2.history
    assert uv_object == uv_object2


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_upsample_downsample_in_time_metadata_only(hera_uvh5):
    """Test round trip works with metadata-only objects"""
    uv_object = hera_uvh5

    # drop the data arrays
    uv_object.data_array = None
    uv_object.flag_array = None
    uv_object.nsample_array = None

    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))

    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")
    uv_object2 = uv_object.copy()

    max_integration_time = np.amin(uv_object.integration_time) / 2.0
    uv_object.upsample_in_time(max_integration_time, blt_order="baseline")
    assert np.amax(uv_object.integration_time) <= max_integration_time

    uv_object.downsample_in_time(
        np.amin(uv_object2.integration_time), blt_order="baseline",
    )

    # increase tolerance on LST if iers.conf.auto_max_age is set to None, as we
    # do in testing if the iers url is down. See conftest.py for more info.
    if iers.conf.auto_max_age is None:
        uv_object._lst_array.tols = (0, 1e-4)

    # make sure that history is correct
    assert (
        "Upsampled data to 0.939524 second integration time using pyuvdata."
        in uv_object.history
    )
    assert (
        "Downsampled data to 1.879048 second integration time using pyuvdata."
        in uv_object.history
    )

    # overwrite history and check for equality
    uv_object.history = uv_object2.history
    assert uv_object == uv_object2


@pytest.mark.filterwarnings("ignore:Telescope mock-HERA is not in known_telescopes")
@pytest.mark.filterwarnings("ignore:There is a gap in the times of baseline")
def test_resample_in_time(bda_test_file):
    """Test the resample_in_time method"""
    # Note this file has slight variations in the delta t between integrations
    # that causes our gap test to issue a warning, but the variations are small
    # We aren't worried about them, so we filter those warnings
    uv_object = bda_test_file

    # save some initial info
    # 2s integration time
    init_data_1_136 = uv_object.get_data((1, 136))
    # 4s integration time
    init_data_1_137 = uv_object.get_data((1, 137))
    # 8s integration time
    init_data_1_138 = uv_object.get_data((1, 138))
    # 16s integration time
    init_data_136_137 = uv_object.get_data((136, 137))

    uv_object.resample_in_time(8)
    # Should have all the target integration time
    assert np.all(np.isclose(uv_object.integration_time, 8))

    # 2s integration time
    out_data_1_136 = uv_object.get_data((1, 136))
    # 4s integration time
    out_data_1_137 = uv_object.get_data((1, 137))
    # 8s integration time
    out_data_1_138 = uv_object.get_data((1, 138))
    # 16s integration time
    out_data_136_137 = uv_object.get_data((136, 137))

    # check array sizes make sense
    assert out_data_1_136.size * 4 == init_data_1_136.size
    assert out_data_1_137.size * 2 == init_data_1_137.size
    assert out_data_1_138.size == init_data_1_138.size
    assert out_data_136_137.size / 2 == init_data_136_137.size

    # check some values
    assert np.isclose(np.mean(init_data_1_136[0:4, 0, 0]), out_data_1_136[0, 0, 0])
    assert np.isclose(np.mean(init_data_1_137[0:2, 0, 0]), out_data_1_137[0, 0, 0])
    assert np.isclose(init_data_1_138[0, 0, 0], out_data_1_138[0, 0, 0])
    assert np.isclose(init_data_136_137[0, 0, 0], out_data_136_137[0, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:Telescope mock-HERA is not in known_telescopes")
@pytest.mark.filterwarnings("ignore:There is a gap in the times of baseline")
def test_resample_in_time_downsample_only(bda_test_file):
    """Test resample_in_time with downsampling only"""
    # Note this file has slight variations in the delta t between integrations
    # that causes our gap test to issue a warning, but the variations are small
    # We aren't worried about them, so we filter those warnings
    uv_object = bda_test_file

    # save some initial info
    # 2s integration time
    init_data_1_136 = uv_object.get_data((1, 136))
    # 4s integration time
    init_data_1_137 = uv_object.get_data((1, 137))
    # 8s integration time
    init_data_1_138 = uv_object.get_data((1, 138))
    # 16s integration time
    init_data_136_137 = uv_object.get_data((136, 137))

    # resample again, with only_downsample set
    uv_object.resample_in_time(8, only_downsample=True)
    # Should have all less than or equal to the target integration time
    assert np.all(
        np.logical_or(
            np.isclose(uv_object.integration_time, 8),
            np.isclose(uv_object.integration_time, 16),
        )
    )

    # 2s integration time
    out_data_1_136 = uv_object.get_data((1, 136))
    # 4s integration time
    out_data_1_137 = uv_object.get_data((1, 137))
    # 8s integration time
    out_data_1_138 = uv_object.get_data((1, 138))
    # 16s integration time
    out_data_136_137 = uv_object.get_data((136, 137))

    # check array sizes make sense
    assert out_data_1_136.size * 4 == init_data_1_136.size
    assert out_data_1_137.size * 2 == init_data_1_137.size
    assert out_data_1_138.size == init_data_1_138.size
    assert out_data_136_137.size == init_data_136_137.size

    # check some values
    assert np.isclose(np.mean(init_data_1_136[0:4, 0, 0]), out_data_1_136[0, 0, 0])
    assert np.isclose(np.mean(init_data_1_137[0:2, 0, 0]), out_data_1_137[0, 0, 0])
    assert np.isclose(init_data_1_138[0, 0, 0], out_data_1_138[0, 0, 0])
    assert np.isclose(init_data_136_137[0, 0, 0], out_data_136_137[0, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:Telescope mock-HERA is not in known_telescopes")
@pytest.mark.filterwarnings("ignore:There is a gap in the times of baseline")
def test_resample_in_time_only_upsample(bda_test_file):
    """Test resample_in_time with only upsampling"""
    # Note this file has slight variations in the delta t between integrations
    # that causes our gap test to issue a warning, but the variations are small
    # We aren't worried about them, so we filter those warnings
    uv_object = bda_test_file

    # save some initial info
    # 2s integration time
    init_data_1_136 = uv_object.get_data((1, 136))
    # 4s integration time
    init_data_1_137 = uv_object.get_data((1, 137))
    # 8s integration time
    init_data_1_138 = uv_object.get_data((1, 138))
    # 16s integration time
    init_data_136_137 = uv_object.get_data((136, 137))

    # again, with only_upsample set
    uv_object.resample_in_time(8, only_upsample=True)
    # Should have all greater than or equal to the target integration time
    assert np.all(
        np.logical_or(
            np.logical_or(
                np.isclose(uv_object.integration_time, 2.0),
                np.isclose(uv_object.integration_time, 4.0),
            ),
            np.isclose(uv_object.integration_time, 8.0),
        )
    )

    # 2s integration time
    out_data_1_136 = uv_object.get_data((1, 136))
    # 4s integration time
    out_data_1_137 = uv_object.get_data((1, 137))
    # 8s integration time
    out_data_1_138 = uv_object.get_data((1, 138))
    # 16s integration time
    out_data_136_137 = uv_object.get_data((136, 137))

    # check array sizes make sense
    assert out_data_1_136.size == init_data_1_136.size
    assert out_data_1_137.size == init_data_1_137.size
    assert out_data_1_138.size == init_data_1_138.size
    assert out_data_136_137.size / 2 == init_data_136_137.size

    # check some values
    assert np.isclose(init_data_1_136[0, 0, 0], out_data_1_136[0, 0, 0])
    assert np.isclose(init_data_1_137[0, 0, 0], out_data_1_137[0, 0, 0])
    assert np.isclose(init_data_1_138[0, 0, 0], out_data_1_138[0, 0, 0])
    assert np.isclose(init_data_136_137[0, 0, 0], out_data_136_137[0, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:Telescope mock-HERA is not in known_telescopes")
@pytest.mark.filterwarnings("ignore:There is a gap in the times of baseline")
def test_resample_in_time_partial_flags(bda_test_file):
    """Test resample_in_time with partial flags"""
    # Note this file has slight variations in the delta t between integrations
    # that causes our gap test to issue a warning, but the variations are small
    # We aren't worried about them, so we filter those warnings
    uv = bda_test_file
    # For ease, select a single baseline
    uv.select(bls=[(1, 136)])
    # Flag one time
    uv.flag_array[0, :, :, :] = True
    uv2 = uv.copy()

    # Downsample in two stages
    uv.resample_in_time(4.0, only_downsample=True)
    uv.resample_in_time(8.0, only_downsample=True)
    # Downsample in a single stage
    uv2.resample_in_time(8.0, only_downsample=True)

    assert uv.history != uv2.history
    uv2.history = uv.history
    assert uv == uv2
    return


@pytest.mark.filterwarnings("ignore:There is a gap in the times of baseline")
def test_downsample_in_time_mwa():
    """
    Test resample in time works with numerical weirdnesses.

    In particular, when min_int_time is not quite an integer mulitple of
    integration_time. This text broke with a prior bug (see issue 773).
    """
    filename = os.path.join(DATA_PATH, "mwa_integration_time.uvh5")
    uv = UVData()
    uv.read(filename)
    uv.phase_to_time(np.mean(uv.time_array))
    uv_object2 = uv.copy()

    # all data within 5 milliseconds of 2 second integrations
    assert np.allclose(uv.integration_time, 2, atol=5e-3)
    min_int_time = 4.0
    uv.resample_in_time(min_int_time, only_downsample=True, keep_ragged=False)

    assert np.all(uv.integration_time > (min_int_time - 5e-3))

    # Now do the human expected thing:
    init_data = uv_object2.get_data((61, 58))
    uv_object2.downsample_in_time(n_times_to_avg=2, keep_ragged=False)

    assert uv_object2.Ntimes == 5

    out_data = uv_object2.get_data((61, 58))

    assert np.isclose(np.mean(init_data[0:2, 0, 0]), out_data[0, 0, 0])


@pytest.mark.filterwarnings("ignore:There is a gap in the times of baseline")
def test_resample_in_time_warning():
    filename = os.path.join(DATA_PATH, "mwa_integration_time.uvh5")
    uv = UVData()
    uv.read(filename)

    uv2 = uv.copy()

    with uvtest.check_warnings(
        UserWarning, match="No resampling will be done because target time"
    ):
        uv.resample_in_time(3, keep_ragged=False)

    assert uv2 == uv


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_frequency_average(uvdata_data):
    """Test averaging in frequency."""
    eq_coeffs = np.tile(
        np.arange(uvdata_data.uv_object.Nfreqs, dtype=np.float),
        (uvdata_data.uv_object.Nants_telescope, 1),
    )
    uvdata_data.uv_object.eq_coeffs = eq_coeffs
    uvdata_data.uv_object.check()

    # check that there's no flagging
    assert np.nonzero(uvdata_data.uv_object.flag_array)[0].size == 0

    with uvtest.check_warnings(UserWarning, "eq_coeffs vary by frequency"):
        uvdata_data.uv_object.frequency_average(2),

    assert uvdata_data.uv_object.Nfreqs == (uvdata_data.uv_object2.Nfreqs / 2)

    # TODO: Spw axis to be collapsed in future release
    expected_freqs = uvdata_data.uv_object2.freq_array.reshape(
        1, int(uvdata_data.uv_object2.Nfreqs / 2), 2
    ).mean(axis=2)
    assert np.max(np.abs(uvdata_data.uv_object.freq_array - expected_freqs)) == 0

    expected_coeffs = eq_coeffs.reshape(
        uvdata_data.uv_object2.Nants_telescope,
        int(uvdata_data.uv_object2.Nfreqs / 2),
        2,
    ).mean(axis=2)
    assert np.max(np.abs(uvdata_data.uv_object.eq_coeffs - expected_coeffs)) == 0

    # no flagging, so the following is true
    expected_data = uvdata_data.uv_object2.get_data(0, 1, squeeze="none")
    # TODO: Spw axis to be collapsed in future release
    reshape_tuple = (
        expected_data.shape[0],
        1,
        int(uvdata_data.uv_object2.Nfreqs / 2),
        2,
        uvdata_data.uv_object2.Npols,
    )
    expected_data = expected_data.reshape(reshape_tuple).mean(axis=3)
    assert np.allclose(
        uvdata_data.uv_object.get_data(0, 1, squeeze="none"), expected_data
    )

    assert np.nonzero(uvdata_data.uv_object.flag_array)[0].size == 0

    assert not isinstance(uvdata_data.uv_object.data_array, np.ma.MaskedArray)
    assert not isinstance(uvdata_data.uv_object.nsample_array, np.ma.MaskedArray)


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_frequency_average_uneven(uvdata_data):
    """Test averaging in frequency with a number that is not a factor of Nfreqs."""
    # check that there's no flagging
    assert np.nonzero(uvdata_data.uv_object.flag_array)[0].size == 0

    with uvtest.check_warnings(
        UserWarning,
        [
            "Nfreqs does not divide by `n_chan_to_avg` evenly. The final 1 "
            "frequencies will be excluded, to control which frequencies to exclude, "
            "use a select to control.",
            "The uvw_array does not match the expected values",
        ],
    ):
        uvdata_data.uv_object.frequency_average(7)

    assert uvdata_data.uv_object2.Nfreqs % 7 != 0

    assert uvdata_data.uv_object.Nfreqs == (uvdata_data.uv_object2.Nfreqs // 7)

    expected_freqs = uvdata_data.uv_object2.freq_array[
        :, np.arange((uvdata_data.uv_object2.Nfreqs // 7) * 7)
    ]

    # TODO: Spw axis to be collapsed in future release
    expected_freqs = expected_freqs.reshape(
        1, int(uvdata_data.uv_object2.Nfreqs // 7), 7
    ).mean(axis=2)
    assert np.max(np.abs(uvdata_data.uv_object.freq_array - expected_freqs)) == 0

    # no flagging, so the following is true
    expected_data = uvdata_data.uv_object2.get_data(0, 1, squeeze="none")
    expected_data = expected_data[
        :, :, 0 : ((uvdata_data.uv_object2.Nfreqs // 7) * 7), :
    ]
    # TODO: Spw axis to be collapsed in future release
    reshape_tuple = (
        expected_data.shape[0],
        1,
        int(uvdata_data.uv_object2.Nfreqs // 7),
        7,
        uvdata_data.uv_object2.Npols,
    )
    expected_data = expected_data.reshape(reshape_tuple).mean(axis=3)
    assert np.allclose(
        uvdata_data.uv_object.get_data(0, 1, squeeze="none"), expected_data
    )

    assert np.nonzero(uvdata_data.uv_object.flag_array)[0].size == 0


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_frequency_average_flagging(uvdata_data):
    """Test averaging in frequency with flagging all samples averaged."""
    # check that there's no flagging
    assert np.nonzero(uvdata_data.uv_object.flag_array)[0].size == 0

    # apply some flagging for testing
    inds01 = uvdata_data.uv_object.antpair2ind(0, 1)
    uvdata_data.uv_object.flag_array[inds01[0], :, 0:2, :] = True
    assert (
        np.nonzero(uvdata_data.uv_object.flag_array)[0].size
        == uvdata_data.uv_object.Npols * 2
    )

    uvdata_data.uv_object.frequency_average(2)

    assert uvdata_data.uv_object.Nfreqs == (uvdata_data.uv_object2.Nfreqs / 2)

    # TODO: Spw axis to be collapsed in future release
    expected_freqs = uvdata_data.uv_object2.freq_array.reshape(
        1, int(uvdata_data.uv_object2.Nfreqs / 2), 2
    ).mean(axis=2)
    assert np.max(np.abs(uvdata_data.uv_object.freq_array - expected_freqs)) == 0

    # TODO: Spw axis to be collapsed in future release
    expected_data = uvdata_data.uv_object2.get_data(0, 1, squeeze="none")
    reshape_tuple = (
        expected_data.shape[0],
        1,
        int(uvdata_data.uv_object2.Nfreqs / 2),
        2,
        uvdata_data.uv_object2.Npols,
    )
    expected_data = expected_data.reshape(reshape_tuple).mean(axis=3)
    assert np.allclose(
        uvdata_data.uv_object.get_data(0, 1, squeeze="none"), expected_data
    )

    assert np.sum(uvdata_data.uv_object.flag_array[inds01[0], :, 0, :]) == 4
    assert (
        np.nonzero(uvdata_data.uv_object.flag_array)[0].size
        == uvdata_data.uv_object.Npols
    )
    assert (
        np.nonzero(uvdata_data.uv_object.flag_array[inds01[1:], :, 0, :])[0].size == 0
    )


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_frequency_average_flagging_partial(uvdata_data):
    """Test averaging in frequency with flagging only one sample averaged."""
    # check that there's no flagging
    assert np.nonzero(uvdata_data.uv_object.flag_array)[0].size == 0

    # apply some flagging for testing
    inds01 = uvdata_data.uv_object.antpair2ind(0, 1)
    uvdata_data.uv_object.flag_array[inds01[0], :, 0, :] = True
    assert (
        np.nonzero(uvdata_data.uv_object.flag_array)[0].size
        == uvdata_data.uv_object.Npols
    )

    uvdata_data.uv_object.frequency_average(2)

    assert uvdata_data.uv_object.Nfreqs == (uvdata_data.uv_object2.Nfreqs / 2)

    # TODO: Spw axis to be collapsed in future release
    expected_freqs = uvdata_data.uv_object2.freq_array.reshape(
        1, int(uvdata_data.uv_object2.Nfreqs / 2), 2
    ).mean(axis=2)
    assert np.max(np.abs(uvdata_data.uv_object.freq_array - expected_freqs)) == 0

    expected_data = uvdata_data.uv_object2.get_data(0, 1, squeeze="none")
    # TODO: Spw axis to be collapsed in future release
    reshape_tuple = (
        expected_data.shape[0],
        1,
        int(uvdata_data.uv_object2.Nfreqs / 2),
        2,
        uvdata_data.uv_object2.Npols,
    )
    expected_data = expected_data.reshape(reshape_tuple).mean(axis=3)
    expected_data[0, :, 0, :] = uvdata_data.uv_object2.data_array[inds01[0], :, 1, :]
    assert np.allclose(
        uvdata_data.uv_object.get_data(0, 1, squeeze="none"), expected_data
    )

    # check that there's no flagging
    assert np.nonzero(uvdata_data.uv_object.flag_array)[0].size == 0


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_frequency_average_flagging_full_and_partial(uvdata_data):
    """
    Test averaging in frequency with flagging all of one and only one of
    another sample averaged.
    """
    # check that there's no flagging
    assert np.nonzero(uvdata_data.uv_object.flag_array)[0].size == 0

    # apply some flagging for testing
    inds01 = uvdata_data.uv_object.antpair2ind(0, 1)
    uvdata_data.uv_object.flag_array[inds01[0], :, 0:3, :] = True
    assert (
        np.nonzero(uvdata_data.uv_object.flag_array)[0].size
        == uvdata_data.uv_object.Npols * 3
    )

    uvdata_data.uv_object.frequency_average(2)

    assert uvdata_data.uv_object.Nfreqs == (uvdata_data.uv_object2.Nfreqs / 2)

    # TODO: Spw axis to be collapsed in future release
    expected_freqs = uvdata_data.uv_object2.freq_array.reshape(
        1, int(uvdata_data.uv_object2.Nfreqs / 2), 2
    ).mean(axis=2)
    assert np.max(np.abs(uvdata_data.uv_object.freq_array - expected_freqs)) == 0

    expected_data = uvdata_data.uv_object2.get_data(0, 1, squeeze="none")
    # TODO: Spw axis to be collapsed in future release
    reshape_tuple = (
        expected_data.shape[0],
        1,
        int(uvdata_data.uv_object2.Nfreqs / 2),
        2,
        uvdata_data.uv_object2.Npols,
    )
    expected_data = expected_data.reshape(reshape_tuple).mean(axis=3)

    expected_data[0, :, 1, :] = uvdata_data.uv_object2.data_array[inds01[0], :, 3, :]

    assert np.allclose(
        uvdata_data.uv_object.get_data(0, 1, squeeze="none"), expected_data
    )
    assert (
        np.nonzero(uvdata_data.uv_object.flag_array)[0].size
        == uvdata_data.uv_object.Npols
    )


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_frequency_average_flagging_partial_twostage(uvdata_data):
    """
    Test averaging in frequency in two stages with flagging only one sample averaged.
    """
    # check that there's no flagging
    assert np.nonzero(uvdata_data.uv_object.flag_array)[0].size == 0

    # apply some flagging for testing
    inds01 = uvdata_data.uv_object.antpair2ind(0, 1)
    uvdata_data.uv_object.flag_array[inds01[0], :, 0, :] = True
    assert (
        np.nonzero(uvdata_data.uv_object.flag_array)[0].size
        == uvdata_data.uv_object.Npols
    )

    uv_object3 = uvdata_data.uv_object.copy()

    uvdata_data.uv_object.frequency_average(2)
    uvdata_data.uv_object.frequency_average(2)

    uv_object3.frequency_average(4)

    assert uvdata_data.uv_object == uv_object3


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_frequency_average_summing_corr_mode(uvdata_data):
    """Test averaging in frequency."""
    # check that there's no flagging
    assert np.nonzero(uvdata_data.uv_object.flag_array)[0].size == 0

    uvdata_data.uv_object.frequency_average(2, summing_correlator_mode=True)

    assert uvdata_data.uv_object.Nfreqs == (uvdata_data.uv_object2.Nfreqs / 2)

    # TODO: Spw axis to be collapsed in future release
    expected_freqs = uvdata_data.uv_object2.freq_array.reshape(
        1, int(uvdata_data.uv_object2.Nfreqs / 2), 2
    ).mean(axis=2)
    assert np.max(np.abs(uvdata_data.uv_object.freq_array - expected_freqs)) == 0

    # no flagging, so the following is true
    expected_data = uvdata_data.uv_object2.get_data(0, 1, squeeze="none")
    # TODO: Spw axis to be collapsed in future release
    reshape_tuple = (
        expected_data.shape[0],
        1,
        int(uvdata_data.uv_object2.Nfreqs / 2),
        2,
        uvdata_data.uv_object2.Npols,
    )
    expected_data = expected_data.reshape(reshape_tuple).sum(axis=3)
    assert np.allclose(
        uvdata_data.uv_object.get_data(0, 1, squeeze="none"), expected_data
    )

    assert np.nonzero(uvdata_data.uv_object.flag_array)[0].size == 0
    assert not isinstance(uvdata_data.uv_object.data_array, np.ma.MaskedArray)
    assert not isinstance(uvdata_data.uv_object.nsample_array, np.ma.MaskedArray)


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_frequency_average_propagate_flags(uvdata_data):
    """
    Test averaging in frequency with flagging all of one and only one of
    another sample averaged, and propagating flags. Data should be identical,
    but flags should be slightly different compared to other test of the same
    name.
    """
    # check that there's no flagging
    assert np.nonzero(uvdata_data.uv_object.flag_array)[0].size == 0

    # apply some flagging for testing
    inds01 = uvdata_data.uv_object.antpair2ind(0, 1)
    uvdata_data.uv_object.flag_array[inds01[0], :, 0:3, :] = True
    assert (
        np.nonzero(uvdata_data.uv_object.flag_array)[0].size
        == uvdata_data.uv_object.Npols * 3
    )

    uvdata_data.uv_object.frequency_average(2, propagate_flags=True)

    assert uvdata_data.uv_object.Nfreqs == (uvdata_data.uv_object2.Nfreqs / 2)

    # TODO: Spw axis to be collapsed in future release
    expected_freqs = uvdata_data.uv_object2.freq_array.reshape(
        1, int(uvdata_data.uv_object2.Nfreqs / 2), 2
    ).mean(axis=2)
    assert np.max(np.abs(uvdata_data.uv_object.freq_array - expected_freqs)) == 0

    expected_data = uvdata_data.uv_object2.get_data(0, 1, squeeze="none")
    # TODO: Spw axis to be collapsed in future release
    reshape_tuple = (
        expected_data.shape[0],
        1,
        int(uvdata_data.uv_object2.Nfreqs / 2),
        2,
        uvdata_data.uv_object2.Npols,
    )
    expected_data = expected_data.reshape(reshape_tuple).mean(axis=3)

    expected_data[0, :, 1, :] = uvdata_data.uv_object2.data_array[inds01[0], :, 3, :]

    assert np.allclose(
        uvdata_data.uv_object.get_data(0, 1, squeeze="none"), expected_data
    )
    # Twice as many flags should exist compared to test of previous name.
    assert (
        np.nonzero(uvdata_data.uv_object.flag_array)[0].size
        == 2 * uvdata_data.uv_object.Npols
    )


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_frequency_average_nsample_precision(uvdata_data):
    """Test averaging in frequency with a half-precision nsample_array."""
    eq_coeffs = np.tile(
        np.arange(uvdata_data.uv_object.Nfreqs, dtype=np.float),
        (uvdata_data.uv_object.Nants_telescope, 1),
    )
    uvdata_data.uv_object.eq_coeffs = eq_coeffs
    uvdata_data.uv_object.check()

    # check that there's no flagging
    assert np.nonzero(uvdata_data.uv_object.flag_array)[0].size == 0

    # change precision of the nsample array
    uvdata_data.uv_object.nsample_array = uvdata_data.uv_object.nsample_array.astype(
        np.float16
    )

    with uvtest.check_warnings(UserWarning, "eq_coeffs vary by frequency"):
        uvdata_data.uv_object.frequency_average(2),

    assert uvdata_data.uv_object.Nfreqs == (uvdata_data.uv_object2.Nfreqs / 2)

    # TODO: Spw axis to be collapsed in future release
    expected_freqs = uvdata_data.uv_object2.freq_array.reshape(
        1, int(uvdata_data.uv_object2.Nfreqs / 2), 2
    ).mean(axis=2)
    assert np.max(np.abs(uvdata_data.uv_object.freq_array - expected_freqs)) == 0

    expected_coeffs = eq_coeffs.reshape(
        uvdata_data.uv_object2.Nants_telescope,
        int(uvdata_data.uv_object2.Nfreqs / 2),
        2,
    ).mean(axis=2)
    assert np.max(np.abs(uvdata_data.uv_object.eq_coeffs - expected_coeffs)) == 0

    # no flagging, so the following is true
    expected_data = uvdata_data.uv_object2.get_data(0, 1, squeeze="none")
    # TODO: Spw axis to be collapsed in future release
    reshape_tuple = (
        expected_data.shape[0],
        1,
        int(uvdata_data.uv_object2.Nfreqs / 2),
        2,
        uvdata_data.uv_object2.Npols,
    )
    expected_data = expected_data.reshape(reshape_tuple).mean(axis=3)
    assert np.allclose(
        uvdata_data.uv_object.get_data(0, 1, squeeze="none"), expected_data
    )

    assert np.nonzero(uvdata_data.uv_object.flag_array)[0].size == 0

    assert not isinstance(uvdata_data.uv_object.data_array, np.ma.MaskedArray)
    assert not isinstance(uvdata_data.uv_object.nsample_array, np.ma.MaskedArray)

    # make sure we still have a half-precision nsample_array
    assert uvdata_data.uv_object.nsample_array.dtype.type is np.float16


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_remove_eq_coeffs_divide(uvdata_data):
    """Test using the remove_eq_coeffs method with divide convention."""
    # give eq_coeffs to the object
    eq_coeffs = np.empty(
        (uvdata_data.uv_object.Nants_telescope, uvdata_data.uv_object.Nfreqs),
        dtype=np.float,
    )
    for i, ant in enumerate(uvdata_data.uv_object.antenna_numbers):
        eq_coeffs[i, :] = ant + 1
    uvdata_data.uv_object.eq_coeffs = eq_coeffs
    uvdata_data.uv_object.eq_coeffs_convention = "divide"
    uvdata_data.uv_object.remove_eq_coeffs()

    # make sure the right coefficients were removed
    for key in uvdata_data.uv_object.get_antpairs():
        eq1 = key[0] + 1
        eq2 = key[1] + 1
        blt_inds = uvdata_data.uv_object.antpair2ind(key)
        norm_data = uvdata_data.uv_object.data_array[blt_inds, 0, :, :]
        unnorm_data = uvdata_data.uv_object2.data_array[blt_inds, 0, :, :]
        assert np.allclose(norm_data, unnorm_data / (eq1 * eq2))

    return


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_remove_eq_coeffs_multiply(uvdata_data):
    """Test using the remove_eq_coeffs method with multiply convention."""
    # give eq_coeffs to the object
    eq_coeffs = np.empty(
        (uvdata_data.uv_object.Nants_telescope, uvdata_data.uv_object.Nfreqs),
        dtype=np.float,
    )
    for i, ant in enumerate(uvdata_data.uv_object.antenna_numbers):
        eq_coeffs[i, :] = ant + 1
    uvdata_data.uv_object.eq_coeffs = eq_coeffs
    uvdata_data.uv_object.eq_coeffs_convention = "multiply"
    uvdata_data.uv_object.remove_eq_coeffs()

    # make sure the right coefficients were removed
    for key in uvdata_data.uv_object.get_antpairs():
        eq1 = key[0] + 1
        eq2 = key[1] + 1
        blt_inds = uvdata_data.uv_object.antpair2ind(key)
        norm_data = uvdata_data.uv_object.data_array[blt_inds, 0, :, :]
        unnorm_data = uvdata_data.uv_object2.data_array[blt_inds, 0, :, :]
        assert np.allclose(norm_data, unnorm_data * (eq1 * eq2))

    return


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_remove_eq_coeffs_errors(uvdata_data):
    """Test errors raised by remove_eq_coeffs method."""
    # raise error when eq_coeffs are not defined
    with pytest.raises(ValueError) as cm:
        uvdata_data.uv_object.remove_eq_coeffs()
    assert str(cm.value).startswith("The eq_coeffs attribute must be defined")

    # raise error when eq_coeffs are defined but not eq_coeffs_convention
    uvdata_data.uv_object.eq_coeffs = np.ones(
        (uvdata_data.uv_object.Nants_telescope, uvdata_data.uv_object.Nfreqs)
    )
    with pytest.raises(ValueError) as cm:
        uvdata_data.uv_object.remove_eq_coeffs()
    assert str(cm.value).startswith(
        "The eq_coeffs_convention attribute must be defined"
    )

    # raise error when convention is not a valid choice
    uvdata_data.uv_object.eq_coeffs_convention = "foo"
    with pytest.raises(ValueError) as cm:
        uvdata_data.uv_object.remove_eq_coeffs()
    assert str(cm.value).startswith("Got unknown convention foo. Must be one of")

    return


@pytest.mark.parametrize(
    "read_func,filelist",
    [
        ("read_miriad", [os.path.join(DATA_PATH, "zen.2457698.40355.xx.HH.uvcA")] * 2),
        (
            "read_mwa_corr_fits",
            [[mwa_corr_files[0:2], [mwa_corr_files[0], mwa_corr_files[2]]]],
        ),
        ("read_uvh5", [os.path.join(DATA_PATH, "zen.2458661.23480.HH.uvh5")] * 2),
        (
            "read_uvfits",
            [os.path.join(DATA_PATH, "day2_TDEM0003_10s_norx_1src_1spw.uvfits")] * 2,
        ),
        (
            "read_ms",
            [
                os.path.join(DATA_PATH, "multi_1.ms"),
                os.path.join(DATA_PATH, "multi_2.ms"),
            ],
        ),
        (
            "read_fhd",
            [
                list(np.array(fhd_files)[[0, 1, 2, 4, 6, 7]]),
                list(np.array(fhd_files)[[0, 2, 3, 5, 6, 7]]),
            ],
        ),
    ],
)
def test_multifile_read_errors(read_func, filelist):
    uv = UVData()
    with pytest.raises(ValueError) as cm:
        getattr(uv, read_func)(filelist)
    assert str(cm.value).startswith(
        "Reading multiple files from class specific read functions is no "
        "longer supported."
    )


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_multifile_read_check(hera_uvh5, tmp_path):
    """Test setting skip_bad_files=True when reading in files"""

    uvTrue = hera_uvh5.copy()

    # Added here because the files seem to have the values flipped
    uvTrue.uvw_array *= -1.0
    uvh5_file = os.path.join(DATA_PATH, "zen.2458661.23480.HH.uvh5")

    # Create a test file and remove header info to 'corrupt' it
    testfile = os.path.join(tmp_path, "zen.2458661.23480.HH.uvh5")

    uvTrue.write_uvh5(testfile)
    with h5py.File(testfile, "r+") as h5f:
        del h5f["Header/ant_1_array"]

    uv = UVData()
    # Test that the expected error arises
    with pytest.raises(KeyError) as cm:
        uv.read(testfile, skip_bad_files=False)
    assert "Unable to open object (object 'ant_1_array' doesn't exist)" in str(cm.value)

    # Test when the corrupted file is at the beggining, skip_bad_files=False
    fileList = [testfile, uvh5_file]
    with pytest.raises(KeyError) as cm:
        with uvtest.check_warnings(UserWarning, match="Failed to read"):
            uv.read(fileList, skip_bad_files=False)
    assert "Unable to open object (object 'ant_1_array' doesn't exist)" in str(cm.value)
    assert uv != uvTrue

    # Test when the corrupted file is at the beggining, skip_bad_files=True
    fileList = [testfile, uvh5_file]
    with uvtest.check_warnings(
        UserWarning,
        match=[
            "Failed to read",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv.read(fileList, skip_bad_files=True)
    assert uv == uvTrue

    # Test when the corrupted file is at the end of a list
    fileList = [uvh5_file, testfile]
    with uvtest.check_warnings(
        UserWarning,
        match=[
            "Failed to read",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv.read(fileList, skip_bad_files=True)
    # Check that the uncorrupted file was still read in
    assert uv == uvTrue

    os.remove(testfile)

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("err_type", ["KeyError", "ValueError"])
def test_multifile_read_check_long_list(hera_uvh5, tmp_path, err_type):
    """
    Test KeyError catching by setting skip_bad_files=True when
    reading in files for a list of length >2
    """
    # Create mini files for testing
    uv = hera_uvh5

    fileList = []
    for i in range(0, 4):
        uv2 = uv.select(
            times=np.unique(uv.time_array)[i * 5 : i * 5 + 4], inplace=False
        )
        fname = str(tmp_path / f"minifile_{i}.uvh5")
        fileList.append(fname)
        uv2.write_uvh5(fname)
    if err_type == "KeyError":
        with h5py.File(fileList[-1], "r+") as h5f:
            del h5f["Header/ant_1_array"]
    elif err_type == "ValueError":
        with h5py.File(fileList[-1], "r+") as h5f:
            h5f["Header/antenna_numbers"][3] = 85
            h5f["Header/ant_1_array"][2] = 1024

    # Test with corrupted file as last file in list, skip_bad_files=True
    uvTest = UVData()
    with uvtest.check_warnings(UserWarning, "Failed to read"):
        uvTest.read(fileList[0:4], skip_bad_files=True)
    uvTrue = UVData()
    uvTrue.read(fileList[0:3], skip_bad_files=True)

    assert uvTest == uvTrue

    # Repeat above test, but with corrupted file as first file in list
    os.remove(fileList[3])
    uv2 = uv.select(times=np.unique(uv.time_array)[15:19], inplace=False)
    fname = str(tmp_path / f"minifile_{3}.uvh5")
    uv2.write_uvh5(fname)
    if err_type == "KeyError":
        with h5py.File(fileList[0], "r+") as h5f:
            del h5f["Header/ant_1_array"]
    elif err_type == "ValueError":
        with h5py.File(fileList[0], "r+") as h5f:
            h5f["Header/antenna_numbers"][3] = 85
            h5f["Header/ant_1_array"][2] = 1024
    uvTest = UVData()
    with uvtest.check_warnings(UserWarning, "Failed to read"):
        uvTest.read(fileList[0:4], skip_bad_files=True)
    uvTrue = UVData()
    uvTrue.read(fileList[1:4], skip_bad_files=True)

    assert uvTest == uvTrue

    # Test with corrupted file first in list, but with skip_bad_files=False
    uvTest = UVData()
    if err_type == "KeyError":
        with pytest.raises(KeyError, match="Unable to open object"):
            with uvtest.check_warnings(UserWarning, match="Failed to read"):
                uvTest.read(fileList[0:4], skip_bad_files=False)
    elif err_type == "ValueError":
        with pytest.raises(ValueError, match="Nants_data must be equal to"):
            with uvtest.check_warnings(UserWarning, match="Failed to read"):
                uvTest.read(fileList[0:4], skip_bad_files=False)
    uvTrue = UVData()
    uvTrue.read([fileList[1], fileList[2], fileList[3]], skip_bad_files=False)

    assert uvTest != uvTrue

    # Repeat above test, but with corrupted file in the middle of the list
    os.remove(fileList[0])
    uv2 = uv.select(times=np.unique(uv.time_array)[0:4], inplace=False)
    fname = str(tmp_path / f"minifile_{0}.uvh5")
    uv2.write_uvh5(fname)
    if err_type == "KeyError":
        with h5py.File(fileList[1], "r+") as h5f:
            del h5f["Header/ant_1_array"]
    elif err_type == "ValueError":
        with h5py.File(fileList[1], "r+") as h5f:
            h5f["Header/antenna_numbers"][3] = 85
            h5f["Header/ant_1_array"][2] = 1024
    uvTest = UVData()
    with uvtest.check_warnings(UserWarning, "Failed to read"):
        uvTest.read(fileList[0:4], skip_bad_files=True)
    uvTrue = UVData()
    uvTrue.read([fileList[0], fileList[2], fileList[3]], skip_bad_files=True)

    assert uvTest == uvTrue

    # Test with corrupted file in middle of list, but with skip_bad_files=False
    uvTest = UVData()
    if err_type == "KeyError":
        with pytest.raises(KeyError, match="Unable to open object"):
            with uvtest.check_warnings(UserWarning, match="Failed to read"):
                uvTest.read(fileList[0:4], skip_bad_files=False)
    elif err_type == "ValueError":
        with pytest.raises(ValueError, match="Nants_data must be equal to"):
            with uvtest.check_warnings(UserWarning, match="Failed to read"):
                uvTest.read(fileList[0:4], skip_bad_files=False)
    uvTrue = UVData()
    uvTrue.read([fileList[0], fileList[2], fileList[3]], skip_bad_files=False)

    assert uvTest != uvTrue

    # Test case where all files in list are corrupted
    os.remove(fileList[1])
    uv2 = uv.select(times=np.unique(uv.time_array)[5:9], inplace=False)
    fname = str(tmp_path / f"minifile_{1}.uvh5")
    uv2.write_uvh5(fname)
    for file in fileList:
        if err_type == "KeyError":
            with h5py.File(file, "r+") as h5f:
                del h5f["Header/ant_1_array"]
        elif err_type == "ValueError":
            with h5py.File(file, "r+") as h5f:
                h5f["Header/antenna_numbers"][3] = 85
                h5f["Header/ant_1_array"][2] = 1024
    uvTest = UVData()
    with uvtest.check_warnings(
        UserWarning,
        match=(
            "########################################################\n"
            "ALL FILES FAILED ON READ - NO READABLE FILES IN FILENAME\n"
            "########################################################"
        ),
    ):
        uvTest.read(fileList[0:4], skip_bad_files=True)
    uvTrue = UVData()

    assert uvTest == uvTrue

    os.remove(fileList[0])
    os.remove(fileList[1])
    os.remove(fileList[2])
    os.remove(fileList[3])

    return


def test_deprecation_warnings_set_phased():
    """
    Test the deprecation warnings in set_phased et al.
    """
    uv = UVData()
    # first call set_phased
    with uvtest.check_warnings(DeprecationWarning, match="`set_phased` is deprecated"):
        uv.set_phased()
    assert uv.phase_type == "phased"
    assert uv._phase_center_epoch.required is True
    assert uv._phase_center_ra.required is True
    assert uv._phase_center_dec.required is True

    # now call set_drift
    with uvtest.check_warnings(DeprecationWarning, match="`set_drift` is deprecated"):
        uv.set_drift()
    assert uv.phase_type == "drift"
    assert uv._phase_center_epoch.required is False
    assert uv._phase_center_ra.required is False
    assert uv._phase_center_dec.required is False

    # now call set_unknown_phase_type
    with uvtest.check_warnings(
        DeprecationWarning, match="`set_unknown_phase_type` is deprecated"
    ):
        uv.set_unknown_phase_type()
    assert uv.phase_type == "unknown"
    assert uv._phase_center_epoch.required is False
    assert uv._phase_center_ra.required is False
    assert uv._phase_center_dec.required is False

    return


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not in known_telescopes.")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_read_background_lsts():
    """Test reading a file with the lst calc in the background."""
    uvd = UVData()
    uvd2 = UVData()
    testfile = os.path.join(DATA_PATH, "day2_TDEM0003_10s_norx_1src_1spw.uvfits")
    uvd.read(testfile, background_lsts=False)
    uvd2.read(testfile, background_lsts=True)
    assert uvd == uvd2


def test_parse_ants_x_orientation_kwarg(hera_uvh5):
    uvd = hera_uvh5
    # call with x_orientation = None to make parse_ants read from the object
    ant_pair, pols = uvutils.parse_ants(uvd, "cross")
    ant_pair2, pols2 = uvd.parse_ants("cross")
    assert np.array_equal(ant_pair, ant_pair2)
    assert np.array_equal(pols, pols2)
