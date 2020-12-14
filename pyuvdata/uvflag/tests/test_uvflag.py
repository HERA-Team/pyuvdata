# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

import pytest
from _pytest.outcomes import Skipped
import os
import numpy as np
import pyuvdata.tests as uvtest
from pyuvdata import UVData, UVCal, utils as uvutils
from pyuvdata.data import DATA_PATH
from pyuvdata import UVFlag
from ..uvflag import lst_from_uv, flags2waterfall, and_rows_cols
from pyuvdata import __version__
import shutil
import copy
import warnings
import h5py
import pathlib

test_d_file = os.path.join(DATA_PATH, "zen.2457698.40355.xx.HH.uvcAA.uvh5")
test_c_file = os.path.join(DATA_PATH, "zen.2457555.42443.HH.uvcA.omni.calfits")
test_f_file = test_d_file.rstrip(".uvh5") + ".testuvflag.h5"

pyuvdata_version_str = "  Read/written with pyuvdata version: " + __version__ + "."

pytestmark = pytest.mark.filterwarnings(
    "ignore:telescope_location is not set. Using known values for HERA.",
    "ignore:antenna_positions is not set. Using known values for HERA.",
)


@pytest.fixture(scope="session")
def uvdata_obj_main():
    uvdata_object = UVData()
    uvdata_object.read(test_d_file)

    yield uvdata_object

    # cleanup
    del uvdata_object

    return


@pytest.fixture(scope="function")
def uvdata_obj(uvdata_obj_main):
    uvdata_object = uvdata_obj_main.copy()

    yield uvdata_object

    # cleanup
    del uvdata_object

    return


# The following three fixtures are used regularly
# to initizize UVFlag objects from standard files
# We need to define these here in order to set up
# some skips for developers who do not have `pytest-cases` installed
@pytest.fixture(scope="function")
def uvf_from_data(uvdata_obj):
    uvf = UVFlag()
    uvf.from_uvdata(uvdata_obj)

    # yield the object for the test
    yield uvf

    # do some cleanup
    del (uvf, uvdata_obj)


@pytest.fixture(scope="function")
def uvf_from_uvcal():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = UVFlag()
    uvf.from_uvcal(uvc)

    # the antenna type test file is large, so downselect to speed up
    if uvf.type == "antenna":
        uvf.select(antenna_nums=uvf.ant_array[:5])

    # yield the object for the test
    yield uvf

    # do some cleanup
    del (uvf, uvc)


@pytest.fixture(scope="function")
def uvf_from_waterfall(uvdata_obj):
    uvf = UVFlag()
    uvf.from_uvdata(uvdata_obj, waterfall=True)

    # yield the object for the test
    yield uvf

    # do some cleanup
    del uvf


# Try to import `pytest-cases` and define decorators used to
# iterate over the three main types of UVFlag objects
# otherwise make the decorators skip the tests that use these iterators
try:
    pytest_cases = pytest.importorskip("pytest_cases", minversion="1.12.1")

    cases_decorator = pytest_cases.parametrize_plus(
        "input_uvf",
        [
            pytest_cases.fixture_ref(uvf_from_data),
            pytest_cases.fixture_ref(uvf_from_uvcal),
            pytest_cases.fixture_ref(uvf_from_waterfall),
        ],
    )

    cases_decorator_no_waterfall = pytest_cases.parametrize_plus(
        "input_uvf",
        [
            pytest_cases.fixture_ref(uvf_from_data),
            pytest_cases.fixture_ref(uvf_from_uvcal),
        ],
    )

    # This warning is raised by pytest_cases
    # It is due to a feature the developer does
    # not know how to handle yet. ignore for now.
    warnings.filterwarnings(
        "ignore",
        message="WARNING the new order is not" + " taken into account !!",
        append=True,
    )

except Skipped:
    cases_decorator = pytest.mark.skipif(
        True, reason="pytest-cases not installed or not required version"
    )
    cases_decorator_no_waterfall = pytest.mark.skipif(
        True, reason="pytest-cases not installed or not required version"
    )


@pytest.fixture()
def test_outfile(tmp_path):
    yield str(tmp_path / "outtest_uvflag.h5")


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_init_bad_mode(uvdata_obj):
    uv = uvdata_obj
    with pytest.raises(ValueError) as cm:
        UVFlag(uv, mode="bad_mode", history="I made a UVFlag object", label="test")
    assert str(cm.value).startswith("Input mode must be within acceptable")

    uv = UVCal()
    uv.read_calfits(test_c_file)
    with pytest.raises(ValueError) as cm:
        UVFlag(uv, mode="bad_mode", history="I made a UVFlag object", label="test")
    assert str(cm.value).startswith("Input mode must be within acceptable")


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_init_uvdata(uvdata_obj):
    uv = uvdata_obj
    uvf = UVFlag(uv, history="I made a UVFlag object", label="test")
    assert uvf.metric_array.shape == uv.flag_array.shape
    assert np.all(uvf.metric_array == 0)
    assert uvf.weights_array.shape == uv.flag_array.shape
    assert np.all(uvf.weights_array == 1)
    assert uvf.type == "baseline"
    assert uvf.mode == "metric"
    assert np.all(uvf.time_array == uv.time_array)
    assert np.all(uvf.lst_array == uv.lst_array)
    assert np.all(uvf.freq_array == uv.freq_array[0])
    assert np.all(uvf.polarization_array == uv.polarization_array)
    assert np.all(uvf.baseline_array == uv.baseline_array)
    assert np.all(uvf.ant_1_array == uv.ant_1_array)
    assert np.all(uvf.ant_2_array == uv.ant_2_array)
    assert "I made a UVFlag object" in uvf.history
    assert 'Flag object with type "baseline"' in uvf.history
    assert pyuvdata_version_str in uvf.history
    assert uvf.label == "test"


def test_add_extra_keywords(uvdata_obj):
    uv = uvdata_obj
    uvf = UVFlag(uv, history="I made a UVFlag object", label="test")
    uvf.extra_keywords = {"keyword1": 1, "keyword2": 2}
    assert "keyword1" in uvf.extra_keywords
    assert "keyword2" in uvf.extra_keywords
    uvf.extra_keywords["keyword3"] = 3
    assert "keyword3" in uvf.extra_keywords
    assert uvf.extra_keywords.get("keyword1") == 1
    assert uvf.extra_keywords.get("keyword2") == 2
    assert uvf.extra_keywords.get("keyword3") == 3


def test_read_extra_keywords(uvdata_obj):
    uv = uvdata_obj
    uv.extra_keywords = {"keyword1": 1, "keyword2": 2}
    assert "keyword1" in uv.extra_keywords
    assert "keyword2" in uv.extra_keywords
    uvf = UVFlag(uv, history="I made a UVFlag object", label="test")
    assert "keyword1" in uvf.extra_keywords
    assert "keyword2" in uvf.extra_keywords


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_init_uvdata_x_orientation(uvdata_obj):
    uv = uvdata_obj
    uv.x_orientation = "east"
    uvf = UVFlag(uv, history="I made a UVFlag object", label="test")
    assert uvf.x_orientation == uv.x_orientation


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_init_uvdata_copy_flags(uvdata_obj, future_shapes):
    uv = uvdata_obj

    if future_shapes:
        uv.use_future_array_shapes()

    with uvtest.check_warnings(UserWarning, 'Copying flags to type=="baseline"'):
        uvf = UVFlag(uv, copy_flags=True, mode="metric")
    #  with copy flags uvf.metric_array should be none
    assert hasattr(uvf, "metric_array")
    assert uvf.metric_array is None
    if future_shapes:
        assert np.array_equal(uvf.flag_array[:, 0, :, :], uv.flag_array)
    else:
        assert np.array_equal(uvf.flag_array, uv.flag_array)
    assert uvf.weights_array is None
    assert uvf.type == "baseline"
    assert uvf.mode == "flag"
    assert np.all(uvf.time_array == uv.time_array)
    assert np.all(uvf.lst_array == uv.lst_array)
    if future_shapes:
        assert np.all(uvf.freq_array == uv.freq_array)
    else:
        assert np.all(uvf.freq_array == uv.freq_array[0])
    assert np.all(uvf.polarization_array == uv.polarization_array)
    assert np.all(uvf.baseline_array == uv.baseline_array)
    assert np.all(uvf.ant_1_array == uv.ant_1_array)
    assert np.all(uvf.ant_2_array == uv.ant_2_array)
    assert 'Flag object with type "baseline"' in uvf.history
    assert pyuvdata_version_str in uvf.history


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_init_uvdata_mode_flag(uvdata_obj):
    uv = uvdata_obj
    uvf = UVFlag()
    uvf.from_uvdata(uv, copy_flags=False, mode="flag")
    #  with copy flags uvf.metric_array should be none
    assert hasattr(uvf, "metric_array")
    assert uvf.metric_array is None
    assert np.array_equal(uvf.flag_array, uv.flag_array)
    assert uvf.weights_array is None
    assert uvf.type == "baseline"
    assert uvf.mode == "flag"
    assert np.all(uvf.time_array == uv.time_array)
    assert np.all(uvf.lst_array == uv.lst_array)
    assert np.all(uvf.freq_array == uv.freq_array[0])
    assert np.all(uvf.polarization_array == uv.polarization_array)
    assert np.all(uvf.baseline_array == uv.baseline_array)
    assert np.all(uvf.ant_1_array == uv.ant_1_array)
    assert np.all(uvf.ant_2_array == uv.ant_2_array)
    assert 'Flag object with type "baseline"' in uvf.history
    assert pyuvdata_version_str in uvf.history


def test_init_uvcal():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = UVFlag(uvc)
    assert uvf.metric_array.shape == uvc.flag_array.shape
    assert np.all(uvf.metric_array == 0)
    assert uvf.weights_array.shape == uvc.flag_array.shape
    assert np.all(uvf.weights_array == 1)
    assert uvf.type == "antenna"
    assert uvf.mode == "metric"
    assert np.all(uvf.time_array == uvc.time_array)
    assert uvf.x_orientation == uvc.x_orientation
    lst = lst_from_uv(uvc)
    assert np.all(uvf.lst_array == lst)
    assert np.all(uvf.freq_array == uvc.freq_array[0])
    assert np.all(uvf.polarization_array == uvc.jones_array)
    assert np.all(uvf.ant_array == uvc.ant_array)
    assert 'Flag object with type "antenna"' in uvf.history
    assert pyuvdata_version_str in uvf.history


def test_init_uvcal_mode_flag():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = UVFlag(uvc, copy_flags=False, mode="flag")
    assert hasattr(uvf, "metric_array")
    assert uvf.metric_array is None
    assert np.array_equal(uvf.flag_array, uvc.flag_array)
    assert uvf.weights_array is None
    assert uvf.type == "antenna"
    assert uvf.mode == "flag"
    assert np.all(uvf.time_array == uvc.time_array)
    lst = lst_from_uv(uvc)
    assert np.all(uvf.lst_array == lst)
    assert np.all(uvf.freq_array == uvc.freq_array[0])
    assert np.all(uvf.polarization_array == uvc.jones_array)
    assert np.all(uvf.ant_array == uvc.ant_array)
    assert 'Flag object with type "antenna"' in uvf.history
    assert pyuvdata_version_str in uvf.history


def test_init_cal_copy_flags():
    uv = UVCal()
    uv.read_calfits(test_c_file)
    with uvtest.check_warnings(UserWarning, 'Copying flags to type=="antenna"'):
        uvf = UVFlag(uv, copy_flags=True, mode="metric")
    #  with copy flags uvf.metric_array should be none
    assert hasattr(uvf, "metric_array")
    assert uvf.metric_array is None
    assert np.array_equal(uvf.flag_array, uv.flag_array)
    assert uvf.type == "antenna"
    assert uvf.mode == "flag"
    assert np.all(uvf.time_array == np.unique(uv.time_array))
    assert np.all(uvf.freq_array == uv.freq_array[0])
    assert np.all(uvf.polarization_array == uv.jones_array)
    assert pyuvdata_version_str in uvf.history


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_init_waterfall_uvd(uvdata_obj, future_shapes):
    uv = uvdata_obj

    if future_shapes:
        uv.use_future_array_shapes()

    uvf = UVFlag(uv, waterfall=True)
    assert uvf.metric_array.shape == (uv.Ntimes, uv.Nfreqs, uv.Npols)
    assert np.all(uvf.metric_array == 0)
    assert uvf.weights_array.shape == (uv.Ntimes, uv.Nfreqs, uv.Npols)
    assert np.all(uvf.weights_array == 1)
    assert uvf.type == "waterfall"
    assert uvf.mode == "metric"
    assert np.all(uvf.time_array == np.unique(uv.time_array))
    assert np.all(uvf.lst_array == np.unique(uv.lst_array))
    if future_shapes:
        assert np.all(uvf.freq_array == uv.freq_array)
    else:
        assert np.all(uvf.freq_array == uv.freq_array[0])
    assert np.all(uvf.polarization_array == uv.polarization_array)
    assert 'Flag object with type "waterfall"' in uvf.history
    assert pyuvdata_version_str in uvf.history


def test_init_waterfall_uvc():
    uv = UVCal()
    uv.read_calfits(test_c_file)
    uvf = UVFlag(uv, waterfall=True, history="input history check")
    assert uvf.metric_array.shape == (uv.Ntimes, uv.Nfreqs, uv.Njones)
    assert np.all(uvf.metric_array == 0)
    assert uvf.weights_array.shape == (uv.Ntimes, uv.Nfreqs, uv.Njones)
    assert np.all(uvf.weights_array == 1)
    assert uvf.type == "waterfall"
    assert uvf.mode == "metric"
    assert np.all(uvf.time_array == np.unique(uv.time_array))
    assert np.all(uvf.freq_array == uv.freq_array[0])
    assert np.all(uvf.polarization_array == uv.jones_array)
    assert 'Flag object with type "waterfall"' in uvf.history
    assert "input history check" in uvf.history
    assert pyuvdata_version_str in uvf.history


def test_init_waterfall_flag_uvcal():
    uv = UVCal()
    uv.read_calfits(test_c_file)
    uvf = UVFlag(uv, waterfall=True, mode="flag")
    assert uvf.flag_array.shape == (uv.Ntimes, uv.Nfreqs, uv.Njones)
    assert not np.any(uvf.flag_array)
    assert uvf.weights_array is None
    assert uvf.type == "waterfall"
    assert uvf.mode == "flag"
    assert np.all(uvf.time_array == np.unique(uv.time_array))
    assert np.all(uvf.freq_array == uv.freq_array[0])
    assert np.all(uvf.polarization_array == uv.jones_array)
    assert 'Flag object with type "waterfall"' in uvf.history
    assert pyuvdata_version_str in uvf.history


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_init_waterfall_flag_uvdata(uvdata_obj):
    uv = uvdata_obj
    uvf = UVFlag(uv, waterfall=True, mode="flag")
    assert uvf.flag_array.shape == (uv.Ntimes, uv.Nfreqs, uv.Npols)
    assert not np.any(uvf.flag_array)
    assert uvf.weights_array is None
    assert uvf.type == "waterfall"
    assert uvf.mode == "flag"
    assert np.all(uvf.time_array == np.unique(uv.time_array))
    assert np.all(uvf.freq_array == uv.freq_array[0])
    assert np.all(uvf.polarization_array == uv.polarization_array)
    assert 'Flag object with type "waterfall"' in uvf.history
    assert pyuvdata_version_str in uvf.history


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_init_waterfall_copy_flags(uvdata_obj):
    uv = UVCal()
    uv.read_calfits(test_c_file)
    with pytest.raises(NotImplementedError) as cm:
        UVFlag(uv, copy_flags=True, mode="flag", waterfall=True)
    assert str(cm.value).startswith("Cannot copy flags when initializing")

    uv = uvdata_obj
    with pytest.raises(NotImplementedError) as cm:
        UVFlag(uv, copy_flags=True, mode="flag", waterfall=True)
    assert str(cm.value).startswith("Cannot copy flags when initializing")


def test_init_invalid_input():
    # input is not UVData, UVCal, path, or list/tuple
    with pytest.raises(ValueError) as cm:
        UVFlag(14)
    assert str(cm.value).startswith("input to UVFlag.__init__ must be one of:")


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_from_uvcal_error(uvdata_obj):
    uv = uvdata_obj
    uvf = UVFlag()
    with pytest.raises(ValueError) as cm:
        uvf.from_uvcal(uv)
    assert str(cm.value).startswith("from_uvcal can only initialize a UVFlag object")


def test_from_udata_error():
    uv = UVCal()
    uv.read_calfits(test_c_file)
    uvf = UVFlag()
    with pytest.raises(ValueError) as cm:
        uvf.from_uvdata(uv)
    assert str(cm.value).startswith("from_uvdata can only initialize a UVFlag object")


def test_init_list_files_weights(tmpdir):
    # Test that weights are preserved when reading list of files
    tmp_path = tmpdir.strpath
    # Create two files to read
    uvf = UVFlag(test_f_file)
    np.random.seed(0)
    wts1 = np.random.rand(*uvf.weights_array.shape)
    uvf.weights_array = wts1.copy()
    uvf.write(os.path.join(tmp_path, "test1.h5"))
    wts2 = np.random.rand(*uvf.weights_array.shape)
    uvf.weights_array = wts2.copy()
    uvf.write(os.path.join(tmp_path, "test2.h5"))
    uvf2 = UVFlag(
        [os.path.join(tmp_path, "test1.h5"), os.path.join(tmp_path, "test2.h5")]
    )
    assert np.all(uvf2.weights_array == np.concatenate([wts1, wts2], axis=0))


def test_init_posix():
    # Test that weights are preserved when reading list of files
    testfile_posix = pathlib.Path(test_f_file)
    uvf1 = UVFlag(test_f_file)
    uvf2 = UVFlag(testfile_posix)
    assert uvf1 == uvf2


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_data_like_property_mode_tamper(uvdata_obj):
    uv = uvdata_obj
    uvf = UVFlag(uv, label="test")
    uvf.mode = "test"
    with pytest.raises(ValueError) as cm:
        list(uvf.data_like_parameters)
    assert str(cm.value).startswith("Invalid mode. Mode must be one of")


def test_read_write_loop(uvdata_obj, test_outfile):
    uv = uvdata_obj
    uvf = UVFlag(uv, label="test")
    uvf.write(test_outfile, clobber=True)
    uvf2 = UVFlag(test_outfile)
    assert uvf.__eq__(uvf2, check_history=True)


def test_read_write_loop_with_optional_x_orientation(uvdata_obj, test_outfile):
    uv = uvdata_obj
    uvf = UVFlag(uv, label="test")
    uvf.x_orientation = "east"
    uvf.write(test_outfile, clobber=True)
    uvf2 = UVFlag(test_outfile)
    assert uvf.__eq__(uvf2, check_history=True)


def test_read_write_loop_waterfal(uvdata_obj, test_outfile):
    uv = uvdata_obj
    uvf = UVFlag(uv, label="test")
    uvf.to_waterfall()
    uvf.write(test_outfile, clobber=True)
    uvf2 = UVFlag(test_outfile)
    assert uvf.__eq__(uvf2, check_history=True)


def test_read_write_loop_ret_wt_sq(test_outfile):
    uvf = UVFlag(test_f_file)
    uvf.weights_array = 2 * np.ones_like(uvf.weights_array)
    uvf.to_waterfall(return_weights_square=True)
    uvf.write(test_outfile, clobber=True)
    uvf2 = UVFlag(test_outfile)
    assert uvf.__eq__(uvf2, check_history=True)


def test_bad_mode_savefile(uvdata_obj, test_outfile):
    uv = uvdata_obj
    uvf = UVFlag(uv, label="test")

    # create the file so the clobber gets tested
    with h5py.File(test_outfile, "w") as h5file:
        h5file.create_dataset("Test", list(range(10)))

    uvf.write(test_outfile, clobber=True)
    # manually re-read and tamper with parameters
    with h5py.File(test_outfile, "a") as h5:
        mode = h5["Header/mode"]
        mode[...] = np.string_("test")

    with pytest.raises(ValueError) as cm:
        uvf = UVFlag(test_outfile)
    assert str(cm.value).startswith("File cannot be read. Received mode")


def test_bad_type_savefile(uvdata_obj, test_outfile):
    uv = uvdata_obj
    uvf = UVFlag(uv, label="test")
    uvf.write(test_outfile, clobber=True)
    # manually re-read and tamper with parameters
    with h5py.File(test_outfile, "a") as h5:
        mode = h5["Header/type"]
        mode[...] = np.string_("test")

    with pytest.raises(ValueError) as cm:
        uvf = UVFlag(test_outfile)
    assert str(cm.value).startswith("File cannot be read. Received type")


def test_write_add_version_str(uvdata_obj, test_outfile):
    uv = uvdata_obj
    uvf = UVFlag(uv, label="test")
    uvf.history = uvf.history.replace(pyuvdata_version_str, "")

    assert pyuvdata_version_str not in uvf.history
    uvf.write(test_outfile, clobber=True)

    with h5py.File(test_outfile, "r") as h5:
        assert h5["Header/history"].dtype.type is np.string_
        hist = h5["Header/history"][()].decode("utf8")
    assert pyuvdata_version_str in hist


def test_read_add_version_str(uvdata_obj, test_outfile):
    uv = uvdata_obj
    uvf = UVFlag(uv, label="test")

    assert pyuvdata_version_str in uvf.history
    uvf.write(test_outfile, clobber=True)

    with h5py.File(test_outfile, "r") as h5:
        hist = h5["Header/history"]
        del hist

    uvf2 = UVFlag(test_outfile)
    assert pyuvdata_version_str in uvf2.history
    assert uvf == uvf2


def test_read_write_ant(test_outfile):
    uv = UVCal()
    uv.read_calfits(test_c_file)
    uvf = UVFlag(uv, mode="flag", label="test")
    uvf.write(test_outfile, clobber=True)
    uvf2 = UVFlag(test_outfile)
    assert uvf.__eq__(uvf2, check_history=True)


def test_read_missing_nants_data(test_outfile):
    uv = UVCal()
    uv.read_calfits(test_c_file)
    uvf = UVFlag(uv, mode="flag", label="test")
    uvf.write(test_outfile, clobber=True)

    with h5py.File(test_outfile, "a") as h5:
        del h5["Header/Nants_data"]

    with uvtest.check_warnings(UserWarning, "Nants_data not available in file,"):
        uvf2 = UVFlag(test_outfile)

    # make sure this was set to None
    assert uvf2.Nants_data == len(uvf2.ant_array)

    uvf2.Nants_data = uvf.Nants_data
    # verify no other elements were changed
    assert uvf.__eq__(uvf2, check_history=True)


def test_read_missing_nspws(test_outfile):
    uv = UVCal()
    uv.read_calfits(test_c_file)
    uvf = UVFlag(uv, mode="flag", label="test")
    uvf.write(test_outfile, clobber=True)

    with h5py.File(test_outfile, "a") as h5:
        del h5["Header/Nspws"]

    uvf2 = UVFlag(test_outfile)
    # make sure Nspws was calculated
    assert uvf2.Nspws == 1

    # verify no other elements were changed
    assert uvf.__eq__(uvf2, check_history=True)


def test_read_write_nocompress(uvdata_obj, test_outfile):
    uv = uvdata_obj
    uvf = UVFlag(uv, label="test")
    uvf.write(test_outfile, clobber=True, data_compression=None)
    uvf2 = UVFlag(test_outfile)
    assert uvf.__eq__(uvf2, check_history=True)


def test_read_write_nocompress_flag(uvdata_obj, test_outfile):
    uv = uvdata_obj
    uvf = UVFlag(uv, mode="flag", label="test")
    uvf.write(test_outfile, clobber=True, data_compression=None)
    uvf2 = UVFlag(test_outfile)
    assert uvf.__eq__(uvf2, check_history=True)


def test_read_write_extra_keywords(uvdata_obj, test_outfile):
    uv = uvdata_obj
    uvf = UVFlag(uv, label="test")
    uvf.extra_keywords = {"keyword1": 1, "keyword2": "string"}
    uvf.write(test_outfile, clobber=True, data_compression=None)
    uvf2 = UVFlag(test_outfile)
    assert uvf2.extra_keywords["keyword1"] == 1
    assert uvf2.extra_keywords["keyword2"] == "string"


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_init_list(uvdata_obj):
    uv = uvdata_obj
    uv.time_array -= 1
    uvf = UVFlag([uv, test_f_file])
    uvf1 = UVFlag(uv)
    uvf2 = UVFlag(test_f_file)
    assert np.array_equal(
        np.concatenate((uvf1.metric_array, uvf2.metric_array), axis=0), uvf.metric_array
    )
    assert np.array_equal(
        np.concatenate((uvf1.weights_array, uvf2.weights_array), axis=0),
        uvf.weights_array,
    )
    assert np.array_equal(
        np.concatenate((uvf1.time_array, uvf2.time_array)), uvf.time_array
    )
    assert np.array_equal(
        np.concatenate((uvf1.baseline_array, uvf2.baseline_array)), uvf.baseline_array
    )
    assert np.array_equal(
        np.concatenate((uvf1.ant_1_array, uvf2.ant_1_array)), uvf.ant_1_array
    )
    assert np.array_equal(
        np.concatenate((uvf1.ant_2_array, uvf2.ant_2_array)), uvf.ant_2_array
    )
    assert uvf.mode == "metric"
    assert np.all(uvf.freq_array == uv.freq_array[0])
    assert np.all(uvf.polarization_array == uv.polarization_array)


def test_read_list(uvdata_obj, test_outfile):
    uv = uvdata_obj
    uv.time_array -= 1
    uvf = UVFlag(uv)
    uvf.write(test_outfile, clobber=True)
    uvf.read([test_outfile, test_f_file])
    uvf1 = UVFlag(uv)
    uvf2 = UVFlag(test_f_file)
    assert np.array_equal(
        np.concatenate((uvf1.metric_array, uvf2.metric_array), axis=0), uvf.metric_array
    )
    assert np.array_equal(
        np.concatenate((uvf1.weights_array, uvf2.weights_array), axis=0),
        uvf.weights_array,
    )
    assert np.array_equal(
        np.concatenate((uvf1.time_array, uvf2.time_array)), uvf.time_array
    )
    assert np.array_equal(
        np.concatenate((uvf1.baseline_array, uvf2.baseline_array)), uvf.baseline_array
    )
    assert np.array_equal(
        np.concatenate((uvf1.ant_1_array, uvf2.ant_1_array)), uvf.ant_1_array
    )
    assert np.array_equal(
        np.concatenate((uvf1.ant_2_array, uvf2.ant_2_array)), uvf.ant_2_array
    )
    assert uvf.mode == "metric"
    assert np.all(uvf.freq_array == uv.freq_array[0])
    assert np.all(uvf.polarization_array == uv.polarization_array)


def test_read_error():
    with pytest.raises(IOError) as cm:
        UVFlag("foo")
    assert str(cm.value).startswith("foo not found")


def test_read_change_type(test_outfile):
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = UVFlag(uvc)
    uvf.write(test_outfile, clobber=True)
    assert hasattr(uvf, "ant_array")
    uvf.read(test_f_file)

    # clear sets these to None now
    assert hasattr(uvf, "ant_array")
    assert uvf.ant_array is None
    assert hasattr(uvf, "baseline_array")
    assert hasattr(uvf, "ant_1_array")
    assert hasattr(uvf, "ant_2_array")
    uvf.read(test_outfile)
    assert hasattr(uvf, "ant_array")
    assert hasattr(uvf, "baseline_array")
    assert uvf.baseline_array is None
    assert hasattr(uvf, "ant_1_array")
    assert uvf.ant_1_array is None
    assert hasattr(uvf, "ant_2_array")
    assert uvf.ant_2_array is None


def test_read_change_mode(uvdata_obj, test_outfile):
    uv = uvdata_obj
    uvf = UVFlag(uv, mode="flag")
    assert hasattr(uvf, "flag_array")
    assert hasattr(uvf, "metric_array")
    assert uvf.metric_array is None
    uvf.write(test_outfile, clobber=True)
    uvf.read(test_f_file)
    assert hasattr(uvf, "metric_array")
    assert hasattr(uvf, "flag_array")
    assert uvf.flag_array is None
    uvf.read(test_outfile)
    assert hasattr(uvf, "flag_array")
    assert hasattr(uvf, "metric_array")
    assert uvf.metric_array is None


def test_write_no_clobber():
    uvf = UVFlag(test_f_file)
    with pytest.raises(ValueError) as cm:
        uvf.write(test_f_file)
    assert str(cm.value).startswith("File " + test_f_file + " exists;")


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_lst_from_uv(uvdata_obj):
    uv = uvdata_obj
    lst_array = lst_from_uv(uv)
    assert np.allclose(uv.lst_array, lst_array)


def test_lst_from_uv_error():
    with pytest.raises(ValueError) as cm:
        lst_from_uv(4)
    assert str(cm.value).startswith("Function lst_from_uv can only operate on")


def test_add():
    uv1 = UVFlag(test_f_file)
    uv2 = copy.deepcopy(uv1)
    uv2.time_array += 1  # Add a day
    uv3 = uv1 + uv2
    assert np.array_equal(
        np.concatenate((uv1.time_array, uv2.time_array)), uv3.time_array
    )
    assert np.array_equal(
        np.concatenate((uv1.baseline_array, uv2.baseline_array)), uv3.baseline_array
    )
    assert np.array_equal(
        np.concatenate((uv1.ant_1_array, uv2.ant_1_array)), uv3.ant_1_array
    )
    assert np.array_equal(
        np.concatenate((uv1.ant_2_array, uv2.ant_2_array)), uv3.ant_2_array
    )
    assert np.array_equal(np.concatenate((uv1.lst_array, uv2.lst_array)), uv3.lst_array)
    assert np.array_equal(
        np.concatenate((uv1.metric_array, uv2.metric_array), axis=0), uv3.metric_array
    )
    assert np.array_equal(
        np.concatenate((uv1.weights_array, uv2.weights_array), axis=0),
        uv3.weights_array,
    )
    assert np.array_equal(uv1.freq_array, uv3.freq_array)
    assert uv3.type == "baseline"
    assert uv3.mode == "metric"
    assert np.array_equal(uv1.polarization_array, uv3.polarization_array)
    assert "Data combined along time axis. " in uv3.history


def test_add_collapsed_pols():
    uvf = UVFlag(test_f_file)
    uvf.weights_array = np.ones_like(uvf.weights_array)
    uvf2 = uvf.copy()
    uvf2.polarization_array[0] = -4
    uvf.__add__(uvf2, inplace=True, axis="pol")  # Concatenate to form multi-pol object
    uvf.collapse_pol()
    uvf3 = uvf.copy()
    uvf3.time_array += 1  # increment the time array
    uvf4 = uvf + uvf3
    assert uvf4.Ntimes == 2 * uvf.Ntimes
    assert uvf4.check()


def test_add_add_version_str():
    uv1 = UVFlag(test_f_file)
    uv1.history = uv1.history.replace(pyuvdata_version_str, "")

    assert pyuvdata_version_str not in uv1.history

    uv2 = copy.deepcopy(uv1)
    uv2.time_array += 1  # Add a day
    uv3 = uv1 + uv2
    assert pyuvdata_version_str in uv3.history


def test_add_baseline():
    uv1 = UVFlag(test_f_file)
    uv2 = copy.deepcopy(uv1)
    uv2.baseline_array += 100  # Arbitrary
    uv3 = uv1.__add__(uv2, axis="baseline")
    assert np.array_equal(
        np.concatenate((uv1.time_array, uv2.time_array)), uv3.time_array
    )
    assert np.array_equal(
        np.concatenate((uv1.baseline_array, uv2.baseline_array)), uv3.baseline_array
    )
    assert np.array_equal(
        np.concatenate((uv1.ant_1_array, uv2.ant_1_array)), uv3.ant_1_array
    )
    assert np.array_equal(
        np.concatenate((uv1.ant_2_array, uv2.ant_2_array)), uv3.ant_2_array
    )
    assert np.array_equal(np.concatenate((uv1.lst_array, uv2.lst_array)), uv3.lst_array)
    assert np.array_equal(
        np.concatenate((uv1.metric_array, uv2.metric_array), axis=0), uv3.metric_array
    )
    assert np.array_equal(
        np.concatenate((uv1.weights_array, uv2.weights_array), axis=0),
        uv3.weights_array,
    )
    assert np.array_equal(uv1.freq_array, uv3.freq_array)
    assert uv3.type == "baseline"
    assert uv3.mode == "metric"
    assert np.array_equal(uv1.polarization_array, uv3.polarization_array)
    assert "Data combined along baseline axis. " in uv3.history


def test_add_antenna():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uv1 = UVFlag(uvc)
    uv2 = copy.deepcopy(uv1)
    uv2.ant_array += 100  # Arbitrary
    uv3 = uv1.__add__(uv2, axis="antenna")
    assert np.array_equal(np.concatenate((uv1.ant_array, uv2.ant_array)), uv3.ant_array)
    assert np.array_equal(
        np.concatenate((uv1.metric_array, uv2.metric_array), axis=0), uv3.metric_array
    )
    assert np.array_equal(
        np.concatenate((uv1.weights_array, uv2.weights_array), axis=0),
        uv3.weights_array,
    )
    assert np.array_equal(uv1.freq_array, uv3.freq_array)
    assert np.array_equal(uv1.time_array, uv3.time_array)
    assert np.array_equal(uv1.lst_array, uv3.lst_array)
    assert uv3.type == "antenna"
    assert uv3.mode == "metric"
    assert np.array_equal(uv1.polarization_array, uv3.polarization_array)
    assert "Data combined along antenna axis. " in uv3.history


def test_add_frequency():
    uv1 = UVFlag(test_f_file)
    uv2 = copy.deepcopy(uv1)
    uv2.freq_array += 1e4  # Arbitrary
    uv3 = uv1.__add__(uv2, axis="frequency")
    assert np.array_equal(
        np.concatenate((uv1.freq_array, uv2.freq_array), axis=-1), uv3.freq_array
    )
    assert np.array_equal(uv1.time_array, uv3.time_array)
    assert np.array_equal(uv1.baseline_array, uv3.baseline_array)
    assert np.array_equal(uv1.ant_1_array, uv3.ant_1_array)
    assert np.array_equal(uv1.ant_2_array, uv3.ant_2_array)
    assert np.array_equal(uv1.lst_array, uv3.lst_array)
    assert np.array_equal(
        np.concatenate((uv1.metric_array, uv2.metric_array), axis=2), uv3.metric_array
    )
    assert np.array_equal(
        np.concatenate((uv1.weights_array, uv2.weights_array), axis=2),
        uv3.weights_array,
    )
    assert uv3.type == "baseline"
    assert uv3.mode == "metric"
    assert np.array_equal(uv1.polarization_array, uv3.polarization_array)
    assert "Data combined along frequency axis. " in uv3.history


def test_add_frequency_with_weights_square():
    # Same test as above, just checking an optional parameter (also in waterfall mode)
    uvf1 = UVFlag(test_f_file)
    uvf1.weights_array = 2 * np.ones_like(uvf1.weights_array)
    uvf1.to_waterfall(return_weights_square=True)
    uvf2 = copy.deepcopy(uvf1)
    uvf2.freq_array += 1e4
    uvf3 = uvf1.__add__(uvf2, axis="frequency")
    assert np.array_equal(
        np.concatenate((uvf1.weights_square_array, uvf2.weights_square_array), axis=1),
        uvf3.weights_square_array,
    )


def test_add_frequency_mix_weights_square():
    # Same test as above, checking some error handling
    uvf1 = UVFlag(test_f_file)
    uvf1.weights_array = 2 * np.ones_like(uvf1.weights_array)
    uvf2 = copy.deepcopy(uvf1)
    uvf1.to_waterfall(return_weights_square=True)
    uvf2.to_waterfall(return_weights_square=False)
    uvf2.freq_array += 1e4
    with pytest.raises(
        ValueError,
        match="weights_square_array optional parameter is missing from second UVFlag",
    ):
        uvf1.__add__(uvf2, axis="frequency", inplace=True)


def test_add_pol():
    uv1 = UVFlag(test_f_file)
    uv2 = copy.deepcopy(uv1)
    uv2.polarization_array += 1  # Arbitrary
    uv3 = uv1.__add__(uv2, axis="polarization")
    assert np.array_equal(uv1.freq_array, uv3.freq_array)
    assert np.array_equal(uv1.time_array, uv3.time_array)
    assert np.array_equal(uv1.baseline_array, uv3.baseline_array)
    assert np.array_equal(uv1.ant_1_array, uv3.ant_1_array)
    assert np.array_equal(uv1.ant_2_array, uv3.ant_2_array)
    assert np.array_equal(uv1.lst_array, uv3.lst_array)
    assert np.array_equal(
        np.concatenate((uv1.metric_array, uv2.metric_array), axis=3), uv3.metric_array
    )
    assert np.array_equal(
        np.concatenate((uv1.weights_array, uv2.weights_array), axis=3),
        uv3.weights_array,
    )
    assert uv3.type == "baseline"
    assert uv3.mode == "metric"
    assert np.array_equal(
        np.concatenate((uv1.polarization_array, uv2.polarization_array)),
        uv3.polarization_array,
    )
    assert "Data combined along polarization axis. " in uv3.history


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_add_flag(uvdata_obj):
    uv = uvdata_obj
    uv1 = UVFlag(uv, mode="flag")
    uv2 = copy.deepcopy(uv1)
    uv2.time_array += 1  # Add a day
    uv3 = uv1 + uv2
    assert np.array_equal(
        np.concatenate((uv1.time_array, uv2.time_array)), uv3.time_array
    )
    assert np.array_equal(
        np.concatenate((uv1.baseline_array, uv2.baseline_array)), uv3.baseline_array
    )
    assert np.array_equal(
        np.concatenate((uv1.ant_1_array, uv2.ant_1_array)), uv3.ant_1_array
    )
    assert np.array_equal(
        np.concatenate((uv1.ant_2_array, uv2.ant_2_array)), uv3.ant_2_array
    )
    assert np.array_equal(np.concatenate((uv1.lst_array, uv2.lst_array)), uv3.lst_array)
    assert np.array_equal(
        np.concatenate((uv1.flag_array, uv2.flag_array), axis=0), uv3.flag_array
    )
    assert np.array_equal(uv1.freq_array, uv3.freq_array)
    assert uv3.type == "baseline"
    assert uv3.mode == "flag"
    assert np.array_equal(uv1.polarization_array, uv3.polarization_array)
    assert "Data combined along time axis. " in uv3.history


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_add_errors(uvdata_obj):
    uv = uvdata_obj
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uv1 = UVFlag(uv)
    # Mismatched classes
    with pytest.raises(ValueError) as cm:
        uv1.__add__(3)
    assert str(cm.value).startswith(
        "Only UVFlag objects can be added to a UVFlag object"
    )

    # Mismatched types
    uv2 = UVFlag(uvc)
    with pytest.raises(ValueError) as cm:
        uv1.__add__(uv2)
    assert str(cm.value).startswith("UVFlag object of type ")

    # Mismatched modes
    uv3 = UVFlag(uv, mode="flag")
    with pytest.raises(ValueError) as cm:
        uv1.__add__(uv3)
    assert str(cm.value).startswith("UVFlag object of mode ")

    # Invalid axes
    with pytest.raises(ValueError) as cm:
        uv1.__add__(uv1, axis="antenna")
    assert str(cm.value).endswith("concatenated along antenna axis.")

    with pytest.raises(ValueError) as cm:
        uv2.__add__(uv2, axis="baseline")
    assert str(cm.value).endswith("concatenated along baseline axis.")


def test_inplace_add():
    uv1a = UVFlag(test_f_file)
    uv1b = copy.deepcopy(uv1a)
    uv2 = copy.deepcopy(uv1a)
    uv2.time_array += 1
    uv1a += uv2
    assert uv1a.__eq__(uv1b + uv2)


def test_clear_unused_attributes():
    uv = UVFlag(test_f_file)
    assert hasattr(uv, "baseline_array")
    assert hasattr(uv, "ant_1_array")
    assert hasattr(uv, "ant_2_array")
    assert hasattr(uv, "Nants_telescope")
    uv._set_type_antenna()
    uv.clear_unused_attributes()
    # clear_unused_attributes now sets these to None
    print(uv._baseline_array.required)
    assert hasattr(uv, "baseline_array")
    assert uv.baseline_array is None
    assert hasattr(uv, "ant_1_array")
    assert uv.ant_1_array is None
    assert hasattr(uv, "ant_2_array")
    assert uv.ant_2_array is None
    assert hasattr(uv, "Nants_telescope")
    assert uv.Nants_telescope is None

    uv._set_mode_flag()
    assert hasattr(uv, "metric_array")
    uv.clear_unused_attributes()
    assert hasattr(uv, "metric_array")
    assert uv.metric_array is None

    # Start over
    uv = UVFlag(test_f_file)
    uv.ant_array = np.array([4])
    uv.flag_array = np.array([5])
    uv.clear_unused_attributes()
    assert hasattr(uv, "ant_array")
    assert uv.ant_array is None
    assert hasattr(uv, "flag_array")
    assert uv.flag_array is None


def test_not_equal():
    uvf1 = UVFlag(test_f_file)
    # different class
    assert not uvf1.__eq__(5)
    # different mode
    uvf2 = uvf1.copy()
    uvf2.mode = "flag"
    assert not uvf1.__eq__(uvf2)
    # different type
    uvf2 = uvf1.copy()
    uvf2.type = "antenna"
    assert not uvf1.__eq__(uvf2)
    # array different
    uvf2 = uvf1.copy()
    uvf2.freq_array += 1
    assert not uvf1.__eq__(uvf2)
    # history different
    uvf2 = uvf1.copy()
    uvf2.history += "hello"
    assert not uvf1.__eq__(uvf2, check_history=True)


def test_to_waterfall_bl():
    uvf = UVFlag(test_f_file)
    uvf.weights_array = np.ones_like(uvf.weights_array)
    uvf.to_waterfall()
    assert uvf.type == "waterfall"
    assert uvf.metric_array.shape == (
        len(uvf.time_array),
        len(uvf.freq_array),
        len(uvf.polarization_array),
    )
    assert uvf.weights_array.shape == uvf.metric_array.shape


def test_to_waterfall_add_version_str():
    uvf = UVFlag(test_f_file)
    uvf.weights_array = np.ones_like(uvf.weights_array)

    uvf.history = uvf.history.replace(pyuvdata_version_str, "")
    assert pyuvdata_version_str not in uvf.history
    uvf.to_waterfall()
    assert pyuvdata_version_str in uvf.history


def test_to_waterfall_bl_multi_pol():
    uvf = UVFlag(test_f_file)
    uvf.weights_array = np.ones_like(uvf.weights_array)
    uvf2 = uvf.copy()
    uvf2.polarization_array[0] = -4
    uvf.__add__(uvf2, inplace=True, axis="pol")  # Concatenate to form multi-pol object
    uvf2 = uvf.copy()  # Keep a copy to run with keep_pol=False
    uvf.to_waterfall()
    assert uvf.type == "waterfall"
    assert uvf.metric_array.shape == (
        len(uvf.time_array),
        len(uvf.freq_array),
        len(uvf.polarization_array),
    )
    assert uvf.weights_array.shape == uvf.metric_array.shape
    assert len(uvf.polarization_array) == 2
    # Repeat with keep_pol=False
    uvf2.to_waterfall(keep_pol=False)
    assert uvf2.type == "waterfall"
    assert uvf2.metric_array.shape == (len(uvf2.time_array), len(uvf.freq_array), 1)
    assert uvf2.weights_array.shape == uvf2.metric_array.shape
    assert len(uvf2.polarization_array) == 1
    assert uvf2.polarization_array[0] == np.str_(
        ",".join(map(str, uvf.polarization_array))
    )


def test_to_waterfall_bl_ret_wt_sq():
    uvf = UVFlag(test_f_file)
    Nbls = uvf.Nbls
    uvf.weights_array = 2 * np.ones_like(uvf.weights_array)
    uvf.to_waterfall(return_weights_square=True)
    assert np.all(uvf.weights_square_array == 4 * Nbls)

    # Switch to flag and check that it is now set to None
    uvf.to_flag()
    assert uvf.weights_square_array is None


def test_collapse_pol(test_outfile):
    uvf = UVFlag(test_f_file)
    uvf.weights_array = np.ones_like(uvf.weights_array)
    uvf2 = uvf.copy()
    uvf2.polarization_array[0] = -4
    uvf.__add__(uvf2, inplace=True, axis="pol")  # Concatenate to form multi-pol object
    uvf2 = uvf.copy()
    uvf2.collapse_pol()
    assert len(uvf2.polarization_array) == 1
    assert uvf2.polarization_array[0] == np.str_(
        ",".join(map(str, uvf.polarization_array))
    )
    assert uvf2.mode == "metric"
    assert hasattr(uvf2, "metric_array")
    assert hasattr(uvf2, "flag_array")
    assert uvf2.flag_array is None

    # test check passes just to be sure
    assert uvf2.check()

    # test writing it out and reading in to make sure polarization_array has
    # correct type
    uvf2.write(test_outfile, clobber=True)
    with h5py.File(test_outfile, "r") as h5:
        assert h5["Header/polarization_array"].dtype.type is np.string_
    uvf = UVFlag(test_outfile)
    assert uvf._polarization_array.expected_type == str
    assert uvf._polarization_array.acceptable_vals is None
    assert uvf == uvf2
    os.remove(test_outfile)


def test_collapse_pol_add_pol_axis():
    uvf = UVFlag(test_f_file)
    uvf.weights_array = np.ones_like(uvf.weights_array)
    uvf2 = uvf.copy()
    uvf2.polarization_array[0] = -4
    uvf.__add__(uvf2, inplace=True, axis="pol")  # Concatenate to form multi-pol object
    uvf2 = uvf.copy()
    uvf2.collapse_pol()
    with pytest.raises(NotImplementedError) as cm:
        uvf2.__add__(uvf2, axis="pol")
    assert str(cm.value).startswith("Two UVFlag objects with their")


def test_collapse_pol_or():
    uvf = UVFlag(test_f_file)
    uvf.to_flag()
    assert uvf.weights_array is None
    uvf2 = uvf.copy()
    uvf2.polarization_array[0] = -4
    uvf.__add__(uvf2, inplace=True, axis="pol")  # Concatenate to form multi-pol object
    uvf2 = uvf.copy()
    uvf2.collapse_pol(method="or")
    assert len(uvf2.polarization_array) == 1
    assert uvf2.polarization_array[0] == np.str_(
        ",".join(map(str, uvf.polarization_array))
    )
    assert uvf2.mode == "flag"
    assert hasattr(uvf2, "flag_array")
    assert hasattr(uvf2, "metric_array")
    assert uvf2.metric_array is None


def test_collapse_pol_add_version_str():
    uvf = UVFlag(test_f_file)
    uvf.to_flag()

    uvf2 = uvf.copy()
    uvf2.polarization_array[0] = -4
    uvf.__add__(uvf2, inplace=True, axis="pol")  # Concatenate to form multi-pol object

    uvf.history = uvf.history.replace(pyuvdata_version_str, "")
    assert pyuvdata_version_str not in uvf.history

    uvf2 = uvf.copy()
    uvf2.collapse_pol(method="or")

    assert pyuvdata_version_str in uvf2.history


def test_collapse_single_pol():
    uvf = UVFlag(test_f_file)
    uvf.weights_array = np.ones_like(uvf.weights_array)
    uvf2 = uvf.copy()
    with uvtest.check_warnings(UserWarning, "Cannot collapse polarization"):
        uvf.collapse_pol()
    assert uvf == uvf2


def test_collapse_pol_flag():
    uvf = UVFlag(test_f_file)
    uvf.to_flag()
    assert uvf.weights_array is None
    uvf2 = uvf.copy()
    uvf2.polarization_array[0] = -4
    uvf.__add__(uvf2, inplace=True, axis="pol")  # Concatenate to form multi-pol object
    uvf2 = uvf.copy()
    uvf2.collapse_pol()
    assert len(uvf2.polarization_array) == 1
    assert uvf2.polarization_array[0] == np.str_(
        ",".join(map(str, uvf.polarization_array))
    )
    assert uvf2.mode == "metric"
    assert hasattr(uvf2, "metric_array")
    assert hasattr(uvf2, "flag_array")
    assert uvf2.flag_array is None


def test_to_waterfall_bl_flags():
    uvf = UVFlag(test_f_file)
    uvf.to_flag()
    uvf.to_waterfall()
    assert uvf.type == "waterfall"
    assert uvf.mode == "metric"
    assert uvf.metric_array.shape == (
        len(uvf.time_array),
        len(uvf.freq_array),
        len(uvf.polarization_array),
    )
    assert uvf.weights_array.shape == uvf.metric_array.shape
    assert len(uvf.lst_array) == len(uvf.time_array)


def test_to_waterfall_bl_flags_or():
    uvf = UVFlag(test_f_file)
    uvf.to_flag()
    assert uvf.weights_array is None
    uvf.to_waterfall(method="or")
    assert uvf.type == "waterfall"
    assert uvf.mode == "flag"
    assert uvf.flag_array.shape == (
        len(uvf.time_array),
        len(uvf.freq_array),
        len(uvf.polarization_array),
    )
    assert len(uvf.lst_array) == len(uvf.time_array)
    uvf = UVFlag(test_f_file)
    uvf.to_flag()
    uvf.to_waterfall(method="or")
    assert uvf.type == "waterfall"
    assert uvf.mode == "flag"
    assert uvf.flag_array.shape == (
        len(uvf.time_array),
        len(uvf.freq_array),
        len(uvf.polarization_array),
    )
    assert len(uvf.lst_array) == len(uvf.time_array)


def test_to_waterfall_ant():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = UVFlag(uvc)
    uvf.weights_array = np.ones_like(uvf.weights_array)
    uvf.to_waterfall()
    assert uvf.type == "waterfall"
    assert uvf.metric_array.shape == (
        len(uvf.time_array),
        len(uvf.freq_array),
        len(uvf.polarization_array),
    )
    assert uvf.weights_array.shape == uvf.metric_array.shape
    assert len(uvf.lst_array) == len(uvf.time_array)


def test_to_waterfall_waterfall():
    uvf = UVFlag(test_f_file)
    uvf.weights_array = np.ones_like(uvf.weights_array)
    uvf.to_waterfall()
    with uvtest.check_warnings(UserWarning, "This object is already a waterfall"):
        uvf.to_waterfall()


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_to_baseline_flags(uvdata_obj):
    uv = uvdata_obj
    uvf = UVFlag(uv)
    uvf.to_waterfall()
    uvf.to_flag()
    uvf.flag_array[0, 10, 0] = True  # Flag time0, chan10
    uvf.flag_array[1, 15, 0] = True  # Flag time1, chan15
    uvf.to_baseline(uv)
    assert uvf.type == "baseline"
    assert np.all(uvf.baseline_array == uv.baseline_array)
    assert np.all(uvf.time_array == uv.time_array)
    times = np.unique(uvf.time_array)
    ntrue = 0.0
    ind = np.where(uvf.time_array == times[0])[0]
    ntrue += len(ind)
    assert np.all(uvf.flag_array[ind, 0, 10, 0])
    ind = np.where(uvf.time_array == times[1])[0]
    ntrue += len(ind)
    assert np.all(uvf.flag_array[ind, 0, 15, 0])
    assert uvf.flag_array.mean() == ntrue / uvf.flag_array.size


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_to_baseline_metric(uvdata_obj, future_shapes):
    uv = uvdata_obj

    if future_shapes:
        uv.use_future_array_shapes()

    uvf = UVFlag(uv)
    uvf.to_waterfall()
    uvf.metric_array[0, 10, 0] = 3.2  # Fill in time0, chan10
    uvf.metric_array[1, 15, 0] = 2.1  # Fill in time1, chan15
    uvf.to_baseline(uv)
    assert np.all(uvf.baseline_array == uv.baseline_array)
    assert np.all(uvf.time_array == uv.time_array)
    times = np.unique(uvf.time_array)
    ind = np.where(uvf.time_array == times[0])[0]
    nt0 = len(ind)
    assert np.all(uvf.metric_array[ind, 0, 10, 0] == 3.2)
    ind = np.where(uvf.time_array == times[1])[0]
    nt1 = len(ind)
    assert np.all(uvf.metric_array[ind, 0, 15, 0] == 2.1)
    assert np.isclose(
        uvf.metric_array.mean(), (3.2 * nt0 + 2.1 * nt1) / uvf.metric_array.size
    )


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_to_baseline_add_version_str(uvdata_obj):
    uv = uvdata_obj
    uvf = UVFlag(uv)
    uvf.to_waterfall()
    uvf.metric_array[0, 10, 0] = 3.2  # Fill in time0, chan10
    uvf.metric_array[1, 15, 0] = 2.1  # Fill in time1, chan15

    uvf.history = uvf.history.replace(pyuvdata_version_str, "")
    assert pyuvdata_version_str not in uvf.history

    uvf.to_baseline(uv)
    assert pyuvdata_version_str in uvf.history


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_baseline_to_baseline(uvdata_obj):
    uv = uvdata_obj
    uvf = UVFlag(uv)
    uvf2 = uvf.copy()
    uvf.to_baseline(uv)
    assert uvf == uvf2


def test_to_baseline_metric_error(uvdata_obj, uvf_from_uvcal):
    uvf = uvf_from_uvcal
    uvf.select(polarizations=uvf.polarization_array[0])
    uv = uvdata_obj
    with pytest.raises(NotImplementedError) as cm:
        uvf.to_baseline(uv, force_pol=True)
    assert str(cm.value).startswith(
        "Cannot currently convert from " "antenna type, metric mode"
    )


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_to_baseline_from_antenna(uvdata_obj, uvf_from_uvcal):
    uvf = uvf_from_uvcal
    uvf.select(polarizations=uvf.polarization_array[0])
    uvf.to_flag()
    uv = uvdata_obj

    ants_data = np.unique(uv.ant_1_array.tolist() + uv.ant_2_array.tolist())
    new_ants = np.setdiff1d(ants_data, uvf.ant_array)

    old_baseline = (uvf.ant_array[0], uvf.ant_array[1])
    old_times = np.unique(uvf.time_array)
    or_flags = np.logical_or(uvf.flag_array[0], uvf.flag_array[1])
    or_flags = np.transpose(or_flags, [2, 0, 1, 3])

    uv2 = copy.deepcopy(uv)
    uvf2 = uvf.copy()

    # hack in the exact times so we can compare some values later
    uv2.select(bls=old_baseline)
    uv2.time_array[: uvf2.time_array.size] = uvf.time_array

    uvf.to_baseline(uv, force_pol=True)
    uvf2.to_baseline(uv2, force_pol=True)
    assert uvf.check()

    uvf2.select(bls=old_baseline, times=old_times)
    assert np.allclose(or_flags, uvf2.flag_array)

    # all new antenna should be completely flagged
    # checks auto correlations
    uvf_new = uvf.select(antenna_nums=new_ants, inplace=False)
    for bl in np.unique(uvf_new.baseline_array):
        uvf2 = uvf_new.select(bls=uv.baseline_to_antnums(bl), inplace=False)
        assert np.all(uvf2.flag_array)

    # check for baselines with one new antenna
    bls = [
        uvf.baseline_to_antnums(bl)
        for bl in uvf.baseline_array
        if np.intersect1d(new_ants, uvf.baseline_to_antnums(bl)).size > 0
    ]
    uvf_new = uvf.select(bls=bls, inplace=False)
    for bl in np.unique(uvf_new.baseline_array):
        uvf2 = uvf_new.select(bls=uv.baseline_to_antnums(bl), inplace=False)
        assert np.all(uvf2.flag_array)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_to_baseline_errors(uvdata_obj):
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uv = uvdata_obj
    uvf = UVFlag(test_f_file)
    uvf.to_waterfall()
    with pytest.raises(ValueError) as cm:
        uvf.to_baseline(7.3)  # invalid matching object
    assert str(cm.value).startswith("Must pass in UVData object or UVFlag object")

    uvf = UVFlag(test_f_file)
    uvf.to_waterfall()
    uvf2 = uvf.copy()
    uvf.polarization_array[0] = -4
    with pytest.raises(ValueError) as cm:
        uvf.to_baseline(uv)  # Mismatched pols
    assert str(cm.value).startswith("Polarizations do not match.")
    uvf.__iadd__(uvf2, axis="polarization")

    with pytest.raises(ValueError) as cm:
        uvf.to_baseline(uv)  # Mismatched pols, can't be forced
    assert str(cm.value).startswith("Polarizations could not be made to match.")


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_to_baseline_force_pol(uvdata_obj):
    uv = uvdata_obj
    uvf = UVFlag(uv)
    uvf.to_waterfall()
    uvf.to_flag()
    uvf.flag_array[0, 10, 0] = True  # Flag time0, chan10
    uvf.flag_array[1, 15, 0] = True  # Flag time1, chan15
    uvf.polarization_array[0] = -4  # Change pol, but force pol anyway
    uvf.to_baseline(uv, force_pol=True)
    assert np.all(uvf.baseline_array == uv.baseline_array)
    assert np.all(uvf.time_array == uv.time_array)
    assert np.array_equal(uvf.polarization_array, uv.polarization_array)
    times = np.unique(uvf.time_array)
    ntrue = 0.0
    ind = np.where(uvf.time_array == times[0])[0]
    ntrue += len(ind)
    assert np.all(uvf.flag_array[ind, 0, 10, 0])
    ind = np.where(uvf.time_array == times[1])[0]
    ntrue += len(ind)
    assert np.all(uvf.flag_array[ind, 0, 15, 0])
    assert uvf.flag_array.mean() == ntrue / uvf.flag_array.size


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_to_baseline_force_pol_npol_gt_1(uvdata_obj):
    uv = uvdata_obj
    uvf = UVFlag(uv)
    uvf.to_waterfall()
    uvf.to_flag()
    uvf.flag_array[0, 10, 0] = True  # Flag time0, chan10
    uvf.flag_array[1, 15, 0] = True  # Flag time1, chan15

    uv2 = copy.deepcopy(uv)
    uv2.polarization_array[0] = -6
    uv += uv2
    uvf.to_baseline(uv, force_pol=True)
    assert np.all(uvf.baseline_array == uv.baseline_array)
    assert np.all(uvf.time_array == uv.time_array)
    assert np.array_equal(uvf.polarization_array, uv.polarization_array)
    assert uvf.Npols == len(uvf.polarization_array)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_to_baseline_metric_force_pol(uvdata_obj):
    uv = uvdata_obj
    uvf = UVFlag(uv)
    uvf.to_waterfall()
    uvf.metric_array[0, 10, 0] = 3.2  # Fill in time0, chan10
    uvf.metric_array[1, 15, 0] = 2.1  # Fill in time1, chan15
    uvf.polarization_array[0] = -4
    uvf.to_baseline(uv, force_pol=True)
    assert np.all(uvf.baseline_array == uv.baseline_array)
    assert np.all(uvf.time_array == uv.time_array)
    assert np.array_equal(uvf.polarization_array, uv.polarization_array)
    times = np.unique(uvf.time_array)
    ind = np.where(uvf.time_array == times[0])[0]
    nt0 = len(ind)
    assert np.all(uvf.metric_array[ind, 0, 10, 0] == 3.2)
    ind = np.where(uvf.time_array == times[1])[0]
    nt1 = len(ind)
    assert np.all(uvf.metric_array[ind, 0, 15, 0] == 2.1)
    assert np.isclose(
        uvf.metric_array.mean(), (3.2 * nt0 + 2.1 * nt1) / uvf.metric_array.size
    )


def test_to_antenna_flags():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = UVFlag(uvc)
    uvf.to_waterfall()
    uvf.to_flag()
    uvf.flag_array[0, 10, 0] = True  # Flag time0, chan10
    uvf.flag_array[1, 15, 0] = True  # Flag time1, chan15
    uvf.to_antenna(uvc)
    assert uvf.type == "antenna"
    assert np.all(uvf.ant_array == uvc.ant_array)
    assert np.all(uvf.time_array == uvc.time_array)
    assert np.all(uvf.flag_array[:, 0, 10, 0, 0])
    assert np.all(uvf.flag_array[:, 0, 15, 1, 0])
    assert uvf.flag_array.mean() == 2.0 * uvc.Nants_data / uvf.flag_array.size


def test_to_antenna_add_version_str():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = UVFlag(uvc)
    uvf.to_waterfall()
    uvf.to_flag()
    uvf.flag_array[0, 10, 0] = True  # Flag time0, chan10
    uvf.flag_array[1, 15, 0] = True  # Flag time1, chan15
    uvf.history = uvf.history.replace(pyuvdata_version_str, "")
    assert pyuvdata_version_str not in uvf.history

    uvf.to_antenna(uvc)
    assert pyuvdata_version_str in uvf.history


def test_to_antenna_metric():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = UVFlag(uvc)
    uvf.to_waterfall()
    uvf.metric_array[0, 10, 0] = 3.2  # Fill in time0, chan10
    uvf.metric_array[1, 15, 0] = 2.1  # Fill in time1, chan15
    uvf.to_antenna(uvc)
    assert np.all(uvf.ant_array == uvc.ant_array)
    assert np.all(uvf.time_array == uvc.time_array)
    assert np.all(uvf.metric_array[:, 0, 10, 0, 0] == 3.2)
    assert np.all(uvf.metric_array[:, 0, 15, 1, 0] == 2.1)
    assert np.isclose(
        uvf.metric_array.mean(), (3.2 + 2.1) * uvc.Nants_data / uvf.metric_array.size
    )


def test_to_antenna_flags_match_uvflag():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = UVFlag(uvc)
    uvf2 = uvf.copy()
    uvf.to_waterfall()
    uvf.to_flag()
    uvf.flag_array[0, 10, 0] = True  # Flag time0, chan10
    uvf.flag_array[1, 15, 0] = True  # Flag time1, chan15
    uvf.to_antenna(uvf2)
    assert np.all(uvf.ant_array == uvc.ant_array)
    assert np.all(uvf.time_array == uvc.time_array)
    assert np.all(uvf.flag_array[:, 0, 10, 0, 0])
    assert np.all(uvf.flag_array[:, 0, 15, 1, 0])
    assert uvf.flag_array.mean() == 2.0 * uvc.Nants_data / uvf.flag_array.size


def test_antenna_to_antenna():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = UVFlag(uvc)
    uvf2 = uvf.copy()
    uvf.to_antenna(uvc)
    assert uvf == uvf2


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_to_antenna_errors(uvdata_obj):
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uv = uvdata_obj
    uvf = UVFlag(test_f_file)
    uvf.to_waterfall()
    with pytest.raises(ValueError) as cm:
        uvf.to_antenna(7.3)  # invalid matching object
    assert str(cm.value).startswith("Must pass in UVCal object or UVFlag object ")

    uvf = UVFlag(uv)
    with pytest.raises(ValueError) as cm:
        uvf.to_antenna(uvc)  # Cannot pass in baseline type
    assert str(cm.value).startswith('Cannot convert from type "baseline" to "antenna".')

    uvf = UVFlag(test_f_file)
    uvf.to_waterfall()
    uvf2 = uvf.copy()
    uvf.polarization_array[0] = -4
    with pytest.raises(ValueError) as cm:
        uvf.to_antenna(uvc)  # Mismatched pols
    assert str(cm.value).startswith("Polarizations do not match. ")

    uvf.__iadd__(uvf2, axis="polarization")
    with pytest.raises(ValueError) as cm:
        uvf.to_antenna(uvc)  # Mismatched pols, can't be forced
    assert str(cm.value).startswith("Polarizations could not be made to match.")


def test_to_antenna_force_pol():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvc.select(jones=-5)
    uvf = UVFlag(uvc)
    uvf.to_waterfall()
    uvf.to_flag()
    uvf.flag_array[0, 10, 0] = True  # Flag time0, chan10
    uvf.flag_array[1, 15, 0] = True  # Flag time1, chan15
    uvf.polarization_array[0] = -4  # Change pol, but force pol anyway
    uvf.to_antenna(uvc, force_pol=True)
    assert np.all(uvf.ant_array == uvc.ant_array)
    assert np.all(uvf.time_array == uvc.time_array)
    assert np.array_equal(uvf.polarization_array, uvc.jones_array)
    assert np.all(uvf.flag_array[:, 0, 10, 0, 0])
    assert np.all(uvf.flag_array[:, 0, 15, 1, 0])
    assert uvf.flag_array.mean() == 2 * uvc.Nants_data / uvf.flag_array.size


def test_to_antenna_metric_force_pol():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvc.select(jones=-5)
    uvf = UVFlag(uvc)
    uvf.to_waterfall()
    uvf.metric_array[0, 10, 0] = 3.2  # Fill in time0, chan10
    uvf.metric_array[1, 15, 0] = 2.1  # Fill in time1, chan15
    uvf.polarization_array[0] = -4
    uvf.to_antenna(uvc, force_pol=True)
    assert np.all(uvf.ant_array == uvc.ant_array)
    assert np.all(uvf.time_array == uvc.time_array)
    assert np.array_equal(uvf.polarization_array, uvc.jones_array)
    assert np.all(uvf.metric_array[:, 0, 10, 0, 0] == 3.2)
    assert np.all(uvf.metric_array[:, 0, 15, 1, 0] == 2.1)
    assert np.isclose(
        uvf.metric_array.mean(), (3.2 + 2.1) * uvc.Nants_data / uvf.metric_array.size
    )


def test_copy():
    uvf = UVFlag(test_f_file)
    uvf2 = uvf.copy()
    assert uvf == uvf2
    # Make sure it's a copy and not just pointing to same object
    uvf.to_waterfall()
    assert uvf != uvf2


def test_or():
    uvf = UVFlag(test_f_file)
    uvf.to_flag()
    uvf2 = uvf.copy()
    uvf2.flag_array = np.ones_like(uvf2.flag_array)
    uvf.flag_array[0] = True
    uvf2.flag_array[0] = False
    uvf2.flag_array[1] = False
    uvf3 = uvf | uvf2
    assert np.all(uvf3.flag_array[0])
    assert not np.any(uvf3.flag_array[1])
    assert np.all(uvf3.flag_array[2:])


def test_or_add_version_str():
    uvf = UVFlag(test_f_file)
    uvf.to_flag()
    uvf.history = uvf.history.replace(pyuvdata_version_str, "")

    assert pyuvdata_version_str not in uvf.history
    uvf2 = uvf.copy()
    uvf2.flag_array = np.ones_like(uvf2.flag_array)
    uvf.flag_array[0] = True
    uvf2.flag_array[0] = False
    uvf2.flag_array[1] = False
    uvf3 = uvf | uvf2

    assert pyuvdata_version_str in uvf3.history


def test_or_error():
    uvf = UVFlag(test_f_file)
    uvf2 = uvf.copy()
    uvf.to_flag()
    with pytest.raises(ValueError) as cm:
        uvf.__or__(uvf2)
    assert str(cm.value).startswith('UVFlag object must be in "flag" mode')


def test_or_add_history():
    uvf = UVFlag(test_f_file)
    uvf.to_flag()
    uvf2 = uvf.copy()
    uvf2.history = "Different history"
    uvf3 = uvf | uvf2
    assert uvf.history in uvf3.history
    assert uvf2.history in uvf3.history
    assert "Flags OR'd with:" in uvf3.history


def test_ior():
    uvf = UVFlag(test_f_file)
    uvf.to_flag()
    uvf2 = uvf.copy()
    uvf2.flag_array = np.ones_like(uvf2.flag_array)
    uvf.flag_array[0] = True
    uvf2.flag_array[0] = False
    uvf2.flag_array[1] = False
    uvf |= uvf2
    assert np.all(uvf.flag_array[0])
    assert not np.any(uvf.flag_array[1])
    assert np.all(uvf.flag_array[2:])


def test_to_flag():
    uvf = UVFlag(test_f_file)
    uvf.to_flag()
    assert hasattr(uvf, "flag_array")
    assert hasattr(uvf, "metric_array")
    assert uvf.metric_array is None
    assert uvf.mode == "flag"
    assert 'Converted to mode "flag"' in uvf.history


def test_to_flag_add_version_str():
    uvf = UVFlag(test_f_file)
    uvf.history = uvf.history.replace(pyuvdata_version_str, "")
    assert pyuvdata_version_str not in uvf.history

    uvf.to_flag()
    assert pyuvdata_version_str in uvf.history


def test_to_flag_threshold():
    uvf = UVFlag(test_f_file)
    uvf.metric_array = np.zeros_like(uvf.metric_array)
    uvf.metric_array[0, 0, 4, 0] = 2.0
    uvf.to_flag(threshold=1.0)
    assert hasattr(uvf, "flag_array")
    assert hasattr(uvf, "metric_array")
    assert uvf.metric_array is None
    assert uvf.mode == "flag"
    assert uvf.flag_array[0, 0, 4, 0]
    assert np.sum(uvf.flag_array) == 1.0
    assert 'Converted to mode "flag"' in uvf.history


def test_flag_to_flag():
    uvf = UVFlag(test_f_file)
    uvf.to_flag()
    uvf2 = uvf.copy()
    uvf2.to_flag()
    assert uvf == uvf2


def test_to_flag_unknown_mode():
    uvf = UVFlag(test_f_file)
    uvf.mode = "foo"
    with pytest.raises(ValueError) as cm:
        uvf.to_flag()
    assert str(cm.value).startswith("Unknown UVFlag mode: foo")


def test_to_metric_baseline():
    uvf = UVFlag(test_f_file)
    uvf.to_flag()
    uvf.flag_array[:, :, 10] = True
    uvf.flag_array[1, :, :] = True
    assert hasattr(uvf, "flag_array")
    assert hasattr(uvf, "metric_array")
    assert uvf.metric_array is None
    assert uvf.mode == "flag"
    uvf.to_metric(convert_wgts=True)
    assert hasattr(uvf, "metric_array")
    assert hasattr(uvf, "flag_array")
    assert uvf.flag_array is None
    assert uvf.mode == "metric"
    assert 'Converted to mode "metric"' in uvf.history
    assert np.isclose(uvf.weights_array[1], 0.0).all()
    assert np.isclose(uvf.weights_array[:, :, 10], 0.0).all()


def test_to_metric_add_version_str():
    uvf = UVFlag(test_f_file)
    uvf.to_flag()
    uvf.flag_array[:, :, 10] = True
    uvf.flag_array[1, :, :] = True
    assert hasattr(uvf, "flag_array")
    assert hasattr(uvf, "metric_array")
    assert uvf.metric_array is None
    assert uvf.mode == "flag"

    uvf.history = uvf.history.replace(pyuvdata_version_str, "")
    assert pyuvdata_version_str not in uvf.history

    uvf.to_metric(convert_wgts=True)
    assert pyuvdata_version_str in uvf.history


def test_to_metric_waterfall():
    uvf = UVFlag(test_f_file)
    uvf.to_waterfall()
    uvf.to_flag()
    uvf.flag_array[:, 10] = True
    uvf.flag_array[1, :, :] = True
    uvf.to_metric(convert_wgts=True)
    assert np.isclose(uvf.weights_array[1], 0.0).all()
    assert np.isclose(uvf.weights_array[:, 10], 0.0).all()


def test_to_metric_antenna():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = UVFlag(uvc, mode="flag")
    uvf.flag_array[10, :, :, 1, :] = True
    uvf.flag_array[15, :, 3, :, :] = True
    uvf.to_metric(convert_wgts=True)
    assert np.isclose(uvf.weights_array[10, :, :, 1, :], 0.0).all()
    assert np.isclose(uvf.weights_array[15, :, 3, :, :], 0.0).all()


def test_metric_to_metric():
    uvf = UVFlag(test_f_file)
    uvf2 = uvf.copy()
    uvf.to_metric()
    assert uvf == uvf2


def test_to_metric_unknown_mode():
    uvf = UVFlag(test_f_file)
    uvf.mode = "foo"
    with pytest.raises(ValueError) as cm:
        uvf.to_metric()
    assert str(cm.value).startswith("Unknown UVFlag mode: foo")


def test_antpair2ind():
    uvf = UVFlag(test_f_file)
    ind = uvf.antpair2ind(uvf.ant_1_array[0], uvf.ant_2_array[0])
    assert np.all(uvf.ant_1_array[ind] == uvf.ant_1_array[0])
    assert np.all(uvf.ant_2_array[ind] == uvf.ant_2_array[0])


def test_antpair2ind_nonbaseline():
    uvf = UVFlag(test_f_file)
    uvf.to_waterfall()
    with pytest.raises(ValueError) as cm:
        uvf.antpair2ind(0, 3)
    assert str(cm.value).startswith(
        "UVFlag object of type "
        + uvf.type
        + " does not contain antenna "
        + "pairs to index."
    )


def test_baseline_to_antnums():
    uvf = UVFlag(test_f_file)
    a1, a2 = uvf.baseline_to_antnums(uvf.baseline_array[0])
    assert a1 == uvf.ant_1_array[0]
    assert a2 == uvf.ant_2_array[0]


def test_get_baseline_nums():
    uvf = UVFlag(test_f_file)
    bls = uvf.get_baseline_nums()
    assert np.array_equal(bls, np.unique(uvf.baseline_array))


def test_get_antpairs():
    uvf = UVFlag(test_f_file)
    antpairs = uvf.get_antpairs()
    for a1, a2 in antpairs:
        ind = np.where((uvf.ant_1_array == a1) & (uvf.ant_2_array == a2))[0]
        assert len(ind) > 0
    for a1, a2 in zip(uvf.ant_1_array, uvf.ant_2_array):
        assert (a1, a2) in antpairs


def test_missing_nants_telescope(tmp_path):
    testfile = str(tmp_path / "test_missing_Nants.h5")
    shutil.copyfile(test_f_file, testfile)

    with h5py.File(testfile, "r+") as f:
        del f["/Header/Nants_telescope"]
    with uvtest.check_warnings(
        UserWarning, match="Nants_telescope not available in file"
    ):
        uvf = UVFlag(testfile)
    uvf2 = UVFlag(test_f_file)
    uvf2.Nants_telescope = 2047
    assert uvf == uvf2
    os.remove(testfile)


def test_combine_metrics_inplace():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = UVFlag(uvc)
    np.random.seed(44)
    uvf.metric_array = np.random.normal(size=uvf.metric_array.shape)
    uvf2 = uvf.copy()
    uvf2.metric_array *= 2
    uvf3 = uvf.copy()
    uvf3.metric_array *= 3
    uvf.combine_metrics([uvf2, uvf3])
    factor = np.sqrt((1 + 4 + 9) / 3.0) / 2.0
    assert np.allclose(uvf.metric_array, np.abs(uvf2.metric_array) * factor)


def test_combine_metrics_not_inplace():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = UVFlag(uvc)
    np.random.seed(44)
    uvf.metric_array = np.random.normal(size=uvf.metric_array.shape)
    uvf2 = uvf.copy()
    uvf2.metric_array *= 2
    uvf3 = uvf.copy()
    uvf3.metric_array *= 3
    uvf4 = uvf.combine_metrics([uvf2, uvf3], inplace=False)
    factor = np.sqrt((1 + 4 + 9) / 3.0)
    assert np.allclose(uvf4.metric_array, np.abs(uvf.metric_array) * factor)


def test_combine_metrics_not_uvflag():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = UVFlag(uvc)
    with pytest.raises(ValueError) as cm:
        uvf.combine_metrics("bubblegum")
    assert str(cm.value).startswith('"others" must be UVFlag or list of UVFlag objects')


def test_combine_metrics_not_metric():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = UVFlag(uvc)
    np.random.seed(44)
    uvf.metric_array = np.random.normal(size=uvf.metric_array.shape)
    uvf2 = uvf.copy()
    uvf2.to_flag()
    with pytest.raises(ValueError) as cm:
        uvf.combine_metrics(uvf2)
    assert str(cm.value).startswith('UVFlag object and "others" must be in "metric"')


def test_combine_metrics_wrong_shape():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = UVFlag(uvc)
    np.random.seed(44)
    uvf.metric_array = np.random.normal(size=uvf.metric_array.shape)
    uvf2 = uvf.copy()
    uvf2.to_waterfall()
    with pytest.raises(ValueError) as cm:
        uvf.combine_metrics(uvf2)
    assert str(cm.value).startswith("UVFlag metric array shapes do not match.")


def test_combine_metrics_add_version_str():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = UVFlag(uvc)
    uvf.history = uvf.history.replace(pyuvdata_version_str, "")

    assert pyuvdata_version_str not in uvf.history
    np.random.seed(44)
    uvf.metric_array = np.random.normal(size=uvf.metric_array.shape)
    uvf2 = uvf.copy()
    uvf2.metric_array *= 2
    uvf3 = uvf.copy()
    uvf3.metric_array *= 3
    uvf4 = uvf.combine_metrics([uvf2, uvf3], inplace=False)

    assert pyuvdata_version_str in uvf4.history


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_super(uvdata_obj):
    class TestClass(UVFlag):
        def __init__(
            self,
            indata,
            mode="metric",
            copy_flags=False,
            waterfall=False,
            history="",
            label="",
            test_property="prop",
        ):

            super(TestClass, self).__init__(
                indata,
                mode=mode,
                copy_flags=copy_flags,
                waterfall=waterfall,
                history=history,
                label=label,
            )

            self.test_property = test_property

    uv = uvdata_obj

    tc = TestClass(uv, test_property="test_property")

    # UVFlag.__init__ is tested, so just see if it has a metric array
    assert hasattr(tc, "metric_array")
    # Check that it has the property
    assert tc.test_property == "test_property"


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_flags2waterfall(uvdata_obj):
    uv = uvdata_obj

    np.random.seed(0)
    uv.flag_array = np.random.randint(0, 2, size=uv.flag_array.shape, dtype=bool)
    wf = flags2waterfall(uv)
    assert np.allclose(np.mean(wf), np.mean(uv.flag_array))
    assert wf.shape == (uv.Ntimes, uv.Nfreqs)

    wf = flags2waterfall(uv, keep_pol=True)
    assert wf.shape == (uv.Ntimes, uv.Nfreqs, uv.Npols)

    # Test external flag_array
    uv.flag_array = np.zeros_like(uv.flag_array)
    f = np.random.randint(0, 2, size=uv.flag_array.shape, dtype=bool)
    wf = flags2waterfall(uv, flag_array=f)
    assert np.allclose(np.mean(wf), np.mean(f))
    assert wf.shape == (uv.Ntimes, uv.Nfreqs)

    # UVCal version
    uvc = UVCal()
    uvc.read_calfits(test_c_file)

    uvc.flag_array = np.random.randint(0, 2, size=uvc.flag_array.shape, dtype=bool)
    wf = flags2waterfall(uvc)
    assert np.allclose(np.mean(wf), np.mean(uvc.flag_array))
    assert wf.shape == (uvc.Ntimes, uvc.Nfreqs)

    wf = flags2waterfall(uvc, keep_pol=True)
    assert wf.shape == (uvc.Ntimes, uvc.Nfreqs, uvc.Njones)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_flags2waterfall_errors(uvdata_obj):

    # First argument must be UVData or UVCal object
    with pytest.raises(ValueError) as cm:
        flags2waterfall(5)
    assert str(cm.value).startswith(
        "flags2waterfall() requires a UVData or " + "UVCal object"
    )

    uv = uvdata_obj
    # Flag array must have same shape as uv.flag_array
    with pytest.raises(ValueError) as cm:
        flags2waterfall(uv, np.array([4, 5]))
    assert str(cm.value).startswith("Flag array must align with UVData or UVCal")


def test_and_rows_cols():
    d = np.zeros((10, 20), np.bool)
    d[1, :] = True
    d[:, 2] = True
    d[5, 10:20] = True
    d[5:8, 5] = True

    o = and_rows_cols(d)
    assert o[1, :].all()
    assert o[:, 2].all()
    assert not o[5, :].all()
    assert not o[:, 5].all()


def test_select_waterfall_errors(uvf_from_waterfall):
    uvf = uvf_from_waterfall
    with pytest.raises(ValueError) as cm:
        uvf.select(antenna_nums=[0, 1, 2])
    assert str(cm.value).startswith("Cannot select on antenna_nums with waterfall")

    with pytest.raises(ValueError) as cm:
        uvf.select(bls=[(0, 1), (0, 2)])
    assert str(cm.value).startswith("Cannot select on bls with waterfall")


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@cases_decorator
@pytest.mark.parametrize("uvf_mode", ["to_flag", "to_metric"])
@pytest.mark.parametrize("dimension", list(range(1, 4)))
def test_select_blt_inds(input_uvf, uvf_mode, dimension):
    uvf = input_uvf

    # used to set the mode depending on which input is given to uvf_mode
    getattr(uvf, uvf_mode)()
    np.random.seed(0)
    if uvf.type == "baseline":
        n_select = uvf.Nblts
    else:
        n_select = uvf.Ntimes

    blt_inds = np.random.choice(n_select, size=n_select // 2, replace=False)
    new_nblts = n_select // 2

    if dimension == 1:
        blt_inds = np.atleast_1d(blt_inds)
    elif dimension == 2:
        blt_inds = np.atleast_2d(blt_inds)
    elif dimension == 3:
        blt_inds = np.atleast_3d(blt_inds)

    uvf1 = uvf.select(blt_inds=blt_inds, inplace=False)

    # test the data was extracted correctly for each case
    for param_name, new_param in zip(uvf._data_params, uvf1.data_like_parameters):
        old_param = getattr(uvf, param_name)
        if uvf.type == "baseline":
            assert np.allclose(old_param[blt_inds.squeeze()], new_param)
        if uvf.type == "antenna":
            assert np.allclose(old_param[:, :, :, blt_inds.squeeze()], new_param)
        if uvf.type == "waterfall":
            assert np.allclose(old_param[blt_inds.squeeze()], new_param)

    if uvf.type == "baseline":
        assert uvf1.Nblts == new_nblts
    else:
        assert uvf1.Ntimes == new_nblts

    # verify that histories are different
    assert not uvutils._check_histories(uvf.history, uvf1.history)
    if uvf.type == "baseline":
        addition_str = "baseline-times"
    else:
        addition_str = "times"

    assert uvutils._check_histories(
        uvf.history + f"  Downselected to specific {addition_str} using pyuvdata.",
        uvf1.history,
    )


@cases_decorator
@pytest.mark.parametrize("uvf_mode", ["to_flag", "to_metric"])
@pytest.mark.parametrize(
    "select_kwargs,err_msg",
    [
        ({"blt_inds": []}, "No baseline-times were found"),
        ({"blt_inds": [int(1e9)]}, "blt_inds contains indices that are too large"),
        ({"blt_inds": [-1]}, "blt_inds contains indices that are negative"),
    ],
)
def test_select_blt_inds_errors(input_uvf, uvf_mode, select_kwargs, err_msg):
    uvf = input_uvf

    # used to set the mode depending on which input is given to uvf_mode
    getattr(uvf, uvf_mode)()

    with pytest.raises(ValueError, match=err_msg):
        uvf.select(**select_kwargs)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@cases_decorator_no_waterfall
@pytest.mark.parametrize("uvf_mode", ["to_flag", "to_metric"])
@pytest.mark.parametrize("dimension", list(range(1, 4)))
def test_select_antenna_nums(input_uvf, uvf_mode, dimension):
    uvf = input_uvf
    # used to set the mode depending on which input is given to uvf_mode
    getattr(uvf, uvf_mode)()

    old_history = copy.deepcopy(uvf.history)
    np.random.seed(0)
    if uvf.type == "baseline":
        unique_ants = np.unique(uvf.ant_1_array.tolist() + uvf.ant_2_array.tolist())
        ants_to_keep = np.random.choice(
            unique_ants, size=unique_ants.size // 2, replace=False
        )

        blts_select = [
            (a1 in ants_to_keep) & (a2 in ants_to_keep)
            for (a1, a2) in zip(uvf.ant_1_array, uvf.ant_2_array)
        ]
        Nblts_selected = np.sum(blts_select)
    else:
        unique_ants = np.unique(uvf.ant_array)
        ants_to_keep = np.random.choice(
            unique_ants, size=unique_ants.size // 2, replace=False
        )
    if dimension == 1:
        ants_to_keep = np.atleast_1d(ants_to_keep)
    elif dimension == 2:
        ants_to_keep = np.atleast_2d(ants_to_keep)
    elif dimension == 3:
        ants_to_keep = np.atleast_3d(ants_to_keep)

    uvf2 = copy.deepcopy(uvf)
    uvf2.select(antenna_nums=ants_to_keep)
    # make 1-D for the remaining iterators in tests
    ants_to_keep = ants_to_keep.squeeze()

    assert ants_to_keep.size == uvf2.Nants_data
    if uvf2.type == "baseline":
        assert Nblts_selected == uvf2.Nblts
        for ant in ants_to_keep:
            assert ant in uvf2.ant_1_array or ant in uvf2.ant_2_array
        for ant in np.unique(uvf2.ant_1_array.tolist() + uvf2.ant_2_array.tolist()):
            assert ant in ants_to_keep
    else:
        for ant in ants_to_keep:
            assert ant in uvf2.ant_array
        for ant in np.unique(uvf2.ant_array):
            assert ant in ants_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to " "specific antennas using pyuvdata.",
        uvf2.history,
    )


@cases_decorator_no_waterfall
@pytest.mark.parametrize("uvf_mode", ["to_flag", "to_metric"])
def test_select_antenna_nums_error(input_uvf, uvf_mode):
    uvf = input_uvf
    # used to set the mode depending on which input is given to uvf_mode
    getattr(uvf, uvf_mode)()
    # also test for error if antenna numbers not present in data
    with pytest.raises(ValueError) as cm:
        uvf.select(antenna_nums=[708, 709, 710])
    assert str(cm.value).startswith("Antenna number 708 is not present")


def sort_bl(p):
    """Sort a tuple that starts with a pair of antennas, and may have stuff after."""
    if p[1] >= p[0]:
        return p
    return (p[1], p[0]) + p[2:]


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@cases_decorator_no_waterfall
@pytest.mark.parametrize("uvf_mode", ["to_flag", "to_metric"])
def test_select_bls(input_uvf, uvf_mode):
    uvf = input_uvf
    # used to set the mode depending on which input is given to uvf_mode
    getattr(uvf, uvf_mode)()
    np.random.seed(0)

    if uvf.type != "baseline":
        with pytest.raises(ValueError) as cm:
            uvf.select(bls=[(0, 1)])
        assert str(cm.value).startswith(
            'Only "baseline" mode UVFlag '
            "objects may select along the "
            "baseline axis"
        )
    else:
        old_history = copy.deepcopy(uvf.history)
        bls_select = np.random.choice(
            uvf.baseline_array, size=uvf.Nbls // 2, replace=False
        )
        first_ants, second_ants = uvf.baseline_to_antnums(bls_select)

        # give the conjugate bls for a few baselines
        first_ants[5:8], second_ants[5:8] = (
            copy.copy(second_ants[5:8]),
            copy.copy(first_ants[5:8]),
        )

        new_unique_ants = np.unique(first_ants.tolist() + second_ants.tolist())
        ant_pairs_to_keep = list(zip(first_ants, second_ants))
        sorted_pairs_to_keep = [sort_bl(p) for p in ant_pairs_to_keep]

        blts_select = [
            sort_bl((a1, a2)) in sorted_pairs_to_keep
            for (a1, a2) in zip(uvf.ant_1_array, uvf.ant_2_array)
        ]
        Nblts_selected = np.sum(blts_select)

        uvf2 = copy.deepcopy(uvf)
        uvf2.select(bls=ant_pairs_to_keep)
        sorted_pairs_object2 = [
            sort_bl(p) for p in zip(uvf2.ant_1_array, uvf2.ant_2_array)
        ]

        assert len(new_unique_ants) == uvf2.Nants_data
        assert Nblts_selected == uvf2.Nblts
        for ant in new_unique_ants:
            assert ant in uvf2.ant_1_array or ant in uvf2.ant_2_array
        for ant in np.unique(uvf2.ant_1_array.tolist() + uvf2.ant_2_array.tolist()):
            assert ant in new_unique_ants
        for pair in sorted_pairs_to_keep:
            assert pair in sorted_pairs_object2
        for pair in sorted_pairs_object2:
            assert pair in sorted_pairs_to_keep

        assert uvutils._check_histories(
            old_history + "  Downselected to " "specific baselines using pyuvdata.",
            uvf2.history,
        )

        # Check with polarization too
        first_ants, second_ants = uvf.baseline_to_antnums(bls_select)
        # conjugate a few bls
        first_ants[5:8], second_ants[5:8] = (
            copy.copy(second_ants[5:8]),
            copy.copy(first_ants[5:8]),
        )

        pols = ["xx"] * len(first_ants)

        new_unique_ants = np.unique(first_ants.tolist() + second_ants.tolist())
        ant_pairs_to_keep = list(zip(first_ants, second_ants, pols))
        sorted_pairs_to_keep = [sort_bl(p) for p in ant_pairs_to_keep]

        blts_select = [
            sort_bl((a1, a2, "xx")) in sorted_pairs_to_keep
            for (a1, a2) in zip(uvf.ant_1_array, uvf.ant_2_array)
        ]
        Nblts_selected = np.sum(blts_select)

        uvf2 = copy.deepcopy(uvf)

        uvf2.select(bls=ant_pairs_to_keep)
        sorted_pairs_object2 = [
            sort_bl(p) + ("xx",) for p in zip(uvf2.ant_1_array, uvf2.ant_2_array)
        ]

        assert len(new_unique_ants) == uvf2.Nants_data
        assert Nblts_selected == uvf2.Nblts
        for ant in new_unique_ants:
            assert ant in uvf2.ant_1_array or ant in uvf2.ant_2_array
        for ant in np.unique(uvf2.ant_1_array.tolist() + uvf2.ant_2_array.tolist()):
            assert ant in new_unique_ants
        for pair in sorted_pairs_to_keep:
            assert pair in sorted_pairs_object2
        for pair in sorted_pairs_object2:
            assert pair in sorted_pairs_to_keep

        assert uvutils._check_histories(
            old_history + "  Downselected to "
            "specific baselines, polarizations using pyuvdata.",
            uvf2.history,
        )

        # check that you can specify a single pair without errors
        assert isinstance(ant_pairs_to_keep[0], tuple)
        uvf2.select(bls=ant_pairs_to_keep[0])
        sorted_pairs_object2 = [
            sort_bl(p) + ("xx",) for p in zip(uvf2.ant_1_array, uvf2.ant_2_array)
        ]
        assert list(set(sorted_pairs_object2)) == [ant_pairs_to_keep[0]]


@cases_decorator_no_waterfall
@pytest.mark.parametrize("uvf_mode", ["to_flag", "to_metric"])
@pytest.mark.parametrize(
    "select_kwargs,err_msg",
    [
        ({"bls": [3]}, "bls must be a list of tuples"),
        ({"bls": [(np.pi, 2 * np.pi)]}, "bls must be a list of tuples of integer"),
        (
            {"bls": (0, 1, "xx"), "polarizations": [-5]},
            "Cannot provide length-3 tuples and also specify polarizations.",
        ),
        (
            {"bls": (0, 1, 5)},
            "The third element in each bl must be a polarization string",
        ),
        ({"bls": (455, 456)}, "Antenna number 455 is not present"),
        ({"bls": (97, 456)}, "Antenna number 456 is not present"),
        (
            {"bls": (97, 97)},
            r"Antenna pair \(97, 97\) does not have any data associated with it.",
        ),
    ],
)
def test_select_bls_errors(input_uvf, uvf_mode, select_kwargs, err_msg):
    uvf = input_uvf
    # used to set the mode depending on which input is given to uvf_mode
    getattr(uvf, uvf_mode)()
    np.random.seed(0)
    if uvf.type != "baseline":
        with pytest.raises(ValueError) as cm:
            uvf.select(bls=[(0, 1)])
        assert str(cm.value).startswith(
            'Only "baseline" mode UVFlag '
            "objects may select along the "
            "baseline axis"
        )
    else:
        if select_kwargs["bls"] == (97, 97):
            uvf.select(bls=[(97, 104), (97, 105), (88, 97)])
        with pytest.raises(ValueError, match=err_msg):
            uvf.select(**select_kwargs)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@cases_decorator
@pytest.mark.parametrize("uvf_mode", ["to_flag", "to_metric"])
def test_select_times(input_uvf, uvf_mode):
    uvf = input_uvf
    # used to set the mode depending on which input is given to uvf_mode
    getattr(uvf, uvf_mode)()
    np.random.seed(0)
    old_history = uvf.history
    unique_times = np.unique(uvf.time_array)
    times_to_keep = np.random.choice(
        unique_times, size=unique_times.size // 2, replace=False
    )

    Nblts_selected = np.sum([t in times_to_keep for t in uvf.time_array])

    uvf2 = copy.deepcopy(uvf)
    uvf2.select(times=times_to_keep)

    assert len(times_to_keep) == uvf2.Ntimes
    if uvf2.type == "baseline":
        n_compare = uvf2.Nblts
    else:
        n_compare = uvf2.Ntimes
    assert Nblts_selected == n_compare
    for t in times_to_keep:
        assert t in uvf2.time_array
    for t in np.unique(uvf2.time_array):
        assert t in times_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to " "specific times using pyuvdata.",
        uvf2.history,
    )
    # check that it also works with higher dimension array
    uvf2 = copy.deepcopy(uvf)
    uvf2.select(times=times_to_keep[np.newaxis, :])

    assert len(times_to_keep) == uvf2.Ntimes
    assert Nblts_selected == n_compare
    for t in times_to_keep:
        assert t in uvf2.time_array
    for t in np.unique(uvf2.time_array):
        assert t in times_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to " "specific times using pyuvdata.",
        uvf2.history,
    )
    # check for errors associated with times not included in data
    with pytest.raises(ValueError) as cm:
        bad_time = [np.min(unique_times) - 0.005]
        uvf.select(times=bad_time)
    assert str(cm.value).startswith(
        "Time {t} is not present in" " the time_array".format(t=bad_time[0])
    )


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@cases_decorator
@pytest.mark.parametrize("uvf_mode", ["to_flag", "to_metric"])
def test_select_frequencies(input_uvf, uvf_mode):
    uvf = input_uvf
    # used to set the mode depending on which input is given to uvf_mode
    getattr(uvf, uvf_mode)()
    np.random.seed(0)
    old_history = uvf.history

    freqs_to_keep = np.random.choice(
        uvf.freq_array.squeeze(), size=uvf.Nfreqs // 10, replace=False
    )

    uvf2 = copy.deepcopy(uvf)
    uvf2.select(frequencies=freqs_to_keep)

    assert len(freqs_to_keep) == uvf2.Nfreqs
    for f in freqs_to_keep:
        assert f in uvf2.freq_array
    for f in np.unique(uvf2.freq_array):
        assert f in freqs_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to " "specific frequencies using pyuvdata.",
        uvf2.history,
    )

    # check that it also works with higher dimension array
    uvf2 = copy.deepcopy(uvf)
    uvf2.select(frequencies=freqs_to_keep[np.newaxis, :])

    assert len(freqs_to_keep) == uvf2.Nfreqs
    for f in freqs_to_keep:
        assert f in uvf2.freq_array
    for f in np.unique(uvf2.freq_array):
        assert f in freqs_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to " "specific frequencies using pyuvdata.",
        uvf2.history,
    )

    # check that selecting one frequency works
    uvf2 = copy.deepcopy(uvf)
    uvf2.select(frequencies=freqs_to_keep[0])
    assert 1 == uvf2.Nfreqs
    assert freqs_to_keep[0] in uvf2.freq_array
    for f in uvf2.freq_array:
        assert f in [freqs_to_keep[0]]

    assert uvutils._check_histories(
        old_history + "  Downselected to " "specific frequencies using pyuvdata.",
        uvf2.history,
    )

    # check for errors associated with frequencies not included in data
    with pytest.raises(ValueError) as cm:
        bad_freq = [np.max(uvf.freq_array) + 100]
        uvf.select(frequencies=bad_freq)
    assert str(cm.value).startswith(
        "Frequency {f} is not present in the freq_array".format(f=bad_freq[0])
    )


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@cases_decorator
@pytest.mark.parametrize("uvf_mode", ["to_flag", "to_metric"])
def test_select_freq_chans(input_uvf, uvf_mode):
    uvf = input_uvf
    # used to set the mode depending on which input is given to uvf_mode
    getattr(uvf, uvf_mode)()
    np.random.seed(0)
    old_history = uvf.history

    old_history = uvf.history
    chans = np.random.choice(uvf.Nfreqs, 2)
    c1, c2 = np.sort(chans)
    chans_to_keep = np.arange(c1, c2)

    uvf2 = copy.deepcopy(uvf)
    uvf2.select(freq_chans=chans_to_keep)

    assert len(chans_to_keep) == uvf2.Nfreqs
    for chan in chans_to_keep:
        if uvf2.type != "waterfall":
            assert uvf.freq_array[0, chan] in uvf2.freq_array
        else:
            assert uvf.freq_array[chan] in uvf2.freq_array

    for f in np.unique(uvf2.freq_array):
        if uvf2.type != "waterfall":
            assert f in uvf.freq_array[0, chans_to_keep]
        else:
            assert f in uvf.freq_array[chans_to_keep]

    assert uvutils._check_histories(
        old_history + "  Downselected to " "specific frequencies using pyuvdata.",
        uvf2.history,
    )

    # check that it also works with higher dimension array
    uvf2 = copy.deepcopy(uvf)
    uvf2.select(freq_chans=chans_to_keep[np.newaxis, :])

    assert len(chans_to_keep) == uvf2.Nfreqs
    for chan in chans_to_keep:
        if uvf2.type != "waterfall":
            assert uvf.freq_array[0, chan] in uvf2.freq_array
        else:
            assert uvf.freq_array[chan] in uvf2.freq_array

    for f in np.unique(uvf2.freq_array):
        if uvf2.type != "waterfall":
            assert f in uvf.freq_array[0, chans_to_keep]
        else:
            assert f in uvf.freq_array[chans_to_keep]

    assert uvutils._check_histories(
        old_history + "  Downselected to " "specific frequencies using pyuvdata.",
        uvf2.history,
    )

    # Test selecting both channels and frequencies
    chans = np.random.choice(uvf.Nfreqs, 2)
    c1, c2 = np.sort(chans)
    chans_to_keep = np.arange(c1, c2)

    if uvf.type != "waterfall":
        freqs_to_keep = uvf.freq_array[0, np.arange(c1 + 1, c2)]  # Overlaps with chans
    else:
        freqs_to_keep = uvf.freq_array[np.arange(c1 + 1, c2)]  # Overlaps with chans

    all_chans_to_keep = np.arange(c1, c2)

    uvf2 = copy.deepcopy(uvf)
    uvf2.select(frequencies=freqs_to_keep, freq_chans=chans_to_keep)

    assert len(all_chans_to_keep) == uvf2.Nfreqs
    for chan in chans_to_keep:
        if uvf2.type != "waterfall":
            assert uvf.freq_array[0, chan] in uvf2.freq_array
        else:
            assert uvf.freq_array[chan] in uvf2.freq_array

    for f in np.unique(uvf2.freq_array):
        if uvf2.type != "waterfall":
            assert f in uvf.freq_array[0, chans_to_keep]
        else:
            assert f in uvf.freq_array[chans_to_keep]


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@cases_decorator
@pytest.mark.parametrize("uvf_mode", ["to_flag", "to_metric"])
def test_select_polarizations(input_uvf, uvf_mode):
    uvf = input_uvf
    # used to set the mode depending on which input is given to uvf_mode
    getattr(uvf, uvf_mode)()
    np.random.seed(0)
    old_history = uvf.history

    pols_to_keep = [-5]

    uvf2 = copy.deepcopy(uvf)
    uvf2.select(polarizations=pols_to_keep)

    assert len(pols_to_keep) == uvf2.Npols
    for p in pols_to_keep:
        assert p in uvf2.polarization_array
    for p in np.unique(uvf2.polarization_array):
        assert p in pols_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to " "specific polarizations using pyuvdata.",
        uvf2.history,
    )

    # check that it also works with higher dimension array
    uvf2 = copy.deepcopy(uvf)
    uvf2.select(polarizations=[pols_to_keep])

    assert len(pols_to_keep) == uvf2.Npols
    for p in pols_to_keep:
        assert p in uvf2.polarization_array
    for p in np.unique(uvf2.polarization_array):
        assert p in pols_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to " "specific polarizations using pyuvdata.",
        uvf2.history,
    )

    # check for errors associated with polarizations not included in data
    with pytest.raises(ValueError) as cm:
        uvf2.select(polarizations=[-3])
    assert str(cm.value).startswith(
        "Polarization {p} is not present in the polarization_array".format(p=-3)
    )


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@cases_decorator
@pytest.mark.parametrize("uvf_mode", ["to_flag", "to_metric"])
def test_select(input_uvf, uvf_mode):
    uvf = input_uvf
    # used to set the mode depending on which input is given to uvf_mode
    getattr(uvf, uvf_mode)()
    np.random.seed(0)
    old_history = uvf.history

    # make new blts
    if uvf.type == "baseline":
        blt_inds = np.arange(uvf.Nblts - 1)
    else:
        blt_inds = np.arange(uvf.Ntimes - 1)

    # new freqs
    freqs_to_keep = np.random.choice(
        uvf.freq_array.squeeze(), size=uvf.Nfreqs - 1, replace=False
    )
    # new ants
    if uvf.type == "baseline":
        unique_ants = np.unique(uvf.ant_1_array.tolist() + uvf.ant_2_array.tolist())
        ants_to_keep = np.random.choice(
            unique_ants, size=unique_ants.size - 1, replace=False
        )

    elif uvf.type == "antenna":
        unique_ants = np.unique(uvf.ant_array)
        ants_to_keep = np.random.choice(
            unique_ants, size=unique_ants.size - 1, replace=False
        )
    else:
        ants_to_keep = None

    if uvf.type == "baseline":
        #  new bls
        bls_select = np.random.choice(
            uvf.baseline_array, size=uvf.Nbls - 1, replace=False
        )
        first_ants, second_ants = uvf.baseline_to_antnums(bls_select)
        # give the conjugate bls for a few baselines
        first_ants[2:4], second_ants[2:4] = second_ants[2:4], first_ants[2:4]

        ant_pairs_to_keep = list(zip(first_ants, second_ants))
        sorted_pairs_to_keep = [sort_bl(p) for p in ant_pairs_to_keep]

    else:
        ant_pairs_to_keep = None

    # new times
    unique_times = np.unique(uvf.time_array)
    times_to_keep = np.random.choice(
        unique_times, size=unique_times.size - 1, replace=False
    )

    # new pols
    pols_to_keep = [-5]

    # Independently count blts that should be selected
    if uvf.type == "baseline":
        blts_blt_select = [i in blt_inds for i in np.arange(uvf.Nblts)]
        blts_ant_select = [
            (a1 in ants_to_keep) & (a2 in ants_to_keep)
            for (a1, a2) in zip(uvf.ant_1_array, uvf.ant_2_array)
        ]
        blts_pair_select = [
            sort_bl((a1, a2)) in sorted_pairs_to_keep
            for (a1, a2) in zip(uvf.ant_1_array, uvf.ant_2_array)
        ]
        blts_time_select = [t in times_to_keep for t in uvf.time_array]
        Nblts_select = np.sum(
            [
                bi & (ai & pi) & ti
                for (bi, ai, pi, ti) in zip(
                    blts_blt_select, blts_ant_select, blts_pair_select, blts_time_select
                )
            ]
        )
    else:
        if uvf.type == "baseline":
            blts_blt_select = [i in blt_inds for i in np.arange(uvf.Nblts)]
        else:
            blts_blt_select = [i in blt_inds for i in np.arange(uvf.Ntimes)]

        blts_time_select = [t in times_to_keep for t in uvf.time_array]
        Nblts_select = np.sum(
            [bi & ti for (bi, ti) in zip(blts_blt_select, blts_time_select)]
        )

    uvf2 = copy.deepcopy(uvf)
    uvf2.select(
        blt_inds=blt_inds,
        antenna_nums=ants_to_keep,
        bls=ant_pairs_to_keep,
        frequencies=freqs_to_keep,
        times=times_to_keep,
        polarizations=pols_to_keep,
    )

    if uvf.type == "baseline":
        assert Nblts_select == uvf2.Nblts
        for ant in np.unique(uvf2.ant_1_array.tolist() + uvf2.ant_2_array.tolist()):
            assert ant in ants_to_keep
    elif uvf.type == "antenna":
        assert Nblts_select == uvf2.Ntimes
        for ant in np.unique(uvf2.ant_array):
            assert ant in ants_to_keep
    else:
        assert Nblts_select == uvf2.Ntimes

    assert len(freqs_to_keep) == uvf2.Nfreqs
    for f in freqs_to_keep:
        assert f in uvf2.freq_array
    for f in np.unique(uvf2.freq_array):
        assert f in freqs_to_keep

    for t in np.unique(uvf2.time_array):
        assert t in times_to_keep

    assert len(pols_to_keep) == uvf2.Npols
    for p in pols_to_keep:
        assert p in uvf2.polarization_array
    for p in np.unique(uvf2.polarization_array):
        assert p in pols_to_keep

    if uvf.type == "baseline":
        assert uvutils._check_histories(
            old_history + "  Downselected to "
            "specific baseline-times, antennas, "
            "baselines, times, frequencies, "
            "polarizations using pyuvdata.",
            uvf2.history,
        )
    elif uvf.type == "antenna":
        assert uvutils._check_histories(
            old_history + "  Downselected to "
            "specific times, antennas, "
            "frequencies, "
            "polarizations using pyuvdata.",
            uvf2.history,
        )
    else:
        assert uvutils._check_histories(
            old_history + "  Downselected to "
            "specific times, "
            "frequencies, "
            "polarizations using pyuvdata.",
            uvf2.history,
        )


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("uvf_mode", ["to_flag", "to_metric"])
def test_parse_ants_error(uvf_from_uvcal, uvf_mode):
    uvf = uvf_from_uvcal
    # used to set the mode depending on which input is given to uvf_mode
    getattr(uvf, uvf_mode)()
    err_msg = (
        "UVFlag objects can only call 'parse_ants' function if type is 'baseline'."
    )
    with pytest.raises(ValueError, match=err_msg):
        uvf.parse_ants("all")


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "select_kwargs,err_msg",
    [
        (
            {"ant_str": "all", "antenna_nums": [1, 2, 3]},
            "Cannot provide ant_str with antenna_nums, bls, or polarizations.",
        ),
        (
            {"ant_str": "all", "bls": [(0, 1), (1, 2)]},
            "Cannot provide ant_str with antenna_nums, bls, or polarizations.",
        ),
        (
            {"ant_str": "all", "polarizations": [-5, -6, -7]},
            "Cannot provide ant_str with antenna_nums, bls, or polarizations.",
        ),
        (
            {"ant_str": "auto"},
            "There is no data matching ant_str=auto in this object.",
        ),
    ],
)
@pytest.mark.parametrize("uvf_mode", ["to_flag", "to_metric"])
def test_select_parse_ants_errors(uvf_from_data, uvf_mode, select_kwargs, err_msg):
    uvf = uvf_from_data
    # used to set the mode depending on which input is given to uvf_mode
    getattr(uvf, uvf_mode)()
    if select_kwargs["ant_str"] == "auto":
        uvf = uvf.select(ant_str="cross", inplace=False)
    with pytest.raises(ValueError, match=err_msg):
        uvf.select(**select_kwargs)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("uvf_mode", ["to_flag", "to_metric"])
def test_select_parse_ants(uvf_from_data, uvf_mode):
    uvf = uvf_from_data
    # used to set the mode depending on which input is given to uvf_mode
    getattr(uvf, uvf_mode)()
    uvf.select(ant_str="97_104,97_105,88_97")
    assert uvf.Nbls == 3
    assert np.array_equiv(
        np.unique(uvf.baseline_array),
        uvutils.antnums_to_baseline(
            *np.transpose([(88, 97), (97, 104), (97, 105)]), uvf.Nants_telescope,
        ),
    )


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_equality_no_history(uvf_from_data):
    uvf = uvf_from_data
    uvf2 = uvf.copy()
    assert uvf.__eq__(uvf2, check_history=True)
    uvf2.history += "different text"
    assert uvf.__eq__(uvf2, check_history=False)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_inequality_different_classes(uvf_from_data):
    uvf = uvf_from_data

    class TestClass(object):
        def __init__(self):
            pass

    other_class = TestClass()

    assert uvf.__ne__(other_class, check_history=False)


def test_to_antenna_collapsed_pols(uvf_from_uvcal):
    uvf = uvf_from_uvcal

    assert not uvf.pol_collapsed
    uvc = UVCal()
    uvc.read_calfits(test_c_file)

    uvf.collapse_pol()
    assert uvf.pol_collapsed
    assert uvf.check()

    uvf.to_waterfall()
    uvf.to_antenna(uvc, force_pol=True)
    assert not uvf.pol_collapsed
    assert uvf.check()


def test_get_ants_error(uvf_from_waterfall):
    uvf = uvf_from_waterfall

    with pytest.raises(
        ValueError, match="A waterfall type UVFlag object has no sense of antennas.",
    ):
        uvf.get_ants()


@cases_decorator_no_waterfall
@pytest.mark.parametrize("uvf_mode", ["to_flag", "to_metric"])
def test_get_ants(input_uvf, uvf_mode):
    uvf = input_uvf
    getattr(uvf, uvf_mode)()
    ants = uvf.get_ants()
    if uvf.type == "baseline":
        expected_ants = np.sort(
            list(set(np.unique(uvf.ant_1_array)).union(np.unique(uvf.ant_2_array)))
        )
    if uvf.type == "antenna":
        expected_ants = np.unique(uvf.ant_array)

    assert np.array_equiv(ants, expected_ants)
