# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for Miriad object.

Note that because of the way that file handling is done in the C++ API, a miriad
file is not closed until the destructor function (tp_dealloc) is called.

The following lines were made before a major rewrite of the miriad interface.
As of April 2020, they should no longer necessarily be true,
but are preserved here in case segfaults arise again.

Due to implementation details of CPython, it is sometimes not enough to `del` the
object--a manual garbage collection may be required. When adding new tests,
proper cleanup consists of: (1) deleting any Miriad objects or UVData objects,
(2) performing garbage collection (with `gc.collect()`), and (3) removing the
directory corresponding to the miriad file. Each test should do this, to avoid
open file handles interfering with other tests. The exception to this is if a
test uses the `uv_in_paper` or `uv_in_uvfits` fixture, as these handle cleanup
on their own.
"""
import os
import shutil
import numpy as np
import pytest
from astropy.time import Time, TimeDelta
from astropy import constants as const

from pyuvdata import UVData
from ..miriad import Miriad
import pyuvdata.utils as uvutils
import pyuvdata.tests as uvtest
from pyuvdata.data import DATA_PATH

aipy_extracts = pytest.importorskip("pyuvdata.uvdata.aipy_extracts")

# always ignore the Altitude not present warning
pytestmark = pytest.mark.filterwarnings("ignore:Altitude is not present in Miriad")

paper_miriad_file = os.path.join(DATA_PATH, "zen.2456865.60537.xy.uvcRREAA")


@pytest.fixture(scope="function")
def uv_in_paper(paper_miriad, tmp_path):
    uv_in = paper_miriad
    write_file = os.path.join(tmp_path, "outtest_miriad.uv")
    uv_in.write_miriad(write_file, clobber=True)
    uv_out = UVData()

    yield uv_in, uv_out, write_file

    # cleanup
    del uv_in, uv_out


@pytest.fixture(scope="function")
def uv_in_uvfits(paper_miriad, tmp_path):
    uv_in = paper_miriad
    write_file = os.path.join(tmp_path, "outtest_miriad.uvfits")

    uv_out = UVData()

    yield uv_in, uv_out, write_file

    # cleanup
    del uv_in, uv_out


@pytest.mark.filterwarnings("ignore:Telescope ATCA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_read_write_read_atca(tmp_path, future_shapes):
    uv_in = UVData()
    uv_out = UVData()
    atca_file = os.path.join(DATA_PATH, "atca_miriad")
    testfile = os.path.join(tmp_path, "outtest_atca_miriad.uv")
    with uvtest.check_warnings(
        UserWarning,
        [
            "Altitude is not present in Miriad file, and "
            "telescope ATCA is not in known_telescopes. ",
            "Altitude is not present",
            "Telescope location is set at sealevel at the file lat/lon "
            "coordinates. Antenna positions are present, but the mean antenna "
            "position does not give a telescope_location on the surface of the "
            "earth. Antenna positions do not appear to be on the surface of the "
            "earth and will be treated as relative.",
            "Telescope ATCA is not in known_telescopes.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv_in.read(atca_file)

    if future_shapes:
        uv_in.use_future_array_shapes()

    uv_in.write_miriad(testfile, clobber=True)
    uv_out.read(testfile)

    if future_shapes:
        uv_out.use_future_array_shapes()

    assert uv_in == uv_out


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_read_nrao_write_miriad_read_miriad(casa_uvfits, tmp_path):
    """Test reading in a CASA tutorial uvfits file, writing and reading as miriad"""
    uvfits_uv = casa_uvfits
    miriad_uv = UVData()
    writefile = os.path.join(tmp_path, "outtest_miriad.uv")
    uvfits_uv.write_miriad(writefile, clobber=True)
    miriad_uv.read(writefile)
    assert uvfits_uv == miriad_uv


@pytest.mark.filterwarnings("ignore:Telescope SZA is not in known_telescopes.")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Telescope location is set at sealevel")
def test_read_write_read_carma(tmp_path):
    uv_in = UVData()
    uv_out = UVData()
    carma_file = os.path.join(DATA_PATH, "carma_miriad")
    testfile = os.path.join(tmp_path, "outtest_carma_miriad.uv")

    # Test file is multi-source, make sure it bugs out
    pytest.raises(
        NotImplementedError, uv_in.read, carma_file,
    )

    with uvtest.check_warnings(
        UserWarning,
        [
            "Altitude is not present in Miriad file, and "
            "telescope SZA is not in known_telescopes. ",
            "Altitude is not present",
            "Telescope location is set at sealevel at the file lat/lon "
            "coordinates. Antenna positions are present, but the mean antenna "
            "position does not give a telescope_location on the surface of the "
            "earth. Antenna positions do not appear to be on the surface of the "
            "earth and will be treated as relative.",
            "Telescope SZA is not in known_telescopes.",
            "File containts more than one source, only using data where source = NOISE",
            "drift RA, Dec is off from lst, latitude by more than 1.0 deg",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "antaz in extra_keywords is a list, array or dict",
            "antel in extra_keywords is a list, array or dict",
            "axisoff in extra_keywords is a list, array or dict",
            "cable in extra_keywords is a list, array or dict",
            "dazim in extra_keywords is a list, array or dict",
            "delev in extra_keywords is a list, array or dict",
            "jyperka in extra_keywords is a list, array or dict",
            "pamatten in extra_keywords is a list, array or dict",
            "phaselo1 in extra_keywords is a list, array or dict",
            "phaselo2 in extra_keywords is a list, array or dict",
            "phasem1 in extra_keywords is a list, array or dict",
            "themt in extra_keywords is a list, array or dict",
            "wfreq in extra_keywords is a list, array or dict",
            "wwidth in extra_keywords is a list, array or dict",
            "wsystemp in extra_keywords is a list, array or dict",
            "psys in extra_keywords is a list, array or dict",
            "psysattn in extra_keywords is a list, array or dict",
            "ambpsys in extra_keywords is a list, array or dict",
            "bfmask in extra_keywords is a list, array or dict",
            "wcorr in extra_keywords is a list, array or dict",
        ],
    ):
        uv_in.read(carma_file, skip_extra_sources=True)

    # Spoofing these values for now, since the data is from the noise source, which
    # doesn't neccessarily have correct values
    uv_in._set_phased()
    uv_in.phase_center_ra = 0.0
    uv_in.phase_center_dec = 0.0
    uv_in.phase_center_epoch = 2000.0

    # Extra keywords cannnot handle lists, dicts, or arrays, so drop them from the
    # dataset, so that the writer doesn't run into issues.
    # TODO: Capture these extra keywords
    for item in list(uv_in.extra_keywords.keys()):
        if isinstance(uv_in.extra_keywords[item], dict):
            uv_in.extra_keywords.pop(item)
        elif isinstance(uv_in.extra_keywords[item], list):
            uv_in.extra_keywords.pop(item)
        elif isinstance(uv_in.extra_keywords[item], np.ndarray):
            uv_in.extra_keywords.pop(item)

    uv_in.write_miriad(testfile, clobber=True)

    uv_out.read(testfile)
    assert uv_in == uv_out


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_read_miriad_write_uvfits(uv_in_uvfits):
    """
    Miriad to uvfits loopback test.

    Read in Miriad files, write out as uvfits, read back in and check for
    object equality.
    """
    miriad_uv, uvfits_uv, testfile = uv_in_uvfits

    miriad_uv.write_uvfits(testfile, spoof_nonessential=True, force_phase=True)
    uvfits_uv.read_uvfits(testfile)
    assert miriad_uv == uvfits_uv


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_miriad_read_warning_lat_lon_corrected():
    miriad_uv = UVData()
    # check warning when correct_lat_lon is set to False
    with uvtest.check_warnings(
        UserWarning,
        [
            "Altitude is not present in Miriad file, using known location "
            "altitude value for PAPER and lat/lon from file.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        miriad_uv.read(paper_miriad_file, correct_lat_lon=False)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "err_type,write_kwargs,err_msg",
    [
        (
            ValueError,
            {"spoof_nonessential": True},
            "The data are in drift mode. Set force_phase",
        ),
        (ValueError, {"force_phase": True}, "Required attribute"),
    ],
)
def test_read_miriad_write_uvfits_phasing_errors(
    uv_in_uvfits, err_type, write_kwargs, err_msg
):
    miriad_uv, uvfits_uv, testfile = uv_in_uvfits

    # check error if phase_type is wrong and force_phase not set
    with pytest.raises(err_type, match=err_msg):
        miriad_uv.write_uvfits(testfile, **write_kwargs)


@pytest.mark.parametrize(
    "err_type,read_kwargs,err_msg",
    [
        (
            ValueError,
            {"phase_type": "phased"},
            'phase_type is "phased" but the RA values are varying',
        ),
        (ValueError, {"phase_type": "foo"}, "The phase_type was not recognized."),
    ],
)
def test_read_miriad_phasing_errors(err_type, read_kwargs, err_msg):
    miriad_uv = UVData()
    # check that setting the phase_type to something wrong errors
    with pytest.raises(err_type, match=err_msg):
        miriad_uv.read(paper_miriad_file, **read_kwargs)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_read_miriad_write_uvfits_phasing_error(uv_in_uvfits):
    miriad_uv, uvfits_uv, testfile = uv_in_uvfits

    miriad_uv._set_unknown_phase_type()
    with pytest.raises(ValueError, match="The phasing type of the data is unknown"):
        miriad_uv.write_uvfits(testfile, spoof_nonessential=True)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_wronglatlon():
    """
    Check for appropriate warnings with incorrect lat/lon values or missing telescope

    To test this, we needed files without altitudes and with wrong lat, lon or
    telescope values.
    These test files were made commenting out the line in miriad.py that adds altitude
    to the file and running the following code:
    import os
    import numpy as np
    from pyuvdata import UVData
    from pyuvdata.data import DATA_PATH
    uv_in = UVData()
    uv_out = UVData()
    miriad_file = os.path.join(DATA_PATH, 'zen.2456865.60537.xy.uvcRREAA')
    latfile = os.path.join(DATA_PATH, 'zen.2456865.60537_wronglat.xy.uvcRREAA')
    lonfile = os.path.join(DATA_PATH, 'zen.2456865.60537_wronglon.xy.uvcRREAA')
    latlonfile = os.path.join(DATA_PATH, 'zen.2456865.60537_wronglatlon.xy.uvcRREAA')
    telescopefile = os.path.join(DATA_PATH,
    'zen.2456865.60537_wrongtelecope.xy.uvcRREAA')
    uv_in.read(miriad_file)
    uv_in.select(times=uv_in.time_array[0])
    uv_in.select(freq_chans=[0])

    lat, lon, alt = uv_in.telescope_location_lat_lon_alt
    lat_wrong = lat + 10 * np.pi / 180.
    uv_in.telescope_location_lat_lon_alt = (lat_wrong, lon, alt)
    uv_in.write_miriad(latfile, clobber=True)
    uv_out.read(latfile)

    lon_wrong = lon + 10 * np.pi / 180.
    uv_in.telescope_location_lat_lon_alt = (lat, lon_wrong, alt)
    uv_in.write_miriad(lonfile, clobber=True)
    uv_out.read(lonfile)

    uv_in.telescope_location_lat_lon_alt = (lat_wrong, lon_wrong, alt)
    uv_in.write_miriad(latlonfile)
    uv_out.read(latlonfile)

    uv_in.telescope_location_lat_lon_alt = (lat, lon, alt)
    uv_in.telescope_name = 'foo'
    uv_in.write_miriad(telescopefile, clobber=True)
    uv_out.read(telescopefile, run_check=False)

    """
    uv_in = UVData()
    latfile = os.path.join(DATA_PATH, "zen.2456865.60537_wronglat.xy.uvcRREAA")
    lonfile = os.path.join(DATA_PATH, "zen.2456865.60537_wronglon.xy.uvcRREAA")
    latlonfile = os.path.join(DATA_PATH, "zen.2456865.60537_wronglatlon.xy.uvcRREAA")
    telescopefile = os.path.join(
        DATA_PATH, "zen.2456865.60537_wrongtelecope.xy.uvcRREAA"
    )
    with uvtest.check_warnings(
        UserWarning,
        [
            "Altitude is not present in file and latitude value does not match",
            "drift RA, Dec is off from lst, latitude by more than 1.0 deg",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv_in.read(latfile)

    with uvtest.check_warnings(
        UserWarning,
        [
            "Altitude is not present in file and longitude value does not match",
            "drift RA, Dec is off from lst, latitude by more than 1.0 deg",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv_in.read(lonfile)

    with uvtest.check_warnings(
        UserWarning,
        [
            "Altitude is not present in file and latitude and longitude "
            "values do not match",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv_in.read(latlonfile, correct_lat_lon=False)

    with uvtest.check_warnings(
        UserWarning,
        [
            "Altitude is not present in Miriad file, and telescope",
            "Altitude is not present in Miriad file, and telescope",
            "Telescope location is not set, but antenna positions are "
            "present. Mean antenna latitude and longitude values match file "
            "values, so telescope_position will be set using the mean of the "
            "antenna altitudes",
            "Telescope foo is not in known_telescopes.",
        ],
    ):
        uv_in.read(telescopefile, run_check=False)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_miriad_location_handling(paper_miriad_main, tmp_path):
    uv_in = paper_miriad_main
    uv_out = UVData()

    testfile = os.path.join(tmp_path, "outtest_miriad.uv")
    aipy_uv = aipy_extracts.UV(paper_miriad_file)

    if os.path.exists(testfile):
        shutil.rmtree(testfile)

    # Test for using antenna positions to get telescope position
    # extract antenna positions and rotate them for miriad
    nants = aipy_uv["nants"]
    rel_ecef_antpos = np.zeros((nants, 3), dtype=uv_in.antenna_positions.dtype)
    for ai, num in enumerate(uv_in.antenna_numbers):
        rel_ecef_antpos[num, :] = uv_in.antenna_positions[ai, :]

    # find zeros so antpos can be zeroed there too
    antpos_length = np.sqrt(np.sum(np.abs(rel_ecef_antpos) ** 2, axis=1))

    ecef_antpos = rel_ecef_antpos + uv_in.telescope_location
    longitude = uv_in.telescope_location_lat_lon_alt[1]
    antpos = uvutils.rotECEF_from_ECEF(ecef_antpos, longitude)

    # zero out bad locations (these are checked on read)
    antpos[np.where(antpos_length == 0), :] = [0, 0, 0]
    antpos = antpos.T.flatten() / const.c.to("m/ns").value

    # make new file
    aipy_uv2 = aipy_extracts.UV(testfile, status="new")
    # initialize headers from old file
    # change telescope name (so the position isn't set from known_telescopes)
    # and use absolute antenna positions
    aipy_uv2.init_from_uv(aipy_uv, override={"telescop": "foo", "antpos": antpos})
    # copy data from old file
    aipy_uv2.pipe(aipy_uv)
    aipy_uv2.close()

    with uvtest.check_warnings(
        UserWarning,
        [
            "Altitude is not present in Miriad file, and "
            "telescope foo is not in known_telescopes. "
            "Telescope location will be set using antenna positions.",
            "Altitude is not present ",
            "Telescope location is not set, but antenna "
            "positions are present. Mean antenna latitude "
            "and longitude values match file values, so "
            "telescope_position will be set using the mean "
            "of the antenna altitudes",
            "Telescope foo is not in known_telescopes.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv_out.read(testfile)

    # Test for handling when antenna positions have a different mean latitude
    # than the file latitude
    # make new file
    if os.path.exists(testfile):
        shutil.rmtree(testfile)
    aipy_uv = aipy_extracts.UV(paper_miriad_file)
    aipy_uv2 = aipy_extracts.UV(testfile, status="new")
    # initialize headers from old file
    # change telescope name (so the position isn't set from known_telescopes)
    # and use absolute antenna positions, change file latitude
    new_lat = aipy_uv["latitud"] * 1.5
    aipy_uv2.init_from_uv(
        aipy_uv, override={"telescop": "foo", "antpos": antpos, "latitud": new_lat}
    )
    # copy data from old file
    aipy_uv2.pipe(aipy_uv)
    aipy_uv2.close()

    with uvtest.check_warnings(
        UserWarning,
        [
            "Altitude is not present in Miriad file, and "
            "telescope foo is not in known_telescopes. "
            "Telescope location will be set using antenna positions.",
            "Altitude is not present in Miriad file, and "
            "telescope foo is not in known_telescopes. "
            "Telescope location will be set using antenna positions.",
            "Telescope location is set at sealevel at the "
            "file lat/lon coordinates. Antenna positions "
            "are present, but the mean antenna latitude "
            "value does not match",
            "drift RA, Dec is off from lst, latitude by more than 1.0 deg",
            "Telescope foo is not in known_telescopes.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv_out.read(testfile)

    # Test for handling when antenna positions have a different mean longitude
    # than the file longitude
    # this is harder because of the rotation that's done on the antenna positions
    # make new file
    if os.path.exists(testfile):
        shutil.rmtree(testfile)
    aipy_uv = aipy_extracts.UV(paper_miriad_file)
    aipy_uv2 = aipy_extracts.UV(testfile, status="new")
    # initialize headers from old file
    # change telescope name (so the position isn't set from known_telescopes)
    # and use absolute antenna positions, change file longitude
    new_lon = aipy_uv["longitu"] + np.pi
    aipy_uv2.init_from_uv(
        aipy_uv, override={"telescop": "foo", "antpos": antpos, "longitu": new_lon}
    )
    # copy data from old file
    aipy_uv2.pipe(aipy_uv)
    aipy_uv2.close()

    with uvtest.check_warnings(
        UserWarning,
        [
            "Altitude is not present in Miriad file, and "
            "telescope foo is not in known_telescopes. "
            "Telescope location will be set using antenna positions.",
            "Altitude is not present in Miriad file, and "
            "telescope foo is not in known_telescopes. "
            "Telescope location will be set using antenna positions.",
            "Telescope location is set at sealevel at the "
            "file lat/lon coordinates. Antenna positions "
            "are present, but the mean antenna longitude "
            "value does not match",
            "drift RA, Dec is off from lst, latitude by more than 1.0 deg",
            "Telescope foo is not in known_telescopes.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv_out.read(testfile)

    # Test for handling when antenna positions have a different mean longitude &
    # latitude than the file longitude
    # make new file
    if os.path.exists(testfile):
        shutil.rmtree(testfile)
    aipy_uv = aipy_extracts.UV(paper_miriad_file)
    aipy_uv2 = aipy_extracts.UV(testfile, status="new")
    # initialize headers from old file
    # change telescope name (so the position isn't set from known_telescopes)
    # and use absolute antenna positions, change file latitude and longitude
    aipy_uv2.init_from_uv(
        aipy_uv,
        override={
            "telescop": "foo",
            "antpos": antpos,
            "latitud": new_lat,
            "longitu": new_lon,
        },
    )
    # copy data from old file
    aipy_uv2.pipe(aipy_uv)
    aipy_uv2.close()

    with uvtest.check_warnings(
        UserWarning,
        [
            "Altitude is not present in Miriad file, and "
            "telescope foo is not in known_telescopes. "
            "Telescope location will be set using antenna positions.",
            "Altitude is not present in Miriad file, and "
            "telescope foo is not in known_telescopes. "
            "Telescope location will be set using antenna positions.",
            "Telescope location is set at sealevel at the "
            "file lat/lon coordinates. Antenna positions "
            "are present, but the mean antenna latitude and "
            "longitude values do not match",
            "drift RA, Dec is off from lst, latitude by more than 1.0 deg",
            "Telescope foo is not in known_telescopes.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv_out.read(testfile)

    # Test for handling when antenna positions are far enough apart to make the
    # mean position inside the earth

    good_antpos = np.where(antpos_length > 0)[0]
    rot_ants = good_antpos[: len(good_antpos) // 2]
    rot_antpos = uvutils.rotECEF_from_ECEF(ecef_antpos[rot_ants, :], longitude + np.pi)
    modified_antpos = uvutils.rotECEF_from_ECEF(ecef_antpos, longitude)
    modified_antpos[rot_ants, :] = rot_antpos
    # zero out bad locations (these are checked on read)
    modified_antpos[np.where(antpos_length == 0), :] = [0, 0, 0]
    modified_antpos = modified_antpos.T.flatten() / const.c.to("m/ns").value

    # make new file
    if os.path.exists(testfile):
        shutil.rmtree(testfile)
    aipy_uv = aipy_extracts.UV(paper_miriad_file)
    aipy_uv2 = aipy_extracts.UV(testfile, status="new")
    # initialize headers from old file
    # change telescope name (so the position isn't set from known_telescopes)
    # and use modified absolute antenna positions
    aipy_uv2.init_from_uv(
        aipy_uv, override={"telescop": "foo", "antpos": modified_antpos}
    )
    # copy data from old file
    aipy_uv2.pipe(aipy_uv)
    aipy_uv2.close()

    with uvtest.check_warnings(
        UserWarning,
        [
            "Altitude is not present in Miriad file, and "
            "telescope foo is not in known_telescopes. "
            "Telescope location will be set using antenna positions.",
            "Altitude is not present ",
            "Telescope location is set at sealevel at the "
            "file lat/lon coordinates. Antenna positions "
            "are present, but the mean antenna position "
            "does not give a telescope_location on the "
            "surface of the earth.",
            "Telescope foo is not in known_telescopes.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv_out.read(testfile)

    # cleanup
    aipy_uv.close()


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_singletimeselect_drift(uv_in_paper, tmp_path):
    """
    Check behavior with writing & reading after selecting a single time from
    a drift file.
    """
    uv_in, uv_out, testfile = uv_in_paper

    uv_in_copy = uv_in.copy()
    uv_in.select(times=uv_in.time_array[0])
    uv_in.write_miriad(testfile, clobber=True)
    uv_out.read(testfile)
    assert uv_in == uv_out

    # check that setting the phase_type works
    uv_out.read(testfile, phase_type="drift")
    assert uv_in == uv_out

    # check again with more than one time but only 1 unflagged time
    time_gt0_array = np.where(uv_in_copy.time_array > uv_in_copy.time_array[0])[0]
    uv_in_copy.flag_array[time_gt0_array, :, :, :] = True

    # get unflagged blts
    blt_good = np.where(~np.all(uv_in_copy.flag_array, axis=(1, 2, 3)))
    assert np.isclose(np.mean(np.diff(uv_in_copy.time_array[blt_good])), 0.0)

    uv_in_copy.write_miriad(testfile, clobber=True)
    uv_out.read(testfile)
    assert uv_in_copy == uv_out

    # check that setting the phase_type works
    uv_out.read(testfile, phase_type="drift")
    assert uv_in_copy == uv_out


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_poltoind(uv_in_paper):
    miriad_uv, uv_out, testfile = uv_in_paper

    pol_arr = miriad_uv.polarization_array

    miriad = miriad_uv._convert_to_filetype("miriad")
    miriad.polarization_array = None
    with pytest.raises(ValueError) as cm:
        miriad._pol_to_ind(pol_arr[0])
    assert str(cm.value).startswith(
        "Can't index polarization -7 because polarization_array is not set"
    )

    miriad.polarization_array = [pol_arr[0], pol_arr[0]]
    with pytest.raises(ValueError) as cm:
        miriad._pol_to_ind(pol_arr[0])
    assert str(cm.value).startswith("multiple matches for pol=-7 in polarization_array")


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "kwd_name,kwd_value,warnstr,errstr",
    (
        [
            "testdict",
            {"testkey": 23},
            "testdict in extra_keywords is a list, array or dict",
            "Extra keyword testdict is of <class 'dict'>",
        ],
        [
            "testlist",
            [12, 14, 90],
            "testlist in extra_keywords is a list, array or dict",
            "Extra keyword testlist is of <class 'list'>",
        ],
        [
            "testarr",
            np.array([12, 14, 90]),
            "testarr in extra_keywords is a list, array or dict",
            "Extra keyword testarr is of <class 'numpy.ndarray'>",
        ],
        [
            "test_long_key",
            True,
            "key test_long_key in extra_keywords is longer than 8 characters",
            None,
        ],
        [
            "complex1",
            np.complex64(5.3 + 1.2j),
            None,
            "Extra keyword complex1 is of <class 'numpy.complex64'>",
        ],
        [
            "complex2",
            6.9 + 4.6j,
            None,
            "Extra keyword complex2 is of <class 'complex'>",
        ],
    ),
)
def test_miriad_extra_keywords_errors(
    uv_in_paper, tmp_path, kwd_name, kwd_value, warnstr, errstr
):
    uv_in, uv_out, testfile = uv_in_paper

    uvw_warn_str = "The uvw_array does not match the expected values"
    # check for warnings & errors with extra_keywords that are dicts, lists or arrays
    uv_in.extra_keywords[kwd_name] = kwd_value
    if warnstr is None:
        warnstr_list = [uvw_warn_str]
    else:
        warnstr_list = [warnstr, uvw_warn_str]

    with uvtest.check_warnings(UserWarning, warnstr_list):
        uv_in.check()

    if errstr is not None:
        with pytest.raises(TypeError, match=errstr):
            uv_in.write_miriad(testfile, clobber=True, run_check=False)
    else:
        with uvtest.check_warnings(UserWarning, warnstr):
            uv_in.write_miriad(testfile, clobber=True, run_check=False)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "kwd_names,kwd_values",
    (
        [["bool", "bool2"], [True, False]],
        [["int1", "int2"], [np.int(5), 7]],
        [["float1", "float2"], [np.int64(5.3), 6.9]],
        [["str", "longstr"], ["hello", "this is a very long string " * 1000]],
    ),
)
def test_miriad_extra_keywords(uv_in_paper, tmp_path, kwd_names, kwd_values):
    uv_in, uv_out, testfile = uv_in_paper

    for name, value in zip(kwd_names, kwd_values):
        uv_in.extra_keywords[name] = value
    uv_in.write_miriad(testfile, clobber=True)
    uv_out.read(testfile)

    assert uv_in == uv_out


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_roundtrip_optional_params(uv_in_paper, tmp_path):
    uv_in, uv_out, testfile = uv_in_paper

    uv_in.x_orientation = "east"
    uv_in.reorder_blts()

    uv_in.write_miriad(testfile, clobber=True)
    uv_out.read(testfile)

    assert uv_in == uv_out

    # test with bda as well (single entry in tuple)
    uv_in.reorder_blts(order="bda")

    uv_in.write_miriad(testfile, clobber=True)
    uv_out.read(testfile)

    assert uv_in == uv_out


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_breakread_miriad(uv_in_paper, tmp_path):
    """Test Miriad file checking."""
    uv_in, uv_out, testfile = uv_in_paper

    with pytest.raises(IOError) as cm:
        uv_in.read("foo", file_type="miriad")
    assert str(cm.value).startswith("foo not found")

    uv_in_copy = uv_in.copy()
    uv_in_copy.Nblts += 10
    uv_in_copy.write_miriad(testfile, clobber=True, run_check=False)
    with uvtest.check_warnings(
        UserWarning, "Nblts does not match the number of unique blts in the data"
    ):
        uv_out.read(testfile, run_check=False)

    uv_in_copy = uv_in.copy()
    uv_in_copy.Nbls += 10
    uv_in_copy.write_miriad(testfile, clobber=True, run_check=False)
    with uvtest.check_warnings(
        UserWarning, "Nbls does not match the number of unique baselines in the data"
    ):
        uv_out.read(testfile, run_check=False)

    uv_in_copy = uv_in.copy()
    uv_in_copy.Ntimes += 10
    uv_in_copy.write_miriad(testfile, clobber=True, run_check=False)
    with uvtest.check_warnings(
        UserWarning, "Ntimes does not match the number of unique times in the data"
    ):
        uv_out.read(testfile, run_check=False)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_read_write_read_miriad(uv_in_paper):
    """
    PAPER file Miriad loopback test.

    Read in Miriad PAPER file, write out as new Miriad file, read back in and
    check for object equality.
    """
    uv_in, uv_out, write_file = uv_in_paper
    uv_out.read(write_file)

    assert uv_in == uv_out

    # check that we can read & write phased data
    uv_in2 = uv_in.copy()
    uv_in2.phase_to_time(Time(np.mean(uv_in2.time_array), format="jd"))
    uv_in2.write_miriad(write_file, clobber=True)
    uv_out.read(write_file)

    assert uv_in2 == uv_out
    del uv_in2

    # check that trying to overwrite without clobber raises an error
    with pytest.raises(OSError) as cm:
        uv_in.write_miriad(write_file, clobber=False)
    assert str(cm.value).startswith("File exists: skipping")

    # check that if x_orientation is set, it's read back out properly
    uv_in.x_orientation = "east"
    uv_in.write_miriad(write_file, clobber=True)
    uv_out.read(write_file)
    assert uv_in == uv_out


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_miriad_antenna_diameters(uv_in_paper):
    # check that if antenna_diameters is set, it's read back out properly
    uv_in, uv_out, write_file = uv_in_paper
    uv_in.antenna_diameters = np.zeros((uv_in.Nants_telescope,), dtype=np.float) + 14.0
    uv_in.write_miriad(write_file, clobber=True)
    uv_out.read(write_file)
    assert uv_in == uv_out

    # check that antenna diameters get written if not exactly float
    uv_in.antenna_diameters = (
        np.zeros((uv_in.Nants_telescope,), dtype=np.float32) + 14.0
    )
    uv_in.write_miriad(write_file, clobber=True)
    uv_out.read(write_file)
    assert uv_in == uv_out


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_miriad_write_miriad_unkonwn_phase_error(uv_in_paper):
    uv_in, uv_out, write_file = uv_in_paper
    # check that trying to write a file with unknown phasing raises an error
    uv_in._set_unknown_phase_type()
    with pytest.raises(ValueError, match="The phasing type of the data is unknown"):
        uv_in.write_miriad(write_file, clobber=True)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_miriad_write_read_diameters(tmp_path):
    uv_in = UVData()
    uv_out = UVData()
    write_file = os.path.join(tmp_path, "outtest_miriad.uv")
    # check for backwards compatibility with old keyword 'diameter' for
    # antenna diameters
    testfile_diameters = os.path.join(DATA_PATH, "zen.2457698.40355.xx.HH.uvcA")
    uv_in.read(testfile_diameters)
    uv_in.write_miriad(write_file, clobber=True)
    uv_out.read(write_file)
    assert uv_in == uv_out


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_miriad_and_aipy_reads(uv_in_paper):
    uv_in, uv_out, write_file = uv_in_paper
    # check that variables 'ischan' and 'nschan' were written to new file
    # need to use aipy, since pyuvdata is not currently capturing these variables
    uv_aipy = aipy_extracts.UV(write_file)
    nfreqs = uv_in.Nfreqs
    nschan = uv_aipy["nschan"]
    ischan = uv_aipy["ischan"]
    assert nschan == nfreqs
    assert ischan == 1

    # cleanup
    uv_aipy.close()


def test_miriad_telescope_locations():
    # test load_telescope_coords w/ blank Miriad
    uv_in = Miriad()
    uv = aipy_extracts.UV(paper_miriad_file)
    uv_in._load_telescope_coords(uv)
    assert uv_in.telescope_location_lat_lon_alt is not None
    uv.close()
    # test load_antpos w/ blank Miriad
    uv_in = Miriad()
    uv = aipy_extracts.UV(paper_miriad_file)
    uv_in._load_antpos(uv)
    assert uv_in.antenna_positions is not None


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_miriad_integration_time_precision(uv_in_paper):
    uv_in, uv_out, write_file = uv_in_paper

    # test that changing precision of integraiton_time is okay
    # tolerance of integration_time (1e-3) is larger than floating point type
    # conversions
    uv_in.integration_time = uv_in.integration_time.astype(np.float32)
    uv_in.write_miriad(write_file, clobber=True)
    new_uv = UVData()
    new_uv.read(write_file)
    assert uv_in == new_uv


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "select_kwargs",
    [
        {"bls": [(0, 0), (0, 1), (4, 2)]},
        {"bls": [(0, 0), (2, 4)], "antenna_nums": [0, 2, 4]},
        {"bls": (2, 4, "xy")},
        {"bls": [(4, 2, "yx")]},
        {"polarizations": [-7], "bls": [(4, 4)]},
        {"bls": [(4, 4, "xy")]},
    ],
)
def test_read_write_read_miriad_partial_bls(uv_in_paper, select_kwargs, tmp_path):
    # check partial read selections
    full, uv_out, write_file = uv_in_paper

    full.write_miriad(write_file, clobber=True)
    uv_in = UVData()

    # test only specified bls were read, and that flipped antpair is loaded too
    uv_in.read(write_file, **select_kwargs)
    antpairs = uv_in.get_antpairs()
    # indexing here is to ignore polarization if present, maybe there is a better way
    bls = select_kwargs["bls"]
    if isinstance(bls, tuple):
        bls = [bls]
    assert np.all([bl[:2] in antpairs or bl[:2][::-1] in antpairs for bl in bls])
    exp_uv = full.select(inplace=False, **select_kwargs)
    assert uv_in == exp_uv


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_read_write_read_miriad_partial_antenna_nums(uv_in_paper, tmp_path):
    full, uv_out, write_file = uv_in_paper

    # check partial read selections
    full.write_miriad(write_file, clobber=True)
    uv_in = UVData()
    # test all bls w/ 0 are loaded
    uv_in.read(write_file, antenna_nums=[0])
    diff = set(full.get_antpairs()) - set(uv_in.get_antpairs())
    assert 0 not in np.unique(diff)
    exp_uv = full.select(antenna_nums=[0], inplace=False)
    assert np.max(exp_uv.ant_1_array) == 0
    assert np.max(exp_uv.ant_2_array) == 0
    assert uv_in == exp_uv


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "select_kwargs",
    [
        {"time_range": [2456865.607, 2456865.609]},
        {"time_range": [2456865.607, 2456865.609], "antenna_nums": [0]},
        {"time_range": [2456865.607, 2456865.609], "polarizations": [-7]},
    ],
)
def test_read_write_read_miriad_partial_times(uv_in_paper, select_kwargs, tmp_path):
    full, uv_out, write_file = uv_in_paper

    # check partial read selections
    full.write_miriad(write_file, clobber=True)
    # test time loading
    uv_in = UVData()
    uv_in.read(write_file, **select_kwargs)
    full_times = np.unique(
        full.time_array[
            (full.time_array > select_kwargs["time_range"][0])
            & (full.time_array < select_kwargs["time_range"][1])
        ]
    )
    assert np.isclose(np.unique(uv_in.time_array), full_times).all()
    # The exact time are calculated above, pop out the time range to compare
    # time range with selecting on exact times
    select_kwargs.pop("time_range", None)
    exp_uv = full.select(times=full_times, inplace=False, **select_kwargs)
    assert uv_in == exp_uv


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("pols", [["xy"], [-7]])
def test_read_write_read_miriad_partial_pols(uv_in_paper, pols, tmp_path):
    full, uv_out, write_file = uv_in_paper

    # check partial read selections
    full.write_miriad(write_file, clobber=True)

    # test polarization loading
    uv_in = UVData()
    uv_in.read(write_file, polarizations=pols)
    assert full.polarization_array == uv_in.polarization_array
    exp_uv = full.select(polarizations=pols, inplace=False)
    assert uv_in == exp_uv


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_read_write_read_miriad_partial_ant_str(uv_in_paper, tmp_path):
    full, uv_out, write_file = uv_in_paper

    # check partial read selections
    full.write_miriad(write_file, clobber=True)
    # test ant_str
    uv_in = UVData()
    uv_in.read(write_file, ant_str="auto")
    assert np.array([blp[0] == blp[1] for blp in uv_in.get_antpairs()]).all()
    exp_uv = full.select(ant_str="auto", inplace=False)
    assert uv_in == exp_uv

    uv_in.read(write_file, ant_str="cross")
    assert np.array([blp[0] != blp[1] for blp in uv_in.get_antpairs()]).all()
    exp_uv = full.select(ant_str="cross", inplace=False)
    assert uv_in == exp_uv

    uv_in.read(write_file, ant_str="all")
    assert uv_in == full


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "err_type,select_kwargs,err_msg",
    [
        (
            AssertionError,
            {"ant_str": "auto", "antenna_nums": [0, 1]},
            "ant_str must be None if antenna_nums or bls is not None",
        ),
        (ValueError, {"bls": "foo"}, "bls must be a list of tuples of antenna numbers"),
        (
            ValueError,
            {"bls": [[0, 1]]},
            "bls must be a list of tuples of antenna numbers",
        ),
        (
            ValueError,
            {"bls": [("foo", "bar")]},
            "bls must be a list of tuples of antenna numbers",
        ),
        (
            ValueError,
            {"bls": [("foo",)]},
            "bls tuples must be all length-2 or all length-3",
        ),
        (
            ValueError,
            {"bls": [(1, 2), (2, 3, "xx")]},
            "bls tuples must be all length-2 or all length-3",
        ),
        (
            ValueError,
            {"bls": [(2, 4, 0)]},
            "The third element in each bl must be a polarization string",
        ),
        (
            ValueError,
            {"bls": [(2, 4, "xy")], "polarizations": ["xy"]},
            "Cannot provide length-3 tuples and also specify polarizations.",
        ),
        (
            AssertionError,
            {"antenna_nums": np.array([(0, 10)])},
            "antenna_nums must be fed as a list of antenna number integers",
        ),
        (
            AssertionError,
            {"polarizations": "xx"},
            "pols must be a list of polarization strings or ints",
        ),
        (
            ValueError,
            {"polarizations": ["yy"]},
            "No data is present, probably as a result of select on read",
        ),
        (
            AssertionError,
            {"time_range": "foo"},
            "time_range must be a len-2 list of Julian Date floats",
        ),
        (
            AssertionError,
            {"time_range": [1, 2, 3]},
            "time_range must be a len-2 list of Julian Date floats",
        ),
        (
            AssertionError,
            {"time_range": ["foo", "bar"]},
            "time_range must be a len-2 list of Julian Date floats",
        ),
        (
            ValueError,
            {"time_range": [10.1, 10.2]},
            "No data is present, probably as a result of select on read",
        ),
        (AssertionError, {"ant_str": 0}, "ant_str must be fed as a string"),
    ],
)
def test_read_write_read_miriad_partial_errors(
    uv_in_paper, err_type, select_kwargs, err_msg, tmp_path
):
    full, uv_out, write_file = uv_in_paper

    # check partial read selections
    full.write_miriad(write_file, clobber=True)
    uv_in = UVData()

    with pytest.raises(err_type) as cm:
        uv_in.read(write_file, **select_kwargs)
    assert str(cm.value).startswith(err_msg)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_read_write_read_miriad_partial_error_special_cases(uv_in_paper, tmp_path):
    full, uv_out, write_file = uv_in_paper

    # check partial read selections
    full.write_miriad(write_file, clobber=True)
    uv_in = UVData()

    with pytest.raises(ValueError) as cm:
        uv_in.read(write_file, polarizations=[1.0])
    assert str(cm.value).startswith(
        "Polarization 1.0 cannot be converted to a polarization number"
    )


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_read_write_read_miriad_partial_with_warnings(uv_in_paper, tmp_path):
    full, uv_out, write_file = uv_in_paper

    # check partial read selections
    full.write_miriad(write_file, clobber=True)

    uv_in = UVData()
    # check handling for generic read selections unsupported by read_miriad
    unique_times = np.unique(full.time_array)
    times_to_keep = unique_times[
        ((unique_times > 2456865.607) & (unique_times < 2456865.609))
    ]
    with uvtest.check_warnings(
        UserWarning,
        [
            "Warning: a select on read keyword is set",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv_in.read(write_file, times=times_to_keep)

    exp_uv = full.select(times=times_to_keep, inplace=False)
    assert uv_in == exp_uv

    uv_in = UVData()
    # check handling for generic read selections unsupported by read_miriad
    blts_select = np.where(full.time_array == unique_times[0])[0]
    ants_keep = [0, 2, 4]
    with uvtest.check_warnings(
        UserWarning,
        [
            "Warning: blt_inds is set along with select on read",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv_in.read(write_file, blt_inds=blts_select, antenna_nums=ants_keep)
    exp_uv = full.select(blt_inds=blts_select, antenna_nums=ants_keep, inplace=False)
    assert uv_in != exp_uv


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_read_write_read_miriad_partial_metadata_only(uv_in_paper, tmp_path):
    uv_in, uv_out, write_file = uv_in_paper
    write_file2 = os.path.join(tmp_path, "outtest_miriad2.uv")

    # try metadata only read
    uv_in_meta = UVData()
    uv_in_meta.read(paper_miriad_file, read_data=False)
    assert uv_in_meta.time_array is None
    assert uv_in_meta.data_array is None
    assert uv_in_meta.integration_time is None
    metadata = [
        "antenna_positions",
        "antenna_names",
        "antenna_positions",
        "channel_width",
        "history",
        "vis_units",
        "telescope_location",
    ]
    for m in metadata:
        assert getattr(uv_in_meta, m) is not None

    # metadata only multiple file read-in
    del uv_in_meta

    new_uv = uv_in.select(freq_chans=np.arange(5), inplace=False)
    new_uv.write_miriad(write_file, clobber=True)
    new_uv = uv_in.select(freq_chans=np.arange(5) + 5, inplace=False)
    new_uv.write_miriad(write_file2, clobber=True)

    uv_in.select(freq_chans=np.arange(10))

    uv_in2 = UVData()
    uv_in2.read(np.array([write_file, write_file2]))

    assert uv_in.history != uv_in2.history
    uv_in2.history = uv_in.history
    assert uv_in == uv_in2


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_read_ms_write_miriad_casa_history(tmp_path):
    """
    Read in .ms file.
    Write to a miriad file, read back in and check for history parameter
    """
    pytest.importorskip("casacore")
    ms_uv = UVData()
    miriad_uv = UVData()
    ms_file = os.path.join(DATA_PATH, "day2_TDEM0003_10s_norx_1src_1spw.ms")
    testfile = os.path.join(tmp_path, "outtest_miriad")
    ms_uv.read(ms_file)

    ms_uv.write_miriad(testfile, clobber=True)
    miriad_uv.read(testfile)

    assert miriad_uv == ms_uv


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_rwr_miriad_antpos_issues(uv_in_paper, tmp_path):
    """
    test warnings and errors associated with antenna position issues in Miriad files

    Read in Miriad PAPER file, mess with various antpos issues and write out as
    a new Miriad file, read back in and check for appropriate behavior.
    """
    uv_in, uv_out, write_file = uv_in_paper

    uv_in_copy = uv_in.copy()
    uv_in_copy.antenna_positions = None
    uv_in_copy.write_miriad(write_file, clobber=True, run_check=False)
    with uvtest.check_warnings(
        UserWarning, "Antenna positions are not present in the file.", nwarnings=2
    ):
        uv_out.read(write_file, run_check=False)

    assert uv_in_copy == uv_out

    uv_in_copy = uv_in.copy()
    ants_with_data = list(set(uv_in_copy.ant_1_array).union(uv_in_copy.ant_2_array))
    ant_ind = np.where(uv_in_copy.antenna_numbers == ants_with_data[0])[0]
    uv_in_copy.antenna_positions[ant_ind, :] = [0, 0, 0]
    uv_in_copy.write_miriad(write_file, clobber=True, no_antnums=True)
    with uvtest.check_warnings(
        UserWarning,
        [
            "antenna number",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv_out.read(write_file)

    assert uv_in_copy == uv_out

    uv_in.antenna_positions = None
    ants_with_data = sorted(set(uv_in.ant_1_array).union(uv_in.ant_2_array))
    new_nums = []
    new_names = []
    for a in ants_with_data:
        new_nums.append(a)
        ind = np.where(uv_in.antenna_numbers == a)[0][0]
        new_names.append(uv_in.antenna_names[ind])
    uv_in.antenna_numbers = np.array(new_nums)
    uv_in.antenna_names = new_names
    uv_in.Nants_telescope = len(uv_in.antenna_numbers)
    uv_in.write_miriad(write_file, clobber=True, no_antnums=True, run_check=False)
    with uvtest.check_warnings(
        UserWarning, "Antenna positions are not present in the file.", nwarnings=2
    ):
        uv_out.read(write_file, run_check=False)

    assert uv_in == uv_out


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_multi_files(casa_uvfits, tmp_path):
    """
    Reading multiple files at once.
    """
    uv_full = casa_uvfits
    testfile1 = os.path.join(tmp_path, "uv1")
    testfile2 = os.path.join(tmp_path, "uv2")
    # rename telescope to avoid name warning
    uv_full.unphase_to_drift()
    uv_full.conjugate_bls("ant1<ant2")

    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(32, 64))
    uv1.write_miriad(testfile1, clobber=True)
    uv2.write_miriad(testfile2, clobber=True)
    del uv1
    uv1 = UVData()
    uv1.read([testfile1, testfile2], file_type="miriad")

    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific frequencies using pyuvdata. "
        "Combined data along frequency axis using"
        " pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # again, setting axis
    del uv1
    uv1 = UVData()
    uv1.read([testfile1, testfile2], axis="freq")
    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific frequencies using pyuvdata. "
        "Combined data along frequency axis using"
        " pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1 == uv_full


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_antpos_units(casa_uvfits, tmp_path):
    """
    Read uvfits, write miriad. Check written antpos are in ns.
    """
    uv = casa_uvfits
    testfile = os.path.join(tmp_path, "uv_antpos_units")
    uv.write_miriad(testfile, clobber=True)
    auv = aipy_extracts.UV(testfile)
    aantpos = auv["antpos"].reshape(3, -1).T * const.c.to("m/ns").value
    aantpos = aantpos[uv.antenna_numbers, :]
    aantpos = (
        uvutils.ECEF_from_rotECEF(aantpos, uv.telescope_location_lat_lon_alt[1])
        - uv.telescope_location
    )
    assert np.allclose(aantpos, uv.antenna_positions)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_readmiriad_write_miriad_check_time_format(tmp_path):
    """
    test time_array is converted properly from Miriad format
    """
    from pyuvdata.uvdata import aipy_extracts

    # test read-in
    fname = os.path.join(DATA_PATH, "zen.2457698.40355.xx.HH.uvcA")
    uvd = UVData()
    uvd.read(fname)
    uvd_t = uvd.time_array.min()
    uvd_l = uvd.lst_array.min()
    uv = aipy_extracts.UV(fname)
    uv_t = uv["time"] + uv["inttime"] / (24 * 3600.0) / 2

    lat, lon, alt = uvd.telescope_location_lat_lon_alt
    t1 = Time(uv["time"], format="jd", location=(lon, lat))
    dt = TimeDelta(uv["inttime"] / 2, format="sec")
    t2 = t1 + dt
    lsts = uvutils.get_lst_for_time(np.array([t1.jd, t2.jd]), lat, lon, alt)
    delta_lst = lsts[1] - lsts[0]
    uv_l = uv["lst"] + delta_lst

    # assert starting time array and lst array are shifted by half integration
    assert np.isclose(uvd_t, uv_t)

    # avoid errors if IERS table is too old (if the iers url is down)
    tolerance = 1e-8
    assert np.allclose(uvd_l, uv_l, atol=tolerance)
    # test write-out
    fout = os.path.join(tmp_path, "ex_miriad")
    uvd.write_miriad(fout, clobber=True)
    # assert equal to original miriad time
    uv2 = aipy_extracts.UV(fout)
    assert np.isclose(uv["time"], uv2["time"])
    assert np.isclose(uv["lst"], uv2["lst"], atol=tolerance)


def test_file_with_bad_extra_words():
    """Test file with bad extra words is iterated and popped correctly."""
    fname = os.path.join(DATA_PATH, "test_miriad_changing_extra.uv")
    uv = UVData()
    warn_message = [
        "Altitude is not present in Miriad file, "
        "using known location values for PAPER.",
        "Mean of empty slice.",
        "invalid value encountered in double_scalars",
        "npols=4 but found 1 pols in data file",
        "Mean of empty slice.",
        "invalid value encountered in double_scalars",
        "antenna number 0 has visibilities associated with it, but it has a "
        "position of (0,0,0)",
        "antenna number 26 has visibilities associated with it, "
        "but it has a position of (0,0,0)",
    ]
    warn_category = (
        [UserWarning]
        + [RuntimeWarning] * 2
        + [UserWarning]
        + [RuntimeWarning] * 2
        + [UserWarning] * 2
    )
    # This is an old PAPER file, run_check must be set to false
    # The antenna positions is (0, 0, 0) vector
    with uvtest.check_warnings(warn_category, warn_message):
        uv.read_miriad(fname, run_check=False)
