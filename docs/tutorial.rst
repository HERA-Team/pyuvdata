Tutorial
========

In the examples below, output files are written to the
``pyuvdata/pyuvdata/data/tutorial_output`` directory, which you will need to
create to run the code exactly as written. Alternatively you can change the
location the output files are saved to by changing the arguments to the
``os.path.join`` calls.

------
UVData
------

UVData: File conversion
-----------------------
Converting between tested data formats

a) miriad -> uvfits
*******************
::

  >>> import os
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> UV = UVData()

  # This miriad file is known to be a drift scan.
  # Use the `read` method, optionally specify the file type. Can also use the
  # file type specific `read_miriad` method, but only if reading a single file.
  >>> miriad_file = os.path.join(DATA_PATH, 'new.uvA')
  >>> UV.read(miriad_file)
  >>> UV.read(miriad_file, file_type='miriad')
  >>> UV.read_miriad(miriad_file)

  # Write out the uvfits file
  >>> write_file = os.path.join(DATA_PATH, 'tutorial_output', 'tutorial.uvfits')
  >>> UV.write_uvfits(write_file, force_phase=True, spoof_nonessential=True)
  The data are in drift mode and do not have a defined phase center. Phasing to zenith of the first timestamp.

b) uvfits -> miriad
*******************
::

  >>> import os
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> import shutil
  >>> UV = UVData()
  >>> uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')

  # Use the `read` method, optionally specify the file type. Can also use the
  # file type specific `read_uvfits` method, but only if reading a single file.
  >>> UV.read(uvfits_file)
  >>> UV.read(uvfits_file, file_type='uvfits')
  >>> UV.read_uvfits(uvfits_file)

  # Write out the miriad file
  >>> write_file = os.path.join(DATA_PATH, 'tutorial_output', 'tutorial.uv')
  >>> if os.path.exists(write_file):
  ...    shutil.rmtree(write_file)
  >>> UV.write_miriad(write_file)

c) FHD -> uvfits
****************
When reading FHD format, we need to point to several files for each observation.
::

  >>> import os
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> UV = UVData()

  # Construct the list of files
  >>> fhd_prefix = os.path.join(DATA_PATH, 'fhd_vis_data/1061316296_')
  >>> fhd_files = [fhd_prefix + f for f in ['flags.sav', 'vis_XX.sav', 'params.sav',
  ...                                       'vis_YY.sav', 'vis_model_XX.sav',
  ...                                       'vis_model_YY.sav', 'settings.txt',
  ...                                       'layout.sav']]

  # Use the `read` method, optionally specify the file type. Can also use the
  # file type specific `read_fhd` method, but only if reading a single observation.
  >>> UV.read(fhd_files)
  >>> UV.read(fhd_files, file_type='fhd')
  >>> UV.read_fhd(fhd_files)
  >>> write_file = os.path.join(DATA_PATH, 'tutorial_output', 'tutorial.uvfits')
  >>> UV.write_uvfits(write_file, spoof_nonessential=True)

d) FHD -> miriad
****************
::

  >>> import os
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> import shutil
  >>> import os
  >>> UV = UVData()

  # Construct the list of files
  >>> fhd_prefix = os.path.join(DATA_PATH, 'fhd_vis_data/1061316296_')
  >>> fhd_files = [fhd_prefix + f for f in ['flags.sav', 'vis_XX.sav', 'params.sav',
  ...                                       'vis_YY.sav', 'vis_model_XX.sav',
  ...                                       'vis_model_YY.sav', 'settings.txt',
  ...                                       'layout.sav']]
  >>> UV.read(fhd_files)
  >>> write_file = os.path.join(DATA_PATH, 'tutorial_output','tutorial.uv')
  >>> if os.path.exists(write_file):
  ...    shutil.rmtree(write_file)
  >>> UV.write_miriad(write_file)

e) CASA -> uvfits
*****************
::

  >>> import os
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> UV = UVData()
  >>> ms_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.ms')

  # Use the `read` method, optionally specify the file type. Can also use the
  # file type specific `read_ms` method, but only if reading a single file.
  # note that reading CASA measurement sets requires casacore to be installed
  >>> UV.read(ms_file)
  >>> UV.read(ms_file, file_type='ms')
  >>> UV.read_ms(ms_file)

  # Write out uvfits file
  >>> write_file = os.path.join(DATA_PATH, 'tutorial_output', 'tutorial.uvfits')
  >>> UV.write_uvfits(write_file, spoof_nonessential=True)

f) CASA -> miriad
*****************
::

  >>> import os
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> import shutil
  >>> import os
  >>> UV=UVData()
  >>> ms_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.ms')

  # note that reading CASA measurement sets requires casacore to be installed
  >>> UV.read(ms_file)

  # Write out Miriad file
  >>> write_file = os.path.join(DATA_PATH, 'tutorial_output', 'tutorial.uv')
  >>> if os.path.exists(write_file):
  ...    shutil.rmtree(write_file)
  >>> UV.write_miriad(write_file)

g) miriad -> uvh5
*****************
::

  >>> import os
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> UV = UVData()

  # This miriad file is known to be a drift scan.
  >>> miriad_file = os.path.join(DATA_PATH, 'new.uvA')
  >>> UV.read(miriad_file)

  # Write out the uvh5 file
  >>> UV.write_uvh5(os.path.join(DATA_PATH, 'tutorial_output', 'tutorial.uvh5'))

h) uvfits -> uvh5
*****************
::

  >>> import os
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> import os
  >>> UV = UVData()
  >>> uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
  >>> UV.read(uvfits_file)

  # Write out the uvh5 file
  >>> write_file = os.path.join(DATA_PATH, 'tutorial_output', 'tutorial.uvh5')
  >>> if os.path.exists(write_file):
  ...    os.remove(write_file)
  >>> UV.write_uvh5(write_file)

  # Read the uvh5 file back in.
  # Use the `read` method, optionally specify the file type. Can also use the
  # file type specific `read_uvh5` method, but only if reading a single file.
  >>> UV.read(write_file)
  >>> UV.read(write_file, file_type='uvh5')
  >>> UV.read_uvh5(write_file)

i) MWA correlator -> uvfits
***************************
The MWA correlator writes FITS files containing the correlator dumps (but
lacking metadata and not conforming to the uvfits format). pyuvdata can read
these files along with MWA metafits files (containing the required metadata)
into a UVData object which can then be written out to uvfits or any other
supported file type. There are also options for applying cable length corrections,
dividing out digital gains, dividing out the coarse band shape, common flagging patterns,
using cotter flag files, and phasing the data to the pointing center. It is also
optional to apply a Van Vleck correction to the data. The default for this correction is
to use a Chebyshev polynomial approximation, and there is an option to instead use a
slower integral implementation.

::

  >>> import os
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> UV = UVData()

  # Construct the list of files
  >>> data_path = os.path.join(DATA_PATH, 'mwa_corr_fits_testfiles/')
  >>> filelist = [data_path + i for i in ['1131733552.metafits',
  ... '1131733552_20151116182537_mini_gpubox01_00.fits']]

  # Use the `read` method, optionally specify the file type. Can also use the
  # file type specific `read_mwa_corr_fits` method, but only if reading files
  # from a single observation.
  # Apply cable corrections and phase data before writing to uvfits
  # Skip routine time/frequency flagging - see flag_init and associated keywords in documentation
  >>> UV.read(filelist, correct_cable_len=True, phase_to_pointing_center=True, flag_init=False)
  >>> UV.read(filelist, file_type='mwa_corr_fits', correct_cable_len=True, phase_to_pointing_center=True, flag_init=False)
  >>> UV.read_mwa_corr_fits(filelist, correct_cable_len=True, phase_to_pointing_center=True, flag_init=False)

  # Write out uvfits file
  >>> write_file = os.path.join(DATA_PATH, 'tutorial_output', 'tutorial.uvfits')
  >>> UV.write_uvfits(write_file, spoof_nonessential=True)


UVData: Quick data access
-------------------------
A small suite of functions are available to quickly access numpy arrays of data,
flags, and nsamples.

a) Data for single antenna pair / polarization combination.
************************************************************
::

  >>> import os
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> import numpy as np
  >>> UV = UVData()
  >>> filename = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
  >>> UV.read(filename)
  >>> data = UV.get_data(1, 2, 'rr')  # data for ant1=1, ant2=2, pol='rr'
  >>> times = UV.get_times(1, 2)  # times corresponding to 0th axis in data
  >>> print(data.shape)
  (9, 64)
  >>> print(times.shape)
  (9,)

  # One can equivalently make any of these calls with the input wrapped in a tuple.
  >>> data = UV.get_data((1, 2, 'rr'))
  >>> times = UV.get_times((1, 2))

b) Flags and nsamples for above data.
*************************************
::

  >>> flags = UV.get_flags(1, 2, 'rr')
  >>> nsamples = UV.get_nsamples(1, 2, 'rr')
  >>> print(flags.shape)
  (9, 64)
  >>> print(nsamples.shape)
  (9, 64)

c) Data for single antenna pair, all polarizations.
***************************************************
::

  >>> data = UV.get_data(1, 2)
  >>> print(data.shape)
  (9, 64, 4)

  # Can also give baseline number
  >>> data2 = UV.get_data(UV.antnums_to_baseline(1, 2))
  >>> print(np.all(data == data2))
  True

d) Data for single polarization, all baselines.
***********************************************
::

  >>> data = UV.get_data('rr')
  >>> print(data.shape)
  (1360, 64)

e) Iterate over all antenna pair / polarizations.
*************************************************
::

  >>> for key, data in UV.antpairpol_iter():
  ...  flags = UV.get_flags(key)
  ...  nsamples = UV.get_nsamples(key)

    # Do something with the data, flags, nsamples

f) Convenience functions to ask what antennas, baselines, and pols are in the data.
***********************************************************************************
::

  # Get all unique antennas in data
  >>> print(UV.get_ants())
  [ 0  1  2  3  6  7  8 11 14 18 19 20 21 22 23 24 26 27]

  # Get all baseline nums in data, print first 10.
  >>> print(UV.get_baseline_nums()[0:10])
  [67586 67587 67588 67591 67592 67593 67596 67599 67603 67604]

  # Get all (ordered) antenna pairs in data (same info as baseline_nums), print first 10.
  >>> print(UV.get_antpairs()[0:10])
  [(0, 1), (0, 2), (0, 3), (0, 6), (0, 7), (0, 8), (0, 11), (0, 14), (0, 18), (0, 19)]

  # Get all antenna pairs and polariations, i.e. keys produced in UV.antpairpol_iter(), print first 5.
  >>> print(UV.get_antpairpols()[0:5])
  [(0, 1, 'rr'), (0, 1, 'll'), (0, 1, 'rl'), (0, 1, 'lr'), (0, 2, 'rr')]

g) Quick access to file attributes of a UV* object (UVData, UVCal, UVBeam)
**************************************************************************
::

  ## in bash ##
  # Print data_array.shape to stdout
  pyuvdata_inspect.py --attr=data_array.shape <uv*_file>

  # Print Ntimes,Nfreqs,Nbls to stdout
  pyuvdata_inspect.py --attr=Ntimes,Nfreqs,Nbls <uv*_file>

  # Load object to instance name "uv" and will remain in interpreter
  pyuvdata_inspect.py -i <uv*_file>

UVData: Phasing
---------------
Phasing/unphasing data
::

  >>> import os
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> from astropy.time import Time
  >>> UV = UVData()
  >>> miriad_file = os.path.join(DATA_PATH, 'new.uvA')
  >>> UV.read(miriad_file)
  >>> print(UV.phase_type)
  drift

  # Phase the data to the zenith at first time step. Can either be specified
  # as a astropy Time object or as a float which is taken to be in JD.
  >>> UV.phase_to_time(Time(UV.time_array[0], format='jd'))
  >>> print(UV.phase_type)
  phased

  # Undo phasing
  >>> UV.unphase_to_drift()
  >>> print(UV.phase_type)
  drift

  # Phase the data to the zenith at first time step using float JD.
  >>> UV.phase_to_time(UV.time_array[0])
  >>> print(UV.phase_type)
  phased

  # Rephase to another phase center (unphases and rephases under the hood)
  # Phase to a specific ra/dec/epoch (in radians)
  >>> UV.phase(5.23368, 0.710940, epoch="J2000")


UVData: Averaging and Resampling
--------------------------------
pyuvdata has methods to average (downsample) in time and frequency and also to
upsample in time (useful to get all baselines on the shortest time integration
for a data set that has had baseline dependent time averaging applied).

Use the ``downsample_in_time``, ``upsample_in_time`` and ``resample_in_time`` methods to
average (downsample) and upsample in time or to do both at once on data
that have had baseline dependent averaging (BDA) applied to put all the baselines
on the same time integrations. Resampling in time is done on phased data by default,
drift mode data are phased, resampled, and then unphased. Set ``allow_drift=True``
to do resampling without phasing.

Use the ``frequency_average`` method to average along the frequency axis.

a) Averaging (Downsampling) in time
***********************************
Use either the `n_times_to_avg` keyword to specify an integer factor to average
by or `min_int_time` to specify a minimum final integration time. Specifying
`min_int_time` is most appropriate when the integration time varies, e.g. if
the data have had baseline-dependent averaging applied.

::

  >>> import os
  >>> import numpy as np
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> uv_object = UVData()
  >>> datafile = os.path.join(DATA_PATH, "zen.2458661.23480.HH.uvh5")
  >>> uv_object.read(datafile)
  >>> uv_object2 = uv_object.copy()
  >>> print("Range of integration times: ", np.amin(uv_object.integration_time),
  ...       "-", np.amax(uv_object.integration_time))
  Range of integration times:  1.879048192 - 1.879048192

  # first use n_times_to_avg to average by a factor of 2 in time.
  >>> uv_object.downsample_in_time(n_times_to_avg=2)
  Data are in drift mode, phasing before resampling.
  Unphasing back to drift mode.

  >>> print("Range of integration times after downsampling: ", np.amin(uv_object.integration_time),
  ...       "-", np.amax(uv_object.integration_time))
  Range of integration times after downsampling:  3.758096384 - 3.758096384

  # Now use min_int_time to average by a factor of 2 in time.
  >>> min_integration_time = np.amax(uv_object2.integration_time) * 2.0
  >>> uv_object2.downsample_in_time(min_int_time=min_integration_time)
  Data are in drift mode, phasing before resampling.
  Unphasing back to drift mode.

  >>> print("Range of integration times after downsampling: ", np.amin(uv_object2.integration_time),
  ...       "-", np.amax(uv_object2.integration_time))
  Range of integration times after downsampling:  3.758096384 - 3.758096384


b) Upsampling in time
*********************
::

  >>> import os
  >>> import numpy as np
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> uv_object = UVData()
  >>> datafile = os.path.join(DATA_PATH, "zen.2458661.23480.HH.uvh5")
  >>> uv_object.read(datafile)
  >>> print("Range of integration times: ", np.amin(uv_object.integration_time),
  ...       "-", np.amax(uv_object.integration_time))
  Range of integration times:  1.879048192 - 1.879048192

  >>> max_integration_time = np.amin(uv_object.integration_time) / 2.0
  >>> uv_object.upsample_in_time(max_integration_time)
  Data are in drift mode, phasing before resampling.
  Unphasing back to drift mode.

  >>> print("Range of integration times after upsampling: ", np.amin(uv_object.integration_time),
  ...       "-", np.amax(uv_object.integration_time))
  Range of integration times after upsampling:  0.939524096 - 0.939524096

c) Resampling a BDA dataset in time
***********************************
::

  >>> import os
  >>> import numpy as np
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> uv_object = UVData()
  >>> testfile = os.path.join(DATA_PATH, "simulated_bda_file.uvh5")
  >>> uv_object.read(testfile)
  >>> print("Range of integration times: ", np.amin(uv_object.integration_time),
  ...       "-", np.amax(uv_object.integration_time))
  Range of integration times:  2.0 - 16.0

  # Resample all baselines to an 8s integration time
  >>> uv_object.resample_in_time(8)
  >>> print("Range of integration times after resampling: ", np.amin(uv_object.integration_time),
  ...       "-", np.amax(uv_object.integration_time))
  Range of integration times after resampling:  8.0 - 8.0

d) Averaging in frequency
*************************
::

  >>> import os
  >>> import numpy as np
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> uv_object = UVData()
  >>> datafile = os.path.join(DATA_PATH, "zen.2458661.23480.HH.uvh5")
  >>> uv_object.read(datafile)
  >>> print("Channel width: ", uv_object.channel_width)
  Channel width:  122070.3125

  # Average by a factor of 2 in frequency
  >>> uv_object.frequency_average(2)
  >>> print("Channel width after frequency averaging: ", uv_object.channel_width)
  Channel width after frequency averaging:  244140.625

UVData: Plotting
----------------
Making a simple waterfall plot.

Note: there is now support for reading in only part of a uvfits, uvh5 or miriad file
(see :ref:`large_files`), so you need not read in the
entire file to plot one waterfall.
::

  >>> import os
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> import numpy as np
  >>> import matplotlib.pyplot as plt # doctest: +SKIP
  >>> UV = UVData()
  >>> filename = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
  >>> UV.read(filename)

  # Note that the length of the array along axis=1 is always 1.
  >>> print(UV.data_array.shape)
  (1360, 1, 64, 4)
  >>> print(UV.Ntimes)
  15
  >>> print(UV.Nfreqs)
  64
  >>> bl = UV.antnums_to_baseline(1, 2)
  >>> print(bl)
  69635
  >>> bl_ind = np.where(UV.baseline_array == bl)[0]

  # Amplitude waterfall for all spectral channels and 0th polarization
  >>> plt.imshow(np.abs(UV.data_array[bl_ind, 0, :, 0])) # doctest: +SKIP
  >>> plt.show() # doctest: +SKIP

  # If using flexible spectral windows (flex_spw=True), plot the 0th window, 0th pol
  >>> if UV.flex_spw: # doctest: +SKIP
  ...     window_sel = UV.flex_spw_id_array == 0 # doctest: +SKIP
  ...     plt.imshow(np.abs(UV.data_array[bl_ind, 0, window_sel, 0])) # doctest: +SKIP
  ...     plt.show() # doctest: +SKIP

  # Update: With new UI features, making waterfalls is easier than ever!
  >>> plt.imshow(np.abs(UV.get_data((1, 2, UV.polarization_array[0])))) # doctest: +SKIP
  >>> plt.show() # doctest: +SKIP


UVData: Location conversions
----------------------------
A number of conversion methods exist to map between different coordinate systems
for locations on the earth.

a) Getting antenna positions in topocentric frame in units of meters
********************************************************************
::

  # directly from UVData object
  >>> import os
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> uvd = UVData()
  >>> data_file = os.path.join(DATA_PATH, 'new.uvA')
  >>> uvd.read(data_file)
  >>> antpos, ants = uvd.get_ENU_antpos()

  # using utils
  >>> from pyuvdata import utils

  # get antennas positions in ECEF
  >>> antpos = uvd.antenna_positions + uvd.telescope_location

  # convert to topocentric (East, North, Up or ENU) coords.
  >>> antpos = utils.ENU_from_ECEF(antpos, *uvd.telescope_location_lat_lon_alt)

UVData: Selecting data
----------------------
The select method lets you select specific antennas (by number or name),
antenna pairs, frequencies (in Hz or by channel number), times (or time_range)
or polarizations to keep in the object while removing others.

Note: The same select interface is now supported on the read for uvfits, uvh5
and miriad files (see :ref:`large_files`), so you need not
read in the entire file before doing the select.

a) Select 3 antennas to keep using the antenna number.
******************************************************
::

  >>> import os
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> import numpy as np
  >>> UV = UVData()
  >>> filename = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
  >>> UV.read(filename)

  # print all the antennas numbers with data in the original file
  >>> print(np.unique(UV.ant_1_array.tolist() + UV.ant_2_array.tolist()))
  [ 0  1  2  3  6  7  8 11 14 18 19 20 21 22 23 24 26 27]
  >>> UV.select(antenna_nums=[0, 11, 20])

  # print all the antennas numbers with data after the select
  >>> print(np.unique(UV.ant_1_array.tolist() + UV.ant_2_array.tolist()))
  [ 0 11 20]

b) Select 3 antennas to keep using the antenna names, also select 5 frequencies to keep.
****************************************************************************************
::

  >>> import os
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> import numpy as np
  >>> UV = UVData()
  >>> filename = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
  >>> UV.read(filename)

  # print all the antenna names with data in the original file
  >>> unique_ants = np.unique(UV.ant_1_array.tolist() + UV.ant_2_array.tolist())
  >>> print([UV.antenna_names[np.where(UV.antenna_numbers==a)[0][0]] for a in unique_ants])
  ['W09', 'E02', 'E09', 'W01', 'N06', 'N01', 'E06', 'E08', 'W06', 'W04', 'N05', 'E01', 'N04', 'E07', 'W05', 'N02', 'E03', 'N08']

  # print how many frequencies in the original file
  >>> print(UV.freq_array.size)
  64
  >>> UV.select(antenna_names=['N02', 'E09', 'W06'], frequencies=UV.freq_array[0,0:4])

  # print all the antenna names with data after the select
  >>> unique_ants = np.unique(UV.ant_1_array.tolist() + UV.ant_2_array.tolist())
  >>> print([UV.antenna_names[np.where(UV.antenna_numbers==a)[0][0]] for a in unique_ants])
  ['E09', 'W06', 'N02']

  # print all the frequencies after the select
  >>> print(UV.freq_array)
  [[3.6304542e+10 3.6304667e+10 3.6304792e+10 3.6304917e+10]]

c) Select a few antenna pairs to keep
*************************************
::

  >>> import os
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> UV = UVData()
  >>> filename = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
  >>> UV.read(filename)

  # print how many antenna pairs with data in the original file
  >>> print(len(set(zip(UV.ant_1_array, UV.ant_2_array))))
  153
  >>> UV.select(bls=[(0, 2), (6, 0), (0, 21)])

  # note that order of the values in the pair does not matter
  # print all the antenna pairs after the select
  >>> print(sorted(set(zip(UV.ant_1_array, UV.ant_2_array))))
  [(0, 2), (0, 6), (0, 21)]

d) Select antenna pairs using baseline numbers
**********************************************
::

  >>> import os
  >>> import numpy as np
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> UV = UVData()
  >>> filename = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
  >>> UV.read(filename)

  # baseline numbers can be found in the baseline_array
  >>> print(len(UV.baseline_array))
  1360

  # select baselines using the baseline numbers
  >>> UV.select(bls=[73736, 73753, 81945])

  # print unique baselines and antennas after select
  >>> print(np.unique(UV.baseline_array))
  [73736 73753 81945]
  >>> print(list(set(zip(UV.ant_1_array, UV.ant_2_array))))
  [(3, 24), (3, 7), (7, 24)]

e) Select antenna pairs and polarizations using ant_str argument
****************************************************************

Basic options are 'auto', 'cross', or 'all'. 'auto' returns just the
autocorrelations (all pols), while 'cross' returns just the cross-correlations
(all pols).  The ant_str can also contain:

1. Individual antenna number(s):
________________________________

- 1: returns all antenna pairs containing antenna number 1 (including the auto correlation)
- 1,2: returns all antenna pairs containing antennas 1 and/or 2

::

  >>> import os
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> UV = UVData()
  >>> filename = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
  >>> UV.read(filename)

  # Print the number of antenna pairs in the original file
  >>> print(len(UV.get_antpairs()))
  153

  # Apply select to UV object
  >>> UV.select(ant_str='1,2,3')

  # Print the number of antenna pairs after the select
  >>> print(len(UV.get_antpairs()))
  48

2. Individual baseline(s):
___________________________

- 1_2: returns only the antenna pair (1,2)
- 1_2,1_3,1_10: returns antenna pairs (1,2),(1,3),(1,10)
- (1,2)_3: returns antenna pairs (1,3),(2,3)
- 1_(2,3): returns antenna pairs (1,2),(1,3)

::

  >>> import os
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> UV = UVData()
  >>> filename = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
  >>> UV.read(filename)

  # Print the number of antenna pairs in the original file
  >>> print(len(UV.get_antpairs()))
  153

  # Apply select to UV object
  >>> UV.select(ant_str='(1,2)_(3,6)')

  # Print the antennas pairs with data after the select
  >>> print(UV.get_antpairs())
  [(1, 3), (1, 6), (2, 3), (2, 6)]

3. Antenna number(s) and polarization(s):
__________________________________________

When polarization information is passed with antenna numbers,
all antenna pairs kept in the object will retain data for each specified polarization

- 1x: returns all antenna pairs containing antenna number 1 and polarizations xx and xy
- 2x_3y: returns the antenna pair (2,3) and polarization xy
- 1r_2l,1l_3l,1r_4r: returns antenna pairs (1,2), (1,3), (1,4) and polarizations rr, ll, and rl.  This yields a complete list of baselines with polarizations of 1r_2l, 1l_2l, 1r_2r, 1r_3l, 1l_3l, 1r_3r, 1r_11l, 1l_11l, and 1r_11r.
- (1x,2y)_(3x,4y): returns antenna pairs (1,3),(1,4),(2,3),(2,4) and polarizations xx, yy, xy, and yx
- 2l_3: returns antenna pair (2,3) and polarizations ll and lr
- 2r_3: returns antenna pair (2,3) and polarizations rr and rl
- 1l_3,2x_3: returns antenna pairs (1,3), (2,3) and polarizations ll, lr, xx, and xy
- 1_3l,2_3x: returns antenna pairs (1,3), (2,3) and polarizations ll, rl, xx, and yx

::

  >>> import os
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> UV = UVData()
  >>> filename = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
  >>> UV.read(filename)

  # Print the number of antennas and polarizations with data in the original file
  >>> print((len(UV.get_antpairs()), UV.get_pols()))
  (153, ['rr', 'll', 'rl', 'lr'])

  # Apply select to UV object
  >>> UV.select(ant_str='1r_2l,1l_3l,1r_6r')

  # Print all the antennas numbers and polarizations with data after the select
  >>> print((UV.get_antpairs(), UV.get_pols()))
  ([(1, 2), (1, 3), (1, 6)], ['rr', 'll', 'rl'])

4. Stokes parameter(s):
________________________

Can be passed lowercase or uppercase

- i,I: keeps only Stokes I
- q,V: keeps both Stokes Q and V

5. Minus sign(s):
________________________

If a minus sign is present in front of an antenna number, it will not be kept in the data

- 1,-3: returns all antenna pairs containing antenna 1, but removes any containing antenna 3
- 1,-1_3: returns all antenna pairs containing antenna 1, except the antenna pair (1,3)
- 1x_(-3y,10x): returns antenna pair (1,10) and polarization xx

::

  >>> import os
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> UV = UVData()
  >>> filename = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
  >>> UV.read(filename)

  # Print the number of antenna pairs in the original file
  >>> print(len(UV.get_antpairs()))
  153

  # Apply select to UV object
  >>> UV.select(ant_str='1,-1_3')

  # Print the number of antenna pairs with data after the select
  >>> print(len(UV.get_antpairs()))
  16

e) Select data and return new object (leaving original intact).
***************************************************************
::

  >>> import os
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> import numpy as np
  >>> UV = UVData()
  >>> filename = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
  >>> UV.read(filename)
  >>> UV2 = UV.select(antenna_nums=[0, 11, 20], inplace=False)

  # print all the antennas numbers with data in the original file
  >>> print(np.unique(UV.ant_1_array.tolist() + UV.ant_2_array.tolist()))
  [ 0  1  2  3  6  7  8 11 14 18 19 20 21 22 23 24 26 27]

  # print all the antennas numbers with data after the select
  >>> print(np.unique(UV2.ant_1_array.tolist() + UV2.ant_2_array.tolist()))
  [ 0 11 20]

UVData: Combining and concatenating data
----------------------------------------
The __add__ method lets you combine UVData objects along
the baseline-time, frequency, and/or polarization axis.

a) Combine frequencies.
***********************
::

  >>> import os
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> import numpy as np
  >>> uv1 = UVData()
  >>> filename = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
  >>> uv1.read(filename)
  >>> uv2 = uv1.copy()

  # Downselect frequencies to recombine
  >>> uv1.select(freq_chans=np.arange(0, 32))
  >>> uv2.select(freq_chans=np.arange(32, 64))
  >>> uv3 = uv1 + uv2
  >>> print((uv1.Nfreqs, uv2.Nfreqs, uv3.Nfreqs))
  (32, 32, 64)

b) Combine times.
*****************
::

  >>> import os
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> import numpy as np
  >>> uv1 = UVData()
  >>> filename = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
  >>> uv1.read(filename)
  >>> uv2 = uv1.copy()

  # Downselect times to recombine
  >>> times = np.unique(uv1.time_array)
  >>> uv1.select(times=times[0:len(times) // 2])
  >>> uv2.select(times=times[len(times) // 2:])
  >>> uv3 = uv1 + uv2
  >>> print((uv1.Ntimes, uv2.Ntimes, uv3.Ntimes))
  (7, 8, 15)
  >>> print((uv1.Nblts, uv2.Nblts, uv3.Nblts))
  (459, 901, 1360)

c) Combine in place.
********************
The following two commands are equivalent, and act on uv1
directly without creating a third uvdata object.
::

  >>> import os
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> import numpy as np
  >>> uv1 = UVData()
  >>> filename = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
  >>> uv1.read(filename)
  >>> uv2 = uv1.copy()
  >>> uv1.select(times=times[0:len(times) // 2])
  >>> uv2.select(times=times[len(times) // 2:])
  >>> uv1.__add__(uv2, inplace=True)

  >>> uv1.read(filename)
  >>> uv2 = uv1.copy()
  >>> uv1.select(times=times[0:len(times) // 2])
  >>> uv2.select(times=times[len(times) // 2:])
  >>> uv1 += uv2

d) Reading multiple files.
**************************
If the ``read`` method is given a list of files
(or list of lists for FHD or MWA correlator files), each file will be read in succession
and combined with the previous file(s).
::

  >>> import os
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> uv = UVData()
  >>> filename = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
  >>> uv.read(filename)
  >>> uv1 = uv.select(freq_chans=np.arange(0, 20), inplace=False)
  >>> uv2 = uv.select(freq_chans=np.arange(20, 40), inplace=False)
  >>> uv3 = uv.select(freq_chans=np.arange(40, 64), inplace=False)
  >>> uv1.write_uvfits(os.path.join(DATA_PATH, 'tutorial_output', 'tutorial1.uvfits'))
  >>> uv2.write_uvfits(os.path.join(DATA_PATH, 'tutorial_output', 'tutorial2.uvfits'))
  >>> uv3.write_uvfits(os.path.join(DATA_PATH, 'tutorial_output', 'tutorial3.uvfits'))
  >>> filenames = [os.path.join(DATA_PATH, 'tutorial_output', f) for f
  ...             in ['tutorial1.uvfits', 'tutorial2.uvfits', 'tutorial3.uvfits']]
  >>> uv.read(filenames)

e) Fast concatenation
*********************
As an alternative to the ``__add__`` operation, the ``fast_concat`` method can
be used. The user specifies a UVData object to combine with the existing one,
along with the axis along which they should be combined. Fast concatenation can
be invoked implicitly when reading in multiple files as above by passing the
``axis`` keyword argument. This will use the ``fast_concat`` method instead of
the ``__add__`` method to combine the data contained in the files into a single
UVData object.

**WARNING**: There is no guarantee that two objects combined in this fashion
will result in a self-consistent object after concatenation. Basic checking is
done, but time-consuming robust check are eschewed for the sake of speed. The
data will also *not* be reordered or sorted as part of the concatenation, and so
this must be done manually by the user if a reordering is desired
(see :ref:`sorting_data`).

The ``fast_concat`` method is significantly faster than ``__add__``, especially
for large UVData objects. Preliminary benchmarking shows that reading in
time-ordered visibilities from disk using the ``axis`` keyword argument can
improve throughput by nearly an order of magnitude for 100 HERA data files
stored in the uvh5 format.
::

  >>> import os
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> uv = UVData()
  >>> filename = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
  >>> uv.read(filename)
  >>> uv1 = uv.select(freq_chans=np.arange(0, 20), inplace=False)
  >>> uv2 = uv.select(freq_chans=np.arange(20, 40), inplace=False)
  >>> uv3 = uv.select(freq_chans=np.arange(40, 64), inplace=False)
  >>> uv1.write_uvfits(os.path.join(DATA_PATH, 'tutorial_output', 'tutorial1.uvfits'))
  >>> uv2.write_uvfits(os.path.join(DATA_PATH, 'tutorial_output', 'tutorial2.uvfits'))
  >>> uv3.write_uvfits(os.path.join(DATA_PATH, 'tutorial_output', 'tutorial3.uvfits'))
  >>> filenames = [os.path.join(DATA_PATH, 'tutorial_output', f) for f
  ...             in ['tutorial1.uvfits', 'tutorial2.uvfits', 'tutorial3.uvfits']]
  >>> uv.read(filenames, axis='freq')


UVData: Summing and differencing visibilities
---------------------------------------------
Simple summing and differencing of visibilities can be done with the ``sum_vis``
and ``diff_vis`` methods.
::

  >>> import os
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> filename = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
  >>> uv1 = UVData()
  >>> uv1.read(filename)
  >>> uv2 = uv1.copy()

  # sum visibilities
  >>> uv1 = uv1.sum_vis(uv2)

  # diff visibilities
  >>> uv1 = uv1.diff_vis(uv2)

  # in place option
  >>> uv1.sum_vis(uv2, inplace=True)

  # override a particular parameter
  >>> uv1.instrument = "test instrument"
  >>> uv1.sum_vis(uv2, inplace=True, override_params=["instrument"])

.. _large_files:

UVData: Working with large files
--------------------------------
To save on memory and time, pyuvdata supports reading only parts of uvfits, uvh5,
miriad, and MWA correlator fits files.

a) Reading just the metadata of a file
******************************************
For uvh5, uvfits, FHD and MWA correlator fits files, reading in the only the
metadata results in a metadata only
UVData object (which has every attribute except the data_array,
flag_array and nsample_array filled out). For Miriad files, less of the
metadata can be read without reading the data, but many of the attributes
are available.

Measurement set (ms) files do not support
reading only the metadata
(the read_data keyword is ignored for ms files).
::

  >>> import os
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> uv = UVData()
  >>> filename = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')

  # read the metadata but not the data
  >>> uv.read(filename, read_data=False)

  >>> print(uv.metadata_only)
  True

  >>> print(uv.time_array.size)
  1360

  >>> print(uv.data_array)
  None

b) Reading only parts of uvfits, uvh5 or miriad data
****************************************************
The same options that are available for the select function can also be passed to
the read method to do the select on the read, saving memory and time if only a
portion of the data are needed.

Note that these keywords can be used for any file type, but for FHD,
MWA correlator FITS files, and
measurement set (ms) files, the select is done after the read, which does not
save memory. Miriad only supports some of the selections on the read, the
unsupported ones are done after the read.
Any of the select keywords can be used for any file type, but selects for keywords
that are not supported by the select on read for a given file type will be
done after the read, which does not save memory.
::

  >>> import os
  >>> import numpy as np
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> uv = UVData()
  >>> filename = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
  >>> uv.read(filename, freq_chans=np.arange(32))
  >>> print(uv.data_array.shape)
  (1360, 1, 32, 4)

  # Reading in the metadata can help with specifying what data to read in
  >>> uv = UVData()
  >>> uv.read(filename, read_data=False)
  >>> unique_times = np.unique(uv.time_array)
  >>> print(unique_times.shape)
  (15,)

  >>> times_to_keep = unique_times[[0, 2, 4]]
  >>> uv.read(filename, times=times_to_keep)
  >>> print(uv.data_array.shape)
  (179, 1, 64, 4)

  # Select a few baselines from a miriad file
  >>> filename = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcA')
  >>> uv.read(filename, bls=[(9, 10), (9, 20)])
  >>> print(uv.get_antpairs())
  [(9, 10), (9, 20)]

  # Select certain frequencies from a uvh5 file
  >>> filename = os.path.join(DATA_PATH, "zen.2458661.23480.HH.uvh5")
  >>> uv.read(filename, freq_chans=np.arange(2))
  >>> print(uv.data_array.shape)
  (200, 1, 2, 2)

c) Writing to a uvh5 file in parts
**********************************

It is possible to write to a uvh5 file in parts, so not all of the file needs to
be in memory at once. This is very useful when combined with partial reading
described above, so that operations that in principle require all of the data,
such as applying calibration solutions, can be performed even in situations where
the available memory is smaller than the size of the file.

Partial writing requires two steps: initializing an empty file on disk with the
correct metadata for the final object, and then subsequently writing the data in
stages to that same file. In this latter stage, the same syntax for performing a
selective read operation is used, so that the user can precisely specify which
parts of the data, flags, and nsample arrays should be written to. The user then
also provides the data, flags, and nsample arrays of the proper size, and they
are written to the appropriate parts of the file on disk.
::

  >>> import os
  >>> import numpy as np
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> uv = UVData()
  >>> filename = os.path.join(DATA_PATH, "zen.2458661.23480.HH.uvh5")
  >>> uv.read(filename, read_data=False)
  >>> partfile = os.path.join(DATA_PATH, 'tutorial_output', 'tutorial_partial_io.uvh5')
  >>> uv.initialize_uvh5_file(partfile, clobber=True)

  # read in the lower and upper halves of the band separately, and apply different scalings
  >>> Nfreqs = uv.Nfreqs
  >>> Hfreqs = Nfreqs // 2
  >>> freq_inds1 = np.arange(Hfreqs)
  >>> freq_inds2 = np.arange(Hfreqs, Nfreqs)
  >>> uv2 = UVData()
  >>> uv2.read(filename, freq_chans=freq_inds1)
  >>> data_array = 0.5 * uv2.data_array
  >>> flag_array = uv2.flag_array
  >>> nsample_array = uv2.nsample_array
  >>> uv.write_uvh5_part(partfile, data_array, flag_array, nsample_array, freq_chans=freq_inds1)

  >>> uv2.read(filename, freq_chans=freq_inds2)
  >>> data_array = 2.0 * uv2.data_array
  >>> flag_array = uv2.flag_array
  >>> nsample_array = uv2.nsample_array
  >>> uv.write_uvh5_part(partfile, data_array, flag_array, nsample_array, freq_chans=freq_inds2)


.. _sorting_data:

UVData: Sorting data along various axes
---------------------------------------
A few methods exist for sorting (and conjugating) data along various axes to
support comparisons between UVData objects and software access patterns.

a) Conjugating baselines
************************

The :meth:`pyuvdata.UVData.conjugate_bls` method will conjugate baselines to conform to various
conventions (``'ant1<ant2'``, ``'ant2<ant1'``, ``'u<0'``, ``'u>0'``, ``'v<0'``, ``'v>0'``) or it can just
conjugate a set of specific baseline-time indices.

::

  >>> import os
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> uv = UVData()
  >>> uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
  >>> uv.read(uvfits_file)
  >>> uv.conjugate_bls('ant1<ant2')
  >>> print(np.min(uv.ant_2_array - uv.ant_1_array) >= 0)
  True

  >>> uv2.conjugate_bls(convention='u<0', use_enu=False)
  >>> print(np.max(uv2.uvw_array[:, 0]) <= 0)
  True

b) Sorting along the baseline-time axis
***************************************

The :meth:`pyuvdata.UVData.reorder_blts` method will reorder the baseline-time axis by sorting by ``'time'``,
``'baseline'``, ``'ant1'`` or ``'ant2'`` or according to an order preferred for data that
have baseline dependent averaging ``'bda'``. A user can also just specify a desired
order by passing an array of baseline-time indices.

::

  >>> import os
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> uv = UVData()
  >>> uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
  >>> uv.read(uvfits_file)

  # The default is to sort first by time, then by baseline
  >>> uv.reorder_blts()
  >>> print(np.min(np.diff(uv.time_array)) >= 0)
  True

  # Explicity sorting by 'time' then 'baseline' gets the same result
  >>> uv2 = uv.copy()
  >>> uv2.reorder_blts(order='time', minor_order='baseline')
  >>> print(uv == uv2)
  True

  >>> uv.reorder_blts(order='ant1', minor_order='ant2')
  >>> print(np.min(np.diff(uv.ant_1_array)) >= 0)
  True

  # You can also sort and conjugate in a single step for the purposes of comparing two objects
  >>> uv.reorder_blts(order='bda', conj_convention='ant1<ant2')
  >>> uv2.reorder_blts(order='bda', conj_convention='ant1<ant2')
  >>> print(uv == uv2)
  True

c) Sorting along the polarization axis
**************************************

The :meth:`pyuvdata.UVData.reorder_pols` method will reorder the polarization axis either following
the ``'AIPS'`` or ``'CASA'`` convention, or by an explicit index ordering set by the user.

::

  >>> import os
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> import pyuvdata.utils as uvutils
  >>> uv = UVData()
  >>> uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
  >>> uv.read(uvfits_file)
  >>> print(uvutils.polnum2str(uv.polarization_array))
  ['rr', 'll', 'rl', 'lr']

  >>> uv.reorder_pols('CASA')
  >>> print(uvutils.polnum2str(uv.polarization_array))
  ['rr', 'rl', 'lr', 'll']

UVData: Working with Redundant Baselines
----------------------------------------

a) Finding Redundant Baselines
******************************
:mod:`utils` contains functions for finding redundant groups of baselines in
an array, either by antenna positions or uvw coordinates. Baselines are
considered redundant if they are within a specified tolerance distance (default is 1 meter).

The :func:`utils.get_baseline_redundancies` function accepts an array of baseline indices
and an array of baseline vectors (ie, uvw coordinates) as input, and finds
redundancies among the vectors as given. If the ``with_conjugates`` option is
selected, it will include baselines that are redundant when reversed in the same group.
In this case, a list of ``conjugates`` is returned as well,
which contains indices for the baselines that were flipped for the redundant groups.
In either mode of operation, this will only return baseline indices that are in the list passed in.

The :func:`utils.get_antenna_redundancies` function accepts an array of
antenna indices and an array of antenna positions as input, defines baseline vectors
and indices in the convention that ``ant1<ant2``, and runs
:func:`utils.get_baseline_redundancies` to find redundant baselines. It will then apply the conjugates
list to the groups it finds.

There is a subtle difference between the purposes of the two functions. `utils.get_antenna_redundancies`
gives you all redundant baselines from the antenna positions, and does not necessarily reflect the baselines
in a file. This is similar to what is written in the `hera_cal` package.
Alternatively, `utils.get_baseline_redundancies` may be given the actual
baseline vectors in a file and it will search for redundancies among those.

The method :meth:`~pyuvdata.UVData.get_redundancies` is provided as a convenience. If
run with the `use_antpos` option, it will mimic the behavior of `utils.get_antenna_redundancies`.
Otherwise it will return redundancies in the existing data using `utils.get_baseline_redundancies`.
If run with `use_antpos` and the `conjugate_bls` option, it will also adjust the data_array and baseline_array
so that the baselines in the returned groups correspond with the baselines listed on the object (i.e., except for
antenna pairs with no associated data).

::

  >>> import os
  >>> import numpy as np
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> from pyuvdata import utils as uvutils
  >>> uvd = UVData()

  # This file contains a HERA19 layout.
  >>> uvd.read(os.path.join(DATA_PATH, 'fewant_randsrc_airybeam_Nsrc100_10MHz.uvfits'))
  >>> uvd.unphase_to_drift(use_ant_pos=True)
  >>> tol = 0.05  # Tolerance in meters
  >>> uvd.select(times=uvd.time_array[0])

  # Returned values: list of redundant groups, corresponding mean baseline vectors, baseline lengths. No conjugates included, so conjugates is None.
  >>> baseline_groups, vec_bin_centers, lengths = uvutils.get_baseline_redundancies(uvd.baseline_array, uvd.uvw_array, tol=tol)
  >>> print(len(baseline_groups))
  19

  # The with_conjugates option includes baselines that are redundant when reversed.
  # If used, the conjugates list will contain a list of indices of baselines that must be flipped to be redundant.
  >>> baseline_groups, vec_bin_centers, lengths, conjugates = uvutils.get_baseline_redundancies(uvd.baseline_array, uvd.uvw_array, tol=tol, with_conjugates=True)
  >>> print(len(baseline_groups))
  19

  # Using antenna positions instead
  >>> antpos, antnums = uvd.get_ENU_antpos()
  >>> baseline_groups, vec_bin_centers, lengths = uvutils.get_antenna_redundancies(antnums, antpos, tol=tol, include_autos=True)
  >>> print(len(baseline_groups))
  20

  # get_antenna_redundancies has the option to ignore autocorrelations.
  >>> baseline_groups, vec_bin_centers, lengths = uvutils.get_antenna_redundancies(antnums, antpos, tol=tol, include_autos=False)
  >>> print(len(baseline_groups))
  19

b) Compressing/inflating on Redundant Baselines
***********************************************
Since redundant baselines should have similar visibilities, some level of data
compression can be achieved by only keeping one out of a set of redundant baselines.
The :meth:`~pyuvdata.UVData.compress_by_redundancy` method will find groups of baselines that are
redundant to a given tolerance and either average over them or select a single
baseline from the redundant group. If the data are identical between redundant
baselines (e.g. if they are from a noiseless simulation) the "select" method
should be used as it is much faster. If the "average" method is used, the data
are combined with a weighted average using the `nsample_array` as weights
and the final `nsample_array` will be a sum of the `nsample_array` of the
combined baselines (so it can be larger than 1).

This action is (almost) inverted by the :meth:`~pyuvdata.UVData.inflate_by_redundancy`
method, which finds all possible baselines from the antenna positions and fills
in the full data array based on redundancy.

::

  >>> import os
  >>> from pyuvdata import UVData
  >>> from pyuvdata.data import DATA_PATH
  >>> import numpy as np
  >>> uv0 = UVData()
  >>> uv0.read(os.path.join(DATA_PATH, 'fewant_randsrc_airybeam_Nsrc100_10MHz.uvfits'))
  >>> tol = 0.02   # In meters

  # Compression can be run in-place or return a separate UVData object.
  >>> uv_backup = uv0.copy()
  >>> uv2 = uv0.compress_by_redundancy(method="select", tol=tol, inplace=False)
  >>> uv0.compress_by_redundancy(method="select", tol=tol)
  >>> uv2 == uv0
  True

  # Note -- Compressing and inflating changes the baseline order, reorder before comparing.
  >>> uv0.inflate_by_redundancy(tol=tol)
  >>> uv_backup.reorder_blts(conj_convention="u>0", uvw_tol=tol)
  >>> uv0.reorder_blts()
  >>> np.all(uv0.baseline_array == uv_backup.baseline_array)
  True

  >>> uv2.inflate_by_redundancy(tol=tol)
  >>> uv2 == uv0
  True

------
UVCal
------

UVCal: Reading/writing
----------------------
Calibration files using UVCal.

a) Reading a cal fits gain calibration file.
********************************************
::

  >>> import os
  >>> from pyuvdata import UVCal
  >>> from pyuvdata.data import DATA_PATH
  >>> import numpy as np
  >>> import matplotlib.pyplot as plt # doctest: +SKIP
  >>> cal = UVCal()
  >>> filename = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.gain.calfits')
  >>> cal.read_calfits(filename)

  # Cal type:
  >>> print(cal.cal_type)
  gain

  # number of antenna polarizations and polarization type.
  >>> print((cal.Njones, cal.jones_array))
  (1, array([-5]))

  # Number of antennas with data
  >>> print(cal.Nants_data)
  19

  # Number of frequencies
  >>> print(cal.Nfreqs)
  10

  # Shape of the gain_array
  >>> print(cal.gain_array.shape)
  (19, 1, 10, 5, 1)

  # plot abs of all gains for first time and first jones polarization.
  >>> for ant in range(cal.Nants_data): # doctest: +SKIP
  ...    plt.plot(cal.freq_array.flatten(), np.abs(cal.gain_array[ant, 0, :, 0, 0]))
  >>> plt.xlabel('Frequency (Hz)') # doctest: +SKIP
  >>> plt.ylabel('Abs(gains)') # doctest: +SKIP
  >>> plt.show() # doctest: +SKIP


b) FHD cal to cal fits
***********************
::

  >>> import os
  >>> from pyuvdata import UVCal
  >>> from pyuvdata.data import DATA_PATH
  >>> import os
  >>> obs_testfile = os.path.join(DATA_PATH, 'fhd_cal_data/1061316296_obs.sav')
  >>> cal_testfile = os.path.join(DATA_PATH, 'fhd_cal_data/1061316296_cal.sav')
  >>> settings_testfile = os.path.join(DATA_PATH, 'fhd_cal_data/1061316296_settings.txt')

  >>> fhd_cal = UVCal()
  >>> fhd_cal.read_fhd_cal(cal_testfile, obs_testfile, settings_file=settings_testfile)
  >>> fhd_cal.write_calfits(os.path.join(DATA_PATH, 'tutorial_output', 'tutorial_cal.fits'), clobber=True)


UVCal: Quick data access
------------------------
Similar methods for quick data access are available for UVCal.
Note that because UVCal has a different gain_array shape,
the data output will have shape (Nfreqs, Ntimes).

a) Data for a single antenna and instrumental polarization
**********************************************************
::

  >>> import os
  >>> from pyuvdata import UVCal
  >>> from pyuvdata.data import DATA_PATH
  >>> import numpy as np
  >>> UVC = UVCal()
  >>> filename = os.path.join(DATA_PATH, 'zen.2457555.42443.HH.uvcA.omni.calfits')
  >>> UVC.read_calfits(filename)
  >>> gain = UVC.get_gains(9, 'Jxx')  # gain for ant=9, pol='Jxx'

  # One can equivalently make any of these calls with the input wrapped in a tuple.
  >>> gain = UVC.get_gains((9, 'Jxx'))

  # If no polarization is fed, then all polarizations are returned
  >>> gain = UVC.get_gains(9)

  # One can also request flags and quality arrays in a similar manner
  >>> flags = UVC.get_flags(9, 'Jxx')
  >>> quals = UVC.get_quality(9, 'Jxx')

UVCal: Calibrating UVData
-------------------------

a) Calibration of UVData by UVCal
*********************************
::

  # We can calibrate directly using a UVCal object
  >>> import os
  >>> from pyuvdata import UVData, UVCal, utils
  >>> from pyuvdata.data import DATA_PATH
  >>> UV = UVData()
  >>> UV.read(os.path.join(DATA_PATH, 'zen.2458116.30448.HH.uvh5'))
  >>> UVC = UVCal()
  >>> UVC.read_calfits(os.path.join(DATA_PATH, 'zen.2458116.30448.HH.flagged_abs.calfits'))
  >>> UV_calibrated = utils.uvcalibrate(UV, UVC, inplace=False)

  # We can also un-calibrate using the same UVCal
  >>> UV_uncalibrated = utils.uvcalibrate(UV_calibrated, UVC, inplace=False, undo=True)

UVCal: Selecting data
---------------------
The select method lets you select specific antennas (by number or name),
frequencies (in Hz or by channel number), times or polarizations
to keep in the object while removing others.

a) Select 3 antennas to keep using the antenna number.
******************************************************
::

  >>> import os
  >>> from pyuvdata import UVCal
  >>> from pyuvdata.data import DATA_PATH
  >>> import numpy as np
  >>> cal = UVCal()
  >>> filename = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.gain.calfits')
  >>> cal.read_calfits(filename)

  # print all the antennas numbers with data in the original file
  >>> print(cal.ant_array)
  [  9  10  20  22  31  43  53  64  65  72  80  81  88  89  96  97 104 105
   112]
  >>> cal.select(antenna_nums=[9, 22, 64])

  # print all the antennas numbers with data after the select
  >>> print(cal.ant_array)
  [ 9 22 64]

b) Select 3 antennas to keep using the antenna names, also select 5 frequencies to keep.
****************************************************************************************
::

  >>> import os
  >>> from pyuvdata import UVCal
  >>> from pyuvdata.data import DATA_PATH
  >>> import numpy as np
  >>> cal = UVCal()
  >>> filename = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.gain.calfits')
  >>> cal.read_calfits(filename)

  # print all the antenna names with data in the original file
  >>> print([cal.antenna_names[np.where(cal.antenna_numbers==a)[0][0]] for a in cal.ant_array[0:9]])
  ['ant9', 'ant10', 'ant20', 'ant22', 'ant31', 'ant43', 'ant53', 'ant64', 'ant65']

  # print all the frequencies in the original file
  >>> print(cal.freq_array)
  [[1.00000000e+08 1.00097656e+08 1.00195312e+08 1.00292969e+08
    1.00390625e+08 1.00488281e+08 1.00585938e+08 1.00683594e+08
    1.00781250e+08 1.00878906e+08]]
  >>> cal.select(antenna_names=['ant31', 'ant81', 'ant104'], freq_chans=np.arange(0, 4))

  # print all the antenna names with data after the select
  >>> print([cal.antenna_names[np.where(cal.antenna_numbers==a)[0][0]] for a in cal.ant_array])
  ['ant31', 'ant81', 'ant104']

  # print all the frequencies after the select
  >>> print(cal.freq_array)
  [[1.00000000e+08 1.00097656e+08 1.00195312e+08 1.00292969e+08]]


UVCal: Adding data
------------------
The __add__ method lets you combine UVCal objects along
the antenna, time, frequency, and/or polarization axis.

a) Add frequencies.
*******************
::

  >>> import os
  >>> from pyuvdata import UVCal
  >>> from pyuvdata.data import DATA_PATH
  >>> import numpy as np
  >>> cal1 = UVCal()
  >>> filename = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.gain.calfits')
  >>> cal1.read_calfits(filename)
  >>> cal2 = cal1.copy()

  # Downselect frequencies to recombine
  >>> cal1.select(freq_chans=np.arange(0, 5))
  >>> cal2.select(freq_chans=np.arange(5, 10))
  >>> cal3 = cal1 + cal2
  >>> print((cal1.Nfreqs, cal2.Nfreqs, cal3.Nfreqs))
  (5, 5, 10)

b) Add times.
****************
::

  >>> import os
  >>> from pyuvdata import UVCal
  >>> from pyuvdata.data import DATA_PATH
  >>> import numpy as np
  >>> cal1 = UVCal()
  >>> filename = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.gain.calfits')
  >>> cal1.read_calfits(filename)
  >>> cal2 = cal1.copy()

  # Downselect times to recombine
  >>> times = np.unique(cal1.time_array)
  >>> cal1.select(times=times[0:len(times) // 2])
  >>> cal2.select(times=times[len(times) // 2:])
  >>> cal3 = cal1 + cal2
  >>> print((cal1.Ntimes, cal2.Ntimes, cal3.Ntimes))
  (2, 3, 5)

c) Adding in place.
*******************
The following two commands are equivalent, and act on cal1
directly without creating a third uvcal object.
::

  >>> import os
  >>> from pyuvdata import UVCal
  >>> from pyuvdata.data import DATA_PATH
  >>> import numpy as np
  >>> cal1 = UVCal()
  >>> filename = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.gain.calfits')
  >>> cal1.read_calfits(filename)
  >>> cal2 = cal1.copy()
  >>> times = np.unique(cal1.time_array)
  >>> cal1.select(times=times[0:len(times) // 2])
  >>> cal2.select(times=times[len(times) // 2:])
  >>> cal1.__add__(cal2, inplace=True)

  >>> cal1.read_calfits(filename)
  >>> cal2 = cal1.copy()
  >>> cal1.select(times=times[0:len(times) // 2])
  >>> cal2.select(times=times[len(times) // 2:])
  >>> cal1 += cal2

d) Reading multiple files.
**************************
If any of the read methods (read_calfits, read_fhd_cal) are given a list of files,
each file will be read in succession and added to the previous.
::

  >>> import os
  >>> from pyuvdata import UVCal
  >>> from pyuvdata.data import DATA_PATH
  >>> import numpy as np
  >>> cal = UVCal()
  >>> filename = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.gain.calfits')
  >>> cal.read_calfits(filename)
  >>> cal1 = cal.select(freq_chans=np.arange(0, 2), inplace=False)
  >>> cal2 = cal.select(freq_chans=np.arange(2, 4), inplace=False)
  >>> cal3 = cal.select(freq_chans=np.arange(4, 7), inplace=False)
  >>> cal1.write_calfits(os.path.join(DATA_PATH, 'tutorial_output', 'tutorial1.fits'))
  >>> cal2.write_calfits(os.path.join(DATA_PATH, 'tutorial_output', 'tutorial2.fits'))
  >>> cal3.write_calfits(os.path.join(DATA_PATH, 'tutorial_output', 'tutorial3.fits'))
  >>> filenames = [os.path.join(DATA_PATH, 'tutorial_output', f) for f
  ...              in ['tutorial1.fits', 'tutorial2.fits', 'tutorial3.fits']]
  >>> cal.read_calfits(filenames)

  # For FHD cal datasets pass lists for each file type
  >>> fhd_cal = UVCal()
  >>> obs_testfiles = [os.path.join(DATA_PATH, f) for f
  ...                  in ['fhd_cal_data/1061316296_obs.sav',
  ...                      'fhd_cal_data/set2/1061316296_obs.sav']]
  >>> cal_testfiles = [os.path.join(DATA_PATH, f) for f
  ...                  in ['fhd_cal_data/1061316296_cal.sav',
  ...                      'fhd_cal_data/set2/1061316296_cal.sav']]
  >>> settings_testfiles = [os.path.join(DATA_PATH, f) for f
  ...                       in ['fhd_cal_data/1061316296_settings.txt',
  ...                           'fhd_cal_data/set2/1061316296_settings.txt']]
  >>> fhd_cal.read_fhd_cal(cal_testfiles, obs_testfiles, settings_file=settings_testfiles)
  diffuse_model parameter value is a string, values are different

UVCal: Changing cal_type from 'delay' to 'gain'
-----------------------------------------------
UVCal includes the method convert_to_gain(), which changes a UVCal object's
cal_type parameter from 'delay' to 'gain,' and accordingly sets the object's
gain_array to one consistent with its pre-existing delay_array.
::

  >>> import os
  >>> from pyuvdata import UVCal
  >>> from pyuvdata.data import DATA_PATH
  >>> cal = UVCal()

  # This file has a cal_type of 'delay'.
  >>> filename = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.delay.calfits')
  >>> cal.read_calfits(filename)
  >>> print(cal.cal_type)
  delay

  # But we can convert it to a 'gain' type calibration.
  >>> cal.convert_to_gain()
  >>> print(cal.cal_type)
  gain

  # If we want the calibration to use a positive value in its exponent, rather
  # than the default negative value:
  >>> cal = UVCal()
  >>> cal.read_calfits(filename)
  >>> cal = cal.convert_to_gain(delay_convention='plus')

  # Convert to gain *without* running the default check that internal arrays are
  # of compatible shapes:
  >>> cal = UVCal()
  >>> cal.read_calfits(filename)
  >>> cal.convert_to_gain(run_check=False)

  # Convert to gain *without* running the default check that optional parameters
  # are properly shaped and typed:
  >>> cal = UVCal()
  >>> cal.read_calfits(filename)
  >>> cal.convert_to_gain(check_extra=False)

  # Convert to gain *without* running the default checks on the reasonableness
  # of the resulting calibration's parameters.
  >>> cal = UVCal()
  >>> cal.read_calfits(filename)
  >>> cal.convert_to_gain(run_check_acceptability=False)

------
UVBeam
------


UVBeam: Reading/writing
-----------------------
Reading and writing beam files using UVBeam.

The text files saved out of CST beam simulations do not have much of the
critical metadata needed for UVBeam objects. When reading in CST files, you
can either provide the required metadata using keywords to the read_cst method
and pass the raw CST files, or you can pass a settings yaml file which lists
the raw files and provides the required metadata to the read_cst method. Both
options are shown in the examples below. More details on creating a new yaml
settings files can be found in :doc:`cst_settings_yaml`.

a) Reading a CST power beam file
********************************
::

  >>> import os
  >>> from pyuvdata import UVBeam
  >>> from pyuvdata.data import DATA_PATH
  >>> import numpy as np
  >>> import matplotlib.pyplot as plt # doctest: +SKIP
  >>> beam = UVBeam()

  # you can pass several filenames and the objects from each file will be
  # combined across the appropriate axis -- in this case frequency
  >>> filenames = [os.path.join(DATA_PATH, f) for f
  ...              in ['NicCSTbeams/HERA_NicCST_150MHz.txt',
  ...                  'NicCSTbeams/HERA_NicCST_123MHz.txt']]

  # You have to specify the telescope_name, feed_name, feed_version, model_name
  # and model_version because they are not included in the raw CST files.
  # You should also specify the polarization that the file represents and you can
  # set rotate_pol to generate the other polarization by rotating by 90 degrees.
  # The feed_pol defaults to 'x' and rotate_pol defaults to True.
  >>> beam.read_cst_beam(filenames, beam_type='power', frequency=[150e6, 123e6],
  ...                    feed_pol='x', rotate_pol=True, telescope_name='HERA',
  ...                    feed_name='PAPER_dipole', feed_version='0.1',
  ...                    model_name='E-field pattern - Rigging height 4.9m',
  ...                    model_version='1.0')
  >>> print(beam.beam_type)
  power
  >>> print(beam.pixel_coordinate_system)
  az_za
  >>> print(beam.data_normalization)
  physical

  # You can also use a yaml settings file.
  # Note that using a yaml file requires that pyyaml is installed.
  >>> settings_file = os.path.join(DATA_PATH, 'NicCSTbeams/NicCSTbeams.yaml')
  >>> beam.read_cst_beam(settings_file, beam_type='power')
  >>> print(beam.beam_type)
  power
  >>> print(beam.pixel_coordinate_system)
  az_za
  >>> print(beam.data_normalization)
  physical

  # number of beam polarizations and polarization type.
  >>> print((beam.Npols, beam.polarization_array))
  (2, array([-5, -6]))
  >>> print(beam.Nfreqs)
  2
  >>> print(beam.data_array.shape)
  (1, 1, 2, 2, 181, 360)

  # plot zenith angle cut through beam
  >>> plt.plot(beam.axis2_array, beam.data_array[0, 0, 0, 0, :, 0]) # doctest: +SKIP
  >>> plt.xscale('log') # doctest: +SKIP
  >>> plt.xlabel('Zenith Angle (radians)') # doctest: +SKIP
  >>> plt.ylabel('Power') # doctest: +SKIP
  >>> plt.show() # doctest: +SKIP

b) Reading a CST E-field beam file
**********************************
::

  >>> import os
  >>> from pyuvdata import UVBeam
  >>> from pyuvdata.data import DATA_PATH
  >>> import numpy as np
  >>> beam = UVBeam()

  # the same interface as for power beams, just specify beam_type = 'efield'
  >>> settings_file = os.path.join(DATA_PATH, 'NicCSTbeams/NicCSTbeams.yaml')
  >>> beam.read_cst_beam(settings_file, beam_type='efield')
  >>> print(beam.beam_type)
  efield

c) Reading in the MWA full embedded element beam
************************************************
::

  # To get all the frequencies available for the MWA full embedded element beam
  # you need to download the output simulation file via
  # `wget http://cerberus.mwa128t.org/mwa_full_embedded_element_pattern.h5`
  # For this tutorial we use the file saved in the test data which only
  # contains a few frequencies.
  # The `read_mwa_beam` method takes delay and amplitude arrays to generate beams
  # pointed any where or with varying gains per dipole. Set a delay to 32
  # to get a beam where that dipole is turned off (or set the amplitude to zero).
  # The native format of the beam is spherical harmonic modes, so there is also
  # an option `pixels_per_deg` to set the output beam resolution
  # (default is 5 pixels per degree).

  >>> import os
  >>> from pyuvdata import UVBeam
  >>> from pyuvdata.data import DATA_PATH
  >>> import numpy as np
  >>> beam = UVBeam()

  >>> mwa_beam_file = os.path.join(DATA_PATH, 'mwa_full_EE_test.h5')
  >>> beam.read_mwa_beam(mwa_beam_file)
  >>> print(beam.beam_type)
  efield

  >>> delays = np.zeros((2, 16), dtype='int')
  >>> delays[:, 0] = 32
  >>> beam.read_mwa_beam(mwa_beam_file, pixels_per_deg=1, delays=delays)


d) Writing a regularly gridded beam FITS file
**********************************************
::

  >>> import os
  >>> from pyuvdata import UVBeam
  >>> from pyuvdata.data import DATA_PATH
  >>> import numpy as np
  >>> beam = UVBeam()
  >>> settings_file = os.path.join(DATA_PATH, 'NicCSTbeams/NicCSTbeams.yaml')
  >>> beam.read_cst_beam(settings_file, beam_type='power')
  >>> write_file = os.path.join(DATA_PATH, 'tutorial_output', 'tutorial.fits')
  >>> beam.write_beamfits(write_file, clobber=True)

e) Writing a HEALPix beam FITS file
***********************************
::

  >>> import os
  >>> from pyuvdata import UVBeam
  >>> from pyuvdata.data import DATA_PATH
  >>> import numpy as np
  >>> beam = UVBeam()
  >>> settings_file = os.path.join(DATA_PATH, 'NicCSTbeams/NicCSTbeams.yaml')
  >>> beam.read_cst_beam(settings_file, beam_type='power')

  # have to specify which interpolation function to use
  >>> beam.interpolation_function = 'az_za_simple'

  # note that the `to_healpix` method requires astropy_healpix to be installed
  # this beam file is very large. Let's cut down the size to ease the computation
  # More on the `select` method is covered in the following section
  >>> za_max = np.deg2rad(10.0)
  >>> za_inds_use = np.nonzero(beam.axis2_array <= za_max)[0]
  >>> beam.select(axis2_inds=za_inds_use)

  >>> beam.to_healpix()
  >>> write_file = os.path.join(DATA_PATH, 'tutorial_output', 'tutorial.fits')
  >>> beam.write_beamfits(write_file, clobber=True)

UVBeam: Selecting data
----------------------
The select method lets you select specific image axis indices (or pixels if
pixel_coordinate_system is HEALPix), frequencies and feeds (or polarizations if
beam_type is power) to keep in the object while removing others.

a) Selecting a range of Zenith Angles
*************************************
::

  >>> import os
  >>> from pyuvdata import UVBeam
  >>> from pyuvdata.data import DATA_PATH
  >>> import numpy as np
  >>> import matplotlib.pyplot as plt # doctest: +SKIP
  >>> beam = UVBeam()
  >>> settings_file = os.path.join(DATA_PATH, 'NicCSTbeams/NicCSTbeams.yaml')
  >>> beam.read_cst_beam(settings_file, beam_type='power')
  >>> new_beam = beam.select(axis2_inds=np.arange(0, 20), inplace=False)

  # plot zenith angle cut through beams
  >>> plt.plot(beam.axis2_array, beam.data_array[0, 0, 0, 0, :, 0], # doctest: +SKIP
  ...         new_beam.axis2_array, new_beam.data_array[0, 0, 0, 0, :, 0], 'r')
  >>> plt.xscale('log') # doctest: +SKIP
  >>> plt.xlabel('Zenith Angle (radians)') # doctest: +SKIP
  >>> plt.ylabel('Power') # doctest: +SKIP
  >>> plt.show() # doctest: +SKIP

UVBeam: Converting to beam types and coordinate systems
-------------------------------------------------------

a) Convert a regularly gridded az_za power beam to HEALpix (leaving original intact).
*************************************************************************************
::

  >>> import os
  >>> from pyuvdata import UVBeam
  >>> from pyuvdata.data import DATA_PATH
  >>> import numpy as np
  >>> from astropy_healpix import HEALPix
  >>> import matplotlib.pyplot as plt # doctest: +SKIP
  >>> from matplotlib.colors import LogNorm # doctest: +SKIP
  >>> beam = UVBeam()
  >>> settings_file = os.path.join(DATA_PATH, 'NicCSTbeams/NicCSTbeams.yaml')
  >>> beam.read_cst_beam(settings_file, beam_type='power')

  # have to specify which interpolation function to use
  >>> beam.interpolation_function = 'az_za_simple'

  # this beam file is very large. Let's cut down the size to ease the computation
  >>> za_max = np.deg2rad(10.0)
  >>> za_inds_use = np.nonzero(beam.axis2_array <= za_max)[0]
  >>> beam.select(axis2_inds=za_inds_use)

  >>> hpx_beam = beam.to_healpix(inplace=False)
  >>> hpx_obj = HEALPix(nside=hpx_beam.nside, order=hpx_beam.ordering)
  >>> lon, lat = hpx_obj.healpix_to_lonlat(hpx_beam.pixel_array)
  >>> plt.scatter(lon, lat, c=hpx_beam.data_array[0,0,0,0,:], norm=LogNorm()) # doctest: +SKIP

b) Convert a regularly gridded az_za efield beam to HEALpix (leaving original intact).
**************************************************************************************
::

  >>> import os
  >>> from pyuvdata import UVBeam
  >>> from pyuvdata.data import DATA_PATH
  >>> import numpy as np
  >>> from astropy_healpix import HEALPix
  >>> import matplotlib.pyplot as plt # doctest: +SKIP
  >>> from matplotlib.colors import LogNorm # doctest: +SKIP
  >>> beam = UVBeam()
  >>> settings_file = os.path.join(DATA_PATH, 'NicCSTbeams/NicCSTbeams.yaml')
  >>> beam.read_cst_beam(settings_file, beam_type='efield')

  # have to specify which interpolation function to use
  >>> beam.interpolation_function = 'az_za_simple'

  # this beam file is very large. Let's cut down the size to ease the computation
  >>> za_max = np.deg2rad(10.0)
  >>> za_inds_use = np.nonzero(beam.axis2_array <= za_max)[0]
  >>> beam.select(axis2_inds=za_inds_use)

  >>> hpx_beam = beam.to_healpix(inplace=False)
  >>> hpx_obj = HEALPix(nside=hpx_beam.nside, order=hpx_beam.ordering)
  >>> lon, lat = hpx_obj.healpix_to_lonlat(hpx_beam.pixel_array)
  >>> plt.scatter(lon, lat, c=hpx_beam.data_array[0,0,0,0,:], norm=LogNorm()) # doctest: +SKIP


c) Convert a regularly gridded efield beam to a power beam (leaving original intact).
*************************************************************************************
::

  >>> import os
  >>> from pyuvdata import UVBeam
  >>> from pyuvdata.data import DATA_PATH
  >>> import numpy as np
  >>> import matplotlib.pyplot as plt # doctest: +SKIP
  >>> beam = UVBeam()
  >>> settings_file = os.path.join(DATA_PATH, 'NicCSTbeams/NicCSTbeams.yaml')
  >>> beam.read_cst_beam(settings_file, beam_type='efield')
  >>> new_beam = beam.efield_to_power(inplace=False)

  # plot zenith angle cut through the beams
  >>> plt.plot(beam.axis2_array, beam.data_array[1, 0, 0, 0, :, 0].real, label='E-field real') # doctest: +SKIP
  >>> plt.plot(beam.axis2_array, beam.data_array[1, 0, 0, 0, :, 0].imag, 'r', label='E-field imaginary') # doctest: +SKIP
  >>> plt.plot(new_beam.axis2_array, np.sqrt(new_beam.data_array[0, 0, 0, 0, :, 0]), 'black', label='sqrt Power') # doctest: +SKIP
  >>> plt.xlabel('Zenith Angle (radians)') # doctest: +SKIP
  >>> plt.ylabel('Magnitude') # doctest: +SKIP
  >>> plt.legend() # doctest: +SKIP
  >>> plt.show() # doctest: +SKIP

Generating pseudo Stokes ('pI', 'pQ', 'pU', 'pV') beams
*******************************************************
::

  >>> import os
  >>> from pyuvdata import UVBeam
  >>> from pyuvdata.data import DATA_PATH
  >>> from pyuvdata import utils as uvutils
  >>> import numpy as np
  >>> from astropy_healpix import HEALPix
  >>> import matplotlib.pyplot as plt # doctest: +SKIP
  >>> from matplotlib.colors import LogNorm # doctest: +SKIP
  >>> beam = UVBeam()
  >>> settings_file = os.path.join(DATA_PATH, 'NicCSTbeams/NicCSTbeams.yaml')
  >>> beam.read_cst_beam(settings_file, beam_type='efield')
  >>> beam.interpolation_function = 'az_za_simple'

  # this beam file is very large. Let's cut down the size to ease the computation
  >>> za_max = np.deg2rad(10.0)
  >>> za_inds_use = np.nonzero(beam.axis2_array <= za_max)[0]
  >>> beam.select(axis2_inds=za_inds_use)

  >>> pstokes_beam = beam.to_healpix(inplace=False)
  >>> pstokes_beam.efield_to_pstokes()
  >>> pstokes_beam.peak_normalize()

  # plotting pseudo-stokes I
  >>> pol_array = pstokes_beam.polarization_array
  >>> pstokes = uvutils.polstr2num('pI')
  >>> pstokes_ind = np.where(np.isin(pol_array, pstokes))[0][0]
  >>> hpx_obj = HEALPix(nside=hpx_beam.nside, order=hpx_beam.ordering)
  >>> lon, lat = hpx_obj.healpix_to_lonlat(hpx_beam.pixel_array)
  >>> plt.scatter(lon, lat, c=np.abs(pstokes_beam.data_array[0, 0, pstokes_ind, 0, :]), norm=LogNorm()) # doctest: +SKIP

Calculating pseudo Stokes ('pI', 'pQ', 'pU', 'pV') beam area and beam squared area
**********************************************************************************
::

  >>> import os
  >>> from pyuvdata import UVBeam
  >>> from pyuvdata.data import DATA_PATH
  >>> import numpy as np
  >>> beam = UVBeam()
  >>> settings_file = os.path.join(DATA_PATH, 'NicCSTbeams/NicCSTbeams.yaml')
  >>> beam.read_cst_beam(settings_file, beam_type='efield')
  >>> beam.interpolation_function = 'az_za_simple'

  # note that the `to_healpix` method requires astropy_healpix to be installed
  # this beam file is very large. Let's cut down the size to ease the computation
  >>> za_max = np.deg2rad(10.0)
  >>> za_inds_use = np.nonzero(beam.axis2_array <= za_max)[0]
  >>> beam.select(axis2_inds=za_inds_use)

  >>> pstokes_beam = beam.to_healpix(inplace=False)
  >>> pstokes_beam.efield_to_pstokes()
  >>> pstokes_beam.peak_normalize()

  # calculating beam area
  >>> freqs = pstokes_beam.freq_array
  >>> pI_area = pstokes_beam.get_beam_area('pI')
  >>> pQ_area = pstokes_beam.get_beam_area('pQ')
  >>> pU_area = pstokes_beam.get_beam_area('pU')
  >>> pV_area = pstokes_beam.get_beam_area('pV')
  >>> pI_area1, pI_area2 = round(pI_area[0].real, 5), round(pI_area[1].real, 5)
  >>> pQ_area1, pQ_area2 = round(pQ_area[0].real, 5), round(pQ_area[1].real, 5)
  >>> pU_area1, pU_area2 = round(pU_area[0].real, 5), round(pU_area[1].real, 5)
  >>> pV_area1, pV_area2 = round(pV_area[0].real, 5), round(pV_area[1].real, 5)

  >>> print ('Beam area at {} MHz for pseudo-stokes\nI: {}\nQ: {}\nU: {}\nV: {}'.format(freqs[0][0]*1e-6, pI_area1, pU_area1, pU_area1, pV_area1))
  Beam area at 123.0 MHz for pseudo-stokes
  I: 0.04674
  Q: 0.02879
  U: 0.02879
  V: 0.0464


  >>> print ('Beam area at {} MHz for pseudo-stokes\nI: {}\nQ: {}\nU: {}\nV: {}'.format(freqs[0][1]*1e-6, pI_area2, pU_area2, pU_area2, pV_area2))
  Beam area at 150.0 MHz for pseudo-stokes
  I: 0.03237
  Q: 0.01956
  U: 0.01956
  V: 0.03186


  # calculating beam squared area
  >>> freqs = pstokes_beam.freq_array
  >>> pI_sq_area = pstokes_beam.get_beam_sq_area('pI')
  >>> pQ_sq_area = pstokes_beam.get_beam_sq_area('pQ')
  >>> pU_sq_area = pstokes_beam.get_beam_sq_area('pU')
  >>> pV_sq_area = pstokes_beam.get_beam_sq_area('pV')
  >>> pI_sq_area1, pI_sq_area2 = round(pI_sq_area[0].real, 5), round(pI_sq_area[1].real, 5)
  >>> pQ_sq_area1, pQ_sq_area2 = round(pQ_sq_area[0].real, 5), round(pQ_sq_area[1].real, 5)
  >>> pU_sq_area1, pU_sq_area2 = round(pU_sq_area[0].real, 5), round(pU_sq_area[1].real, 5)
  >>> pV_sq_area1, pV_sq_area2 = round(pV_sq_area[0].real, 5), round(pV_sq_area[1].real, 5)

  >>> print ('Beam squared area at {} MHz for pseudo-stokes\nI: {}\nQ: {}\nU: {}\nV: {}'.format(freqs[0][0]*1e-6, pI_sq_area1, pU_sq_area1, pU_sq_area1, pV_sq_area1))
  Beam squared area at 123.0 MHz for pseudo-stokes
  I: 0.02474
  Q: 0.01179
  U: 0.01179
  V: 0.0246


  >>> print ('Beam squared area at {} MHz for pseudo-stokes\nI: {}\nQ: {}\nU: {}\nV: {}'.format(freqs[0][1]*1e-6, pI_sq_area2, pU_sq_area2, pU_sq_area2, pV_sq_area2))
  Beam squared area at 150.0 MHz for pseudo-stokes
  I: 0.01696
  Q: 0.00792
  U: 0.00792
  V: 0.01686
