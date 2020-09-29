# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Added `propagate_coarse_flags` option to `read_mwa_corr_fits`, as well as new flagging defaults.
- Added support for `extra_keywords` in UVFlag construction from UVData objects (and paths)

### Changed
- Improved memory usage in reading MWA correlator fits files.
- Speed improvement in redundant group finder.

### Fixed
- Fixed a bug in `compress_by_redundancy` with `method='average'` that resulted in more unique times on the compressed object than on the original object.
- Fixed bug causing MemoryError when finding redundant baselines for over 100 antennas.

## [2.1.3] - 2020-12-15

### Added
- Added support for `telescope_location`, `antenna_positions` and `lst_array` in UVCal objects and file types.
- Added support for a `metadata_only` mode in UVCal, including options to only read the metadata when reading in calibration files.
- Added a `copy` method for UVCal objects.
- Added `antenna_positions`, `antenna_names` and `antenna_numbers` as optional metadata to our known telescopes list and added HERA antenna position information.
- Added handling for `extra_keywords` in UVFlag objects, including read/write to/from files.
- Added handling for `object_name` and `extra_keywords` to `sum_vis` and `diff_vis` methods and added the `override_params` option to override other parameters.

### Changed
- Changed to use Astropy sites for telescope locations when avaliable. This results in a small change for our known position for the MWA.
- Modified `UVData.read` to do faster concatenation of files, changed the interface to `UVData.fast_concat` to allow lists of `UVData` objects to be passed in.

### Fixed
- Fixed a bug where telescope positions from MWA uvfits files created by Cotter were not identified as being in the ITRF frame because of a missing FITS keyword.
- Fixed a bug where `antenna_positions` from FHD files were interpreted as being in the relative ECEF frame rather than the rotated ECEF frame
- Fixed a bug where the `lst_array` was not updated in `compress_by_redundancy` when using `method='average'`.
- Fixed an undefined parameter bug when interpolating beams that do not cover the sky.
- Fixed an indexing error bug in `compress_by_redundancy` when using `method='average'`.

### Deprecated
- UVCal objects now require `telescope_location`, `antenna_positions` and `lst_array` parameters. Not setting them will cause an error starting in pyuvdata 2.3.

## [2.1.2] - 2020-10-07

### Added
- Capability to apply Van Vleck correction to mwa_corr_fits files.
- Added support for multiple spectral windows for UVH5, UVFITS, MIR, and MIRIAD files.
- Added the `flex_spw` attribute to the `UVData` class, which can be set to True by using the new `_set_flex_spw` method.
- Added the optional `flex_spw_id_array` attribute, of type=int and shape=(`Nfreqs`,), which indexes individual channels along the frequency axis to `spw_array`.
- Adjustment to digital gain removal from mwa_corr_fits files to account for a division by 64 due to a bit selection shift.
- Options to remove coarse band shape and digital gains from mwa_corr_fits files.
- Support for cotter flags in mwa_corr_fits files.

### Changed
- When `flex_spw=True`, the `channel_width` attribute of UVData is expected to be an array of type=float and shape=(`Nfreqs`,). Individual channels are allowed to have different channel widths.
- Changed `freq_array` to be of shape (1, `Nfreqs`) (formerley (`Nspws`, `Nfreqs`)).
- Changed  `data_array`, `flag_array`, `nsample_array` to be of shape (`Nblts`, 1, `Nfreqs`, `Npols`) (formerly (`Nblts`, `Nspws`, `Nfreqs`, `Npols`)).
- Changed UVH5 metadata byte conversion method from tobytes() to bytes()

## [2.1.1] - 2020-8-07

### Added
- Added read-only support for MIR files, adding the `Mir` class to `UVData`.
- Added the `MirParser` class, which allows for python access to MIR files.
- A new `check_warnings` method in our test module that behaves like `pytest.warns`
but adds the ability to check for multiple warnings.
- The 'ant_str' option when making selects on UVFlag objects
- A check that the uvws match the antenna positions, as part of the acceptability checking.

### Changed
- Upated the numpy requirement to >= 1.18
- Moved `parse_ants` to `utils.py` to allow any UVBased object the potential to use it.

### Deprecated
- The `checkWarnings` function in our test module, use the new `check_warnings`
function instead.

## [2.1.0] - 2020-7-08

### Added
- `UVFlag` can now take `PosixPath` objects as input.
- An option to average rather than select a single baseline in `UVData.compress_by_redundancy`,
controlled by the `method` keyword.
- `nsample_array_dtype` keyword for `UVData.read`, which sets the datatype for the nsample_array. Currently only used for mwa_corr_fits files.
- `utils` cython module to speed up some common utility functions.
- Added support for installing the package on Windows.
- 'background_lsts' keyword in `uvdata.read` to calculate lsts in the background while reading. Defaults to True.
- `background` keyword in `uvdata.set_lsts_from_time_array` to allow lst calculation in a background thread.
- `read_data` keyword to read_mwa_corr_fits, allows from metadata only reads of MWA correlator fits files.
- `propagate_flags` keyword for `UVData.frequency_average` which flags averaged samples if any contributing samples were flagged

### Changed
- Nants data calculation changed to use numpy functions for small speed up.
- Changed `input` variable to `indata` in UVFlag to avoid shadowing python builtin.
- `select` now also accepts a list of baseline indices for the `bls` parameter.
- `data_array_dtype` keyword for `UVData.read` is now also respected by mwa_corr_fits files. Previously it was only used by uvh5 files.
- The `time_range` parameter on UVCal is no longer required and it is suggested to only set it if the Ntimes axis is length one.
- FHD now supports metadata only reads.
- Updated formatting and added more explicit typing in corr_fits.pyx
- Updated Nants calculations for speed up.
- updated HERA telescope location to match the HERA defined center of array.
- `utils.uvcalibrate` now incorporates many more consistency checks between the uvcal and uvdata object. The new keywords `time_check` and `ant_check` were added to control some of these checks.

### Deprecated
- The UVBeam methods `set_cs_params`, `set_efield`, `set_power`, `set_simple`, and `set_phased_array` have been made private. The public methods will be removed in version 2.2.
- The UVCal methods `set_gain`, `set_delay`, `set_unknown_cal_type`, `set_sky`, and `set_redundant` have been made private. The public methods will be removed in version 2.2.
- `utils.uvcalibrate` will error rather than warn if some of the newly added checks do not pass, including if antenna names do not match between the objects, starting in version 2.2. In addition, the `flag_missing` keyword is deprecated and will be removed in version 2.2.

### Fixed
- UVFlag.__add__ now properly concatenates all existing data-like parameters of the object, including optional ones.
- A bug in `UVData.downsample_in_time` where the data were not being properly weighted by the nsample array and the nsample_array was not being properly weighted by the integration times.
- A bug in `UVData.downsample_in_time` that lead to duplicated data on the final object if a baseline had varying integration times and some integration times were greater than or equal to the requested minimum integration time.

## [2.0.2] - 2020-4-29

### Added
- New `_corr_fits` C extension for performing the index and conjugation mapping calculation for read_mwa_corr_fits
- `copy` method to UVCal and UVBeam objects
- `chunks` to `UVH5.write_uvh5` and `UVData.write_uvh5` for HDF5 dataset chunking.
- `multidim_index` to `UVH5.read_uvh5` and `UVData.read` for multidimensional slicing into HDF5 datasets

### Changed
- Various serialized calculations in uvdata.py, uvbeam.py, and utils.py to be vectorized with numpy
- Miriad interface was re-written from a hand written CPython interface to a Cython interface which dynamically creates the CPython during package setup/build.
- uvfits, calfits and beamfits files should now be read faster due to a simplified handling of fits header objects, especially when the history is very long.
- The UVData methods `set_drift`, `set_phased`, and `set_unknown_phase_type` have been made private. The public methods will be removed in version 2.2.

## [2.0.1] - 2020-3-24

### Changed
- UVData allows conjugation of metadata only objects.
- Handling of strings in UVFlag files has been made more widely compatible.

### Added
- Option to provide parameters for RectBivariatespline through interp
- `weights_square_array` (optional) parameter on UVFlag - stores sum of squares of weights when converting to waterfall
- `frequency_average` method on UVData to average data along the frequency axis.

### Fixed
- A bug in `select` where baseline-polarization tuples would not be conjugated correctly in `UVData` and `UVFlag`, and select on read for Miriad files.
- `metafits_ppds.fits` files can now be passed to `mwa_corr_fits.read` without throwing an error.
- UVParameters that are array_like and have NaNs are now properly identified as equal if the NaNs are in the same locations and all non-NaN entries are equal.
- A bug in `mwa_corr_fits.read` in filling flag and nsample arrays.
- A bug in `UVData.downsample_in_time` in calculating the number of new blts.

## [2.0.0] - 2020-2-12

### Changed
- All references to Python 2 removed from codebase.

### Added
- Routine flagging of MWA coarse band edges and center channels, as well as beginning and end integrations. Only done during `read_mwa_corr_fits`.
- Added a return_weights_square option to `utils.collapse` function, which returns the sum of the squares of the weights.

### Removed
- Previously deprecated code marked for removal in version > 1.5:
 - reading multiple files with file format specific read functions (e.g.  read_mirad). Multi-file reads can only be performed with `uvdata.read`
 - `read_metadata` keyword in various specific read functions (e.g. `read_miriad`)
 - `metadata_only` keyword in `select` and `get_redundancies`
 - `uvdata.miriad.read_miriad_metadata`
 - `phase_center` keyword in `uvdata.read_mwa_corr_fits`
 - `phase_data` keyword in `uvdata.read_mwa_corr_fits`
 - `uvdata.get_antenna_redundancies`
 - `uvdata.get_baseline_redundancies`
 - `uvdata.uvfits.read_uvfits_metadata`
 - `uvdata.uvfits.read_uvfits_data`

## [1.5.0] - 2020-1-15

### Added
- Support for reading the MWA full embedded element beam into a UVBeam object.
- New `time_range` keyword to `select` so exact times don't need to be specified (also added to `read` methods for select on read).
- Support for rephasing phased data including on `read`, `__add__` and `fast_concat` so that files with different phasing can be read in together.
- `sum_vis` and `diff_vis` for summing or differencing visibilities in the data_array.
- `read_mwa_corr_fits` for reading in MWA correlator gpubox files and applying cable corrections.
- `eq_coeffs` for storing equalization coefficients and `remove_eq_coeffs` for removing them.
- `utils.apply_uvflag` for applying UVFlag objects to UVData objects

### Fixed
- Fixed `utils.uvcalibrate` to handle `x_orientation` attribute
- Arrays of file names can now be passed to `read` (not just lists).
- run_check is no longer turned off in `read_uvfits` and `read_uvh5` when `read_data` is False.
- Redundancy finder will now error if any baselines appear in multiple groups.
- A bug in `UVCal` objects that prevented them from properly getting data with `ee`/`nn`-style polarizations.
- a bug in `UVFlag` where `x_orientation` was not set during initialization.
- A bug in `UVCal.read_fhd_cal` that caused calibration solutions to be approximately doubled.
- A bug in UVFlag where polarization array states were not updated when using `force_pol` keyword in `to_antenna` and `to_baseline`
- A bug in UVFlag.to_baseline() where force_pol kwarg did not work for UVData Npols > 1
- `UVData.read_uvfits` no longer breaks if there are non-ascii bytes in antenna names (which CASA sometimes writes).

### Deprecated
- Reading in multiple files (or file sets) using file-type specific read methods (e.g. `read_uvfits`) in favor of the generic `read` method.
- The `phase_center` and `phase_data` keywords to `read_mwa_corr_fits` in favor of `phase_to_pointing_center` and the `phase_center_radec` keyword in the generic `read` method.
- Support for reading only the header (not all the metadata) of uvfits files.
- The `read_uvfits_metadata` and `read_uvfits_data` methods on the UVFITS object.
- The `read_miriad_metadata` method on the Mirad object.

## [1.4.2] - 2019-10-15

### Added
- `copy` method for `UVData`, which can optionally make a copy of just metadata
- `upsample_in_time`, `downsample_in_time`, and `resample_in_time` methods on `UVData` objects
- `utils.uvcalibrate(.., undo=True)` kwarg for undo-ing a calibration.
- `utils.uvcalibrate` updates `UVData.vis_units` if `UVCal.gain_scale` is set.
- `UVCal.gain_scale` non-required attribute
- UVData.get_redundancies method to replace old get_baseline_redundancies and get_antenna_redundancies.
- option for `UVBeam.interp` to return a new beam object.
- `UVFlag` information on Read The Docs

### Changed
- `UVData.phase_to_time` now accepts a float as an input. Assumes float represents a JD.
- Added optional acceptability check for `utils.LatLonAlt_from_XYZ`
- Use `astropy_healpix` rather than `healpy` for HEALPix functionality because `healpy` has a GPL license which is incompatible with ours.
- `h5py` is now a required package instead of an optional one.
- Phasing now supports metadata only `UVData` objects
- utils.get_baseline_redundancies uses scipy pdist functions instead of for loops (faster)
- UVData.get_antenna_redundancies will no longer automatically conjugate baselines.
- UVData.get_baseline_redundancies and UVData.get_antenna_redundancies have been combined.
- `UVFlag` inherits from `UVBase` object.
- `UVFlag` objects can now convert from antenna type to baseline type
- `UVFlag` objects can now be initialized without inputs

### Fixed
- A bug in UVBeam._interp_freq where kind parameter was not passed for real-only beams
- A bug in get_antenna_redundancies for nearly N-S baselines.
- A bug where `conj_pol` could not handle cardinal direction polarizations.
- A bug that gave the wrong error message when calling `UVData.phase_to_time` without an Astropy Time object.

### Deprecated
- UVData.get_baseline_redundancies and UVData.get_antenna_redundancies will be deprecated in version 1.6.1

## [1.4.1] - 2019-08-2

### Added
- `metadata_only` property on `UVData` to automatically detect if data-like arrays are present
- support for combining metadata only objects and reading in multiple files as metadata only
- `utils.uvcalibrate` flag propagation bug fix
- `UVCal.ant2ind` indexing bug fix
- `UVCal.get_*` methods for accessing data arrays with antenna-polarization keys
- `utils.uvcalibrate` for automated calibration of `UVData` by `UVCal`

### Fixed
- Fixed a bug in select that caused bls and antenna_names/numbers to be or'ed rather than and'ed together.
- Fixed a bug where `baseline_to_antnums` could accept a numpy array as input but not other array_like objects.

### Changed
- removed `new_object` keyword from `UVBeam._interp_freq` in favor of new functionality in `UVBeam.interp`

## [1.4.0] - 2019-05-23

### Added
- Option in UVBase.check() to ignore whether required parameters are set.
- made an option to not save the `lst_array` to uvfits files.
- `conjugate_bls` option to `UVData.get_antenna_redundancies`
- `UVData.conjugate_bls` method to conjugate baselines to get the desired baseline directions.
- `UVData.reorder_blts` method to reorder the data along the blt axis (and optionally also conjugate baselines), and a new `blt_order` optional parameter on UVData objects to track the ordering (including through read/writes).
- `lst_array` is now saved to UVFITS files (even though it's not a standard parameter) so that it doesn't have to be recalculated

### Fixed
- Fixed init logic in UVFlag.
- Fixed a bug in how FHD uvw vectors were oriented (and visibilities were conjugated)
- Fixed a bug in `UVData.inflate_by_redundancy` when Nblts is not equal to Nbls * Ntimes.
- Fixed a bug in UVData when reading in an FHD file with a single time integration.
- Fixed a bug in how the longitudinal branch cut was handled in beam interpolation
- Changed the way interpolation splines are saved in UVBeam to fix errors related to polarization selections.
- Python 3: `np.string_` call now uses `keepdims=True` to guard against `antenna_names` being cast as an array.

### Changed
- Testing framework changed from `nose` to `pytest`.
- `uvdata.set_lsts_from_time_array` only calculates lsts for unique elements in `time_array`.

## [1.3.8] - 2019-05-01

### Added
- Optional `x_orientation` parameter to utils functions polstr2num, polnum2str, jstr2num and jnum2str to allow for E/N based polarization strings (rather than just x/y based ones)
- New optional `x_orientation` parameter on UVBeam (paralleling UVData and UVCal), with read/write support in beamfits
- Added `x_orientation` as an optional parameter in read_cst_beam and in cst settings yaml files.
- All str2num or num2str calls on UVData and UVBeam pass the object's x_orientation
- New `UVData.fast_concat` method to allow fast concatenation of UVData objects (or files) along a particular axis.
- Added preliminary `UVFlag` module from hera_qm to pyuvdata. Will eventually promote to `UVBase` object, but for now this is undocumented functionality.

### Deprecated
- Defined 'east' and 'north' as the allowed 'x_orientation' values in UVData and UVCal, Backwards compatiblity support exists for 'E' and 'N' values
- `UVData.order_pols` method in favor of `UVData.reorder_pols`.

### Fixed
- Building pyuvdata on macOS now targets minimum macOS 10.9 if run on macOS 10.9 or above
- Possible bug where `check_variables` dictionary can change size during `read_miriad` call


## [1.3.7] - 2019-04-02

### Added
- Added `add_to_history` kwarg to UVH5.write_uvh5_part
- `_healpix_interp_bilinear` as a new interpolation method in `UVBeam`
- `freq_interpolation_kind` added as an attribute to `UVBeam`
- `tol` added as keyword argument to `UVBeam._interp_freq` which allows for a fast return of `data_array` slice if nearest-neighbor frequencies are all within the distance tolerance.
- `polarizations` added as keyword argument to `UVBeam` interpolation methods.
- Support for a yaml settings file to collect and propagate metadata for CST beam files.

### Changed
- `UVBeam._interp_freq` returns both `interp_data` and `interp_bandpass`, instead of just the former.

### Fixed
- Combining overlapping data along multiple axes (most common when reading in multiple files) no longer errors.


## [1.3.6] - 2019-02-15

### Added
- `keep_all_metadata` keyword for optionally discarding unused metadata when performing a select operation.

### Changed
- Extends `run_acceptability_check` for UVH5 metadata in `check_header` function.

### Fixed
- Antenna numbering bug in redundancy methods. It wasn't using the correct antenna numbers to make baseline indices.
- Redundancy code returns one group if all baselines are redundant. Previously returned each baseline as a separate group
- Redundancy code finds unique baselines along baseline_array without assuming Nblts = Nbls \* Ntimes. Previously assumed Nblts = Nbls \* Ntimes and attempted to slice array.
- "inflate_by_redundancy" method errored when phase_type == phased, due to _set_u_positive using phased uvw coordinates. It now uses ENU frame uvw coordinates.

## [1.3.5] - 2018-12-20

## [1.3.4] - 2018-12-19
### Added
- Methods on UVData objects to compress/inflate data by redundant baselines.
- Convenience functions on UVData for finding redundant baselines (calling the corresponding utils functions)
- memo describing the UVH5 format
- read/write support for uvh5 files with integer datatypes for visibilities
- Option to only do the select on the metadata. This is useful for partially defined objects as in pyuvsim setup or after reading only the metadata from a file.
- support for python3.7

### Changed
- UVdata.get_ENU_antpos() now defaults to using the telescope_location as the center rather than the median antenna position.
- UVBeam.efield_to_pstokes() no longer restricted to healpix coordinates
- latitude and longitude in uvh5 files are written in degrees instead of radians.
- Fixes a bug in redundancy methods for when there are no redundant baselines.

### Fixed
- `_key2inds` now properly reorders polarization axis for conjugated visibilities. This also effects the `get_data` function.
- long strings are saved correctly in miriad files from python3

## [1.3.3] - 2018-11-01
### Added
- option to save splines for reuse in UVBeam.interp function

### Changed
- improve string handling for uvh5 files
- changed top-level import structure to exclude file-specific class (e.g. `UVFITS`, `CALFITS`) and base classes (`UVBase`, `UVParameter`) and to not import utility functions into the top-level namespace

### Deprecated
- Support for UVData objects without antenna_positions. Antenna positions will be required in a future version.

## [1.3.2] - 2018-09-27
### Added
- Utility functions to identify redundant baselines (either by baseline or antenna pair)
- Reading FHD layout files to get antenna positions
- Polarization dict constants and parsing functions in utils.py for mapping between polarization strings and numbers

### Changed
- LST array is now optional in uvh5 files
- polarization string capitalization was unified throughout: use lower case except for Stokes parameters
- integration_time is now a vector of length NBlts instead of a scalar

### Deprecated
- Support for FHD inputs without layout files (containing antenna positions).

### Fixed
- flags are always returned as a boolean array from `UVData.get_flags`
- integration_time, uvw_array and lst_array are now always checked for consistency when adding UVData objects
- consistency checks on baseline lengths now uses the uvw_array tolerances

## [1.3.1] - 2018-07-24

## [1.3] - 2018-07-22
### Added
- compatibility with python 3 (retaining python 2 compatibility)
- generic `UVData.read` method for all input file types, including select on read options
- partial write support for uvh5 file format
- partial read support for uvfits, miriad and uvh5 file formats (including only reading metadata and select on read)
- new uvh5 file format: an HDF5 file standard matched to UVData objects
- new method to calculate uvws from antenna positions
- `UVBeam.get_beam_area` and `UVBeam.get_beam_sq_area` functions to calculate beam integrals, including for pseudo-Stokes beams
- beam interpolation methods, to any set of points and to healpix pixel centers
- a script to renumber antennas for CASA compatiblity if there are fewer than 256 antennas but numbers higher than that
- memo describing the beam fits file format
- method to peak normalize UVBeam objects
- support for reading FHD calibrations into UVCal objects
- support for sky-based calibration metadata in UVCal and the calfits file format
- method to convert E-field beams to power beams
- `UVData.get_ENU_antpos` method to get ENU coordinates from antenna positions
- support for `extra_keywords` on UVCal objects

### Changed
- Major overhaul of phasing code, see the phasing memo in docs/references for more details
- Make all earth location coordinate conversions use same axes order (n_points, 3)
- replaced `ant_pair_nums` keyword in `UVData.select` and partial read methods with `bls` which supports lists like [(0,1,'xx'), (2,3,'yy')]
- extracted miriad wrappers from aipy, removing aipy as a dependency

### Fixed
- Fixed error when combining auto-correlation only and cross-correlation only UVData objects
- Fixed error with the `UVData.select` function using a single antenna name
- Fixed bugs with `UVData.get_data`, `UVData.get_flags`, and `UVData.get_nsamples` for conjugated baseline polarizations
- Fixed a bug that caused a memory error in `UVData.write_uvfits`
- Fixed bugs in interpreting the uvw direction convention for uvfits
- Fixed a bug in reading azimuth locations from CST beam files
- Fixed a bug reading in single frequency uvfits files
- Fixed a bug in reading MWA Cotter measurement sets
- Fixed units errors in cal fits files
- Fixed a serious bug where data was overwritten in the add functions if the axes were out of canonical order
- Fixed scrambled data ordering in add function

## [1.2.1] - 2017-11-09
### Fixed
- Fixed a bug in parsing frequencies from CST file names

## [1.2] - 2017-11-08
### Added
- support doctest in the tutorial
- utility functions for converting between Jones numbers and polarization strings
- support for `antdiam` variable in miriad files
- module for reading a CST file into a UVBeam object
- utility functions for converting to rotated ECEF coordinates from ECEF coordinates
- support reading in a list of calfits files
- include `extra_keywords` to miriad files
- `__add__` method for UVCal objects
- `antenna_diameters` value to UVData objects
- `parse_ants` method on UVData objects
- `x_orientation` value to UVData objects
- "smart slicing" functionality to UVData objects
- convenience methods on UVData objects for easily getting data and metadata
- UVBeam object
- in-place selection for UVData objects
- `total_quality_array` value on UVCal objects
- `__add__` method for UVData objects
- utility functions for converting to local ENU coordinates from ECEF coordinates
- `convert_to_gain` method on UVCal delay-type objects
- read-only support for CASA measurement sets into UVData objects
- `select` method on UVCal objects

### Changed
- antenna names in miriad saved as strings instead of arrays of ASCII hex values

### Fixed
- baseline-time axis mis-ordering in add function
- handling of antenna positions in miriad and uvfits files
- selecting autocorrelation data from UVData objects
- indexing of spectral windows in calfits files
- handling of `total_quality_array` in UVCal objects when selecting a subset of data

## [1.1] - 2017-04-14
(historical information needs to be filled in)
