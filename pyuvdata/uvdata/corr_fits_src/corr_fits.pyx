# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2020 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

# distutils: language = c
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
# python imports
import numpy as np
# cython imports
cimport cython
cimport numpy

cpdef dict input_output_mapping():
  """Build a mapping dictionary from pfb input to output numbers."""
  # the polyphase filter bank maps inputs to outputs, which the MWA
  # correlator then records as the antenna indices.
  # the following is taken from mwa_build_lfiles/mwac_utils.c
  # inputs are mapped to outputs via pfb_mapper as follows
  # (from mwa_build_lfiles/antenna_mapping.h):
  # floor(index/4) + index%4 * 16 = input
  # for the first 64 outputs, pfb_mapper[output] = input
  cdef int p, i
  cdef dict pfb_inputs_to_outputs = {}
  # fmt: off
  pfb_mapper = [0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51,
                4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38, 54, 7, 23, 39, 55,
                8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59,
                12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47,
                63]

  # fmt: on
  # build a mapper for all 256 inputs
  for p in range(4):
    for i in range(64):
      pfb_inputs_to_outputs[pfb_mapper[i] + p * 64] = p * 64 + i

  return pfb_inputs_to_outputs

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void generate_map(
  dict ants_to_pf,
  numpy.int32_t[::1] map_inds,
  numpy.npy_bool[::1] conj,
):
  """Compute the map between pfb inputs and antenna numbersself.

  This function operates on input `map_inds` and `conj` arrays inplace.

  Parameters
  ----------
  map_inds : 1D numpy array of type int32
    The array into which mapping indices will be populated.
  conj : 1D numpy array of type np.bool_
    The array into which indices of baselines to conjugate will be populated.

  """
  cdef int ant1, ant2, p1, p2, pol_ind, bls_ind, out_ant1, out_ant2
  cdef int out_p1, out_p2, ind1_1, ind1_2, ind2_1, ind2_2, data_index

  cdef dict in_to_out = input_output_mapping()

  for ant1 in range(128):
    for ant2 in range(ant1, 128):
      for p1 in range(2):
        for p2 in range(2):
          # generate the indices in self.data_array for this combination
          # baselines are ordered (0,0),(0,1),...,(0,127),(1,1),.....
          # polarizion of 0 (1) corresponds to y (x)
          pol_ind = int(2 * p1 + p2)
          bls_ind = int(128 * ant1 - ant1 * (ant1 + 1) / 2 + ant2)
          # find the pfb input indices for this combination
          ind1_1 = ants_to_pf[(ant1, p1)]
          ind1_2 = ants_to_pf[(ant2, p2)]

          # find the pfb output indices
          ind2_1 = in_to_out[ind1_1]
          ind2_2 = in_to_out[ind1_2]

          out_ant1 = int(ind2_1 / 2)
          out_ant2 = int(ind2_2 / 2)
          out_p1 = ind2_1 % 2
          out_p2 = ind2_2 % 2
          # the correlator has ind2_2 <= ind2_1 except for
          # redundant data. The redundant data is not perfectly
          # redundant; sometimes the values of redundant data
          # are off by one in the imaginary part.
          # For consistency, we are ignoring the redundant values
          # that have ind2_2 > ind2_1
          if ind2_2 > ind2_1:
            # get the index for the data
            data_index = int(
              2 * out_ant2 * (out_ant2 + 1)
              + 4 * out_ant1
              + 2 * out_p2
              + out_p1
            )
            # need to take the complex conjugate of the data
            map_inds[bls_ind * 4 + pol_ind] = data_index
            conj[bls_ind * 4 + pol_ind] = True
          else:
            data_index = int(
              2 * out_ant1 * (out_ant1 + 1)
              + 4 * out_ant2
              + 2 * out_p1
              + out_p2
            )
            map_inds[bls_ind * 4 + pol_ind] = data_index

  return

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef numpy.ndarray[ndim=1, dtype=numpy.float64_t] get_cable_len_diffs(
  int Nblts,
  numpy.int_t[::1] ant1_array,
  numpy.int_t[::1] ant2_array,
  numpy.ndarray cable_lens,
):
  """Computer the difference in cable lengths for each baseline.

  The inputs are one dimensional and will be both C and F contiguous but
  we want to declare the memory layout in a consistent way with the rest of the code.

  Parameters
  ----------
  Nblts : int
    The number of baseline times
  ant1_array : numpy array of type int_t
    Array of antenna 1 numbers for each baseline.
  ant2_array : numpy array of type int_t
    Array of antenna 2 numbers for each baseline.
  cable_lens : numpy array
    Array of strings of the length of the cable for each antenna.

  Returns
  -------
  cable_diffs : numpy array of type float64
    Array of lenght Nblts with the difference of cable lengths for each baseline.

  """
  cdef Py_ssize_t i
  cdef int n_cables = cable_lens.shape[0]
  cdef numpy.float64_t[::1] cable_array = np.zeros(n_cables, dtype=np.float64)
  cdef numpy.float64_t[::1] cable_diffs = np.zeros(Nblts, dtype=np.float64)
  # "the velocity factor of electic fields in RG-6 like coax"
  # from MWA_Tools/CONV2UVFITS/convutils.h
  cdef float v_factor = 1.204

  # check if the cable length already has the velocity factor applied
  # this loop is very python-y but there is no real gain to trying to make it C-esque
  for i in range(n_cables):
    if cable_lens[i][:3] == "EL_":
      cable_array[i] = float(cable_lens[i][3:])
    else:
      cable_array[i] = float(cable_lens[i]) * v_factor

  # build array of differences
  for i in range(Nblts):
    cable_diffs[i] = cable_array[ant2_array[i]] - cable_array[ant1_array[i]]

  return np.asarray(cable_diffs)
