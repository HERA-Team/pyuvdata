# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2020 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

# distutils: language = c++
# python imports
import numpy as np
# cython imports
cimport cython
cimport numpy
cimport libcpp.complex
from libc.string cimport strcmp

DEF PREAMBLE_SIZE = 5
DEF MAXVAR = 32768

cdef extern from "miriad.h":
  cdef int H_BYTE  "H_BYTE"
  cdef int H_INT   "H_INT"
  cdef int H_INT2  "H_INT2 "
  cdef int H_REAL  "H_REAL "
  cdef int H_DBLE  "H_DBLE "
  cdef int H_TXT   "H_TXT  "
  cdef int H_CMPLX "H_CMPLX"
  cdef int H_BYTE_SIZE "H_BYTE_SIZE"
  cdef int  H_INT_SIZE  "H_INT_SIZE"
  cdef int  H_INT2_SIZE  "H_INT2_SIZE"
  cdef int  H_INT8_SIZE  "H_INT8_SIZE"
  cdef int  H_REAL_SIZE  "H_REAL_SIZE"
  cdef int  H_DBLE_SIZE  "H_DBLE_SIZE"
  cdef int  H_TXT_SIZE  "H_TXT_SIZE"
  cdef int  H_CMPLX_SIZE  "H_CMPLX_SIZE"

  void uvclose_c(int tno)
  void uvopen_c(int *tno, const char *name, const char *status)
  void uvset_c(int tno, const char *object, const char *type, int n, double p1, double p2, double p3)
  void uvrewind_c(int tno)
  void uvwrite_c(int tno, double *preamble, float *data, int *flags, int n)
  void uvread_c   (int tno, double *preamble, float *data, int *flags, int n, int *nread);
  void uvcopyvr_c (int tin, int tout)
  void uvtrack_c  (int tno, const char *name, const char *switches)
  void uvprobvr_c (int tno, const char *var, char *type, int *length, int *updated);
  void uvgetvr_c  (int tno, int type, const char *var, char *data, int n);
  void uvputvr_c  (int tno, int type, const char *var, const char *data, int n);
  void uvselect_c (int tno, const char *object, double p1, double p2, int datasel);
  void haccess_c(int tno, int *ihandle, const char *keyword, const char *status, int *iostat);
  void hdaccess_c(int ihandle, int *iostat);

  void hwriteb_c(int item, char buf[4], int offset, int length, int *iostat)
  void hreadb_c(int item, char buf[4], int offset, int length, int *iostat)

  void hreadi_c(int item, int *buf, int offset, int length, int *iostat)
  void hwritei_c(int item, int *buf, int offset, int length, int *iostat)

  void hwritej_c(int item, short *buf, int offset, int length, int *iostat)
  void hreadj_c(int item, short *buf, int offset, int length, int *iostat)

  void hwritel_c(int item, long *buf, int offset, int length, int *iostat)
  void hreadl_c(int item, long *buf, int offset, int length, int *iostat)

  void hwriter_c(int item, float *buf, int offset, int length, int *iostat)
  void hreadr_c(int item, float *buf, int offset,int length, int *iostat)

  void hreadd_c(int item, double *buf, int offset, int length, int *iostat)
  void hwrited_c(int item, double *buf, int offset, int length, int *iostat)

  void hwritec_c(int item, float buf[2], int offset, int length, int *iostat)
  void hreadc_c(int item, float buf[2], int offset, int length, int *iostat)

cdef extern from "io.h":
  cdef const int ITEM_HDR_SIZE "ITEM_HDR_SIZE"
  cdef char binary_item[ITEM_HDR_SIZE]
  cdef char real_item[ITEM_HDR_SIZE]
  cdef char int_item[ITEM_HDR_SIZE]
  cdef char int2_item[ITEM_HDR_SIZE]
  cdef char int8_item[ITEM_HDR_SIZE]
  cdef char char_item[ITEM_HDR_SIZE]
  cdef char dble_item[ITEM_HDR_SIZE]
  cdef char cmplx_item[ITEM_HDR_SIZE]


cdef extern from "hio.h":
  int mroundup(int a, int b)

cdef extern from "maxdimc.h":
  cpdef int _MAXCHAN "MAXCHAN"

MAXCHAN = _MAXCHAN

ctypedef numpy.int_t DTYPE_t
ctypedef numpy.complex64_t DTYPE_c
ctypedef numpy.float64_t DTYPE_f64

cdef inline int GETI(int bl):
  return (bl - 65536) // 2048 - 1 if bl > 65536 else (bl >> 8) - 1

cdef inline int GETJ(int bl):
  return (bl - 65536) % 2048 - 1 if bl > 65536 else (bl & 255) - 1

cdef inline float MKBL(int i, int j):
  return (i + 1) << 8 | (j + 1) if (i + 1 < 256 and j + 1 < 256) else (i + 1) * 2048 + (j + 1 + 65536)

cdef inline void CHK_IO(int i) except *:
  if (i != 0):
    raise IOError("IO failed.")

cpdef void hdaccess(int item_hdl) except +:
  cdef int iostat
  hdaccess_c(item_hdl, &iostat)
  return

cdef int INIT(int item_hdl, char type_item[4], int size):
  cdef int offset, iostat
  hwriteb_c(item_hdl, type_item, 0, ITEM_HDR_SIZE, &iostat)
  CHK_IO(iostat)
  offset = mroundup(ITEM_HDR_SIZE, size)
  return offset

cpdef int hwrite_init(int item_hdl, str type) except +:
  cdef int offset

  if type[0] == "a":
    offset = INIT(item_hdl, char_item, H_BYTE_SIZE)

  elif type[0] == "b":
    offset = INIT(item_hdl, binary_item, ITEM_HDR_SIZE)

  elif type[0] == "i":
    offset = INIT(item_hdl, int_item, H_INT_SIZE)

  elif type[0] == "j":
    offset = INIT(item_hdl, int2_item, H_INT2_SIZE)

  elif type[0] == "l":
    offset = INIT(item_hdl, int8_item, H_INT8_SIZE)

  elif type[0] == "r":
    offset = INIT(item_hdl, real_item, H_REAL_SIZE)

  elif type[0] == "d":
    offset = INIT(item_hdl, dble_item, H_DBLE_SIZE)

  elif type[0] == "c":
    offset = INIT(item_hdl, cmplx_item, H_CMPLX_SIZE)
  else:
    raise ValueError(f"Unknown type {type[0]}")

  return offset

# hread_init surmises the type of a header item from the first few bytes
cdef int FIRSTINT(char s[4]):
  return (<int *>s)[0]

cpdef int hread_init(int item_hdl) except +:
  cdef int offset, iostat, code
  cdef char s[ITEM_HDR_SIZE]

  hreadb_c(item_hdl, s, 0, ITEM_HDR_SIZE, &iostat)

  CHK_IO(iostat)
  code = FIRSTINT(s)
  if (code == FIRSTINT(char_item)):
      offset = mroundup(ITEM_HDR_SIZE,H_BYTE_SIZE)
      return "a", offset

  elif (code == FIRSTINT(binary_item)):
    return "b", ITEM_HDR_SIZE

  elif (code == FIRSTINT(int_item)):
    offset = mroundup(ITEM_HDR_SIZE,H_INT_SIZE)
    return "i", offset

  elif (code == FIRSTINT(int2_item)):
    offset = mroundup(ITEM_HDR_SIZE,H_INT2_SIZE)
    return "j", offset

  elif (code == FIRSTINT(int8_item)):
    offset = mroundup(ITEM_HDR_SIZE,H_INT8_SIZE)
    return  "l", offset

  elif (code == FIRSTINT(real_item)):
    offset = mroundup(ITEM_HDR_SIZE,H_REAL_SIZE)
    return "r", offset

  elif (code == FIRSTINT(dble_item)):
    offset = mroundup(ITEM_HDR_SIZE,H_DBLE_SIZE)
    return  "d", offset
  elif (code == FIRSTINT(cmplx_item)):
    offset = mroundup(ITEM_HDR_SIZE,H_CMPLX_SIZE)
    return "c", offset
  else:
    raise RuntimeError("unknown item type.")

cpdef int hwrite(int item_hdl, int offset, val, str type) except +:
  cdef int iostat
  cdef int int_1
  cdef long lg
  cdef float fl
  cdef double db
  cdef float cx[2]
  cdef char *st

  if type[0] == "a":
    if not isinstance(val, (str, bytes)):
      raise ValueError("expected a string")
    if isinstance(val, str):
      val = val.encode()

    st = <char *>val
    int_1 = len(val)
    hwriteb_c(item_hdl, st, offset, int_1, &iostat)
    CHK_IO(iostat)
    offset = H_BYTE_SIZE * int_1

  elif type[0] == "i":
    if not isinstance(val, (np.int, np.int_, np.intc)):
      raise ValueError("expected an int")
    int_1 = <int>val
    hwritei_c(item_hdl, &int_1, offset, H_INT_SIZE, &iostat)
    CHK_IO(iostat)
    offset = H_INT_SIZE

  elif type[0] == "j":
    if not isinstance(val, (np.int, np.int_, np.intc)):
      raise ValueError("expected an int")
    sh = <short>val
    hwritej_c(item_hdl, &sh, offset, H_INT2_SIZE, &iostat)
    CHK_IO(iostat)
    offset = H_INT2_SIZE

  elif type[0] == "l":
    if not isinstance(val, (np.int, np.intc, np.int_)):
      raise ValueError("expected a  long")
    lg = <long>val
    hwritel_c(item_hdl, &lg, offset, H_INT8_SIZE, &iostat)
    CHK_IO(iostat)
    offset = H_INT8_SIZE

  elif type[0] == "r":
    if not isinstance(val, (np.float, np.float32)):
      raise ValueError("expected a float")
    fl = <float>val
    hwriter_c(item_hdl, &fl, offset, H_REAL_SIZE, &iostat)
    CHK_IO(iostat)
    offset = H_REAL_SIZE

  elif type[0] == "d":
    if not isinstance(val, (np.float, np.float32, np.float_)):
      raise ValueError("expected a double")
    db = <double>val
    hwrited_c(item_hdl, &db, offset, H_DBLE_SIZE, &iostat)
    CHK_IO(iostat)
    offset = H_DBLE_SIZE

  elif type[0] == "c":
    if not isinstance(val, np.complex64):
      raise ValueError("expected a complex")

    cx[0] = <float>val.real
    cx[1] = <float>val.imag
    hwritec_c(item_hdl, cx, offset, H_CMPLX_SIZE, &iostat)
    CHK_IO(iostat)
    offset = H_CMPLX_SIZE

  else:
    raise ValueError(f"unknown item type: {type [0]}")

  return offset

cpdef hread(int item_hdl, int offset, str type) except +:
  cdef int iostat
  cdef int int_1
  cdef short sh
  cdef long lg
  cdef float fl
  cdef double db
  cdef float cx[2]
  cdef char st[2]

  if type[0] == "a":
     hreadb_c(item_hdl, st, offset, H_BYTE_SIZE, &iostat)
     CHK_IO(iostat)
     return st, H_BYTE_SIZE

  elif type[0] == "i":
    hreadi_c(item_hdl, &int_1, offset, H_INT_SIZE, &iostat)
    CHK_IO(iostat)
    return int_1, H_INT_SIZE

  elif type[0] == "j":
    hreadj_c(item_hdl, &sh, offset, H_INT2_SIZE, &iostat)
    CHK_IO(iostat)
    return sh, H_INT2_SIZE

  elif type[0] == "l":
    hreadl_c(item_hdl, &lg, offset, H_INT8_SIZE, &iostat)
    CHK_IO(iostat)
    return lg, H_INT8_SIZE

  elif type[0] == "r":
    hreadr_c(item_hdl, &fl, offset, H_REAL_SIZE, &iostat)
    CHK_IO(iostat)
    return fl, H_REAL_SIZE

  elif type[0] == "d":
    hreadd_c(item_hdl, &db, offset, H_DBLE_SIZE, &iostat)
    CHK_IO(iostat)
    return db, H_DBLE_SIZE

  elif type[0] == "c":
    hreadc_c(item_hdl, cx, offset, H_CMPLX_SIZE, &iostat)
    CHK_IO(iostat)
    # check this line here
    return <float>cx[0] + 1j * <float>cx[1], H_CMPLX_SIZE

  else:
    raise ValueError(f"unknown item type: {type[0]}")

cdef class UV:
  cpdef int tno
  cpdef long decimate
  cpdef long decphase
  cpdef long intcnt
  cpdef double curtime

  def __init__(self, str filename, str status, str corrmode):
    self.tno = -1
    self.decimate = 1
    self.decphase = 0
    self.intcnt = -1
    self.curtime = -1

    if corrmode[0] not in ["r", "j"]:
      raise ValueError(f"UV corrmode must be 'r' or 'j' but received {corrmode}.")

    try:
      uvopen_c(&self.tno, filename.encode(), status.encode())
      uvset_c(self.tno, "preamble", "uvw/time/baseline", 0, 0., 0., 0.)
      uvset_c(self.tno, "corr", corrmode.encode(), 0, 0., 0., 0.)
    except:
      self.tno = -1
      raise

  cpdef void close(self):
    if self.tno != -1:
      uvclose_c(self.tno)

    self.tno = -1
    return

  def __dealloc__(self):
    self.close()
    return

  @cython.boundscheck(False)
  cdef _get_j_type(self, int htype, char *name, int length):
    cdef numpy.ndarray[dtype=int, ndim=1] arr = np.zeros((length,), dtype=np.int16)
    uvgetvr_c(self.tno, htype, name, <char *>&arr[0], length)
    if length == 1:
      return arr.item(0)
    return arr

  @cython.boundscheck(False)
  cdef _get_i_type(self, int htype, char *name, int length):
    cdef numpy.ndarray[dtype=numpy.int32_t, ndim=1] arr = np.zeros((length,), dtype=np.int32)
    uvgetvr_c(self.tno, htype, name, <char *>&arr[0], length)
    if length == 1:
      return arr.item(0)
    return arr

  @cython.boundscheck(False)
  cdef _get_d_type(self, int htype, char *name, int length):
    cdef numpy.ndarray[dtype=DTYPE_f64, ndim=1] arr = np.zeros((length,), dtype=np.float64)
    uvgetvr_c(self.tno, htype, name, <char *>&arr[0], length)
    if length == 1:
      return arr.item(0)
    return arr

  @cython.boundscheck(False)
  cdef _get_r_type(self, int htype, char *name, int length):
    cdef numpy.ndarray[dtype=numpy.float32_t, ndim=1] arr = np.zeros((length,), dtype=np.float32)
    uvgetvr_c(self.tno, htype, name, <char *>&arr[0], length)
    if length == 1:
      return arr.item(0)
    return arr

  @cython.boundscheck(False)
  cdef _get_c_type(self, int htype, char *name, int length):
    cdef numpy.ndarray[dtype=DTYPE_c, ndim=1] arr = np.zeros((length,), dtype=np.complex64)
    uvgetvr_c(self.tno, htype, name, <char *>&arr[0], length)
    if length == 1:
      return arr.item(0)
    return arr

  @cython.boundscheck(False)
  cdef void _store_j_type(self, int htype, char *name, numpy.ndarray[dtype=int] value):
    uvputvr_c(self.tno, htype, name, <char *>&value[0], value.size)
    return

  @cython.boundscheck(False)
  cdef void _store_i_type(self, int htype, char *name, numpy.ndarray[dtype=numpy.int32_t] value):
    uvputvr_c(self.tno, htype, name, <char *>&value[0], value.size)
    return

  @cython.boundscheck(False)
  cdef void _store_d_type(self, int htype, char *name, numpy.ndarray[dtype=DTYPE_f64] value):
    uvputvr_c(self.tno, htype, name, <char *>&value[0], value.size)
    return

  @cython.boundscheck(False)
  cdef void _store_r_type(self, int htype, char *name, numpy.ndarray[dtype=numpy.float32_t] value):
    uvputvr_c(self.tno, htype, name, <char *>&value[0], value.size)
    return

  @cython.boundscheck(False)
  cdef void _store_c_type(self, int htype, char *name, numpy.ndarray[dtype=DTYPE_c] value):
    uvputvr_c(self.tno, htype, name, <char *>&value[0], value.size)
    return

  cpdef void rewind(self):
    uvrewind_c(self.tno)
    self.intcnt = -1
    self.curtime = -1
    return

  @cython.boundscheck(False)
  cpdef raw_read(self, int n2read) except +:
    cdef int nread, i, j
    cdef double preamble[PREAMBLE_SIZE]
    cdef numpy.ndarray[numpy.complex64_t , ndim=1] data = np.zeros((n2read,), dtype=np.complex64)
    cdef numpy.ndarray[int, ndim=1] flags = np.zeros((n2read,), dtype=np.intc)
    cdef numpy.ndarray[DTYPE_f64, ndim=1] uvw = np.zeros((3,), dtype=np.float64)

    while True:

      uvread_c(self.tno, preamble, <float *>&data[0], <int *>&flags[0], n2read, &nread)

      if (preamble[3] != self.curtime):
        self.intcnt += 1
        self.curtime = preamble[3]

      if ((self.intcnt - self.decphase) % self.decimate == 0 or nread == 0):
        break

    uvw[0] = preamble[0]
    uvw[1] = preamble[1]
    uvw[2] = preamble[2]

    i = GETI(<int>preamble[4])
    j = GETJ(<int>preamble[4])

    return (uvw, preamble[3], (i, j)), data, flags, nread

  cpdef void raw_write(self, object input_preamble, numpy.ndarray[dtype=DTYPE_c, ndim=1] data, numpy.ndarray[dtype=int, ndim=1] flags):
    cdef int nread
    cdef double preamble[PREAMBLE_SIZE]
    cdef double t = input_preamble[1]
    cdef int i = input_preamble[2][0], j = input_preamble[2][1]

    if len(input_preamble[0]) != 3:
      raise ValueError(f"uvw must have shape (3,) but got {len(input_preamble[0])}")

    preamble[0] = input_preamble[0][0]
    preamble[1] = input_preamble[0][1]
    preamble[2] = input_preamble[0][2]
    preamble[3] = t
    preamble[4] = MKBL(i, j)

    uvwrite_c(self.tno, preamble, <float *>&data[0], <int *>&flags[0], data.size)

    return

  cpdef void copyvr(self, UV uv):
    uvcopyvr_c(uv.tno, self.tno)
    return

  cpdef void trackvr(self, str name, str switches):
    uvtrack_c(self.tno, name.encode(), switches.encode())
    return

  cpdef _rdvr(self, str name, str type) except +:
    cdef char value[MAXVAR]
    cdef int length, updated, elem_size

    uvprobvr_c(self.tno, name.encode(), value, &length, &updated)

    if type[0] == "a":
      elem_size = 1
    elif type[0] == "j":
      elem_size = 2
    elif type[0] in ["i", "r"]:
      elem_size = 4
    elif type[0] in ["d", "c"]:
      elem_size = 8
    else:
      raise ValueError(f"unknown type of UV variable {name}: {type[0]}")

    if length * elem_size > MAXVAR:
      raise ValueError(f"UV variable {name} is too big for pyuvdata's internal buffers")

    if type[0] == "a":
      uvgetvr_c(self.tno, H_BYTE, name.encode(), value, length + 1)

      return value.decode("utf-8")

    elif type[0] == "j":
      return self._get_j_type(H_INT2, name.encode(), length)

    elif type[0] == "i":
      return self._get_i_type(H_INT, name.encode(), length)

    elif type[0] == "r":
      return self._get_r_type(H_REAL, name.encode(), length)

    elif type[0] == "d":
      return self._get_d_type(H_DBLE, name.encode(), length)

    elif type[0] == "c":
      return self._get_c_type(H_CMPLX, name.encode(), length)

    return

  cpdef _wrvr(self, str name, str type, value) except +:
    cdef char c_value[MAXVAR]
    cdef char *st
    if isinstance(value, np.ndarray):
      if value.ndim != 1:
        raise ValueError(f"Input variable {name} has rank {value.ndim} but must be rank 1.")

    if type[0] == "a":

      if not isinstance(value, (str, bytes)):
        raise ValueError("Expected a string")

      if isinstance(value, str):
        value = value.encode()

      st = <char *>value
      uvputvr_c(self.tno, H_BYTE, name.encode(), st, len(value) + 1)

    elif type[0] == "j":
      return self._store_j_type(H_INT2, name.encode(), np.atleast_1d(value).astype(np.int16))

    elif type[0] == "i":
      return self._store_i_type(H_INT, name.encode(), np.atleast_1d(value).astype(np.int32))

    elif type[0] == "r":
      return self._store_r_type(H_REAL, name.encode(), np.atleast_1d(value).astype(np.float32))

    elif type[0] == "d":
      return self._store_d_type(H_DBLE, name.encode(), np.atleast_1d(value).astype(np.float64))

    elif type[0] == "c":
      return self._store_c_type(H_CMPLX, name.encode(), np.atleast_1d(value).astype(np.complex64))

    else:
      raise ValueError(f"Unkown UV variable type {type[0]}")

    return

  cpdef void _select(self, str name, numpy.float64_t ind1, numpy.float64_t ind2, int include_flag) except +:
    # we used to only call strncmp(name, decimation, 5) so only look at first 5 letters
    if strcmp(name[:5].encode(), "decimation"[:5]) == 0:
      self.decimate = <long> ind1
      self.decphase = <long> ind2
    else:
      uvselect_c(self.tno, name.encode(), <double>ind1, <double>ind2, include_flag)

    return

  cpdef int haccess(self, str name, str mode) except +:
    cdef int item_hdl, iostat
    haccess_c(self.tno, &item_hdl, name.encode(), mode.encode(), &iostat)
    CHK_IO(iostat)
    return item_hdl
