from . import pyeep
###############################################################################
class cnt_base:
  def __init__(self, handle):
    self._handle = handle
    if self._handle == -1:
      raise Exception('not a valid libeep handle')

  def __del__(self):
    if self._handle != -1:
      pyeep.close(self._handle)
###############################################################################
class cnt_in(cnt_base):
  def __init__(self, handle):
    cnt_base.__init__(self, handle)

  def get_channel_count(self):
    return pyeep.get_channel_count(self._handle)

  def get_channel(self, index):
    return (pyeep.get_channel_label(self._handle, index), pyeep.get_channel_unit(self._handle, index), pyeep.get_channel_reference(self._handle, index))

  def get_sample_frequency(self):
    return pyeep.get_sample_frequency(self._handle)

  def get_sample_count(self):
    return pyeep.get_sample_count(self._handle)

  def get_samples(self, fro, to):
    return pyeep.get_samples(self._handle, fro, to)

  def get_trigger_count(self):
    return pyeep.get_trigger_count(self._handle)

  def get_trigger(self, index):
    return pyeep.get_trigger(self._handle, index)
###############################################################################
class cnt_out(cnt_base):
  def __init__(self, handle, channel_count):
    cnt_base.__init__(self, handle)
    self._channel_count=channel_count

  def add_samples(self, samples):
    return pyeep.add_samples(self._handle, samples, self._channel_count)
###############################################################################
def read_cnt(filename):
  if not filename.endswith('.cnt'):
    raise Exception('unsupported extension')
  return cnt_in(pyeep.read(filename))
###############################################################################
def write_cnt(filename, rate, channels, rf64 = 0):
  """
  create an object for writing a .cnt file.

  rate -- sampling rate in Herz
  channels -- list of tuples, where tuples contains three strings:
              channel label, channel reference and unit, i,e,:
              ['Cz', 'ref', 'uV')]
  rf64 -- if 0, create default 32-bit cnt data. otherwise 64 bit(for larger tan 2GB files)
  """
  if not filename.endswith('.cnt'):
    raise Exception('unsupported extension')

  channels_handle = pyeep.create_channel_info()
  for c in channels:
    pyeep.add_channel(channels_handle, c[0], c[1], c[2])

  rv = cnt_out(pyeep.write_cnt(filename, rate, channels_handle, rf64), len(channels))

  pyeep.close_channel_info(channels_handle)

  return rv
