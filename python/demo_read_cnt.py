#!/usr/bin/python

import libeep

import sys

for f in sys.argv[1:]:
  cnt = libeep.read_cnt(f)
  print('file: %s' % (f))
  print('  channels: %i' % (cnt.get_channel_count()))
  print(['{0}'.format(cnt.get_channel(i)) for i in range(cnt.get_channel_count())])
  print('  sample frequency: %i' % (cnt.get_sample_frequency()))
  print('  sample count: %i' % (cnt.get_sample_count()))
  print('  samples(0, 2):')
  print(["{0:0.2f}".format(i) for i in cnt.get_samples(0, 2)])
  print('  triggers: %i' % (cnt.get_trigger_count()))
  print(['{0}'.format(cnt.get_trigger(i)) for i in range(cnt.get_trigger_count())])
