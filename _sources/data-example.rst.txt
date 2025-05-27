Example
=======

Header (YAML):

>>> import audformat.testing
>>> db = audformat.testing.create_db()
>>> db
name: unittest
description: A database for unit testing.
source: internal
usage: unrestricted
languages: [deu, eng]
author: J. Wagner, H. Wierstorf
organization: audEERING GmbH
license: CC0-1.0
attachments:
  file: {path: extra/file.txt}
  folder: {path: extra/folder}
media:
  microphone: {type: other, format: wav, channels: 1, bit_depth: 16, sampling_rate: 16000}
  webcam:
    type: other
    format: avi
    video_fps: 25
    video_resolution: [800, 600]
    video_channels: 3
    video_depth: 8
raters:
  gold: {description: Gold standard by taking the average ratings., type: human}
  machine: {description: Predictions made by the machine., type: machine, classifier: LibSVM,
    features: ComParE_2016}
schemes:
  bool: {dtype: bool}
  date: {dtype: date}
  float: {dtype: float, minimum: -1.0, maximum: 1.0}
  int: {dtype: int, minimum: 0, maximum: 100}
  label:
    dtype: str
    labels: [label1, label2, label3]
  label_map_int:
    dtype: int
    labels: {1: a, 2: b, 3: c}
  label_map_misc: {dtype: str, labels: misc}
  label_map_str:
    dtype: str
    labels:
      label1: {prop1: 1, prop2: a}
      label2: {prop1: 2, prop2: b}
      label3: {prop1: 3, prop2: c}
  string: {dtype: str}
  time: {dtype: time}
splits:
  dev: {type: dev}
  test: {type: test}
  train: {type: train}
tables:
  files:
    type: filewise
    split_id: train
    media_id: microphone
    columns:
      bool: {scheme_id: bool, rater_id: gold}
      date: {scheme_id: date, rater_id: gold}
      float: {scheme_id: float, rater_id: gold}
      int: {scheme_id: int, rater_id: gold}
      label: {scheme_id: label, rater_id: gold}
      label_map_int: {scheme_id: label_map_int, rater_id: gold}
      label_map_misc: {scheme_id: label_map_misc, rater_id: gold}
      label_map_str: {scheme_id: label_map_str, rater_id: gold}
      string: {scheme_id: string, rater_id: gold}
      time: {scheme_id: time, rater_id: gold}
      no_scheme: {}
  segments:
    type: segmented
    split_id: dev
    media_id: microphone
    columns:
      bool: {scheme_id: bool, rater_id: gold}
      date: {scheme_id: date, rater_id: gold}
      float: {scheme_id: float, rater_id: gold}
      int: {scheme_id: int, rater_id: gold}
      label: {scheme_id: label, rater_id: gold}
      label_map_int: {scheme_id: label_map_int, rater_id: gold}
      label_map_misc: {scheme_id: label_map_misc, rater_id: gold}
      label_map_str: {scheme_id: label_map_str, rater_id: gold}
      string: {scheme_id: string, rater_id: gold}
      time: {scheme_id: time, rater_id: gold}
      no_scheme: {}
misc_tables:
  misc:
    levels: {labels: str}
    columns:
      int: {scheme_id: int}
      label: {scheme_id: label}
audformat: https://github.com/audeering/audformat

Filewise table as :class:`pd.DataFrame`:

>>> db["files"].get()
                bool                    date  ...                   time   no_scheme
file                                          ...
audio/001.wav  False                     NaT  ... 0 days 00:00:00.280000  nOqgBuJRkn
audio/002.wav   <NA>                     NaT  ... 0 days 00:00:00.960000  iMKa6qC99q
audio/003.wav  False 1970-01-01 00:00:00.640  ... 0 days 00:00:00.520000  MgqdWkARaq
audio/004.wav  False 1970-01-01 00:00:00.090  ... 0 days 00:00:00.060000        None
audio/005.wav  False                     NaT  ... 0 days 00:00:00.540000  l8x3NmGtNB
...              ...                     ...  ...                    ...         ...
audio/096.wav   True 1970-01-01 00:00:00.160  ... 0 days 00:00:00.180000  Vgv37YqafU
audio/097.wav   <NA> 1970-01-01 00:00:00.780  ... 0 days 00:00:00.210000  Nr3hn17JSB
audio/098.wav   True 1970-01-01 00:00:00.750  ... 0 days 00:00:00.620000  FFSFTcMjMu
audio/099.wav  False 1970-01-01 00:00:00.470  ... 0 days 00:00:00.920000  V49rd5YUQW
audio/100.wav   <NA> 1970-01-01 00:00:00.610  ... 0 days 00:00:00.280000  zeTXeIT7wb
<BLANKLINE>
[100 rows x 11 columns]

Segmented table as :class:`pd.DataFrame`:

>>> db["segments"].get()
                                                                    bool  ...   no_scheme
file          start                     end                               ...
audio/001.wav 0 days 00:00:00.082561829 0 days 00:00:04.832983387   True  ...        None
              0 days 00:00:09.907641513 0 days 00:00:13.087561565   <NA>  ...        None
              0 days 00:00:13.422086186 0 days 00:00:16.376171043   True  ...  ZYixPedAyB
              0 days 00:00:19.276700122 0 days 00:00:25.737048646   True  ...        None
              0 days 00:00:30.918073408 0 days 00:00:35.168424756   <NA>  ...  s5pS1eh4k7
...                                                                  ...  ...         ...
audio/010.wav 0 days 00:00:18.786962756 0 days 00:00:29.510792049  False  ...        None
              0 days 00:00:30.691782517 0 days 00:00:33.635262781  False  ...  jWjGWGeIYX
              0 days 00:00:34.058263452 0 days 00:00:34.149017810  False  ...        None
              0 days 00:00:45.930182929 0 days 00:00:49.444541600  False  ...  Zz7W7ZlYXJ
              0 days 00:00:51.577189767 0 days 00:00:56.646463227   True  ...        None
<BLANKLINE>
[100 rows x 11 columns]

Misc table as :class:`pd.DataFrame`:

>>> db["misc"].get()
        int   label
labels
label1   26  label2
label2    0  label3
label3   56  label3
