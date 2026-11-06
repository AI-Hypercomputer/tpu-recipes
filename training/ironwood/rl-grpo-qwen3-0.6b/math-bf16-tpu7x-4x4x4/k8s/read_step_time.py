#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Step time of one run from its TensorBoard event files. Standard library only (no TensorFlow, no pip install).

usage:
  python3 read_step_time.py <directory-with-events.out.tfevents-files>

It prints ONE line:
  perf/train/global_step_time   median <seconds>  p10 <seconds>  p90 <seconds>  n=<steps used>

Rules it applies (do not change them when comparing with the reference numbers):
  - metric = perf/train/global_step_time  (the full step, INCLUDING the trainer->sampler weight sync;
    the PHASETIME lines in the pod log do NOT include the weight sync, so never quote a step time from them)
  - TensorBoard numbers the steps from 1. Steps 1 and 2 are dropped (compilation and warm-up), so a finished
    20-step run gives n=18. (The PHASETIME lines in the pod log count from 0 instead; do not mix the two.)
  - the median is reported, not the mean
  - if a step was written twice (pod restart) the last write wins
"""
import argparse, collections, glob, os, statistics, struct, sys


def varint(b, i):
    r = s = 0
    while True:
        x = b[i]; i += 1
        r |= (x & 0x7F) << s
        if not (x & 0x80): return r, i
        s += 7

def fields(b, start=0, end=None):
    """Yield (field_no, wire_type, payload_bytes_or_int)."""
    i, end = start, len(b) if end is None else end
    while i < end:
        key, i = varint(b, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:   v, i = varint(b, i)
        elif wt == 1: v = b[i:i+8]; i += 8
        elif wt == 2:
            n, i = varint(b, i); v = b[i:i+n]; i += n
        elif wt == 5: v = b[i:i+4]; i += 4
        else: raise ValueError(f"wiretype {wt}")
        yield fn, wt, v

def parse_value(vb):
    tag, simple, tensor = None, None, None
    for fn, wt, v in fields(vb):
        if fn == 1 and wt == 2: tag = v.decode('utf-8', 'replace')
        elif fn == 2 and wt == 5: simple = struct.unpack('<f', v)[0]
        elif fn == 8 and wt == 2: tensor = v
    if simple is None and tensor is not None:
        # pull a scalar out of the TensorProto
        for fn, wt, v in fields(tensor):
            if fn == 5 and wt == 5: simple = struct.unpack('<f', v)[0]
            elif fn == 9 and wt == 1: simple = struct.unpack('<d', v)[0]
            elif fn == 4 and wt == 2 and len(v) in (4, 8):
                simple = struct.unpack('<f' if len(v) == 4 else '<d', v)[0]
    return tag, simple

def read(path):
    data = open(path, 'rb').read()
    i, out = 0, []
    while i + 12 <= len(data):
        (n,) = struct.unpack_from('<Q', data, i)
        i += 12                       # length + length-crc
        if i + n + 4 > len(data): break
        rec = data[i:i+n]; i += n + 4  # payload + payload-crc
        wall = step = None; vals = []
        for fn, wt, v in fields(rec):
            if fn == 1 and wt == 1: wall = struct.unpack('<d', v)[0]
            elif fn == 2 and wt == 0: step = v
            elif fn == 5 and wt == 2:
                for f2, w2, v2 in fields(v):
                    if f2 == 1 and w2 == 2: vals.append(parse_value(v2))
        if vals: out.append((wall, step, vals))
    return out



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("event_dir")
    ap.add_argument("--tag", default="perf/train/global_step_time")
    ap.add_argument("--min-step", type=int, default=3)
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(a.event_dir, "**", "events.out.tfevents.*"), recursive=True))
    if not paths:
        sys.exit("ERROR: no events.out.tfevents.* file under " + a.event_dir)
    series = {}
    for path in paths:
        for _, step, vals in read(path):
            for tag, value in vals:
                if tag is not None and value is not None:
                    series.setdefault(tag, {})[step] = value
    if a.tag not in series:
        sys.exit("ERROR: tag %r not found. Tags present:\n  %s" % (a.tag, "\n  ".join(sorted(series))))
    v = sorted(val for step, val in series[a.tag].items() if step >= a.min_step)
    if not v:
        sys.exit("ERROR: no step >= %d recorded yet (run too short or still compiling)" % a.min_step)
    print("%s   median %.2f  p10 %.2f  p90 %.2f  n=%d" % (a.tag, statistics.median(v), v[len(v)//10], v[len(v)*9//10], len(v)))


if __name__ == "__main__":
    main()
