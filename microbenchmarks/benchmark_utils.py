"""Utilities for benchmarks."""

from collections import defaultdict
from dataclasses import dataclass
import gzip
import json
import os
import pathlib
import re
import time
from typing import Any, Callable
import jax
import numpy as np


@dataclass
class TimingStats:
  """The timing statistics of the benchmark.

  Attributes:
    time_median: the median completion time of the benchmark function or the
      trace if a trace matcher is specified.
  """

  time_median: float = 0


def get_trace(log_dir: str) -> dict[str, Any]:
  """Extract the trace object from the log directory.

  If multiple profiles exist in the directory, the lastest one will be used.

  Args:
    log_dir: log directory created by jax.profiler.trace().

  Returns:
    A trace object in JSON format.
  """
  # Navigate to the folder with the latest trace dump to find `trace.json.jz`
  trace_folders = (
      pathlib.Path(log_dir).absolute() / "plugins" / "profile"
  ).iterdir()
  latest_trace_folder = max(trace_folders, key=os.path.getmtime)
  trace_jsons = latest_trace_folder.glob("*.trace.json.gz")
  try:
    (trace_json,) = trace_jsons
  except ValueError as value_error:
    raise ValueError(
        f"Invalid trace folder: {latest_trace_folder}"
    ) from value_error

  with gzip.open(trace_json, "rb") as f:
    trace = json.load(f)

  return trace


def get_eligible_events(
    trace: dict[str, Any],
    trace_matcher: re.Pattern[str],
) -> list[dict[str, Any]]:
  """Filter the trace events eligible for benchmarking.

  Args:
    trace: a trace object in JSON format.
    trace_matcher: a regex-based trace name matcher to filter the evnets.

  Returns:
    A list of events objects in JSON format.
  """
  if "traceEvents" not in trace:
    raise KeyError("Key 'traceEvents' not found in trace.")

  ret = []
  for e in trace["traceEvents"]:
    if "name" in e and trace_matcher.match(e["name"]):
      ret.append(e)
  return ret


def calculate_timing_stats(events: list[dict[str, Any]]) -> TimingStats:
  """Calculate the timing statistics from the given list of trace events.

  Args:
    events: a list of trace events.

  Returns:
    Timing statistics.
  """
  # Data could be distributed onto multiple cores. We approximate the runtime to
  # be the maximum duration of all events with the same run_id.
  events_by_run_id = defaultdict(list)
  for e in events:
    run_id = (
        e["args"]["run_id"] if "args" in e and "run_id" in e["args"] else "0"
    )
    events_by_run_id[run_id].append(e)

  try:
    durations = [
        max([e["dur"] for e in es]) / 1e6
        for run_id, es in events_by_run_id.items()
    ]
  except KeyError:
    print("KeyError: Key 'dur' not found in the event object")
    raise

  return TimingStats(
      time_median=np.median(durations),
  )


def run_bench(
    fn: Callable[..., Any],
    *args,
    num_iter: int,
    warmup_iter: int,
    log_dir: str,
    func_label: str,
    trace_matcher: re.Pattern[str] = None,
    clear_caches: bool = False,
) -> TimingStats:
  """Runs a function `num_iter` times to measure the runtime of benchmark function.

  A jax profiler trace is captured in order to measure the timing at the event
  level within the function.

  Args:
    fn: the function to be benchmarked.
    *args: arguments to the function `fn`.
    num_iter: number of times `fn` will be run.
    warmup_iter: number of times `fn` will be run before the acutal timing
      measurement.
    log_dir: the directory to save the profiler trace to.
    func_label: the trace name of `fn` in the profiler.
    trace_matcher: a regex-based trace matcher to filter the events eligible for
      benchmarking. If None, timing result will be derived from time() wrapper.
    clear_caches: call jax.clear_caches() every time before executing `fn`,
      which clears all compilation and staging caches.

  Returns:
    Timing statistics of the benchmark.
  """
  # warm up
  for _ in range(warmup_iter):
    fn(*args)

  durations = []
  with jax.profiler.trace(log_dir):
    for _ in range(num_iter):
      if clear_caches:
        jax.clear_caches()
      with jax.profiler.TraceAnnotation(func_label):
        start_t = time.time()
        jax.block_until_ready(fn(*args)),
        durations.append(time.time() - start_t)

  if trace_matcher:
    trace = get_trace(log_dir)
    events = get_eligible_events(trace, trace_matcher)
    time_stats = calculate_timing_stats(events)
  else:
    time_stats = TimingStats(
        time_median=np.median(durations),
    )

  return time_stats
