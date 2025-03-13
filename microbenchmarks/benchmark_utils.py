"""Utilities for benchmarks."""

from collections import defaultdict
from dataclasses import dataclass
import gzip
import json
import os
import pathlib
import re
from typing import Any, Callable
import jax
import numpy as np


@dataclass
class BenchmarkResult:
  """The result of a benchmark run.

  Attributes:
    time_median: the median elapsed time of the benchmark at the event level. By
      default, the event to be measured is equivalent to the function to be
      benchmarked, unless an event_matcher is provided.
    time_min: the minimum elapsed time of the benchmark.
  """

  time_median: float = 0
  time_min: float = 0
  time_fn_median: float = 0


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
    event_matcher: re.Pattern[str],
) -> list[dict[str, Any]]:
  """Filter the trace events eligible for benchmarking.

  Args:
    trace: a trace object in JSON format.
    event_matcher: a regex-based event name matcher to filter the evnets.

  Returns:
    A list of events objects in JSON format.
  """
  if "traceEvents" not in trace:
    raise KeyError("Key 'traceEvents' not found in trace.")

  ret = []
  for e in trace["traceEvents"]:
    if "name" in e and event_matcher.match(e["name"]):
      ret.append(e)
  return ret


def get_benchmark_result(events: list[dict[str, Any]]) -> BenchmarkResult:
  """Derive the benchmark result from the given list of trace events.

  Args:
    events: a list of trace events.

  Returns:
    A summary of the benchmark result.
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

  return BenchmarkResult(
      time_median=np.median(durations),
      time_min=np.min(durations),
  )


def run_bench(
    fn: Callable[..., Any],
    num_iter: int,
    warmup_iter: int,
    log_dir: str,
    func_label: str,
    event_matcher: re.Pattern[str] = None,
) -> BenchmarkResult:
  """Runs a function `num_iter` times to measure the runtime for benchmarking.

  A jax profiler trace is captured in order to measure the timing at the event
  level within the function.

  Args:
    fn: the function to be benchmarked.
    num_iter: number of times `fn` will be run.
    warmup_iter: number of times `fn` will be run before the acutal timing
      measurement.
    log_dir: the directory to save the profiler trace to.
    func_label: a label to identify the function in the trace events.
    event_matcher: a regex-based event matcher to filter the evnets eligible for
      benchmarking. If None, the runtime of the function will be reported.

  Returns:
    A summary of the benchmark result.
  """
  # warm up
  for _ in range(warmup_iter):
    fn()

  with jax.profiler.trace(log_dir):
    for _ in range(num_iter):
      jax.clear_caches()
      with jax.profiler.TraceAnnotation(func_label):
        fn()

  trace = get_trace(log_dir)

  if not event_matcher:
    event_matcher = re.compile(func_label)
  events = get_eligible_events(trace, event_matcher)
  result = get_benchmark_result(events)

  fn_matcher = re.compile(func_label)
  fn_events = get_eligible_events(trace, fn_matcher)
  tmp = get_benchmark_result(fn_events)
  result.time_fn_median = tmp.time_median

  return result
