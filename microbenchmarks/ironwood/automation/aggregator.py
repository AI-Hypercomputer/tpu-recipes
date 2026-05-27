import argparse
import os
import glob
import pandas as pd
import gcsfs

columns_mapping = {
    "collectives": [
        "topology", "op_type", "input_num_elements", "transferred_data (GB)", "dtype", "step_time_ms_num_runs",
        "achieved_bw (GB/s)_p50", "achieved_bw (GB/s)_p90", "achieved_bw (GB/s)_p95", "achieved_bw (GB/s)_p99", "achieved_bw (GB/s)_avg", "achieved_bw (GB/s)_min", "achieved_bw (GB/s)_max",
        "step_time_ms_p50", "step_time_ms_p90", "step_time_ms_p95", "step_time_ms_p99", "step_time_ms_avg", "step_time_ms_min", "step_time_ms_max",
    ],
    "hbm": [
        "dtype", "tensor_size_gbytes", "time_ms_num_runs",
        "bw_gbyte_sec_p50", "bw_gbyte_sec_p90", "bw_gbyte_sec_p95", "bw_gbyte_sec_p99", "bw_gbyte_sec_avg", "bw_gbyte_sec_min", "bw_gbyte_sec_max",
        "time_ms_p50", "time_ms_p90", "time_ms_p95", "time_ms_p99", "time_ms_avg", "time_ms_min", "time_ms_max",
    ],
    "host_device": [
        "data_size_mib", "transfer_type", "H2D_bw (GiB/s)_num_runs",
        "H2D_bw (GiB/s)_p50", "H2D_bw (GiB/s)_p90", "H2D_bw (GiB/s)_p95", "H2D_bw (GiB/s)_p99", 
        "H2D_bw (GiB/s)_avg", "H2D_bw (GiB/s)_min", "H2D_bw (GiB/s)_max",
        "D2H_bw (GiB/s)_p50", "D2H_bw (GiB/s)_p90", "D2H_bw (GiB/s)_p95", "D2H_bw (GiB/s)_p99", 
        "D2H_bw (GiB/s)_avg", "D2H_bw (GiB/s)_min", "D2H_bw (GiB/s)_max",
    ],
    "gemm": [
        "m", "n", "k", "dtype", "step_time_ms_num_runs",
        "tflops_per_sec_per_device_p50", "tflops_per_sec_per_device_p90",
        "tflops_per_sec_per_device_p95", "tflops_per_sec_per_device_p99",
        "tflops_per_sec_per_device_avg", "tflops_per_sec_per_device_min",
        "tflops_per_sec_per_device_max",
        "step_time_ms_p50", "step_time_ms_p90", "step_time_ms_p95", "step_time_ms_p99", "step_time_ms_avg", "step_time_ms_min", "step_time_ms_max",
    ],
    "bmm": [
        "b", "m", "n", "k", "dtype", "step_time_ms_num_runs",
        "tflops_per_sec_per_device_p50", "tflops_per_sec_per_device_p90",
        "tflops_per_sec_per_device_p95", "tflops_per_sec_per_device_p99",
        "tflops_per_sec_per_device_avg", "tflops_per_sec_per_device_min",
        "tflops_per_sec_per_device_max",
        "step_time_ms_p50", "step_time_ms_p90", "step_time_ms_p95", "step_time_ms_p99", "step_time_ms_avg", "step_time_ms_min", "step_time_ms_max",
    ],
    "gemm_all_reduce": [
        "topology", "m", "n", "k", "dtype", "step_time_ms_num_runs",
        "tflops_per_sec_per_device_p50", "tflops_per_sec_per_device_p90",
        "tflops_per_sec_per_device_p95", "tflops_per_sec_per_device_p99",
        "tflops_per_sec_per_device_avg", "tflops_per_sec_per_device_min",
        "tflops_per_sec_per_device_max",
        "step_time_ms_p50", "step_time_ms_p90", "step_time_ms_p95", "step_time_ms_p99", "step_time_ms_avg", "step_time_ms_min", "step_time_ms_max",
    ],
    "attention": [
        "batch_size", "q_seq_len", "kv_seq_len", "q_heads", "kv_heads", "qk_head_dim", "v_head_dim", "mode", "causal", "has_optimized", "time_ms_num_runs", 
        "time_ms_p50", "time_ms_p90",
        "time_ms_p95", "time_ms_p99",
        "time_ms_avg", "time_ms_min",
        "time_ms_max",
    ],
}

def download_from_gcs(bucket_path: str, local_dir: str):
    """
    Downloads the content of the GCS bucket path to a local directory.
    """
    fs = gcsfs.GCSFileSystem()
    gcs_path = bucket_path.replace("gs://", "").rstrip("/") + "/"

    print(f"Downloading from gs://{gcs_path} to {local_dir}...")
    os.makedirs(local_dir, exist_ok=True)
    fs.get(gcs_path, local_dir, recursive=True)

def aggregate_collectives(directories: list[str], picked_columns: list[str]) -> pd.DataFrame:
    if len(directories) == 0:
        return None
    aggregated_df = pd.DataFrame()
    for directory in directories:
        files = glob.glob(f"{directory}/*.tsv")
        for file in files:
            df = pd.read_csv(file, sep='\t')
            df["topology"] = [file.split('/')[-4].split('-')[1] for _ in range(df.shape[0])]
            aggregated_df = pd.concat([aggregated_df, df[picked_columns].rename(columns={"step_time_ms_num_runs": "num_runs"})], ignore_index=True)
    return aggregated_df

def aggregate_hbm(directories: list[str], picked_columns: list[str]) -> pd.DataFrame:
    if len(directories) == 0:
        return None
    aggregated_df = pd.DataFrame()
    for directory in directories:
        files = glob.glob(f"{directory}/*.tsv")
        for file in files:
            df = pd.read_csv(file, sep='\t')
            aggregated_df = pd.concat([aggregated_df, df[picked_columns].rename(columns={"time_ms_num_runs": "num_runs"})], ignore_index=True)
    return aggregated_df

def aggregate_host_device(directories: list[str], picked_columns: list[str]) -> pd.DataFrame:
    if len(directories) == 0:
        return None
    aggregated_df = pd.DataFrame()
    for directory in directories:
        files = glob.glob(f"{directory}/*.tsv")
        for file in files:
            df = pd.read_csv(file, sep='\t')
            aggregated_df = pd.concat([aggregated_df, df[picked_columns].rename(columns={"H2D_bw (GiB/s)_num_runs": "num_runs"})], ignore_index=True)
    return aggregated_df

def aggregate_gemm(directories: list[str], picked_columns: list[str]) -> pd.DataFrame:
    if len(directories) == 0:
        return None
    aggregated_df = pd.DataFrame()
    for directory in directories:
        files = glob.glob(f"{directory}/*.tsv")
        for file in files:
            df = pd.read_csv(file, sep='\t')
            aggregated_df = pd.concat([aggregated_df, df[picked_columns].rename(columns={"step_time_ms_num_runs": "num_runs"})], ignore_index=True)
    return aggregated_df

def aggregate_gemm_all_reduce(directories: list[str], picked_columns: list[str]) -> pd.DataFrame:
    if len(directories) == 0:
        return None
    aggregated_df = pd.DataFrame()
    for directory in directories:
        files = glob.glob(f"{directory}/*.tsv")
        for file in files:
            df = pd.read_csv(file, sep='\t')
            df["topology"] = [file.split('/')[-4].split('-')[1] for _ in range(df.shape[0])]
            aggregated_df = pd.concat([aggregated_df, df[picked_columns].rename(columns={"step_time_ms_num_runs": "num_runs"})], ignore_index=True)
    return aggregated_df

def aggregate_bmm(directories: list[str], picked_columns: list[str]) -> pd.DataFrame:
    if len(directories) == 0:
        return None
    aggregated_df = pd.DataFrame()
    for directory in directories:
        files = glob.glob(f"{directory}/*.tsv")
        for file in files:
            df = pd.read_csv(file, sep='\t')
            aggregated_df = pd.concat([aggregated_df, df[picked_columns].rename(columns={"step_time_ms_num_runs": "num_runs"})], ignore_index=True)
    return aggregated_df

def aggregate_attention(directories: list[str], picked_columns: list[str]) -> pd.DataFrame:
    if len(directories) == 0:
        return None
    aggregated_df = pd.DataFrame()
    for directory in directories:
        files = glob.glob(f"{directory}/*.tsv")
        for file in files:
            df = pd.read_csv(file, sep='\t')
            aggregated_df = pd.concat([aggregated_df, df[picked_columns].rename(columns={"step_time_ms_num_runs": "num_runs"})], ignore_index=True)
    return aggregated_df

aggregate_function = {
    "collectives": aggregate_collectives,
    "hbm": aggregate_hbm,
    "host_device": aggregate_host_device,
    "gemm": aggregate_gemm,
    "bmm": aggregate_bmm,
    "attention": aggregate_attention,
    "gemm_all_reduce": aggregate_gemm_all_reduce,
}

def aggregate_results(bucket_path: str, local_dir: str):
    categories = ["collectives", "hbm", "host_device", "gemm", "bmm", "gemm_all_reduce", "attention"]
    directories = {}
    results = {}
    for category in categories:
        directories[category] = sorted(glob.glob(f"{local_dir}/*/{category}/*", recursive=True))
        results[category] = aggregate_function[category](directories[category], columns_mapping[category])
        if results[category] is not None:
            print(f"Writing {category} results to {bucket_path}/aggregated_results/{category}.tsv")
            results[category].to_csv(f"{bucket_path}/aggregated_results/{category}.tsv", index=False, sep='\t')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download from GCS and aggregate results locally.")
    parser.add_argument("--bucket_path", type=str, required=True, help="The GCS bucket path (gs://...)")
    parser.add_argument("--local_dir", type=str, default="./results", help="Local directory to download and aggregate results.")
    args = parser.parse_args()
    
    download_from_gcs(args.bucket_path, args.local_dir)
    aggregate_results(args.bucket_path, args.local_dir)
