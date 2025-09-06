"""RP_2026 logging utilities."""

import logging
import os
import re
import subprocess
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import torch


def setup_logger_and_tensorboard(run_name = None, log_dir = "logs", base_dir = ".", tensorboard = True):
    start_time_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    if run_name is None or run_name == "":
        run_name = f"run_{start_time_string}"

    log_file_path = os.path.join(log_dir, f"{run_name}.log")

    if os.path.exists(log_file_path):
        raise FileExistsError(f"Log file {log_file_path} already exists. Please choose a different run name.")
    print(f"Logging to {log_file_path}")

    if tensorboard:
        tensorboard_dir = os.path.join(base_dir, "runs", run_name)
        if os.path.exists(tensorboard_dir):
            print(f"Warning: TensorBoard log directory {tensorboard_dir} already exists and may be overwritten.")
        writer = SummaryWriter(tensorboard_dir)
        os.makedirs(tensorboard_dir, exist_ok=True)
        print(f"TensorBoard logs to {tensorboard_dir} - visualize with `tensorboard --logdir {tensorboard_dir}`")

    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel("INFO")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, writer

def set_seed(seed: int = 42):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensures deterministic algorithms (may be slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def report_time_and_memory_of_script(script_path, argparse_flags=None, output_file=None, script_title=None):
    # Run the command and capture stderr, where `/usr/bin/time -l` outputs its results
    system = os.uname().sysname
    time_flag = "-v" if system == "Linux" else "-l"
    command = f"/usr/bin/time {time_flag} python3 {script_path}"

    if argparse_flags:
        command += f" {argparse_flags}"

    try:
        start_time = time.perf_counter()
        result = subprocess.run(command, shell=True, text=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, check=True)
        runtime = time.perf_counter() - start_time
    except Exception as e:
        print(f"Error running command {command}: {e}")
        return None

    minutes, seconds = divmod(runtime, 60)
    time_message = f"Runtime: {minutes} minutes, {seconds:.2f} seconds"
    if script_title:
        time_message = f"{script_title } " + time_message
    print(time_message)

    # Extract the "maximum resident set size" line using a regex
    memory_re = r"Maximum resident set size \(kbytes\): (\d+)" if system == "Linux" else r"\s+(\d+)\s+maximum resident set size"
    match_memory = re.search(memory_re, result.stderr)
    if match_memory:
        peak_memory = int(match_memory.group(1))  # Capture the numeric value
        # Determine units (bytes or KB)
        if "kbytes" in match_memory.group(0):
            peak_memory *= 1024  # Convert KB to bytes
        peak_memory_readable_units = peak_memory / (1024**2)  # MB
        unit = "MB"
        if peak_memory_readable_units > 1000:
            peak_memory_readable_units = peak_memory_readable_units / 1024  # GB
            unit = "GB"
        memory_message = f"Peak memory usage: {peak_memory_readable_units:.2f} {unit}"
        if script_title:
            memory_message = f"{script_title } " + memory_message
        print(memory_message)
    else:
        raise ValueError("Failed to find 'maximum resident set size' in output.")

    if output_file:
        if os.path.dirname(output_file):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
        file_mode = "a" if os.path.isfile(output_file) else "w"
        with open(output_file, file_mode, encoding="utf-8") as f:
            f.write(time_message + "\n")
            f.write(memory_message + "\n")

    return (runtime, peak_memory)  # Return the runtime (seconds) and peak memory usage (bytes)