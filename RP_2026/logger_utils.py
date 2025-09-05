"""RP_2026 logging utilities."""

import logging
import os
import re
import subprocess
import time

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