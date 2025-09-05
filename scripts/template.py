import os
import logging
import pandas as pd
import argparse
from RP_2026.constants import report_time_and_memory_of_script

logger = logging.getLogger(__name__)
logger.setLevel("INFO")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S")
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(script_dir), "data")

### ARGUMENTS ###


### ARGUMENTS ###

# parser = argparse.ArgumentParser(description="Run VarScan on a set of reads and report the time and memory usage")
# parser.add_argument("-a", "--arg1", default="", help="arg1")

# args = parser.parse_args()

# arg1 = args.arg1

# argparse_flags = f"--index {index_path} --t2g {t2g_path} --technology bulk ..."
# _ = report_time_and_memory_of_script(script_path, output_file = output_file_path, argparse_flags = argparse_flags, script_title = "Test script")
