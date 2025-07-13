#!/usr/bin/env python3
"""NumPy selector script with NPU Manager integration for P/E core optimization"""

import os
import sys
import subprocess
import psutil
from pathlib import Path


class CoreSelector:
    """Manages core selection for P-cores vs E-cores."""

    def __init__(self):
        self.p_cores = []
        self.e_cores = []
        self._detect_cores()

    def _detect_cores(self):
        """Detect P-cores and E-cores using lscpu."""
        try:
            result = subprocess.run(['lscpu', '-p=CPU,CORE,MAXMHZ'],
                                  capture_output=True, text=True)

            if result.returncode == 0:
                core_info = {}
                for line in result.stdout.strip().split('\n'):
                    if line.startswith('#'):
                        continue
                    parts = line.split(',')
                    if len(parts) >= 3:
                        cpu_id = int(parts[0])
                        max_freq = int(parts[2]) if parts[2] else 0
                        core_info[cpu_id] = max_freq

                # P-cores have higher frequency
                if core_info:
                    avg_freq = sum(core_info.values()) / len(core_info)
                    self.p_cores = [cpu for cpu, freq in core_info.items()
                                   if freq > avg_freq * 1.1]
                    self.e_cores = [cpu for cpu, freq in core_info.items()
                                   if freq <= avg_freq * 1.1]
        except:
            # Fallback
            cpu_count = psutil.cpu_count(logical=False)
            self.p_cores = list(range(cpu_count // 2))
            self.e_cores = list(range(cpu_count // 2, cpu_count))

    def set_affinity(self, core_type: str):
        """Set CPU affinity for current process."""
        if core_type == 'p':
            cores = self.p_cores
            print(f"Setting affinity to P-cores: {cores}")
        elif core_type == 'e':
            cores = self.e_cores
            print(f"Setting affinity to E-cores: {cores}")
        else:  # auto
            # Use P-cores for low latency by default
            cores = self.p_cores[:4] if self.p_cores else [0, 1, 2, 3]
            print(f"Auto-selecting cores: {cores}")

        # Set affinity
        p = psutil.Process()
        p.cpu_affinity(cores)

        # Set OpenMP environment variables
        os.environ['OMP_NUM_THREADS'] = str(len(cores))
        os.environ['MKL_NUM_THREADS'] = str(len(cores))
        os.environ['NUMEXPR_NUM_THREADS'] = str(len(cores))

        # Set KMP affinity for Intel MKL
        cpu_list = ','.join(map(str, cores))
        os.environ['KMP_AFFINITY'] = f"granularity=fine,proclist=[{cpu_list}],explicit"

        # For OpenVINO
        os.environ['OV_CPU_THREADS_NUM'] = str(len(cores))

        return cores


def main():
    if len(sys.argv) < 2:
        print("Usage: numpy_select.py [p|e|auto] [command...]")
        print("  p    - Use P-cores (performance)")
        print("  e    - Use E-cores (efficiency)")
        print("  auto - Automatic selection")
        sys.exit(1)

    core_type = sys.argv[1].lower()
    if core_type not in ['p', 'e', 'auto']:
        print(f"Invalid core type: {core_type}")
        sys.exit(1)

    # Initialize selector
    selector = CoreSelector()

    print(f"Detected {len(selector.p_cores)} P-cores: {selector.p_cores}")
    print(f"Detected {len(selector.e_cores)} E-cores: {selector.e_cores}")

    # Set affinity
    selected_cores = selector.set_affinity(core_type)

    # If there's a command to run
    if len(sys.argv) > 2:
        command = sys.argv[2:]
        print(f"\nExecuting: {' '.join(command)}")
        print("-" * 60)

        # Execute the command with the set affinity
        try:
            result = subprocess.run(command)
            sys.exit(result.returncode)
        except KeyboardInterrupt:
            print("\nInterrupted")
            sys.exit(130)
        except Exception as e:
            print(f"Error executing command: {e}")
            sys.exit(1)
    else:
        # Just print the configuration
        print("\nEnvironment configured for NumPy/OpenVINO optimization")
        print("Current process affinity:", psutil.Process().cpu_affinity())


if __name__ == "__main__":
    main()
