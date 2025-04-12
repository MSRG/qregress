#! /bin/bash
salloc -t 0-01:00:00  -J 0.1_A2_HWE-CNOT  -N 1  --cpus-per-task=16  --account=rrg-jacobsen-ab
