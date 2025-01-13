#! /bin/bash
salloc -t 7-00:00:00 -J ${errorname}_linear_${name} -N 1 -n 64 --account=rrg-jacobsen-ab
