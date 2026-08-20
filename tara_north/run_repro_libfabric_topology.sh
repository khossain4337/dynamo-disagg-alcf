#!/bin/bash
#
source env_for_libfabric_topology_error.sh                                                                                                                                                                                                           
LD_PRELOAD=$PWD/fi_getinfo_shim.so python3 repro_libfabric_topology.py
#source env_for_libfabric_topology_error.sh && python3 repro_libfabric_topology.py
