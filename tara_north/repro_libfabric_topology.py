#!/usr/bin/env python3
"""
Minimal reproducer for: NIXL LIBFABRIC backend topology discovery is
EFA-only and hard-fails with NIXL_ERR_BACKEND on HPE Slingshot/CXI hardware.

No vLLM, no GPU work, no multi-node setup, no KV transfer -- this is the
exact call vLLM's NixlPullConnectorWorker.__init__ makes internally
(nixl_wrapper_cls(str(uuid.uuid4()), config) -> agent.createBackend(...)),
isolated down to just backend construction.

Run on ONE node with the cxi provider present (same conda env as the
NixlConnector run):
    python3 repro_libfabric_topology.py

Expected on AWS EFA hardware: succeeds, prints SUCCESS.
Actual on HPE Slingshot/CXI (GH200, libfabric 2.3.1 -- see accompanying
report for full environment): raises nixl_cu13._bindings.nixlBackendError:
NIXL_ERR_BACKEND, with this on stderr from the C++ layer (glog/absl,
printed before the Python exception surfaces):

    E... libfabric_topology.cpp:500] fi_getinfo failed for PCIe mapping
         with provider cxi: No data available
    E... libfabric_topology.cpp:86] Failed to build PCIe to libfabric
         mapping (provider: cxi)
    E... libfabric_topology.cpp:53] Topology discovery failed - no
         suitable network providers found
    E... backend_plugin.h:102] Failed to create engine: Failed to discover
         system topology - cannot proceed without topology information
    E... nixl_agent.cpp:404] createBackend: backend creation failed for
         'LIBFABRIC'
"""
import uuid

from nixl._api import nixl_agent, nixl_agent_config

print("Creating NIXL agent, requesting LIBFABRIC backend only...")
agent = nixl_agent(str(uuid.uuid4()), nixl_agent_config(backends=["LIBFABRIC"]))
print("SUCCESS: LIBFABRIC backend created without error.")
