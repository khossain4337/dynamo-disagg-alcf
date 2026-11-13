# Run list — `multi_model/`

One line per script: what it does, why it exists. **Current** means used by
today's Muse-Spark runs. Each `qsub_*` launches the `run_*` beside it in its
table; submit with plain `qsub <file>`.

## Environment

| File | What | Why | Status |
|---|---|---|---|
| `get_env_vllm_0.32.1rc1_disaggpp_nixl_1.5.2.sh` | Builds the conda env: vLLM `0.32.1rc1.dev562`, NIXL 1.5.2, then runs the decode-PP patch. | One env serves colocated, router and P:D, with NIXL on the decode side under PP. | Current |
| `patch_env_vllm_0.32.1rc1_disaggpp_nixl_1.5.2.sh` | Restores clean vLLM files in the env, then applies `muse_spark/nixl_decode_pp_9e3db48873.diff`. | Upstream vLLM rejects NIXL with decode-side PP; the diff lifts that. Never run while a job uses the env. | Current |
| `env_vllm_0.32.1rc1_disaggpp_nixl_1.5.2.sh` | Activates the disaggpp env; puts libfabric 2.7.0 ahead of the Cray 2.3.1 module. | libfabric 2.7.0 carries the cxi write-with-data that the NIXL push needs. | Current |
| `emit_common_env_disaggpp_nixl_1.5.2.sh` | Writes a run's `common_env.sh`: every variable the vLLM ranks and the bench client see. | mpiexec and ssh start bare shells; one file keeps every rank on the same env. | Current |
| `get_env_vllm_0.32.1rc1_nixl_1.5.2.sh` | Same env build without the decode-PP patch. | Base for the disaggpp env. | Superseded by disaggpp |
| `env_vllm_0.32.1rc1_nixl_1.5.2.sh` | Activates the unpatched 1.5.2 env. | — | Superseded by disaggpp |
| `emit_common_env_nixl_1.5.2.sh` | `common_env.sh` writer for the unpatched 1.5.2 env. | — | Superseded by disaggpp |
| `get_env_vllm_0.32.1rc1_nixl_1.5.1.sh` | Env build with NIXL 1.5.1 from source against Cray libfabric 2.3.1. | The 1.5.1 wheel needed a newer libfabric symbol than Cray ships. | Superseded by 1.5.2 |
| `env_vllm_0.32.1rc1_nixl_1.5.1.sh` | Activates the 1.5.1 env. | — | Superseded by 1.5.2 |
| `download_model_vllm_0.32.1rc1_nixl_1.5.2.sh` | Fetches a model into the shared HF cache in the background; `VERIFY=1` only checks it. | Survives the ssh session; checks every shard is complete. Defaults to Muse-Spark. | Current |

## Builds

| File | What | Why | Status |
|---|---|---|---|
| `build_libfabric_v2.7.0.sh` | Builds upstream libfabric 2.7.0 (cxi + CUDA) with json-c and curl, on a compute node. | Cray 2.3.1 lacks the cxi write-with-data the NIXL push uses. | Current |
| `libfabric_v2.7.0_build_check.sh` | Checks the 2.7.0 build, then runs the `cq_data_write` reproducer. | Proves the build and the write path before any vLLM run. | One-off check |
| `build_nccl_tests_nccl_2.30.7.sh` | Builds nccl-tests against the env's wheel NCCL. | Needed by the NCCL plugin checks. | One-off build |

## Fabric checks

| File | What | Why | Status |
|---|---|---|---|
| `run_nccl_plugin_check_nixl_1.5.2.sh` | aws-ofi-nccl plugin vs socket, same 2 nodes. | Showed the plugin works; socket gets 7–10% of its rate (`-10-30c`). | Done |
| `run_nccl_plugin_check_nixl_1.5.2_lf2.7.0.sh` | Same check, plugin arm only, over libfabric 2.7.0. | Confirms NCCL still works after the libfabric swap. | Done |
| `run_nixl_2rank_transfer_nixl_1.5.2_lf2.7.0.sh` | 2-rank NIXL transfer test over libfabric 2.7.0; `--op WRITE` supported. | Tests the NIXL push path outside vLLM. | Current |
| `run_nixl_2rank_transfer_nixl_1.5.2.sh` | 2-rank NIXL transfer test, NIXL 1.5.2 on Cray 2.3.1. | READ path only; WRITE is blocked on Cray 2.3.1. | Superseded by lf2.7.0 |
| `run_nixl_2rank_transfer_nixl_1.5.1.sh` | 2-rank NIXL transfer test, NIXL 1.5.1 env. | — | Superseded by 1.5.2 |

## `muse_spark/` — sweep (current)

Muse-Spark-Open-816B-A42B, TP4 x PP8 per instance, EP off, 16 serving nodes.

| File | What | Why | Status |
|---|---|---|---|
| `run_muse_spark_router_sweep_tp4pp8_N16.sh` | Router arm: 2 colocated instances behind vllm-router (round robin). One boot, one warmup, then every cell in `CELLS`. | The router x2 baseline the P:D arm is compared against. | Current |
| `run_muse_spark_pd_sweep_tp4pp8_N16.sh` | P:D arm, 1P:1D: P and D instances, NIXL push of the KV, toy proxy in front. One boot, one warmup, then every cell in `CELLS`. `engine_ttl:0` since 11-12 (edited in place). | The disaggregated arm under test. | Current |
| `qsub_muse_spark_sweep_rt_smoke_N16.sh`, `..._pd_smoke_N16.sh` | Six short-to-long cells ending in one (8k, 1M) request per stream, GPU watch kept on. | Proves both arms, the NIXL 1M push, the hang watch and the 48 h client timeout before the 48 h pilots. | Done (11417, 11418) |
| `grid/qsub_muse_spark_grid_<rt\|pd>_<prefill\|decode>_N16_osl<O>[_isl<I>].sh` (59) | Grid sweep: ISL, OSL in 8k–512k, every concurrency up to the KV cap, 2 prompts per stream. One whole (ISL, OSL) ladder per job; prefill-heavy (ISL > OSL) jobs hold one OSL row. `RUN_TAG` = `grid-<arm>-<kind>-osl<O>[-isl<I>]`; `#PBS -N` = `g<r\|p><p\|d><O>[_<I>]`. 48 h, six long pairs 54 h. | The main Muse-Spark comparison, router x2 vs P:D. | Current |
| `grid/test_pbs_chain.sh` | Three 1-node jobs chained with `afterany`, the middle one failing; `check` prints their end states. | Proves a failed job releases the next before the grid runs as two chains (40-node limit). | Done (11532–11534, pass) |
| `grid/submit_grid_chains.sh` | Submits the 59 grid qsubs as two `afterany` chains, router (30) and P:D (29); prefill-heavy rows first, then decode-heavy by OSL rising. Stamp file `RUNS/grid_chains_submitted.txt` blocks a second run and logs `jobid chain file`. | Site limit 40 nodes: one router and one P:D job at a time (38). | Current |
| `qsub_muse_spark_lean_rt_prefill_N16.sh`, `..._pd_prefill_N16.sh` | Lean pilot (1M, OSL) at c = 8, OSL = 8k, 16k, 32k, 40k. 12 h. | Router decode under prefill stalls vs P:D; where P:D shifts from P-bound to D-bound. | Done (11434, 11435) |
| `qsub_muse_spark_lean_rt_dec8_N16.sh`, `..._pd_dec8_N16.sh` | Lean pilot (8k, 1M) at c = 8. Router 24 h, P:D 48 h. | 8 streams on D (the PP depth); router checks the step-time model's ~30 ms. | Done (11436; P:D 11511 after 11437 failed on `engine_ttl`) |
| `qsub_muse_spark_sweep_rt_prefill_N16.sh`, `..._pd_prefill_N16.sh` | Pilot (1M, 8k) at c = 4, 8, 16, 32. | Prefill rate at full length; what happens past the KV cap. | Superseded by `lean_*` |
| `qsub_muse_spark_sweep_rt_decode_a_N16.sh`, `..._pd_decode_a_N16.sh` | Pilot (8k, 1M) at c = 4, 8. | Decode cost over a full 1M context. | Superseded by `lean_*` |
| `qsub_muse_spark_sweep_rt_decode_b_N16.sh`, `..._pd_decode_b_N16.sh` | Pilot (8k, 1M) at c = 16, at or over the cap. | Shows queueing vs preemption past the KV cap. | Superseded by `lean_*` |
| `hang_watch.sh` | Started by both sweep runners. Fires on a vLLM ERROR line, a rank death, or 30 min with no progress while requests run; stops the bench. | A dead PP stage can leave the server answering `/health` while nothing moves. | Current |
| `bench_serve_long_timeout.py` | `vllm bench serve` with a 48 h request timeout instead of the built-in 6 h. | A (8k, 1M) request runs for hours. | Current |
| `toy_proxy_server_9e3db48873.py` | Verbatim copy of vLLM's NIXL test proxy at our vLLM pin. | Routes each request to P, then D, in the P:D arms. | Current |
| `nixl_decode_pp_9e3db48873.diff` | Patch to vLLM's NIXL push connector (3 files), applied by `../patch_env_*`. | Lets the D side run with PP. | Current |

## `muse_spark/` — diagnostics

| File | What | Why | Status |
|---|---|---|---|
| `run_muse_spark_diag_mem_tp4pp8_N8.sh` | One instance, an ISL ladder in `CELLS`, GPU memory traced through every cell. | Finds where long inputs run out of GPU memory. | Current tool |
| `qsub_muse_spark_diag_mem_mnbt16k_N8.sh` | 16k–128k at batched tokens 16,384. | OOM'd at 128k: the pilots' failure, reproduced. | Done |
| `qsub_muse_spark_diag_mem_mnbt8k_N8.sh` | 16k–128k at batched tokens 8,192. | Served 128k; set 8,192 for the sweeps. | Done |
| `qsub_muse_spark_diag_mem_mnbt8k_1m_N8.sh` | 128k–1M at 8,192. | Memory flat to 1M; 1M time to first token ≤ 152 s. | Done |
| `run_muse_spark_pilot_tp16pp2_N8.sh` | One TP16 x PP2 instance, one wave per cell (num_prompts = c). | Tests whether 16 GPUs per stage decode faster than TP4 x PP8 despite cross-node all-reduces. | Pilot |
| `qsub_muse_spark_pilot_tp16pp2_short_N8.sh` | (1024, 256) at c 1, 4, 16. | Clean decode speed at batch c; read first. | Pilot |
| `qsub_muse_spark_pilot_tp16pp2_long_N8.sh` | Warmup shape (128k, 128) c=16, then 128k / 512k / 1M with 1,024 out at c=4. | Peak memory and long-context decode; run after the short one reads. | Pilot |

## `muse_spark/` — correctness

| File | What | Why | Status |
|---|---|---|---|
| `run_muse_spark_parity_tp4pp8_N8.sh` + `qsub_muse_spark_parity_tp4pp8_N8.sh` | Greedy outputs from one colocated instance. | Reference for the P:D parity check. | Done; rerun on a vLLM bump |
| `run_muse_spark_parity_pd_tp4pp8_N16.sh` + `qsub_muse_spark_parity_pd_tp4pp8_N16.sh` | Same prompts through P:D. | Shows the patched NIXL push gives the same tokens. | Done; rerun on a vLLM bump |
| `parity_client.py` | Fixed prompts, temperature 0, 256 tokens, top-2 logprobs. | Used by both parity runners. | Done |
| `parity_compare.py` | First token where two runs differ, with the logprob gap there. | Separates real divergence from near-ties. | Done |

## `muse_spark/` — earlier arms (one ladder per boot)

| File | What | Why | Status |
|---|---|---|---|
| `run_muse_spark_router_tp4pp8_N16.sh` + `qsub_muse_spark_router_tp4pp8_ep{0,1}_N16.sh` | Router arm, one (ISL, OSL) ladder per boot, EP off or on. | First router runs. | Superseded by `router_sweep` |
| `run_muse_spark_pd_tp4pp8_N16.sh` + `qsub_muse_spark_pd_tp4pp8_ep{0,1}_N16.sh` | P:D arm, one ladder per boot, EP off or on. | First P:D bench runs. | Superseded by `pd_sweep` |

## `muse_spark/` — single-instance pilots, (1024, 256)

All one instance on 8 nodes, one ladder; each answered one question.
`qsub_...` is the qsub with the runner's suffix.

| File | What | Why | Status |
|---|---|---|---|
| `run_muse_spark_pilot_tp4pp8_N8.sh` + `qsub_..._pilot_tp4pp8_N8.sh` | TP4 x PP8, EP on. | First baseline. | Done |
| `run_muse_spark_pilot_tp4pp8_noep_N8.sh` + `qsub_..._pilot_tp4pp8_noep_N8.sh` | Same, EP off. | EP on vs off. | Done |
| `run_muse_spark_pilot_tp32_N8.sh` + `qsub_..._pilot_tp32_N8.sh` | TP32 + EP32. | Alternative layout. Failed at model build. | Rejected |
| `run_muse_spark_pilot_tp4dp8_N8.sh` + `qsub_..._pilot_tp4dp8_N8.sh` | TP4 x DP8 x EP32, needs `plugins/muse_dp_patch`. | Alternative layout for 1M. Failed on memory (`-11-03b`). | Rejected |
| `plugins/muse_dp_patch.py` | vLLM plugin that allows DP > 1 with EP for Muse-Spark. | Only for the DP8 pilot. | Rejected with it |
| `run_muse_spark_pilot_tp4pp8_noep_mrc_N8.sh` + `qsub_..._noep_mrc{on,off}_N8.sh` | EP off, libfabric memory-registration cache monitors on or off, disaggpp env. | Cache on (`userfaultfd`) kept for the sweeps. | Done |
| `run_muse_spark_pilot_tp4pp8_nathan_N8.sh` + `qsub_..._nathan_mrc{on,off}_N8.sh` | Nathan's serve args verbatim, cache on or off. | Compare against Nathan's faster decode. | Done; chase parked |
| `run_muse_spark_pilot_tp4pp8_nathan_kdreg2_N8.sh` + `qsub_..._nathan_kdreg2_N8.sh` | Nathan's args, kdreg2 cache monitor. | kdreg2 = `userfaultfd` within noise. | Done |
| `run_muse_spark_pilot_tp4pp8_nathan_no_ncclplugin_N8.sh` + `qsub_...` | Nathan's args, NCCL over TCP sockets instead of the aws-ofi-nccl plugin. | Plugin cleared as a cause (`-11-09`). | Done |
| `run_muse_spark_pilot_tp4pp8_noep_no_ncclplugin_N8.sh` + `qsub_...` | Our args, no NCCL plugin. | Same question, our args. | Done |
| `run_muse_spark_pilot_tp4pp8_noep_mns128_no_ncclplugin_N8.sh` + `qsub_...` | Our args with Nathan's 128 seqs / 8,192 batched tokens, no NCCL plugin. | Isolates Nathan's batch limits. | Done |
| `run_muse_spark_pilot_tp4pp8_noep_ncclvars_N8.sh` + `qsub_...` | Adds `NCCL_GRAPH_REGISTER=0 NCCL_PROTO=^LL128`. | Nathan's NCCL variables, tested alone. | Done |
| `run_muse_spark_pilot_tp4pp8_noep_pfxcache_N8.sh` + `qsub_...` | Prefix caching on; a repeated c hits the cache. | Sweeps keep it off (random inputs). | Done |
