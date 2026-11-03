def register():
    from vllm.logger import init_logger
    from vllm.models.muse_spark.nvidia import muse_moe_runner as m

    log = init_logger("muse_dp_patch")
    orig = m._validate_route_major_deployment

    def guard(moe_config):
        pc = moe_config.moe_parallel_config
        if (pc.dp_size > 1 and pc.tp_size == 1 and pc.pcp_size == 1
                and pc.use_ag_rs_all2all_kernels):
            m.validate_deployment(moe_config, kernel_routes=False)
            log.warning_once("muse_dp_patch: DP%d allowed (EP, ag_rs)", pc.dp_size)
            return
        orig(moe_config)

    m._validate_route_major_deployment = guard
