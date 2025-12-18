import os
import platform
import socket

afqmc_config = {"use_gpu": False, "single_precision": False}


def is_jupyter_notebook():
    try:
        from IPython.core.getipython import get_ipython

        ipython = get_ipython()
        if ipython is not None and "IPKernelApp" in ipython.config:
            return True
        else:
            return False
    except ImportError:
        return False


def setup_jax():
    from jax import config

    if afqmc_config["single_precision"] == False:
        config.update("jax_enable_x64", True)
    # breaking change in random number generation in jax v0.5
    config.update("jax_threefry_partitionable", False)

    if afqmc_config["use_gpu"] == True:
        config.update("jax_platform_name", "gpu")
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        hostname = socket.gethostname()
        system_type = platform.system()
        machine_type = platform.machine()
        processor = platform.processor()
        print(f"# Hostname: {hostname}")
        print(f"# System Type: {system_type}")
        print(f"# Machine Type: {machine_type}")
        print(f"# Processor: {processor}")
        uname_info = platform.uname()
        print("# Using GPU.")
        print(f"# System: {uname_info.system}")
        print(f"# Node Name: {uname_info.node}")
        print(f"# Release: {uname_info.release}")
        print(f"# Version: {uname_info.version}")
        print(f"# Machine: {uname_info.machine}")
        print(f"# Processor: {uname_info.processor}")
    else:
        afqmc_config["use_gpu"] = False
        config.update("jax_platform_name", "cpu")
        os.environ["XLA_FLAGS"] = (
            "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
        )
