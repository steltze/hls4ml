import json
import os

from hls4ml.model.optimizer.optimizer import ConfigurableOptimizerPass, ModelOptimizerPass


def initialize_large_fifos(model, profiling_fifo_depth):
    """Setting all FIFO depths equal to a large value so that they can be profiled.

    Args:
        model (ModelGraph): The model to which FIFO depth optimization is applied.
        profiling_fifo_depth (int): A large non-negative integer, must be larger than the max expected depth of the FIFOs.
    """
    # initialize all the fifos to `profiling_fifo_depth` so that they will be automatically implemented in BRAMs and so
    # they will be profiled. Alternatively, "config_dataflow -override_user_fifo_depth profiling_fifo_depth" can be
    # used inside build_prj.tcl to override all FIFO depths with the specified value
    vars_to_profile = {
        k: v
        for k, v in model.output_vars.items()
        if v != model.get_output_variables()[0] and v != model.get_input_variables()[0]
    }
    for v in vars_to_profile.values():
        if v.pragma:
            v.pragma = (v.pragma[0], profiling_fifo_depth)
    return


def override_test_bench(model):
    """In order for the FIFO depth profiling to produce correct results, it is necessary for the cosimulation to
    call the top function - Vitis IP at **least twice**. The test bench produced by the Vivado Writer is
    overwritten by adding a for-loop over the top function.

    Args:
        model (ModelGraph): The model to which FIFO depth optimization is applied.
    """
    indent = "    "
    path_to_old_test_bench = f"{model.config.get_output_dir()}/{model.config.get_project_name()}_test.cpp"
    path_to_new_test_bench = f"{model.config.get_output_dir()}/{model.config.get_project_name()}_new_test.cpp"

    newline = ""
    second_part_of_testbench = False
    with open(path_to_old_test_bench) as old_test_bench:
        file_iterator = iter(old_test_bench)
        for line in file_iterator:

            if "// hls-fpga-machine-learning insert zero" in line:
                newline += indent + indent + "const unsigned BATCH_SIZE = 2;\n"
                newline += (
                    indent
                    + indent
                    + "for(unsigned batch_iteration = 0; batch_iteration < BATCH_SIZE; ++batch_iteration) {\n"
                )
                newline += line
                second_part_of_testbench = True
            elif ("// hls-fpga-machine-learning insert tb-output" in line) and second_part_of_testbench:
                newline += line
                newline += next(file_iterator)
                newline += indent + "}\n"
            else:
                newline += line

    with open(path_to_new_test_bench, "w+") as new_test_bench:
        new_test_bench.write(newline)

    # replace the old test bench with the new test bench that includes a for-loop
    os.system(f"mv {path_to_new_test_bench} {path_to_old_test_bench}")
    return


def execute_cosim_to_profile_fifos(model):
    """Execute a cosimulation with a testh bench that calls the top function - Vitis IP at **least twice**,
    to properly profile the max FIFO depths. The function will momentarily replace the initial test bench
    with a suitable one for the optimization, and after the optimizer pass, the original test bench reinitialized.

    Args:
        model (ModelGraph): The model to which FIFO depth optimization is applied.
    """
    model.write()

    override_test_bench(model)

    model.build(
        reset=False,
        csim=True,
        synth=True,
        cosim=True,
        validation=False,
        export=False,
        vsynth=False,
        fifo_opt=True,
    )

    return


def get_vitis_optimized_fifo_depths(model):
    """Parse the files generated by the cosimulation to retrieve the optimized depths for the FIFOs.
    Attention, only the FIFOs between the layers are profiled!

    Args:
        model (ModelGraph): The model to which FIFO depth optimization is applied.

    Returns:
        Dict[str, int]: A dictionary that contains the FIFO names as keys and the optimized depths as values.
    """
    # channel.zip is generated after the cosimulation and contains the chan_status*.csv files
    # in the chan_status*.csv files the max depth achieved during cosimulation can be found at the last (4th) line
    path_to_zip_file = (
        model.config.get_output_dir()
        + "/"
        + model.config.get_project_name()
        + "_prj"
        + "/solution1/.autopilot/db/channel_depth_info/"
    )
    os.system(f"unzip -q -o {path_to_zip_file}channel.zip -d {path_to_zip_file}")

    # the channel_info.csv file contains the mapping of each fifo name (i.e layer4_out_U) to the respective
    # chan_status*.csv file
    names_file_path = (
        model.config.get_output_dir()
        + "/"
        + model.config.get_project_name()
        + "_prj"
        + "/solution1/.autopilot/db/channel_info.csv"
    )

    csv_fifo_depth_files = {}
    with open(names_file_path) as names_file:
        for line in names_file:
            # if "layer" in line:
            layer_name = line.split(",")[1]
            csv_file_name = line.split(",")[3][:-1]
            csv_fifo_depth_files[layer_name] = csv_file_name

    optmized_fifo_depths = {}
    for layer_name, file_name in csv_fifo_depth_files.items():
        with open(path_to_zip_file + file_name) as chan_status_file:
            lines = chan_status_file.readlines()
            optmized_fifo_depths[layer_name[:-2]] = int(
                lines[-1]
            )  # remove "_U" from the layer name string and keep the last line of the file that contains the max depth

    return optmized_fifo_depths


def generate_max_depth_file(model, optmized_fifo_depths):
    """Generate a json file with the names of the FIFOs and their optimized depths for post-processing.
    The json file is not used by the rest of the pipeline, it is only produced for the user.

    Args:
        model (ModelGraph): The model to which FIFO depth optimization is applied.
        optmized_fifo_depths (Dict[str, int]): A dictionary that contains the FIFO names as keys and the optimized
        depths as values.
    """
    with open(model.config.get_output_dir() + "/max_depth.json", "w") as f:
        json.dump(optmized_fifo_depths, f, indent=4)


def set_optimized_fifo_depths(model, optmized_fifo_depths):
    """Set the new optimized FIFO depths.

    Args:
        model (ModelGraph): The model to which FIFO depth optimization is applied.
        optmized_fifo_depths (Dict[str, int]): A dictionary that contains the FIFO names as keys and the optimized
        depths as values.
    """

    # iterate through the layer output FIFOs
    for v in model.output_vars.values():
        if v.pragma:
            if v.name not in optmized_fifo_depths.keys():
                continue

            filtered_depth = optmized_fifo_depths[v.name]
            v.pragma = (v.pragma[0], filtered_depth)
    return


class FifoDepthOptimization(ConfigurableOptimizerPass, ModelOptimizerPass):
    def __init__(self):
        pass

    def transform(self, model):
        """Perform FIFO depth optimization between the FIFOs of all layers to reduce resource utilization as the
        initial FIFOs set by hls4ml might be larger than required. At the end of the optimization the FIFOs will
        have the largest depths achieved during cosimulation without causing any deadlocks between the layers
        (producer-consumer), thus no additional delays between the layers. In some cases, this optimization
        might lead to bigger FIFOs than initially set by the hls4ml tool in order to prevent deadlocks.

        Args:
            model (ModelGraph): The model to which FIFO depth optimization is applied.

        Raises:
            ValueError: If the FIFO depth for profiling provided by the user is not a non-negative integer.
            RuntimeError: If the IO type is not set to "io_stream".

        Returns:
            bool: The execution state of the Optimzer Pass
        """

        # use `large_fifo_depth = 0` to keep the default fifo depth
        # consider changing 100_000 either with a very very large value > of any total bram storage space
        # or via vitis 2023.2 c-simulation
        profiling_fifo_depth = getattr(self, "profiling_fifo_depth", 100_000)

        if not isinstance(profiling_fifo_depth, int) or profiling_fifo_depth < 0:
            raise ValueError("The FIFO depth for profiling (profiling_fifo_depth variable) must be a non-negative integer")

        # check axi-stream or io-stream
        if not (model.config.get_config_value("IOType") == "io_stream"):
            raise RuntimeError("To use this optimization you have to set `IOType` field to `io_stream` in the HLS config")

        initialize_large_fifos(model, profiling_fifo_depth)

        execute_cosim_to_profile_fifos(model)

        optimized_fifo_depths = get_vitis_optimized_fifo_depths(model)

        generate_max_depth_file(model, optimized_fifo_depths)

        set_optimized_fifo_depths(model, optimized_fifo_depths)

        print("[hls4ml] - FIFO optimization completed")
        return False
