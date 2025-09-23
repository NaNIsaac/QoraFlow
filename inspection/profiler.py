import cProfile
import io
import logging
import pstats
from pathlib import Path

from CordaX.config import ConfigManager, ExpConfig
from CordaX.filesystem import get_run_scan_dir
from CordaX.integrator.loader import PalXFELLoader


def main() -> None:
    """Profile program with cProfile module and visualize with tuna."""
    config: ExpConfig = ConfigManager.load_config()
    load_dir: str = config.path.load_dir
    file: Path = get_run_scan_dir(load_dir, 161, 1, sub_path="p0010.h5")

    logging_file: Path = Path("logs/profiling/profiling.log")
    # logging Setting
    logging.basicConfig(filename=logging_file, level=logging.INFO, format="%(message)s")

    # Create a profiler object
    profiler = cProfile.Profile()

    # Enable the profiler
    profiler.enable()

    # Run the main function from the other file
    PalXFELLoader(file)
    # processing_main.main()

    # Disable the profiler
    profiler.disable()

    # Create a Stats object and sort the results by cumulative time
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats("cumulative")

    # Redirect the stats output to a StringIO object
    output_stream = io.StringIO()
    stats.stream = output_stream

    # Print the stats to the StringIO object
    stats.print_stats()

    # Get the captured output
    profiling_results = output_stream.getvalue()

    # Log the captured output
    logging.info(profiling_results)

    # Save the profiling results to a file
    stats_file: Path = Path("logs/profiling/profiling.prof")
    stats.dump_stats(stats_file)

    print(f"Profiling results logged to '{logging_file}'")
    print(f"Profiling results logged to '{stats_file}'")


if __name__ == "__main__":
    ConfigManager.initialize("config.yaml")
    main()
