import h5py
import numpy as np
import numpy.typing as npt
from scipy.io import savemat


def load_npz(file: str) -> dict[str, npt.NDArray]:
    return dict(np.load(file))


def load_lcls_cube(file: str) -> dict[str, npt.NDArray]:
    with h5py.File(file, "r") as hf:
        images = np.array(hf["jungfrau512k_data"])
        if "delay" in hf:
            delays = np.array(hf["delay"])
        # FIXME: Find How to get angles
        elif "scan__xcs_gon_rot" in hf:
            delays = np.zeros(images.shape[0])
        else:
            delays = np.zeros(images.shape[0])
    return {"poff": images, "delay": delays}


def save_npz(file: str, data: dict[str, npt.NDArray]) -> None:
    np.savez(file, **data)


def save_mat(file: str, data: dict[str, npt.NDArray]) -> None:
    savemat(file, {"data": data["poff"], "delay": data["delay"]})


def convert_data(
    input_format: str, output_format: str, input_file: str, save_file: str
) -> None:
    match input_format:
        case "lcls_cube":
            data = load_lcls_cube(input_file)
        case "npz":
            data = load_npz(input_file)
        case _:
            raise ValueError(f"Unsupported input format: {input_format}")

    match output_format:
        case "npz":
            save_npz(save_file, data)
        case "mat":
            save_mat(save_file, data)
        case _:
            raise ValueError(f"Unsupported output format: {output_format}")


if __name__ == "__main__":
    import os

    root: str = "Z:\\241016_LCLS\\cube"
    for load_file_name in os.listdir(root):
        base_name, extension = os.path.splitext(load_file_name)
        if extension != ".h5":
            continue

        names: list[str] = base_name.split("_")
        run_n: int = int(names[2][3:])
        scan_type: str = "_".join(names[3:])
        if "delay" not in scan_type:
            continue

        save_file_name: str = "_".join(names) + ".mat"
        save_file: str = os.path.join(
            "Z:\\241016_LCLS\\analysis_data\\mat_files", save_file_name
        )
        load_file: str = os.path.join(root, load_file_name)
        convert_data("lcls_cube", "mat", load_file, save_file)
        print(save_file)
