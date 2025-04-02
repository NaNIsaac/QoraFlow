import os

import pandas as pd
import matplotlib.pyplot as plt

from src.config.config import load_config, ExpConfig
from src.filesystem import get_run_scan_dir


def power_compare(power_run: dict[int, int]) -> pd.DataFrame:
    config: ExpConfig = load_config()

    power_df = pd.DataFrame(columns=["power", "delay", "com_y", "com_x", "intensity"])

    output_dir: str = config.path.output_dir
    for power, run_num in power_run.items():
        data_file = os.path.join(get_run_scan_dir(output_dir, run_num, 1), "roi_small", "data.csv")
        data_df = pd.read_csv(data_file, index_col=0)
        delays = data_df.index.values
        com_y = data_df["poff_com_y"].values - data_df["pon_com_y"].values
        com_x = data_df["poff_com_x"].values - data_df["pon_com_x"].values
        intensity = data_df["poff_intensity"].values - data_df["pon_intensity"].values

        temp_df = pd.DataFrame({
            "power": power,
            "delay": delays,
            "com_y": com_y,
            "com_x": com_x,
            "intensity": intensity
        })

        power_df = pd.concat([power_df, temp_df], ignore_index=True)

    return power_df


def plot_power_compare(power_df: pd.DataFrame):

    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'serif'

    fig, axs = plt.subplots(3, 1, figsize=(8, 8))

    for power, group in power_df.groupby("power"):
        axs[0].plot(group["delay"], group["com_y"], ".-", label=f'{power}%')
        axs[1].plot(group["delay"], group["com_x"], ".-", label=f'{power}%')
        axs[2].plot(group["delay"], group["intensity"], ".-", label=f'{power}%')
        axs[0].text(group["delay"].iloc[-1] + 1, group["com_y"].iloc[-1], f'{power}%', va='center', ha='left')
        axs[1].text(group["delay"].iloc[-1] + 1, group["com_x"].iloc[-1], f'{power}%', va='center', ha='left')
        axs[2].text(group["delay"].iloc[-1] + 1, group["intensity"].iloc[-1], f'{power}%', va='center', ha='left')

    fig.suptitle("Different Powers", fontsize=14)

    axs[0].set_title('COM x over Delays for Different Powers', fontsize=12)
    axs[1].set_title('COM y over Delays for Different Powers', fontsize=12)
    axs[2].set_title('Intensity over Delays for Different Powers', fontsize=12)

    axs[0].set_ylabel('COM x', fontsize=12)
    axs[1].set_ylabel('COM y', fontsize=12)
    axs[2].set_ylabel('Intensity', fontsize=12)

    axs[2].set_xlabel('Delays', fontsize=12)

    for ax in axs:
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    return fig


def save_power_df(power_df: pd.DataFrame, root: str):
    for power, group in power_df.groupby("power"):
        df_filtered = group[['delay', 'com_y', 'com_x', 'intensity']]
        file = os.path.join(root, f'power={power:0>2}_run={power_run_dict[power]:0>3}.csv')
        df_filtered.to_csv(file, index=False)


if __name__ == "__main__":
    power_runs = [89, 90]
    power_run_dict = dict(enumerate(power_runs, 1))

    power_df = power_compare(power_run_dict)
    power_df = power_df[power_df["delay"] < 5]
    plot_power_compare(power_df)

    plt.show()
    # save_power_df(power_df, ".")
