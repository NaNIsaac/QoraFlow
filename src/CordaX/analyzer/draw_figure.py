from typing import TYPE_CHECKING

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def patch_rectangle(image: npt.NDArray, y1: int, y2: int, x1: int, x2: int) -> "Figure":
    patched_image = np.copy(image)

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(patched_image)

    patch = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="r", facecolor="none"
    )

    ax.add_patch(patch)

    ax.set_title("Patched Image")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

    return fig


class Visualizer:
    def __init__(self, data_df: pd.DataFrame, title: str = ""):
        self.data_df = data_df
        self.title = title

    def draw_intensity_figure(self) -> "Figure":
        delay = self.data_df.index
        poff_intensity = self.data_df["poff_intensity"]
        pon_intensity = self.data_df["pon_intensity"]

        fig, ax = plt.subplots()
        fig.suptitle(self.title)
        ax.plot(delay, poff_intensity, label="poff_intensity", marker="o")
        ax.plot(delay, pon_intensity, label="pon_intensity", marker="x")

        ax.set_xlabel("Delay [ps]")
        ax.set_ylabel("Intensity [a.u.]")
        ax.set_title("Intensity vs Delay")
        ax.legend()

        plt.tight_layout()

        return fig


    def draw_com_figure(self) -> "Figure":
        delay = self.data_df.index
        poff_com_x = self.data_df["poff_com_x"]
        pon_com_x = self.data_df["pon_com_x"]
        poff_com_y = self.data_df["poff_com_y"]
        pon_com_y = self.data_df["pon_com_y"]

        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(self.title)
        # Plot poff_com_x and pon_com_x
        axs[0].plot(delay, poff_com_x, label="poff_com_x", marker="o", color="b")
        axs[0].plot(delay, pon_com_x, label="pon_com_x", marker="x", color="r")
        axs[0].set_title("COM X Position")
        axs[0].set_ylabel("Position X($Q_z$) [Å$^{-1}$]")
        axs[0].legend()

        # Plot poff_com_y and pon_com_y
        axs[1].plot(delay, poff_com_y, label="poff_com_y", marker="o", color="g")
        axs[1].plot(delay, pon_com_y, label="pon_com_y", marker="x", color="y")
        axs[1].set_title("COM Y Position")
        axs[1].set_xlabel("Delay [ps]")
        axs[1].set_ylabel("Position Y(2$\\theta$) [Å$^{-1}$]")
        axs[1].legend()

        plt.tight_layout()

        return fig


    def draw_intensity_diff_figure(self) -> "Figure":
        delay = self.data_df.index
        poff_intensity = self.data_df["poff_intensity"]
        pon_intensity = self.data_df["pon_intensity"]

        # Calculate the difference between pon_intensity and poff_intensity
        intensity_difference = pon_intensity - poff_intensity

        fig, ax = plt.subplots()
        fig.suptitle(self.title)
        ax.plot(
            delay,
            intensity_difference,
            label="Intensity Difference (poff - pon)",
            marker="o",
        )

        ax.set_xlabel("Delay [ps]")
        ax.set_ylabel("Intensity Difference [a.u.]")
        ax.set_title("Intensity Difference (pon - poff) vs Delay")
        ax.legend()

        plt.tight_layout()

        return fig


    def draw_com_diff_figure(self) -> "Figure":
        delay = self.data_df.index
        poff_com_x = self.data_df["poff_com_x"]
        pon_com_x = self.data_df["pon_com_x"]
        poff_com_y = self.data_df["poff_com_y"]
        pon_com_y = self.data_df["pon_com_y"]

        # Calculate the difference between pon_com_x and poff_com_x
        com_x_difference = poff_com_x - pon_com_x

        # Calculate the difference between pon_com_y and poff_com_y
        com_y_difference = poff_com_y - pon_com_y

        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(self.title)
        # Plot com_x_difference
        axs[0].plot(
            delay,
            com_x_difference,
            label="COM X Difference (poff - pon)",
            marker="o",
            color="b",
        )
        axs[0].set_title("COM X Position Difference (poff - pon)")
        axs[0].set_ylabel("Position X Difference($Q_z$) [Å$^{-1}$]")
        axs[0].legend()

        # Plot com_y_difference
        axs[1].plot(
            delay,
            com_y_difference,
            label="COM Y Difference (poff - pon)",
            marker="o",
            color="g",
        )
        axs[1].set_title("COM Y Position Difference (poff - pon)")
        axs[1].set_xlabel("Delay [ps]")
        axs[1].set_ylabel("Position Y Difference($Q_xy$) [Å$^{-1}$]")
        axs[1].legend()

        plt.tight_layout()

        return fig
