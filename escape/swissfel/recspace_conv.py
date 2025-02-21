import xrayutilities as xu
from collections import deque
import jungfrau_utils
import numpy as np
from functools import lru_cache
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


class SixCircleBernina:
    def __init__(self, pixsz=0.075, JF_ID="JF01T03V01"):
        self.pixsz = pixsz
        self.JF_ID = JF_ID
        self.angles = ["mu", "eta", "chi", "phi", "gamma", "delta"]
        self.qconv = xu.QConversion(("y+", "x-", "z+", "x-"), ("y+", "x-"), (0, 0, 1))

        self.rot_sf2you = (
            Rotation.from_rotvec(-np.pi / 2 * np.array([0, 0, 1])).as_matrix()
            @ Rotation.from_rotvec(-np.pi / 2 * np.array([0, 1, 0])).as_matrix()
        )
        self.rot_you2sf = np.linalg.inv(self.rot_sf2you)

    @property
    @lru_cache(maxsize=1)
    def det_sz_px(self):
        jfh = jungfrau_utils.JFDataHandler(self.JF_ID)
        return np.asarray(jfh.get_shape_out())

    @property
    def beam_center_x_px(self):
        return self.det_sz_px[1] / 2

    @property
    def beam_center_y_px(self):
        return self.det_sz_px[0] / 2

    @property
    def beam_center_x_frac(self):
        return self.beam_center_x_px / self.det_sz_px[1]

    @property
    def beam_center_y_frac(self):
        return self.beam_center_y_px / self.det_sz_px[0]

    def plot_geom(
        self,
        mu=None,
        eta=None,
        chi=None,
        phi=None,
        gamma=None,
        delta=None,
        detector_distance=None,
        energy=8000,
        ub_matrix=None,
        fig="Recspace vis",
        ax=None,
    ):
        pixsz_plot = self.pixsz * 80
        det_sz_px_plot = np.round(self.det_sz_px * self.pixsz / pixsz_plot).astype(int)

        self.qconv.init_area(
            "x-",
            "y-",
            self.beam_center_x_frac * det_sz_px_plot[1],
            self.beam_center_y_frac * det_sz_px_plot[0],
            *det_sz_px_plot[::-1],
            distance=detector_distance,
            pwidth1=pixsz_plot,
            pwidth2=pixsz_plot,
        )

        if ub_matrix is None:
            ub_matrix = self.qconv.UB

        pxpos = deque(
            self.qconv.area(mu, eta, chi, phi, gamma, delta, en=energy, UB=ub_matrix)
        )
        pxpos.rotate(0)

        orpos = deque(
            self.qconv.point(mu, eta, chi, phi, gamma, delta, en=energy, UB=ub_matrix)
        )
        orpos.rotate(0)
        if ax is None:
            if not isinstance(fig, plt.Figure):
                fig = plt.figure(fig)
            ax = fig.add_subplot(projection="3d")

        ret = ax.plot_surface(
            *pxpos,
            alpha=0.6,
            # facecolors=fcolors,
            # vmin=minn,vmax=maxx.
        )
        ax.plot(*orpos, "ro", ms=10)
        ax.xaxis.label.set_color("r")
        ax.yaxis.label.set_color("g")
        ax.zaxis.label.set_color("b")
        plt.xlabel("h")
        plt.ylabel("k")
        ax.set_zlabel("l")

        arrowlength = 0.3 * np.max(
            [np.diff(ax.get_xlim()), np.diff(ax.get_ylim()), np.diff(ax.get_zlim())]
        )
        ax.quiver(*orpos, 0, 0, arrowlength, color="b", arrow_length_ratio=0.2)
        ax.quiver(*orpos, 0, arrowlength, 0, color="g", arrow_length_ratio=0.2)
        ax.quiver(*orpos, arrowlength, 0, 0, color="r", arrow_length_ratio=0.2)

        ax.axis("equal")

    def get_px_hkl(
        self,
        mu=None,
        eta=None,
        chi=None,
        phi=None,
        gamma=None,
        delta=None,
        detector_distance=None,
        energy=8000,
        ub_matrix=None,
    ):
        pixsz_plot = self.pixsz * 80
        det_sz_px_plot = np.round(self.det_sz_px * self.pixsz / pixsz_plot).astype(int)

        self.qconv.init_area(
            "x-",
            "y-",
            self.beam_center_x_px,
            self.beam_center_y_px,
            *self.det_sz_px[::-1],
            distance=detector_distance,
            pwidth1=self.pixsz,
            pwidth2=self.pixsz,
        )

        pxpos = deque(
            self.qconv.area(mu, eta, chi, phi, gamma, delta, en=energy, UB=ub_matrix)
        )
        pxpos.rotate(0)
        return pxpos


class THC_robot:
    def __init__(self, pixsz=0.075, JF_ID="JF01T03V01"):
        self.pixsz = pixsz
        self.JF_ID = JF_ID
        self.angles = ["mu", "eta", "chi", "phi", "gamma", "delta"]
        self.qconv = xu.QConversion(("y+", "x-", "z+", "x-"), ("y+", "x-"), (0, 0, 1))

        self.rot_sf2you = (
            Rotation.from_rotvec(-np.pi / 2 * np.array([0, 0, 1])).as_matrix()
            @ Rotation.from_rotvec(-np.pi / 2 * np.array([0, 1, 0])).as_matrix()
        )
        self.rot_you2sf = np.linalg.inv(self.rot_sf2you)

    @property
    @lru_cache(maxsize=1)
    def det_sz_px(self):
        jfh = jungfrau_utils.JFDataHandler(self.JF_ID)
        return np.asarray(jfh.get_shape_out())

    @property
    def beam_center_x_px(self):
        return self.det_sz_px[1] / 2

    @property
    def beam_center_y_px(self):
        return self.det_sz_px[0] / 2

    @property
    def beam_center_x_frac(self):
        return self.beam_center_x_px / self.det_sz_px[1]

    @property
    def beam_center_y_frac(self):
        return self.beam_center_y_px / self.det_sz_px[0]

    def plot_geom(
        self,
        mu=None,
        eta=None,
        phi=None,
        gamma=None,
        delta=None,
        detector_distance=None,
        energy=8000,
        ub_matrix=None,
        fig="Recspace vis",
        ax=None,
    ):
        pixsz_plot = self.pixsz * 80
        det_sz_px_plot = np.round(self.det_sz_px * self.pixsz / pixsz_plot).astype(int)

        self.qconv.init_area(
            "x-",
            "y-",
            self.beam_center_x_frac * det_sz_px_plot[1],
            self.beam_center_y_frac * det_sz_px_plot[0],
            *det_sz_px_plot[::-1],
            distance=detector_distance,
            pwidth1=pixsz_plot,
            pwidth2=pixsz_plot,
        )

        if ub_matrix is None:
            ub_matrix = self.qconv.UB

        pxpos = deque(
            self.qconv.area(mu, eta, 90, phi, gamma, delta, en=energy, UB=ub_matrix)
        )
        pxpos.rotate(0)

        orpos = deque(
            self.qconv.point(mu, eta, 90, phi, gamma, delta, en=energy, UB=ub_matrix)
        )
        orpos.rotate(0)
        if ax is None:
            if not isinstance(fig, plt.Figure):
                fig = plt.figure(fig)
            ax = fig.add_subplot(projection="3d")

        ret = ax.plot_surface(
            *pxpos,
            alpha=0.6,
            # facecolors=fcolors,
            # vmin=minn,vmax=maxx.
        )
        ax.plot(*orpos, "ro", ms=10)
        ax.xaxis.label.set_color("r")
        ax.yaxis.label.set_color("g")
        ax.zaxis.label.set_color("b")
        plt.xlabel("h")
        plt.ylabel("k")
        ax.set_zlabel("l")

        arrowlength = 0.3 * np.max(
            [np.diff(ax.get_xlim()), np.diff(ax.get_ylim()), np.diff(ax.get_zlim())]
        )
        ax.quiver(*orpos, 0, 0, arrowlength, color="b", arrow_length_ratio=0.2)
        ax.quiver(*orpos, 0, arrowlength, 0, color="g", arrow_length_ratio=0.2)
        ax.quiver(*orpos, arrowlength, 0, 0, color="r", arrow_length_ratio=0.2)

        ax.axis("equal")

    def get_px_hkl(
        self,
        mu=None,
        eta=None,
        phi=None,
        gamma=None,
        delta=None,
        detector_distance=None,
        energy=8000,
        ub_matrix=None,
    ):
        pixsz_plot = self.pixsz * 80
        det_sz_px_plot = np.round(self.det_sz_px * self.pixsz / pixsz_plot).astype(int)

        self.qconv.init_area(
            "x-",
            "y-",
            self.beam_center_x_px,
            self.beam_center_y_px,
            *self.det_sz_px[::-1],
            distance=detector_distance,
            pwidth1=self.pixsz,
            pwidth2=self.pixsz,
        )

        pxpos = deque(
            self.qconv.area(mu, eta, 90, phi, gamma, delta, en=energy, UB=ub_matrix)
        )
        pxpos.rotate(0)
        return pxpos


class GIC_robot:
    def __init__(self, pixsz=0.075, JF_ID="JF01T03V01", jf_util_keyword_dict={}):
        self.pixsz = pixsz
        self.JF_ID = JF_ID
        self.angles = ["mu", "eta", "chi", "phi", "gamma", "delta"]
        self.qconv = xu.QConversion(("y+", "x-", "z+", "x-"), ("y+", "x-"), (0, 0, 1))

        self.rot_sf2you = (
            Rotation.from_rotvec(-np.pi / 2 * np.array([0, 0, 1])).as_matrix()
            @ Rotation.from_rotvec(-np.pi / 2 * np.array([0, 1, 0])).as_matrix()
        )
        self.rot_you2sf = np.linalg.inv(self.rot_sf2you)
        self.JF_UTIL_KWARGS = jf_util_keyword_dict

    @property
    @lru_cache(maxsize=1)
    def det_sz_px(self):
        jfh = jungfrau_utils.JFDataHandler(self.JF_ID, **self.JF_UTIL_KWARGS)
        return np.asarray(jfh.get_shape_out())

    @property
    def beam_center_x_px(self):
        return self.det_sz_px[1] / 2

    @property
    def beam_center_y_px(self):
        return self.det_sz_px[0] / 2

    @property
    def beam_center_x_frac(self):
        return self.beam_center_x_px / self.det_sz_px[1]

    @property
    def beam_center_y_frac(self):
        return self.beam_center_y_px / self.det_sz_px[0]

    def plot_geom(
        self,
        mu=None,
        eta=None,
        gamma=None,
        delta=None,
        detector_distance=None,
        energy=8000,
        ub_matrix=None,
        fig="Recspace vis",
        ax=None,
    ):
        pixsz_plot = self.pixsz * 80
        det_sz_px_plot = np.round(self.det_sz_px * self.pixsz / pixsz_plot).astype(int)

        self.qconv.init_area(
            "x-",
            "y-",
            self.beam_center_x_frac * det_sz_px_plot[1],
            self.beam_center_y_frac * det_sz_px_plot[0],
            *det_sz_px_plot[::-1],
            distance=detector_distance,
            pwidth1=pixsz_plot,
            pwidth2=pixsz_plot,
        )

        if ub_matrix is None:
            ub_matrix = self.qconv.UB

        pxpos = deque(
            self.qconv.area(mu, eta, 180, 0, gamma, delta, en=energy, UB=ub_matrix)
        )
        pxpos.rotate(0)

        orpos = deque(
            self.qconv.point(mu, eta, 180, 0, gamma, delta, en=energy, UB=ub_matrix)
        )
        orpos.rotate(0)
        if ax is None:
            if not isinstance(fig, plt.Figure):
                fig = plt.figure(fig)
            ax = fig.add_subplot(projection="3d")

        ret = ax.plot_surface(
            *pxpos,
            alpha=0.6,
            # facecolors=fcolors,
            # vmin=minn,vmax=maxx.
        )
        ax.plot(*orpos, "ro", ms=10)
        ax.xaxis.label.set_color("r")
        ax.yaxis.label.set_color("g")
        ax.zaxis.label.set_color("b")
        plt.xlabel("h")
        plt.ylabel("k")
        ax.set_zlabel("l")

        arrowlength = 0.3 * np.max(
            [np.diff(ax.get_xlim()), np.diff(ax.get_ylim()), np.diff(ax.get_zlim())]
        )
        ax.quiver(*orpos, 0, 0, arrowlength, color="b", arrow_length_ratio=0.2)
        ax.quiver(*orpos, 0, arrowlength, 0, color="g", arrow_length_ratio=0.2)
        ax.quiver(*orpos, arrowlength, 0, 0, color="r", arrow_length_ratio=0.2)

        ax.axis("equal")

    def get_px_hkl(
        self,
        mu=None,
        eta=None,
        gamma=None,
        delta=None,
        detector_distance=None,
        energy=8000,
        ub_matrix=None,
    ):
        pixsz_plot = self.pixsz * 80
        det_sz_px_plot = np.round(self.det_sz_px * self.pixsz / pixsz_plot).astype(int)

        self.qconv.init_area(
            "x-",
            "y-",
            self.beam_center_x_px,
            self.beam_center_y_px,
            *self.det_sz_px[::-1],
            distance=detector_distance,
            pwidth1=self.pixsz,
            pwidth2=self.pixsz,
        )

        pxpos = deque(
            self.qconv.area(mu, eta, 180, 0, gamma, delta, en=energy, UB=ub_matrix)
        )
        pxpos.rotate(0)
        return pxpos
