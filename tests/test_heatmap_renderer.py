from __future__ import annotations

import unittest

import numpy as np

from apps.pyqt6.views.heatmap_renderer import HeatmapRenderer, HeatmapRenderConfig


class HeatmapRendererTest(unittest.TestCase):
    def test_empty_points_return_transparent_rgba(self) -> None:
        renderer = HeatmapRenderer(80, 120)
        rgba = renderer.build_heatmap_rgba([], color_mode="blue")

        self.assertEqual(rgba.shape, (120, 80, 4))
        self.assertEqual(int(rgba[..., 3].max()), 0)

    def test_filters_bad_points_and_builds_colored_alpha(self) -> None:
        renderer = HeatmapRenderer(
            80,
            120,
            config=HeatmapRenderConfig(sigma=4, show_contours=False, heatmap_opacity=1.0),
        )
        rgba = renderer.build_heatmap_rgba(
            [
                (305.0, 670.0),
                (np.nan, 400.0),
                None,
                (900.0, 2000.0),
            ],
            color_mode="red",
        )

        self.assertGreater(int(rgba[..., 3].max()), 0)
        self.assertGreater(int(rgba[..., 0].max()), int(rgba[..., 2].max()))

    def test_contours_add_visible_light_lines(self) -> None:
        renderer = HeatmapRenderer(
            96,
            160,
            config=HeatmapRenderConfig(sigma=7, contour_levels=5, contour_alpha=120, heatmap_opacity=0.9),
        )
        rgba = renderer.build_heatmap_rgba(
            [(300.0 + offset, 650.0) for offset in range(-45, 46, 15)],
            color_mode="blue",
            show_contours=True,
        )

        light_pixels = (
            (rgba[..., 0] > 170)
            & (rgba[..., 1] > 170)
            & (rgba[..., 2] > 170)
            & (rgba[..., 3] > 0)
        )
        self.assertTrue(bool(light_pixels.any()))

    def test_recent_points_have_more_weight_when_decay_enabled(self) -> None:
        renderer = HeatmapRenderer(
            120,
            180,
            court_width=610.0,
            court_height=1340.0,
            config=HeatmapRenderConfig(sigma=1.5, show_contours=False, temporal_decay_floor=0.25),
        )
        density = renderer.build_density(
            [
                (60.0, 120.0),
                (550.0, 1220.0),
            ]
        )
        old_x = round(60.0 / 610.0 * (renderer.width - 1))
        old_y = round(120.0 / 1340.0 * (renderer.height - 1))
        recent_x = round(550.0 / 610.0 * (renderer.width - 1))
        recent_y = round(1220.0 / 1340.0 * (renderer.height - 1))

        self.assertGreater(float(density[recent_y, recent_x]), float(density[old_y, old_x]))


if __name__ == "__main__":
    unittest.main()
