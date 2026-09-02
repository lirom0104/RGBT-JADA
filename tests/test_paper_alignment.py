import unittest

import torch
from torch import nn

from scene.gaussian_model import GaussianModel


class PaperAlignmentTests(unittest.TestCase):
    @staticmethod
    def _opacity_model(base, rgb_bias, thermal_bias):
        model = GaussianModel.__new__(GaussianModel)
        model.opacity_activation = torch.sigmoid
        model._opacity_base = nn.Parameter(torch.tensor(base, dtype=torch.float32))
        model._at_gom_opacity_bias_rgb = nn.Parameter(torch.tensor(rgb_bias, dtype=torch.float32))
        model._at_gom_opacity_bias_th = nn.Parameter(torch.tensor(thermal_bias, dtype=torch.float32))
        return model

    def test_legacy_opacity_migration_preserves_renders_and_makes_rgb_canonical(self):
        model = self._opacity_model(
            [[-2.0], [0.25]],
            [[0.5], [-0.75]],
            [[-0.25], [0.4]],
        )
        old_rgb = torch.sigmoid(model._opacity_base + model._at_gom_opacity_bias_rgb).detach()
        old_thermal = torch.sigmoid(model._opacity_base + model._at_gom_opacity_bias_th).detach()

        model._canonicalize_rgb_opacity()

        torch.testing.assert_close(model.get_rgb_opacity, model.get_opacity_base)
        torch.testing.assert_close(model.get_rgb_opacity, old_rgb)
        torch.testing.assert_close(model.get_thermal_opacity, old_thermal)
        self.assertFalse(model._at_gom_opacity_bias_rgb.requires_grad)
        self.assertEqual(torch.count_nonzero(model._at_gom_opacity_bias_rgb).item(), 0)

    def test_size_cannot_bypass_cmo_and_low_opacity_pruning_conditions(self):
        model = self._opacity_model(
            [[-6.0], [-6.0], [2.0]],
            [[0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0]],
        )
        model.max_radii2D = torch.tensor([100.0, 100.0, 100.0])
        model.get_cmo_scores = lambda iteration=None: {
            "cmo_prune_mask": torch.tensor([False, True, True])
        }

        prune_mask = model._build_prune_mask(
            min_opacity=0.01,
            extent=1.0,
            max_screen_size=20,
            iteration=10_000,
        )

        torch.testing.assert_close(prune_mask, torch.tensor([False, True, False]))

    def test_thermal_scale_is_bounded_without_pruning_anchors(self):
        model = GaussianModel.__new__(GaussianModel)
        model.use_at_gom = True
        model.scaling_activation = torch.exp
        model._scaling = nn.Parameter(torch.log(torch.tensor([[1.0, 10.0, 0.5]])))
        model._at_gom_log_scale_residual = nn.Parameter(
            torch.log(torch.tensor([[2.0, 3.0, 4.0]]))
        )

        model.constrain_thermal_scaling(scene_extent=5.0)

        self.assertEqual(model.get_thermal_scaling.shape[0], 1)
        self.assertLessEqual(model.get_thermal_scaling.max().item(), 3.0 + 1e-6)
        torch.testing.assert_close(
            model.get_thermal_scaling,
            torch.tensor([[2.0, 3.0, 2.0]]),
        )


if __name__ == "__main__":
    unittest.main()
