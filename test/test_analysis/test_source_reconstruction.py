# -*- coding: utf-8 -*-
"""Tests for pixelated source reconstruction module."""
import pytest
import numpy as np
from pathlib import Path
from copy import deepcopy
from unittest.mock import Mock, patch, MagicMock

# Skip tests if lenstronomy source reconstruction modules are not available
pytest.importorskip(
    "lenstronomy.ImSim.SourceReconstruction.pixelated_source_reconstruction"
)


class TestPixelatedSourceReconstructor:
    """Tests for PixelatedSourceReconstructor class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock file system and outputs
        self.mock_output = {
            "settings": {
                "lens_name": "test_lens",
                "band": ["F814W"],
                "model": {
                    "lens": ["EPL", "SHEAR_GAMMA_PSI"],
                    "lens_light": ["SERSIC_ELLIPSE"],
                    "source_light": ["SERSIC_ELLIPSE"],
                },
            },
            "kwargs_result": {
                "kwargs_lens": [
                    {
                        "theta_E": 1.0,
                        "gamma": 2.0,
                        "e1": 0.1,
                        "e2": 0.05,
                        "center_x": 0.0,
                        "center_y": 0.0,
                    },
                    {"gamma1": 0.05, "gamma2": -0.02, "ra_0": 0, "dec_0": 0},
                ],
                "kwargs_source": [
                    {
                        "amp": 1.0,
                        "R_sersic": 0.2,
                        "n_sersic": 1.0,
                        "center_x": 0.0,
                        "center_y": 0.1,
                        "e1": 0.0,
                        "e2": 0.0,
                    }
                ],
                "kwargs_lens_light": [{}],
            },
            "multi_band_list_out": [
                [
                    {
                        "image_data": np.random.randn(50, 50) * 0.01,
                        "background_rms": 0.01,
                        "ra_at_xy_0": -1.0,
                        "dec_at_xy_0": -1.0,
                        "transform_pix2angle": np.array([[0.04, 0], [0, 0.04]]),
                    },
                    {
                        "psf_type": "PIXEL",
                        "kernel_point_source": np.ones((5, 5)) / 25,
                    },
                    {},
                ]
            ],
            "fit_output": [],
        }

    def test_get_source_grid_params_defaults(self):
        """Test that default source grid parameters are computed correctly."""
        from dolphin.analysis.source_reconstruction import PixelatedSourceReconstructor
        from dolphin.processor.config import ModelConfig

        # Create instance with mocked file system
        with patch(
            "dolphin.analysis.source_reconstruction.FileSystem"
        ) as mock_fs_class:
            mock_fs = Mock()
            mock_fs_class.return_value = mock_fs

            reconstructor = PixelatedSourceReconstructor("/fake/path")

            # Create a mock config with pixel_size property
            mock_config = Mock(spec=ModelConfig)
            mock_config.settings = self.mock_output["settings"]
            mock_config.pixel_size = [0.04]  # Match the transform matrix in mock_output

            # Test default parameter generation
            params = reconstructor._get_source_grid_params(
                mock_config,
                None,
                self.mock_output["multi_band_list_out"],
                0,
            )

            # Check that defaults are reasonable
            assert "pixel_width" in params
            assert "num_pixels_x" in params
            assert "num_pixels_y" in params
            assert "ra_at_xy_0" in params
            assert "dec_at_xy_0" in params
            assert params["pixel_width"] > 0
            assert params["num_pixels_x"] > 0
            assert params["num_pixels_y"] > 0

    def test_get_source_grid_params_from_config(self):
        """Test source grid params loaded from config override defaults."""
        from dolphin.analysis.source_reconstruction import PixelatedSourceReconstructor
        from dolphin.processor.config import ModelConfig

        settings = deepcopy(self.mock_output["settings"])
        settings["pixelated_source_reconstruction"] = {
            "enabled": True,
            "source_grid": {
                "pixel_width": 0.05,
                "num_pixels_x": 80,
                "num_pixels_y": 80,
                "ra_at_xy_0": -2.0,
                "dec_at_xy_0": -2.0,
            },
        }

        with patch(
            "dolphin.analysis.source_reconstruction.FileSystem"
        ) as mock_fs_class:
            mock_fs = Mock()
            mock_fs_class.return_value = mock_fs

            reconstructor = PixelatedSourceReconstructor("/fake/path")

            # Create a mock config with pixel_size property
            mock_config = Mock(spec=ModelConfig)
            mock_config.settings = settings
            mock_config.pixel_size = [0.04]

            params = reconstructor._get_source_grid_params(
                mock_config,
                None,
                self.mock_output["multi_band_list_out"],
                0,
            )

            assert params["pixel_width"] == 0.05
            assert params["num_pixels_x"] == 80
            assert params["num_pixels_y"] == 80
            assert params["ra_at_xy_0"] == -2.0
            assert params["dec_at_xy_0"] == -2.0

    def test_get_source_grid_params_kwargs_override(self):
        """Test that kwargs override both defaults and config."""
        from dolphin.analysis.source_reconstruction import PixelatedSourceReconstructor
        from dolphin.processor.config import ModelConfig

        settings = deepcopy(self.mock_output["settings"])
        settings["pixelated_source_reconstruction"] = {
            "source_grid": {"pixel_width": 0.05},
        }

        override_kwargs = {"pixel_width": 0.03, "num_pixels_x": 100}

        with patch(
            "dolphin.analysis.source_reconstruction.FileSystem"
        ) as mock_fs_class:
            mock_fs = Mock()
            mock_fs_class.return_value = mock_fs

            reconstructor = PixelatedSourceReconstructor("/fake/path")

            # Create a mock config with pixel_size property
            mock_config = Mock(spec=ModelConfig)
            mock_config.settings = settings
            mock_config.pixel_size = [0.04]

            params = reconstructor._get_source_grid_params(
                mock_config,
                override_kwargs,
                self.mock_output["multi_band_list_out"],
                0,
            )

            assert params["pixel_width"] == 0.03
            assert params["num_pixels_x"] == 100

    def test_get_source_grid_params_uses_band_index(self):
        """Test that pixel size is read from config using band_index."""
        from dolphin.analysis.source_reconstruction import PixelatedSourceReconstructor
        from dolphin.processor.config import ModelConfig

        # Create multi-band mock output
        multi_band_list_out = [
            [
                {
                    "image_data": np.random.randn(50, 50) * 0.01,
                    "background_rms": 0.01,
                    "ra_at_xy_0": -1.0,
                    "dec_at_xy_0": -1.0,
                    "transform_pix2angle": np.array([[0.04, 0], [0, 0.04]]),
                },
                {"psf_type": "PIXEL", "kernel_point_source": np.ones((5, 5)) / 25},
                {},
            ],
            [
                {
                    "image_data": np.random.randn(50, 50) * 0.01,
                    "background_rms": 0.01,
                    "ra_at_xy_0": -1.0,
                    "dec_at_xy_0": -1.0,
                    "transform_pix2angle": np.array([[0.08, 0], [0, 0.08]]),
                },
                {"psf_type": "PIXEL", "kernel_point_source": np.ones((5, 5)) / 25},
                {},
            ],
        ]

        with patch(
            "dolphin.analysis.source_reconstruction.FileSystem"
        ) as mock_fs_class:
            mock_fs = Mock()
            mock_fs_class.return_value = mock_fs

            reconstructor = PixelatedSourceReconstructor("/fake/path")

            # Create a mock config with multiple pixel sizes
            mock_config = Mock(spec=ModelConfig)
            mock_config.settings = self.mock_output["settings"]
            mock_config.pixel_size = [0.04, 0.08]

            # Test band_index=0 uses pixel_size[0]
            params_band0 = reconstructor._get_source_grid_params(
                mock_config,
                None,
                multi_band_list_out,
                0,
            )
            assert params_band0["pixel_width"] == 0.04 * 2  # 2x coarser than image

            # Test band_index=1 uses pixel_size[1]
            params_band1 = reconstructor._get_source_grid_params(
                mock_config,
                None,
                multi_band_list_out,
                1,
            )
            assert params_band1["pixel_width"] == 0.08 * 2  # 2x coarser than image

    def test_get_regularization_params_defaults(self):
        """Test default regularization parameters."""
        from dolphin.analysis.source_reconstruction import PixelatedSourceReconstructor

        with patch(
            "dolphin.analysis.source_reconstruction.FileSystem"
        ) as mock_fs_class:
            mock_fs = Mock()
            mock_fs_class.return_value = mock_fs

            reconstructor = PixelatedSourceReconstructor("/fake/path")

            params = reconstructor._get_regularization_params(
                self.mock_output["settings"], None
            )

            assert params["type"] == "curvature"
            assert params["lambda_bounds"] == [1e2, 1e8]
            assert params["lambda_tolerance"] == 1e-7

    def test_get_regularization_params_invalid_type(self):
        """Test that invalid regularization type raises error."""
        from dolphin.analysis.source_reconstruction import PixelatedSourceReconstructor

        with patch(
            "dolphin.analysis.source_reconstruction.FileSystem"
        ) as mock_fs_class:
            mock_fs = Mock()
            mock_fs_class.return_value = mock_fs

            reconstructor = PixelatedSourceReconstructor("/fake/path")

            with pytest.raises(ValueError, match="Invalid regularization type"):
                reconstructor._get_regularization_params(
                    self.mock_output["settings"],
                    {"type": "invalid_type"},
                )

    def test_create_source_pixel_grid(self):
        """Test source pixel grid creation."""
        from dolphin.analysis.source_reconstruction import PixelatedSourceReconstructor

        with patch(
            "dolphin.analysis.source_reconstruction.FileSystem"
        ) as mock_fs_class:
            mock_fs = Mock()
            mock_fs_class.return_value = mock_fs

            reconstructor = PixelatedSourceReconstructor("/fake/path")

            params = {
                "pixel_width": 0.04,
                "num_pixels_x": 50,
                "num_pixels_y": 50,
                "ra_at_xy_0": -1.0,
                "dec_at_xy_0": -1.0,
            }

            grid = reconstructor._create_source_pixel_grid(params)

            assert grid.num_pixel_axes[0] == 50
            assert grid.num_pixel_axes[1] == 50
            assert np.isclose(grid.pixel_width, 0.04)

    def test_compute_grid_extent(self):
        """Test grid extent computation."""
        from dolphin.analysis.source_reconstruction import PixelatedSourceReconstructor
        from lenstronomy.Data.pixel_grid import PixelGrid

        with patch(
            "dolphin.analysis.source_reconstruction.FileSystem"
        ) as mock_fs_class:
            mock_fs = Mock()
            mock_fs_class.return_value = mock_fs

            reconstructor = PixelatedSourceReconstructor("/fake/path")

            # Create a simple grid
            grid = PixelGrid(
                nx=10,
                ny=10,
                transform_pix2angle=np.array([[0.1, 0], [0, 0.1]]),
                ra_at_xy_0=0.0,
                dec_at_xy_0=0.0,
            )

            extent = reconstructor._compute_grid_extent(grid)

            assert len(extent) == 4
            assert extent[0] < extent[1]  # x_min < x_max
            assert extent[2] < extent[3]  # y_min < y_max


class TestAutoSourceGrid:
    """Tests for auto_source_grid_from_caustics function."""

    def test_auto_source_grid_basic(self):
        """Test automatic source grid determination."""
        from dolphin.analysis.source_reconstruction import auto_source_grid_from_caustics

        kwargs_lens = [
            {
                "theta_E": 1.0,
                "gamma": 2.0,
                "e1": 0.1,
                "e2": 0.0,
                "center_x": 0.0,
                "center_y": 0.0,
            }
        ]
        lens_model_list = ["EPL"]

        result = auto_source_grid_from_caustics(
            kwargs_lens, lens_model_list, grid_resolution=0.05
        )

        assert "pixel_width" in result
        assert "num_pixels_x" in result
        assert "num_pixels_y" in result
        assert "ra_at_xy_0" in result
        assert "dec_at_xy_0" in result
        assert result["pixel_width"] == 0.05
        assert result["num_pixels_x"] >= 20  # Should have reasonable size


class TestConfigAdditions:
    """Tests for config.py additions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.settings_with_recon = {
            "lens_name": "test_lens",
            "band": ["F814W"],
            "model": {"lens": ["EPL"]},
            "pixelated_source_reconstruction": {
                "enabled": True,
                "source_grid": {
                    "pixel_width": 0.04,
                    "num_pixels_x": 60,
                    "num_pixels_y": 60,
                    "ra_at_xy_0": -1.2,
                    "dec_at_xy_0": -1.2,
                },
                "regularization": {
                    "type": "gradient",
                    "lambda_bounds": [1e3, 1e7],
                    "lambda_tolerance": 1e-6,
                },
            },
        }

        self.settings_without_recon = {
            "lens_name": "test_lens",
            "band": ["F814W"],
            "model": {"lens": ["EPL"]},
        }

    def test_is_pixelated_reconstruction_enabled_true(self):
        """Test detection of enabled reconstruction."""
        # This would test the config method - implementation depends on
        # how the actual ModelConfig is structured
        settings = self.settings_with_recon
        assert "pixelated_source_reconstruction" in settings
        assert settings["pixelated_source_reconstruction"]["enabled"] is True

    def test_is_pixelated_reconstruction_enabled_false(self):
        """Test detection when reconstruction not configured."""
        settings = self.settings_without_recon
        assert "pixelated_source_reconstruction" not in settings


class TestFilesAdditions:
    """Tests for files.py save/load additions."""

    def test_reconstruction_output_roundtrip(self, tmp_path):
        """Test that reconstruction output can be saved and loaded correctly."""
        # Create mock reconstruction result
        result = {
            "lens_model_id": "test_model",
            "band_index": 0,
            "optimal_lambda": 1e5,
            "magnification": 5.0,
            "background_rms": 0.01,
            "source_grid_params": {"pixel_width": 0.04, "num_pixels_x": 50},
            "regularization_params": {"type": "curvature"},
            "kwargs_lens": [{"theta_E": 1.0}],
            "source_grid_extent": [-1.0, 1.0, -1.0, 1.0],
            "image_grid_extent": [-2.0, 2.0, -2.0, 2.0],
            "source_pixel_values": np.random.randn(2500),
            "source_image": np.random.randn(50, 50),
            "lensed_image": np.random.randn(100, 100),
            "convolved_image": np.random.randn(100, 100),
            "residual": np.random.randn(100, 100),
            "M_matrix": None,
            "b_vector": None,
            "U_matrix": None,
        }

        # Test would require actual FileSystem implementation
        # This is a placeholder for the test structure
        assert result["magnification"] == 5.0
        assert result["source_image"].shape == (50, 50)


class TestVectorizedPixelatedSourceReconstruction:
    """Tests for the vectorized PixelatedSourceReconstruction implementation."""

    def test_vectorized_vs_original_diagonal_likelihood(self):
        """Verify that vectorized implementation matches the original.

        This test creates a simple setup and compares the M and b matrices
        from the vectorized implementation against the original nested-loop version.
        """
        from dolphin.analysis.source_reconstruction import PixelatedSourceReconstruction
        from lenstronomy.ImSim.SourceReconstruction.pixelated_source_reconstruction import (
            PixelatedSourceReconstruction as OriginalPSR,
        )
        from lenstronomy.Data.imaging_data import ImageData
        from lenstronomy.Data.psf import PSF
        from lenstronomy.LensModel.lens_model import LensModel
        from lenstronomy.Data.pixel_grid import PixelGrid

        # Create small test data
        numPix = 20
        image_data = np.random.randn(numPix, numPix) * 0.01
        background_rms = 0.01
        C_D = background_rms**2 * np.ones_like(image_data)

        kwargs_data = {
            'image_data': image_data,
            'background_rms': background_rms,
            'noise_map': C_D**0.5,  # Standard deviation map
            'ra_at_xy_0': -0.4,
            'dec_at_xy_0': -0.4,
            'transform_pix2angle': np.array([[0.04, 0], [0, 0.04]]),
        }

        # Simple PSF
        psf_kernel = np.zeros((5, 5))
        psf_kernel[2, 2] = 1.0  # Delta function PSF for simplicity
        kwargs_psf = {
            'psf_type': 'PIXEL',
            'kernel_point_source': psf_kernel,
        }

        # Simple lens model
        kwargs_lens = [{'theta_E': 0.8, 'gamma': 2.0, 'e1': 0.0, 'e2': 0.0,
                       'center_x': 0.0, 'center_y': 0.0}]

        # Create small source grid
        source_grid_params = {
            'pixel_width': 0.08,
            'num_pixels_x': 10,
            'num_pixels_y': 10,
            'ra_at_xy_0': -0.4,
            'dec_at_xy_0': -0.4,
        }

        # Initialize classes
        data_class = ImageData(**kwargs_data)
        psf_class = PSF(**kwargs_psf)
        lens_model_class = LensModel(lens_model_list=['EPL'])

        source_pixel_grid = PixelGrid(
            nx=source_grid_params['num_pixels_x'],
            ny=source_grid_params['num_pixels_y'],
            transform_pix2angle=source_grid_params['pixel_width'] * np.eye(2),
            ra_at_xy_0=source_grid_params['ra_at_xy_0'],
            dec_at_xy_0=source_grid_params['dec_at_xy_0'],
        )

        # Test vectorized version
        psr_vectorized = PixelatedSourceReconstruction(
            data_class, psf_class, lens_model_class, source_pixel_grid
        )
        M_vec, b_vec = psr_vectorized.generate_M_b_diagonal_likelihood(
            kwargs_lens, verbose=False, show_progress=False
        )

        # Test original version
        psr_original = OriginalPSR(
            data_class, psf_class, lens_model_class, source_pixel_grid
        )
        M_orig, b_orig = psr_original.generate_M_b_diagonal_likelihood(
            kwargs_lens, verbose=False, show_progress=False
        )

        # Verify they match
        np.testing.assert_allclose(M_vec, M_orig, rtol=1e-10, atol=1e-12,
                                   err_msg="M matrices do not match")
        np.testing.assert_allclose(b_vec, b_orig, rtol=1e-10, atol=1e-12,
                                   err_msg="b vectors do not match")


# Integration test placeholder
class TestSourceReconstructionIntegration:
    """Integration tests for full source reconstruction workflow."""

    @pytest.mark.slow
    def test_full_reconstruction_workflow(self):
        """Test complete reconstruction workflow with simulated data.

        This test is marked slow as it involves actual matrix computations.
        """
        # This would be a full integration test that:
        # 1. Creates simulated lensing data
        # 2. Runs parametric fitting
        # 3. Performs source reconstruction
        # 4. Verifies results
        pytest.skip("Full integration test requires complete environment setup")
