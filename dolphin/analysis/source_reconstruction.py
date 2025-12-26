# -*- coding: utf-8 -*-
"""This module provides pixelated source reconstruction functionality.

This module enables semi-linear source reconstruction where the lens model
is fixed and the source light is reconstructed on a pixel grid. This is
particularly useful for sources with complex morphology that cannot be
adequately described by simple parametric profiles.

References:
    - Zhang et al. (2025), arXiv:2508.08393
    - Suyu et al. (2006), MNRAS, 371, 983
"""
__author__ = "dolphin contributors"

import numpy as np
import scipy.signal
from copy import deepcopy

from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.psf import PSF
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.ImSim.SourceReconstruction.pixelated_source_reconstruction import (
    PixelatedSourceReconstruction as _PixelatedSourceReconstructionBase,
)
from lenstronomy.ImSim.SourceReconstruction.regularization_matrix_pixel import (
    pixelated_regularization_matrix,
)
from lenstronomy.ImSim.SourceReconstruction.solve_regularization_strength import (
    d_log_evi_d_lambda,
    solve_optimal_lambda,
)

try:
    from scipy.optimize import minimize_scalar
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class PixelatedSourceReconstruction(_PixelatedSourceReconstructionBase):
    """Optimized version of lenstronomy's PixelatedSourceReconstruction.

    This subclass overrides the M and b matrix generation methods to use
    vectorized NumPy operations instead of nested loops, providing significant
    speedup (typically 10-100x faster depending on matrix size).
    """

    def generate_M_b_diagonal_likelihood(
        self, kwargs_lens, verbose=False, show_progress=True
    ):
        """Generates M and b matrices using vectorized NumPy operations.

        This is a drop-in replacement for the parent class method but uses
        einsum for efficient computation of the matrix products.

        :param kwargs_lens: List of keyword arguments for the lens_model_class.
        :param verbose: If True, print progress messages during matrix generation steps.
        :param show_progress: If True, show progress messages (for compatibility).
        :returns: (M, b) tuple, where M is the matrix and b is the vector.
        """
        if verbose:
            print("Step 1: Lensing the source pixels")
        lensed_sp = self.lens_pixel_source_of_a_rectangular_region(kwargs_lens)
        if verbose:
            print("Step 1: Finished!")

        if verbose:
            print("Step 2: Convolve the lensed pixels")
        N_lensed = len(lensed_sp)
        lensed_pixel_conv_set = np.zeros((N_lensed, self._numPix, self._numPix))
        for i in range(N_lensed):
            lensed_pixel_conv_set[i] = self.sparse_convolution(
                lensed_sp[i], self._kernel
            )
        if verbose:
            print("Step 2: Finished!")

        if verbose:
            print("Step 3: Compute the matrix M and vector b (vectorized)")

        # Vectorized computation of b
        # b[i] = sum over pixels of lensed_pixel_conv_set[i] * image_data / C_D
        b = np.einsum('ijk,jk->i', lensed_pixel_conv_set, self._image_data / self._C_D)

        # Vectorized computation of M
        # M[i,j] = sum over pixels of lensed_pixel_conv_set[i] * lensed_pixel_conv_set[j] / C_D
        # Use einsum for efficient computation
        M = np.einsum(
            'ijk,ljk->il',
            lensed_pixel_conv_set,
            lensed_pixel_conv_set / self._C_D[None, :, :]
        )

        if verbose:
            print("Step 3: Finished!")

        return M, b


class PixelatedSourceReconstructor:
    """Performs pixelated source reconstruction given a fixed lens model.

    This class encapsulates the workflow for semi-linear source reconstruction:
    1. Set up the source pixel grid
    2. Compute the M matrix and b vector (lensing + convolution response)
    3. Generate the regularization matrix U
    4. Solve for the optimal regularization strength λ
    5. Solve the linear system (M + λU)a = b for source pixel values

    Example usage:
        >>> from dolphin.analysis.source_reconstruction import PixelatedSourceReconstructor
        >>> reconstructor = PixelatedSourceReconstructor(io_directory)
        >>> result = reconstructor.reconstruct(
        ...     lens_name="my_lens",
        ...     model_id="parametric_fit_v1",
        ...     band_index=0
        ... )
        >>> print(f"Magnification: {result['magnification']:.2f}")
    """

    def __init__(self, io_directory):
        """Initialize the reconstructor.

        :param io_directory: path to the input/output directory
        :type io_directory: `str`
        """
        # Import here to avoid circular imports
        from dolphin.processor.files import FileSystem
        from dolphin.processor.config import ModelConfig

        self.io_directory = io_directory
        self._file_system = FileSystem(io_directory)

    def reconstruct(
        self,
        lens_name,
        model_id,
        band_index=0,
        kwargs_lens=None,
        source_grid_kwargs=None,
        regularization_kwargs=None,
        verbose=True,
        show_progress=True,
    ):
        """Perform pixelated source reconstruction.

        This method loads a previously fitted lens model and reconstructs the
        source light on a pixel grid using semi-linear inversion.

        :param lens_name: name of the lens system
        :type lens_name: `str`
        :param model_id: identifier of the parametric fitting run to use for
            the lens model parameters
        :type model_id: `str`
        :param band_index: index of band to reconstruct (for multi-band data)
        :type band_index: `int`
        :param kwargs_lens: optional override for lens parameters. If None,
            uses the best-fit parameters from the specified model_id
        :type kwargs_lens: `list` of `dict` or `None`
        :param source_grid_kwargs: optional override for source grid settings.
            If None, uses settings from config file or defaults
        :type source_grid_kwargs: `dict` or `None`
        :param regularization_kwargs: optional override for regularization
            settings. If None, uses settings from config file or defaults
        :type regularization_kwargs: `dict` or `None`
        :param verbose: if True, print progress messages
        :type verbose: `bool`
        :param show_progress: if True, show progress bar during M,b computation
        :type show_progress: `bool`
        :return: dictionary containing reconstruction results
        :rtype: `dict`
        """
        from dolphin.processor.config import ModelConfig

        # Load the model output and config
        output = self._file_system.load_output(lens_name, model_id)
        settings = output["settings"]
        kwargs_result = output["kwargs_result"]
        multi_band_list_out = output["multi_band_list_out"]

        # Use provided kwargs_lens or load from fitted model
        if kwargs_lens is None:
            kwargs_lens = kwargs_result["kwargs_lens"]

        # Get config for lens model list
        config = ModelConfig(
            lens_name=lens_name,
            file_system=self._file_system,
            settings=settings,
        )

        # Set up source grid parameters
        source_grid_params = self._get_source_grid_params(
            settings, source_grid_kwargs, multi_band_list_out, band_index
        )

        # Set up regularization parameters
        reg_params = self._get_regularization_params(settings, regularization_kwargs)

        # Get data and PSF for the specified band
        band_data = multi_band_list_out[band_index]
        kwargs_data = band_data[0]
        kwargs_psf = band_data[1]

        # Create lenstronomy classes
        data_class = ImageData(**kwargs_data)
        psf_class = PSF(**kwargs_psf)

        lens_model_list = config.get_lens_model_list()
        lens_model_class = LensModel(lens_model_list=lens_model_list)

        # Create source pixel grid
        source_pixel_grid = self._create_source_pixel_grid(source_grid_params)

        if verbose:
            print(f"Source grid: {source_grid_params['num_pixels_x']} x "
                  f"{source_grid_params['num_pixels_y']} pixels")
            print(f"Pixel width: {source_grid_params['pixel_width']:.4f} arcsec")

        # Initialize the PixelatedSourceReconstruction class
        psr = PixelatedSourceReconstruction(
            data_class, psf_class, lens_model_class, source_pixel_grid
        )

        # Compute M matrix and b vector
        if verbose:
            print("Computing M matrix and b vector...")
        M, b = psr.generate_M_b(
            kwargs_lens, verbose=verbose, show_progress=show_progress
        )

        # Generate regularization matrix
        if verbose:
            print(f"Generating {reg_params['type']} regularization matrix...")
        U = pixelated_regularization_matrix(
            source_grid_params["num_pixels_x"],
            source_grid_params["num_pixels_y"],
            reg_params["type"],
        )

        # Diagnose M matrix condition
        if verbose:
            M_condition = np.linalg.cond(M)
            M_eigenvalues = np.linalg.eigvalsh(M)
            print(f"M matrix diagnostics:")
            print(f"  Condition number: {M_condition:.2e}")
            print(f"  Min eigenvalue: {M_eigenvalues[0]:.2e}")
            print(f"  Max eigenvalue: {M_eigenvalues[-1]:.2e}")
            print(f"  Rank deficiency: {np.sum(M_eigenvalues < 1e-10)} near-zero eigenvalues")

        # Solve for optimal regularization strength
        if verbose:
            print("Solving for optimal regularization strength...")
        optimal_lambda = self._solve_lambda(
            U, M, b, reg_params["lambda_bounds"], reg_params["lambda_tolerance"], verbose=verbose
        )
        if verbose:
            print(f"Optimal λ: {optimal_lambda:.2e}")

        # Solve for source pixel values
        if verbose:
            print("Solving for source pixel values...")
        Lambda_U = optimal_lambda * U
        source_pixel_values = np.linalg.solve(M + Lambda_U, b)

        # Reshape to 2D source image
        source_image = source_pixel_values.reshape(
            source_grid_params["num_pixels_y"], source_grid_params["num_pixels_x"]
        )

        if verbose:
            print(f"Source image statistics:")
            print(f"  Shape: {source_image.shape}")
            print(f"  Min/Max: {source_image.min():.6e} / {source_image.max():.6e}")
            print(f"  Sum: {source_image.sum():.6e}")
            print(f"  Mean: {source_image.mean():.6e}")

        np.save('/grad/bwedig/dolphin/notebooks/source_image.npy', source_image)
        if verbose:
            print('Saved source image to /grad/bwedig/dolphin/notebooks/source_image.npy')

        # Get lens light model from the parametric fit
        # We need to add lens light to get the full model
        lens_light_model = np.zeros_like(kwargs_data["image_data"])
        lens_convolved = np.zeros_like(kwargs_data["image_data"])

        # Get pixel coordinates and create lens plane grid
        # (needed for both lens light and source reconstruction)
        nx, ny = kwargs_data["image_data"].shape
        transform = kwargs_data.get(
            "transform_pix2angle", np.eye(2) * 0.04
        )
        ra_0 = kwargs_data.get("ra_at_xy_0", 0)
        dec_0 = kwargs_data.get("dec_at_xy_0", 0)

        lens_plane_pixel_grid = PixelGrid(
            nx=nx,
            ny=ny,
            transform_pix2angle=transform,
            ra_at_xy_0=ra_0,
            dec_at_xy_0=dec_0,
        )

        if "kwargs_lens_light" in kwargs_result and kwargs_result["kwargs_lens_light"]:
            try:
                from lenstronomy.LightModel.light_model import LightModel
                from lenstronomy.Data.imaging_data import ImageData as LenstroImageData

                lens_light_model_list = config.get_lens_light_model_list()
                if lens_light_model_list:
                    lens_light_model = LightModel(light_model_list=lens_light_model_list)

                    # Create coordinate grid for evaluating lens light
                    x_grid, y_grid = np.meshgrid(np.arange(nx), np.arange(ny))
                    ra_grid = ra_0 + transform[0, 0] * x_grid + transform[0, 1] * y_grid
                    dec_grid = dec_0 + transform[1, 0] * x_grid + transform[1, 1] * y_grid

                    # Compute lens light surface brightness
                    # lens_light_sb = light_model.surface_brightness(
                    #     ra_grid, dec_grid, kwargs_result["kwargs_lens_light"]
                    # )

                    # Convolve with PSF
                    # lens_light_model = scipy.signal.fftconvolve(
                    #     lens_light_sb, psf_class.kernel_point_source, mode="same"
                    # )

                    image_model = ImageModel(data_class=lens_plane_pixel_grid,
                                 psf_class=psf_class,
                                 lens_model_class=lens_model_class,
                                #  source_model_class=source_model_class,
                                 lens_light_model_class=lens_light_model,
                                #  kwargs_numerics=kwargs_numerics
                                 )
                    lens_convolved = image_model.lens_surface_brightness(kwargs_lens_light=kwargs_result["kwargs_lens_light"], unconvolved=False) 
                    print('Computed lens light model for source reconstruction.')
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not compute lens light model: {e}")
                lens_light_model = np.zeros_like(kwargs_data["image_data"])

        # Generate model images
        if verbose:
            print("Generating model images...")

        # Use lenstronomy's built-in method to lens the source image
        # This directly uses the same ray-tracing that was used to build the M matrix
        lensed_image = psr.lens_an_image_by_rayshooting(kwargs_lens, source_image)

        if verbose:
            print(f"Lensed image statistics:")
            print(f"  Shape: {lensed_image.shape}")
            print(f"  Min/Max: {lensed_image.min():.6e} / {lensed_image.max():.6e}")
            print(f"  Sum: {lensed_image.sum():.6e}")
            print(f"  Mean: {lensed_image.mean():.6e}")

        source_convolved = scipy.signal.fftconvolve(
            lensed_image, psf_class.kernel_point_source, mode="same"
        )

        if verbose:
            print(f"Convolved source statistics:")
            print(f"  Sum: {source_convolved.sum():.6e}")
            print(f"Lens light convolved sum: {lens_convolved.sum():.6e}")

        # Full model = lens light + source light
        convolved_image = lens_convolved + source_convolved
        residual = kwargs_data["image_data"] - convolved_image

        # Compute magnification (source only)
        magnification = lensed_image.sum() / source_image.sum()

        if verbose:
            print(f"Computed magnification: {magnification:.4f}")

        # Compute source grid extent for plotting
        source_grid_extent = self._compute_grid_extent(source_pixel_grid)
        image_grid_extent = self._compute_grid_extent(data_class)

        result = {
            "source_pixel_values": source_pixel_values,
            "source_image": source_image,
            "lensed_image": lensed_image,
            "convolved_image": convolved_image,
            "residual": residual,
            "optimal_lambda": optimal_lambda,
            "magnification": magnification,
            "M_matrix": M,
            "b_vector": b,
            "U_matrix": U,
            "source_grid_params": source_grid_params,
            "regularization_params": reg_params,
            "source_grid_extent": source_grid_extent,
            "image_grid_extent": image_grid_extent,
            "kwargs_lens": kwargs_lens,
            "kwargs_result": kwargs_result,  # Store full kwargs_result for lens light
            "lens_model_id": model_id,
            "band_index": band_index,
            "background_rms": kwargs_data["background_rms"],
        }

        if verbose:
            print("Source reconstruction complete!")
            print(f"Magnification: {magnification:.3f}")

        return result

    def _get_source_grid_params(
        self, settings, source_grid_kwargs, multi_band_list_out, band_index
    ):
        """Get source grid parameters from settings or kwargs.

        :param settings: model settings dictionary
        :type settings: `dict`
        :param source_grid_kwargs: optional override kwargs
        :type source_grid_kwargs: `dict` or `None`
        :param multi_band_list_out: multi-band data list
        :type multi_band_list_out: `list`
        :param band_index: band index
        :type band_index: `int`
        :return: source grid parameters
        :rtype: `dict`
        """
        pixel_size = source_grid_kwargs["pixel_width"]

        # Default parameters based on image data
        kwargs_data = multi_band_list_out[band_index][0]
        image_data = kwargs_data["image_data"]
        image_size = image_data.shape[0]

        # Try to get pixel size from transform matrix
        transform = kwargs_data.get("transform_pix2angle", np.eye(2) * pixel_size)

        # Default source grid: same size as image but coarser pixels
        default_params = {
            "pixel_width": pixel_size * 2,  # 2x coarser than image
            "num_pixels_x": image_size // 2,
            "num_pixels_y": image_size // 2,
            "ra_at_xy_0": kwargs_data.get("ra_at_xy_0", 0) / 2,
            "dec_at_xy_0": kwargs_data.get("dec_at_xy_0", 0) / 2,
        }

        # Check for settings in config file
        if (
            "pixelated_source_reconstruction" in settings
            and settings["pixelated_source_reconstruction"] is not None
        ):
            recon_settings = settings["pixelated_source_reconstruction"]
            if "source_grid" in recon_settings and recon_settings["source_grid"]:
                grid_settings = recon_settings["source_grid"]
                for key in default_params:
                    if key in grid_settings:
                        default_params[key] = grid_settings[key]

        # Override with provided kwargs
        if source_grid_kwargs is not None:
            for key in default_params:
                if key in source_grid_kwargs:
                    default_params[key] = source_grid_kwargs[key]

        return default_params

    def _get_regularization_params(self, settings, regularization_kwargs):
        """Get regularization parameters from settings or kwargs.

        :param settings: model settings dictionary
        :type settings: `dict`
        :param regularization_kwargs: optional override kwargs
        :type regularization_kwargs: `dict` or `None`
        :return: regularization parameters
        :rtype: `dict`
        """
        # Default parameters
        default_params = {
            "type": "curvature",
            "lambda_bounds": [1e4, 1e8],
            "lambda_tolerance": 1e-7,
            "max_iterations": 30,
        }

        # Check for settings in config file
        if (
            "pixelated_source_reconstruction" in settings
            and settings["pixelated_source_reconstruction"] is not None
        ):
            recon_settings = settings["pixelated_source_reconstruction"]
            if "regularization" in recon_settings and recon_settings["regularization"]:
                reg_settings = recon_settings["regularization"]
                for key in default_params:
                    if key in reg_settings:
                        default_params[key] = reg_settings[key]

        # Override with provided kwargs
        if regularization_kwargs is not None:
            for key in default_params:
                if key in regularization_kwargs:
                    default_params[key] = regularization_kwargs[key]

        # Validate regularization type
        valid_types = ["zeroth_order", "gradient", "curvature"]
        if default_params["type"] not in valid_types:
            raise ValueError(
                f"Invalid regularization type: {default_params['type']}. "
                f"Must be one of {valid_types}"
            )

        return default_params

    def _create_source_pixel_grid(self, source_grid_params):
        """Create lenstronomy PixelGrid for source reconstruction.

        :param source_grid_params: source grid parameters
        :type source_grid_params: `dict`
        :return: PixelGrid instance
        :rtype: `lenstronomy.Data.pixel_grid.PixelGrid`
        """
        transform_pix2angle = source_grid_params["pixel_width"] * np.eye(2)

        return PixelGrid(
            nx=source_grid_params["num_pixels_x"],
            ny=source_grid_params["num_pixels_y"],
            transform_pix2angle=transform_pix2angle,
            ra_at_xy_0=source_grid_params["ra_at_xy_0"],
            dec_at_xy_0=source_grid_params["dec_at_xy_0"],
        )

    def _solve_lambda_lcurve(self, U, M, b, bounds, verbose=False):
        """Solve for optimal lambda using L-curve method.

        This is more robust than Bayesian evidence for ill-conditioned problems.
        The L-curve plots ||Ma - b||² vs ||Ua||² as λ varies. The optimal λ
        is at the "corner" of the L.

        :param U: regularization matrix
        :param M: M matrix from lensing response
        :param b: b vector from data
        :param bounds: [lower, upper] bounds for lambda search
        :param verbose: print diagnostic information
        :return: optimal lambda value
        """
        if verbose:
            print("  Using L-curve method for λ optimization...")

        # Sample λ values logarithmically
        n_samples = 20
        lambda_values = np.logspace(np.log10(bounds[0]), np.log10(bounds[1]), n_samples)

        residual_norms = []
        regularization_norms = []

        for lam in lambda_values:
            # Solve (M + λU)a = b
            try:
                a = np.linalg.solve(M + lam * U, b)

                # Compute norms
                residual = np.linalg.norm(M @ a - b)
                regularization = np.linalg.norm(U @ a)

                residual_norms.append(residual)
                regularization_norms.append(regularization)
            except:
                residual_norms.append(np.inf)
                regularization_norms.append(np.inf)

        residual_norms = np.array(residual_norms)
        regularization_norms = np.array(regularization_norms)

        # Find the corner of the L-curve using maximum curvature
        # Convert to log scale for better curvature detection
        valid = np.isfinite(residual_norms) & np.isfinite(regularization_norms)

        if np.sum(valid) < 3:
            if verbose:
                print("  L-curve method failed, using upper bound")
            return bounds[1]

        log_residual = np.log10(residual_norms[valid])
        log_regularization = np.log10(regularization_norms[valid])
        lambda_valid = lambda_values[valid]

        # Compute curvature using finite differences
        # Curvature κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
        dx = np.gradient(log_residual)
        dy = np.gradient(log_regularization)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5

        # Find maximum curvature (the corner)
        corner_idx = np.argmax(curvature)
        optimal_lambda = lambda_valid[corner_idx]

        if verbose:
            print(f"  L-curve corner at λ = {optimal_lambda:.2e}")
            print(f"    Residual norm: {residual_norms[valid][corner_idx]:.2e}")
            print(f"    Regularization norm: {regularization_norms[valid][corner_idx]:.2e}")

        return optimal_lambda

    def _solve_lambda(self, U, M, b, bounds, tolerance, verbose=False):
        """Solve for optimal regularization strength.

        :param U: regularization matrix
        :type U: `numpy.ndarray`
        :param M: M matrix from lensing response
        :type M: `numpy.ndarray`
        :param b: b vector from data
        :type b: `numpy.ndarray`
        :param bounds: [lower, upper] bounds for lambda search
        :type bounds: `list`
        :param tolerance: convergence tolerance
        :type tolerance: `float`
        :param verbose: print diagnostic information
        :type verbose: `bool`
        :return: optimal lambda value
        :rtype: `float`
        """
        try:
            optimal_lambda = solve_optimal_lambda(
                d_log_evi_d_lambda,
                U,
                M,
                b,
                bounds[0],
                bounds[1],
                tolerance=tolerance,
                max_iterations=30,
                check_initial_bounds=True,
            )
        except Exception as e:
            # If optimization fails, try expanded bounds
            print(f"Warning: Lambda optimization failed ({e}).")

            # Evaluate derivative at a few points to understand the landscape
            if verbose:
                test_lambdas = [1e4, 1e6, 1e8]
                print("  Testing derivative at different λ values:")
                for test_lambda in test_lambdas:
                    try:
                        deriv = d_log_evi_d_lambda(test_lambda, U, M, b)
                        print(f"    λ={test_lambda:.0e}: d(log evi)/dλ = {deriv:.6f}")
                    except:
                        print(f"    λ={test_lambda:.0e}: derivative computation failed")

            # Check if the derivative suggests we need STRONGER regularization
            # instead of weaker
            try:
                deriv_at_lower = d_log_evi_d_lambda(bounds[0], U, M, b)
                deriv_at_upper = d_log_evi_d_lambda(bounds[1], U, M, b)

                if verbose:
                    print(f"  Derivative at lower bound ({bounds[0]:.0e}): {deriv_at_lower:.6f}")
                    print(f"  Derivative at upper bound ({bounds[1]:.0e}): {deriv_at_upper:.6f}")

                # If derivative is negative at lower bound and positive at upper,
                # the evidence function is decreasing: optimal λ should be HIGHER
                if deriv_at_lower < 0 and deriv_at_upper > 0:
                    print("  WARNING: Evidence derivative suggests optimal λ is ABOVE upper bound!")
                    print("  Using STRONG regularization (upper bound) to avoid noise.")
                    return bounds[1]

            except Exception as deriv_error:
                if verbose:
                    print(f"  Could not evaluate derivatives: {deriv_error}")

            # Try wider bounds
            expanded_bounds = [bounds[0] * 0.01, bounds[1] * 100]
            print(f"Retrying with expanded bounds: [{expanded_bounds[0]:.2e}, {expanded_bounds[1]:.2e}]")

            try:
                optimal_lambda = solve_optimal_lambda(
                    d_log_evi_d_lambda,
                    U,
                    M,
                    b,
                    expanded_bounds[0],
                    expanded_bounds[1],
                    tolerance=tolerance,
                    max_iterations=30,
                    check_initial_bounds=False,  # Don't check bounds
                )

                # Check if we hit the lower bound (sign of trouble)
                if abs(optimal_lambda - expanded_bounds[0]) / expanded_bounds[0] < 0.1:
                    print(f"  WARNING: Optimal λ={optimal_lambda:.2e} is near lower bound!")
                    print(f"  This suggests the evidence method is unreliable for this problem.")
                    print(f"  Switching to L-curve method...")

                    # Use L-curve method instead
                    optimal_lambda = self._solve_lambda_lcurve(U, M, b, bounds, verbose=verbose)

                    if verbose:
                        print(f"  L-curve selected λ={optimal_lambda:.2e}")
                else:
                    print(f"Success with expanded bounds! Optimal λ: {optimal_lambda:.2e}")

            except Exception as e2:
                # If still fails, use a stronger default based on empirical testing
                print(f"Expanded bounds also failed ({e2}). Using conservative default.")
                # Use upper bound as it provides stronger regularization
                optimal_lambda = bounds[1]

        return optimal_lambda

    def _compute_grid_extent(self, grid_or_data):
        """Compute the extent of a pixel grid for plotting.

        :param grid_or_data: PixelGrid or ImageData instance
        :type grid_or_data: `PixelGrid` or `ImageData`
        :return: [x_min, x_max, y_min, y_max] extent
        :rtype: `list`
        """
        ra_0, dec_0 = grid_or_data.radec_at_xy_0
        pixel_width = grid_or_data.pixel_width
        nx, ny = grid_or_data.num_pixel_axes

        extent = [
            ra_0 - 0.5 * pixel_width,
            ra_0 + (nx - 0.5) * pixel_width,
            dec_0 - 0.5 * pixel_width,
            dec_0 + (ny - 0.5) * pixel_width,
        ]
        return extent

    def save_reconstruction(self, lens_name, reconstruction_id, result):
        """Save reconstruction results to file.

        :param lens_name: name of the lens system
        :type lens_name: `str`
        :param reconstruction_id: identifier for this reconstruction
        :type reconstruction_id: `str`
        :param result: reconstruction result dictionary
        :type result: `dict`
        """
        self._file_system.save_reconstruction_output(
            lens_name, reconstruction_id, result
        )

    def load_reconstruction(self, lens_name, reconstruction_id):
        """Load reconstruction results from file.

        :param lens_name: name of the lens system
        :type lens_name: `str`
        :param reconstruction_id: identifier for the reconstruction
        :type reconstruction_id: `str`
        :return: reconstruction result dictionary
        :rtype: `dict`
        """
        return self._file_system.load_reconstruction_output(
            lens_name, reconstruction_id
        )


def auto_source_grid_from_caustics(
    kwargs_lens,
    lens_model_list,
    grid_resolution=0.02,
    padding_factor=1.5,
    num_points=200,
):
    """Automatically determine source grid extent from lens caustics.

    This function computes the caustics of the lens model and returns
    suggested source grid parameters that will cover the caustic region.

    :param kwargs_lens: lens model parameters
    :type kwargs_lens: `list` of `dict`
    :param lens_model_list: list of lens model names
    :type lens_model_list: `list` of `str`
    :param grid_resolution: desired pixel size in arcsec
    :type grid_resolution: `float`
    :param padding_factor: factor to expand grid beyond caustics
    :type padding_factor: `float`
    :param num_points: number of points for caustic computation
    :type num_points: `int`
    :return: suggested source grid parameters
    :rtype: `dict`
    """
    from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions

    lens_model = LensModel(lens_model_list=lens_model_list)
    lens_ext = LensModelExtensions(lens_model)

    # Get caustics
    try:
        ra_crit, dec_crit, ra_caustic, dec_caustic = lens_ext.critical_curve_caustics(
            kwargs_lens, compute_window=5, grid_scale=0.01
        )
    except Exception:
        # If caustic computation fails, return conservative defaults
        return {
            "pixel_width": grid_resolution,
            "num_pixels_x": 100,
            "num_pixels_y": 100,
            "ra_at_xy_0": -1.0,
            "dec_at_xy_0": -1.0,
        }

    # Find extent of caustics
    all_ra = np.concatenate(ra_caustic) if ra_caustic else np.array([0])
    all_dec = np.concatenate(dec_caustic) if dec_caustic else np.array([0])

    ra_min, ra_max = all_ra.min(), all_ra.max()
    dec_min, dec_max = all_dec.min(), all_dec.max()

    # Add padding
    ra_center = (ra_min + ra_max) / 2
    dec_center = (dec_min + dec_max) / 2
    ra_range = (ra_max - ra_min) * padding_factor
    dec_range = (dec_max - dec_min) * padding_factor

    # Ensure minimum size
    ra_range = max(ra_range, 1.0)
    dec_range = max(dec_range, 1.0)

    # Compute grid parameters
    num_pixels_x = int(np.ceil(ra_range / grid_resolution))
    num_pixels_y = int(np.ceil(dec_range / grid_resolution))

    return {
        "pixel_width": grid_resolution,
        "num_pixels_x": num_pixels_x,
        "num_pixels_y": num_pixels_y,
        "ra_at_xy_0": ra_center - ra_range / 2,
        "dec_at_xy_0": dec_center - dec_range / 2,
    }


def plot_caustics_and_source_grid(
    kwargs_lens,
    lens_model_list,
    source_grid_params=None,
    figsize=(8, 8),
):
    """Plot caustic curves and optionally overlay the source grid.

    This helps visualize whether the source grid appropriately covers
    the caustic region where the source is constrained by lensed images.

    :param kwargs_lens: lens model parameters
    :type kwargs_lens: `list` of `dict`
    :param lens_model_list: list of lens model names
    :type lens_model_list: `list` of `str`
    :param source_grid_params: optional source grid parameters to overlay
    :type source_grid_params: `dict` or `None`
    :param figsize: figure size
    :type figsize: `tuple`
    :return: matplotlib figure
    :rtype: `matplotlib.figure.Figure`
    """
    import matplotlib.pyplot as plt
    from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions

    lens_model = LensModel(lens_model_list=lens_model_list)
    lens_ext = LensModelExtensions(lens_model)

    # Compute caustics
    ra_crit, dec_crit, ra_caustic, dec_caustic = lens_ext.critical_curve_caustics(
        kwargs_lens, compute_window=5, grid_scale=0.01
    )

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot caustics
    for i, (ra, dec) in enumerate(zip(ra_caustic, dec_caustic)):
        label = "Caustic" if i == 0 else None
        ax.plot(ra, dec, 'r-', linewidth=2, label=label)

    # Plot source grid if provided
    if source_grid_params is not None:
        ra_0 = source_grid_params['ra_at_xy_0']
        dec_0 = source_grid_params['dec_at_xy_0']
        pixel_width = source_grid_params['pixel_width']
        nx = source_grid_params['num_pixels_x']
        ny = source_grid_params['num_pixels_y']

        # Grid boundaries
        ra_max = ra_0 + nx * pixel_width
        dec_max = dec_0 + ny * pixel_width

        # Draw grid rectangle
        ax.plot([ra_0, ra_max, ra_max, ra_0, ra_0],
                [dec_0, dec_0, dec_max, dec_max, dec_0],
                'b--', linewidth=2, label='Source Grid')

        # Draw grid lines
        for i in range(0, nx + 1, max(1, nx // 10)):
            ra_line = ra_0 + i * pixel_width
            ax.plot([ra_line, ra_line], [dec_0, dec_max], 'b-', alpha=0.2, linewidth=0.5)
        for j in range(0, ny + 1, max(1, ny // 10)):
            dec_line = dec_0 + j * pixel_width
            ax.plot([ra_0, ra_max], [dec_line, dec_line], 'b-', alpha=0.2, linewidth=0.5)

    ax.set_xlabel('RA offset (arcsec)', fontsize=12)
    ax.set_ylabel('Dec offset (arcsec)', fontsize=12)
    ax.set_title('Caustic and Source Grid', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    return fig
