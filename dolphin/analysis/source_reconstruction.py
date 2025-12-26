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
from scipy.linalg import cho_factor, cho_solve
from copy import deepcopy

from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.ImSim.SourceReconstruction.pixelated_source_reconstruction import (
    PixelatedSourceReconstruction,
)
from lenstronomy.ImSim.SourceReconstruction.regularization_matrix_pixel import (
    pixelated_regularization_matrix,
)
from lenstronomy.ImSim.SourceReconstruction.solve_regularization_strength import (
    d_log_evi_d_lambda,
    solve_optimal_lambda,
)


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
        use_source_mask=False,
        source_mask_type="snr",
        source_mask_threshold=0.5,
        source_mask_sigma=2.0,
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
        :param use_source_mask: if True, apply smoothing mask to source
        :type use_source_mask: `bool`
        :param source_mask_type: type of mask ('snr' or 'caustic')
        :type source_mask_type: `str`
        :param source_mask_threshold: threshold for S/N-based masking (e.g., 0.5
            means mask pixels with reconstructed flux < 0.5 * peak)
        :type source_mask_threshold: `float`
        :param source_mask_sigma: Gaussian smoothing sigma for soft mask (in pixels)
        :type source_mask_sigma: `float`
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
            config, source_grid_kwargs, multi_band_list_out, band_index
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

        # Solve for optimal regularization strength
        if verbose:
            print("Solving for optimal regularization strength...")
        optimal_lambda = self._solve_lambda(
            U, M, b, reg_params["lambda_bounds"], reg_params["lambda_tolerance"]
        )
        if verbose:
            print(f"Optimal λ: {optimal_lambda:.2e}")

        # Solve for source pixel values
        if verbose:
            print("Solving for source pixel values...")
        Lambda_U = optimal_lambda * U
        A = M + Lambda_U

        # Use Cholesky decomposition for symmetric positive definite matrix
        # This is ~2x faster than general LU decomposition used by np.linalg.solve
        try:
            c, low = cho_factor(A, lower=True)
            source_pixel_values = cho_solve((c, low), b)
            if verbose:
                print("  Used Cholesky decomposition (optimized for symmetric matrix)")
        except np.linalg.LinAlgError:
            # Fallback to standard solver if Cholesky fails (shouldn't happen for well-conditioned problems)
            if verbose:
                print("  Warning: Cholesky failed, using standard LU solver")
            source_pixel_values = np.linalg.solve(A, b)

        # Reshape to 2D source image
        source_image = source_pixel_values.reshape(
            source_grid_params["num_pixels_y"], source_grid_params["num_pixels_x"]
        )

        # Apply source mask if requested
        source_mask = None
        if use_source_mask:
            if verbose:
                print(f"Computing {source_mask_type} mask...")

            if source_mask_type == "snr":
                source_mask = self._compute_snr_mask(
                    source_image,
                    threshold=source_mask_threshold,
                    sigma=source_mask_sigma,
                    verbose=verbose,
                )
            elif source_mask_type == "caustic":
                source_mask = self._compute_caustic_mask(
                    kwargs_lens,
                    lens_model_class,
                    source_pixel_grid,
                    sigma=source_mask_sigma,
                    verbose=verbose,
                )
            else:
                raise ValueError(f"Unknown source_mask_type: {source_mask_type}")

            # Apply mask to source image
            source_image = source_image * source_mask
            if verbose:
                effective_pixels = np.sum(source_mask)
                total_pixels = source_mask.size
                print(f"  Effective source pixels: {effective_pixels:.1f}/{total_pixels} "
                      f"({100*effective_pixels/total_pixels:.1f}%)")

        # Generate model images
        if verbose:
            print("Generating model images...")
        lensed_image = psr.lens_an_image_by_rayshooting(kwargs_lens, source_image)
        convolved_source_image = scipy.signal.fftconvolve(
            lensed_image, psf_class.kernel_point_source, mode="same"
        )

        # Add lens light if present in the model
        lens_light_image = self._render_lens_light(
            config, kwargs_result, multi_band_list_out, band_index, verbose
        )

        # Total model image includes both source and lens light
        convolved_image = convolved_source_image + lens_light_image
        residual = convolved_image - kwargs_data["image_data"]

        # Compute magnification
        magnification = lensed_image.sum() / source_image.sum()

        # Compute source grid extent for plotting
        source_grid_extent = self._compute_grid_extent(source_pixel_grid)
        image_grid_extent = self._compute_grid_extent(data_class)

        result = {
            "source_pixel_values": source_pixel_values,
            "source_image": source_image,
            "lensed_image": lensed_image,
            "convolved_source_image": convolved_source_image,
            "lens_light_image": lens_light_image,
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
            "lens_model_id": model_id,
            "band_index": band_index,
            "background_rms": kwargs_data["background_rms"],
            "source_mask": source_mask,
        }

        if verbose:
            print("Source reconstruction complete!")
            print(f"Magnification: {magnification:.3f}")

        return result

    def _get_source_grid_params(
        self, config, source_grid_kwargs, multi_band_list_out, band_index
    ):
        """Get source grid parameters from config or kwargs.

        :param config: ModelConfig instance
        :type config: `ModelConfig`
        :param source_grid_kwargs: optional override kwargs
        :type source_grid_kwargs: `dict` or `None`
        :param multi_band_list_out: multi-band data list
        :type multi_band_list_out: `list`
        :param band_index: band index
        :type band_index: `int`
        :return: source grid parameters
        :rtype: `dict`
        """
        # Default parameters based on image data
        kwargs_data = multi_band_list_out[band_index][0]
        image_data = kwargs_data["image_data"]
        image_size = image_data.shape[0]

        # Get pixel size from config using band_index
        config_pixel_sizes = config.pixel_size
        image_pixel_size = config_pixel_sizes[band_index]

        # Default source grid: same size as image but coarser pixels
        default_params = {
            "pixel_width": image_pixel_size * 2,  # 2x coarser than image
            "num_pixels_x": image_size // 2,
            "num_pixels_y": image_size // 2,
            "ra_at_xy_0": kwargs_data.get("ra_at_xy_0", 0) / 2,
            "dec_at_xy_0": kwargs_data.get("dec_at_xy_0", 0) / 2,
        }

        # Check for settings in config file
        if (
            "pixelated_source_reconstruction" in config.settings
            and config.settings["pixelated_source_reconstruction"] is not None
        ):
            recon_settings = config.settings["pixelated_source_reconstruction"]
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
            "lambda_bounds": [1e2, 1e8],
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

    def _solve_lambda(self, U, M, b, bounds, tolerance):
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
            # If optimization fails, use geometric mean of bounds
            print(f"Warning: Lambda optimization failed ({e}). Using default.")
            optimal_lambda = np.sqrt(bounds[0] * bounds[1])

        return optimal_lambda

    def _render_lens_light(
        self, config, kwargs_result, multi_band_list_out, band_index, verbose
    ):
        """Render the lens light component of the image.

        :param config: ModelConfig instance
        :type config: `ModelConfig`
        :param kwargs_result: result dictionary with fitted parameters
        :type kwargs_result: `dict`
        :param multi_band_list_out: multi-band data list
        :type multi_band_list_out: `list`
        :param band_index: index of the band
        :type band_index: `int`
        :param verbose: if True, print progress
        :type verbose: `bool`
        :return: convolved lens light image
        :rtype: `numpy.ndarray`
        """
        from lenstronomy.ImSim.MultiBand.single_band_multi_model import SingleBandMultiModel
        from lenstronomy.Data.imaging_data import ImageData

        # Get lens light model list
        lens_light_model_list = config.get_lens_light_model_list()

        # If no lens light in the model, return zeros
        if not lens_light_model_list or len(lens_light_model_list) == 0:
            band_data = multi_band_list_out[band_index]
            kwargs_data = band_data[0]
            data_class = ImageData(**kwargs_data)
            return np.zeros_like(data_class.data)

        # Get lens light parameters
        kwargs_lens_light = kwargs_result.get("kwargs_lens_light", None)
        if kwargs_lens_light is None or len(kwargs_lens_light) == 0:
            band_data = multi_band_list_out[band_index]
            kwargs_data = band_data[0]
            data_class = ImageData(**kwargs_data)
            return np.zeros_like(data_class.data)

        if verbose:
            print("  Rendering lens light component...")

        # Create SingleBandMultiModel to properly handle multi-linear lens light
        kwargs_model = config.get_kwargs_model()

        # Create the single band model
        single_band_model = SingleBandMultiModel(
            multi_band_list=multi_band_list_out,
            kwargs_model=kwargs_model,
            likelihood_mask_list=None,
            band_index=band_index,
        )

        # Get the lens light image (already PSF-convolved)
        # Use unconvolved=False, source_add=False, lens_light_add=True, point_source_add=False
        lens_light_convolved = single_band_model.image_linear_solve(
            kwargs_lens=kwargs_result.get("kwargs_lens", []),
            kwargs_source=kwargs_result.get("kwargs_source", []),
            kwargs_lens_light=kwargs_lens_light,
            kwargs_ps=kwargs_result.get("kwargs_ps", []),
            inv_bool=False,
        )[1]  # [1] returns the model image

        # Now extract just the lens light component
        # Re-render with only lens light
        from lenstronomy.ImSim.image_model import ImageModel
        from lenstronomy.LensModel.lens_model import LensModel
        from lenstronomy.LightModel.light_model import LightModel
        from lenstronomy.Data.psf import PSF

        band_data = multi_band_list_out[band_index]
        kwargs_data = band_data[0]
        kwargs_psf = band_data[1]

        data_class = ImageData(**kwargs_data)
        psf_class = PSF(**kwargs_psf)
        lens_model_class = LensModel(lens_model_list=config.get_lens_model_list())
        lens_light_model_class = LightModel(light_model_list=lens_light_model_list)

        image_model = ImageModel(
            data_class=data_class,
            psf_class=psf_class,
            lens_model_class=lens_model_class,
            lens_light_model_class=lens_light_model_class,
        )

        lens_light_convolved = image_model.image(
            kwargs_lens=kwargs_result.get("kwargs_lens", []),
            kwargs_source=[],
            kwargs_lens_light=kwargs_lens_light,
            kwargs_ps=[],
            unconvolved=False,
            source_add=False,
            lens_light_add=True,
            point_source_add=False,
        )

        return lens_light_convolved

    def _compute_snr_mask(
        self, source_image, threshold=0.5, sigma=2.0, verbose=False
    ):
        """Compute a smooth S/N-based mask for the source image.

        This creates a soft mask that smoothly transitions from 1 (bright regions)
        to 0 (faint regions), helping to suppress noise artifacts.

        :param source_image: reconstructed source image
        :type source_image: `numpy.ndarray`
        :param threshold: relative threshold (fraction of peak flux)
        :type threshold: `float`
        :param sigma: Gaussian smoothing sigma for soft edges (in pixels)
        :type sigma: `float`
        :param verbose: if True, print progress
        :type verbose: `bool`
        :return: 2D soft mask array (values between 0 and 1)
        :rtype: `numpy.ndarray`
        """
        from scipy.ndimage import gaussian_filter

        # Compute flux threshold
        peak_flux = np.max(source_image)
        flux_threshold = threshold * peak_flux

        if verbose:
            print(f"  Peak flux: {peak_flux:.3e}, threshold: {flux_threshold:.3e}")

        # Create initial binary mask
        binary_mask = (source_image >= flux_threshold).astype(float)

        # Smooth the mask for soft edges
        if sigma > 0:
            smooth_mask = gaussian_filter(binary_mask, sigma=sigma, mode='constant')
            # Normalize to [0, 1]
            if smooth_mask.max() > 0:
                smooth_mask = smooth_mask / smooth_mask.max()
        else:
            smooth_mask = binary_mask

        return smooth_mask

    def _compute_caustic_mask(
        self, kwargs_lens, lens_model_class, source_pixel_grid, sigma=2.0, verbose=False
    ):
        """Compute a smooth mask for source pixels based on caustic region.

        :param kwargs_lens: lens model parameters
        :type kwargs_lens: `list` of `dict`
        :param lens_model_class: LensModel instance
        :type lens_model_class: `LensModel`
        :param source_pixel_grid: source plane pixel grid
        :type source_pixel_grid: `PixelGrid`
        :param sigma: Gaussian smoothing sigma for soft edges (in pixels)
        :type sigma: `float`
        :param verbose: if True, print progress
        :type verbose: `bool`
        :return: 2D soft mask array (values between 0 and 1)
        :rtype: `numpy.ndarray`
        """
        from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
        from scipy.ndimage import gaussian_filter

        lens_ext = LensModelExtensions(lens_model_class)

        # Compute caustics
        try:
            ra_crit, dec_crit, ra_caustic, dec_caustic = lens_ext.critical_curve_caustics(
                kwargs_lens, compute_window=5, grid_scale=0.01
            )
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not compute caustics ({e}). Using full grid.")
            return np.ones((source_pixel_grid.num_pixel_axes[1], source_pixel_grid.num_pixel_axes[0]))

        if not ra_caustic or len(ra_caustic) == 0:
            if verbose:
                print("  Warning: No caustics found. Using full grid.")
            return np.ones((source_pixel_grid.num_pixel_axes[1], source_pixel_grid.num_pixel_axes[0]))

        # Concatenate all caustic curves
        all_ra = np.concatenate(ra_caustic)
        all_dec = np.concatenate(dec_caustic)

        # Compute distance from each pixel to nearest caustic point
        x_grid, y_grid = source_pixel_grid.pixel_coordinates

        # For efficiency, compute distance to caustic center with smooth falloff
        ra_center = (all_ra.min() + all_ra.max()) / 2
        dec_center = (all_dec.min() + all_dec.max()) / 2
        ra_size = (all_ra.max() - all_ra.min()) / 2
        dec_size = (all_dec.max() - all_dec.min()) / 2

        # Elliptical distance from center
        dist_x = (x_grid - ra_center) / (ra_size + sigma * source_pixel_grid.pixel_width)
        dist_y = (y_grid - dec_center) / (dec_size + sigma * source_pixel_grid.pixel_width)
        dist = np.sqrt(dist_x**2 + dist_y**2)

        # Smooth transition: 1 at center, 0 far away
        mask = np.exp(-0.5 * (dist - 1.0)**2 / (sigma * 0.5)**2)
        mask = np.clip(mask, 0, 1)

        return mask

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
