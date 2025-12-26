# -*- coding: utf-8 -*-
"""This module has a class for maintaining the file system."""
__author__ = "ajshajib"

from pathlib import Path
import json
import numpy as np
import h5py
import gdown


class FileSystem(object):
    """This class contains the method to handle the file system and directory
    addresses."""

    def __init__(self, io_directory):
        """Initiates a FileSystem object with `io_directory` as root.

        :param io_directory: path to input/output directory
        :type io_directory: str
        """
        self._root_path = Path(io_directory)
        self.root = io_directory

    @staticmethod
    def path2str(path):
        """Converts a pathlib Path into string.

        :param path: path to a file or directory
        :type path: `Path`
        :return: path to a file or directory
        :rtype: `str`
        """
        return str(path.resolve())

    def get_lens_list_file_path(self):
        """Get the address for the lens_list.txt file.

        :return:
        :rtype:
        """
        return self.path2str(self._root_path / "lens_list.txt")

    def get_lens_list(self):
        """Get the list of lenses from lens_list.txt.

        :return:
        :rtype:
        """
        lens_list = []

        for line in open(self.get_lens_list_file_path(), "r"):
            if not line.startswith("#"):
                lens_list.append(line.rstrip("\n").rstrip("\r"))

        return lens_list

    def get_config_file_path(self, lens_name):
        """Get the file path to the config file for `lens_name`.

        :param lens_name: lens name
        :type lens_name: `str`
        :return: path to the config file
        :rtype: `str`
        """
        return self.path2str(self.get_settings_directory() / f"{lens_name}_config.yaml")

    def get_logs_directory(self):
        """Get directory for logs folder. If the directory doesn't exist, a folder is
        created.

        :return:
        :rtype:
        """
        logs_dir = self.path2str(self._root_path / "logs")

        # commenting out, as this directory needs to be created by user
        # if not os.path.isdir(logs_dir):
        #    os.mkdir(logs_dir)

        return logs_dir

    def get_settings_directory(self):
        """Get directory for settings folder. If the directory doesn't exist, a folder
        is created.

        :return:
        :rtype:
        """
        return self._root_path / "settings"

        # commenting out, as this directory needs to be created by user
        # if not os.path.isdir(settings_dir):
        #     os.mkdir(settings_dir)

    def get_outputs_directory(self):
        """Get directory for settings folder. If the directory doesn't exist, a folder
        is created.

        :return:
        :rtype:
        """
        outputs_dir = self.path2str(self._root_path / "outputs")

        # commenting out, as this directory needs to be created by user
        # if not os.path.isdir(outputs_dir):
        #    os.mkdir(outputs_dir)

        return outputs_dir

    def get_data_directory(self):
        """Get directory for data folder. If the directory doesn't exist, a folder is
        created.

        :return:
        :rtype:
        """
        data_dir = self.path2str(self._root_path / "data")

        # commenting out, as this directory needs to be created by user
        # if not os.path.isdir(data_dir):
        #     os.mkdir(data_dir)

        return data_dir

    def get_image_file_path(self, lens_name, band):
        """Get the file path for the imaging data for `lens_name`.

        :param lens_name: lens name
        :type lens_name: `str`
        :param band: band name
        :type band: `str`
        :return: file path
        :rtype: `str`
        """
        return self.path2str(
            Path(self.get_data_directory())
            / f"{lens_name}"
            / f"image_{lens_name}_{band}.h5"
        )

    def get_psf_file_path(self, lens_name, band):
        """Get the file path for the PSF data for `lens_name`.

        :param lens_name: lens name
        :type lens_name: `str`
        :param band: band name
        :type band: `str`
        :return: file path
        :rtype: `str`
        """
        return self.path2str(
            Path(self.get_data_directory())
            / f"{lens_name}"
            / f"psf_{lens_name}_{band}.h5"
        )

    def get_log_file_path(self, lens_name, model_id):
        """Get the file path for the PSF data for `lens_name`.

        :param lens_name: lens name
        :type lens_name: `str`
        :param model_id: identifier for run model
        :type model_id: `str`
        :return: file path
        :rtype: `str`
        """
        return (
            self.path2str(Path(self.get_logs_directory()))
            + f"/log_{lens_name}_{model_id}.txt"
        )

    def get_output_file_path(self, lens_name, model_id, file_type="json"):
        """Get the file path for the PSF data for `lens_name`.

        :param lens_name: lens name
        :type lens_name: `str`
        :param model_id: identifier for run model
        :type model_id: `str`
        :param file_type: type of file, options: 'json', 'h5'
        :type file_type: `str`
        :return: file path
        :rtype: `str`
        """
        return (
            self.path2str(Path(self.get_outputs_directory()))
            + f"/output_{lens_name}_{model_id}.{file_type}"
        )

    def save_output(self, lens_name, model_id, output, file_type="h5"):
        """Save output from fitting sequence.

        :param lens_name: name of the lens
        :type lens_name: `str`
        :param model_id: identifier for model run
        :type model_id: `str`
        :param output: output dictionary
        :type output: `dict`
        :param file_type: type of file to save, 'h5' or 'json'
        :type file_type: `str`
        :return: None
        :rtype:
        """
        if file_type == "h5":
            self.save_output_h5(lens_name, model_id, output)
        elif file_type == "json":
            self.save_output_json(lens_name, model_id, output)
        else:
            raise ValueError(f"File type {file_type} not recognized!")

    def save_output_json(self, lens_name, model_id, output):
        """Save output from fitting sequence.

        :param lens_name: name of the lens
        :type lens_name: `str`
        :param model_id: identifier for model run
        :type model_id: `str`
        :param output: output dictionary
        :type output: `dict`
        :return: None
        :rtype:
        """
        save_file = self.get_output_file_path(lens_name, model_id, file_type="json")
        with open(save_file, "w") as f:
            json.dump(self.encode_numpy_arrays(output), f, ensure_ascii=False, indent=4)

    def save_output_h5(self, lens_name, model_id, output):
        """Save output from fitting sequence to h5 format.

        :param lens_name: name of the lens
        :type lens_name: `str`
        :param model_id: identifier for model run
        :type model_id: `str`
        :param output: output dictionary
        :type output: `dict`
        :return: None
        :rtype:
        """
        save_file = self.get_output_file_path(lens_name, model_id, file_type="h5")

        with h5py.File(save_file, "w") as f:
            f.attrs["settings"] = json.dumps(
                self.encode_numpy_arrays(output["settings"]), ensure_ascii=False
            )
            f.attrs["kwargs_result"] = json.dumps(
                self.encode_numpy_arrays(output["kwargs_result"]), ensure_ascii=False
            )

            f.attrs["multi_band_list_out"] = json.dumps(
                self.encode_numpy_arrays(output["multi_band_list_out"]),
                ensure_ascii=False,
            )

            group = f.create_group("fit_output")
            for i, single_output in enumerate(output["fit_output"]):
                subgroup = group.create_group(f"{i}")
                subgroup.attrs["fitting_type"] = np.bytes_(single_output[0])

                if single_output[0] == "PSO":
                    subgroup.create_dataset("chi2", data=np.array(single_output[1][0]))
                    subgroup.create_dataset(
                        "position", data=np.array(single_output[1][1])
                    )
                    subgroup.create_dataset(
                        "velocity", data=np.array(single_output[1][2])
                    )
                    subgroup.create_dataset(
                        "param_list", data=np.array(single_output[2], dtype="S25")
                    )
                elif single_output[0] == "emcee":
                    subgroup.create_dataset(
                        "samples",
                        data=np.array(
                            single_output[1],
                        ),
                    )
                    subgroup.create_dataset(
                        "param_list", data=np.array(single_output[2], dtype="S25")
                    )
                    subgroup.create_dataset(
                        "log_likelihood",
                        data=np.array(
                            single_output[3],
                        ),
                    )
                else:
                    raise ValueError(
                        f"Fitting type {single_output[0]} not recognized for "
                        "saving output!"
                    )

    def load_output(self, lens_name, model_id, file_type="h5"):
        """Load from saved output file.

        :param lens_name: lens name
        :type lens_name: `str`
        :param model_id: model identifier provided at run initiation
        :type model_id: `str`
        :param file_type: type of file, 'h5' or 'json'
        :return: output dictionary
        :rtype: `dict`
        """
        if file_type == "h5":
            return self.load_output_h5(lens_name, model_id)
        elif file_type == "json":
            return self.load_output_json(lens_name, model_id)
        else:
            raise ValueError(f"File type {file_type} not recognized!")

    def load_output_json(self, lens_name, model_id):
        """Load from saved output file.

        :param lens_name: lens name
        :type lens_name: `str`
        :param model_id: model identifier provided at run initiation
        :type model_id: `str`
        :return: output dictionary
        :rtype: `dict`
        """
        load_file = self.get_output_file_path(lens_name, model_id, file_type="json")

        with open(load_file, "r") as f:
            output = json.load(f)

        return self.decode_numpy_arrays(output)

    def load_output_h5(self, lens_name, model_id):
        """Load from saved output file.

        :param lens_name: lens name
        :type lens_name: `str`
        :param model_id: model identifier provided at run initiation
        :type model_id: `str`
        :return: output dictionary
        :rtype: `dict`
        """
        load_file = self.get_output_file_path(lens_name, model_id, file_type="h5")

        with h5py.File(load_file, "r") as f:
            settings = self.decode_numpy_arrays(json.loads(str(f.attrs["settings"])))

            kwargs_result = self.decode_numpy_arrays(
                json.loads(str(f.attrs["kwargs_result"]))
            )

            multi_band_list_out = self.decode_numpy_arrays(
                json.loads(str(f.attrs["multi_band_list_out"]))
            )

            fit_output = []
            group = f["fit_output"]

            n = len(f["fit_output"].keys())
            for index in [f"{i}" for i in range(n)]:
                fitting_step = [
                    str(group[index].attrs["fitting_type"], encoding="utf-8")
                ]

                if fitting_step[0] == "PSO":
                    fitting_step.append(
                        [
                            group[index]["chi2"][:],
                            group[index]["position"][:],
                            group[index]["velocity"][:],
                        ]
                    )
                    fitting_step.append(
                        [
                            str(s, encoding="utf-8")
                            for s in group[index]["param_list"][:]
                        ]
                    )
                elif fitting_step[0] == "emcee":
                    fitting_step.append(group[index]["samples"][:])
                    fitting_step.append(
                        [
                            str(s, encoding="utf-8")
                            for s in group[index]["param_list"][:]
                        ]
                    )
                    fitting_step.append(group[index]["log_likelihood"][:])
                # else:
                #     raise ValueError('Fitting type {} not recognized for '
                #                      'loading output!'.format(fitting_step[0]
                #                      ))

                fit_output.append(fitting_step)

            output = {
                "settings": settings,
                "kwargs_result": kwargs_result,
                "fit_output": fit_output,
                "multi_band_list_out": multi_band_list_out,
            }

            return output

    def get_reconstruction_output_file_path(self, lens_name, reconstruction_id):
        """Get the file path for a source reconstruction output.

        :param lens_name: lens name
        :type lens_name: `str`
        :param reconstruction_id: identifier for the reconstruction run
        :type reconstruction_id: `str`
        :return: file path
        :rtype: `str`
        """
        return (
            self.path2str(Path(self.get_outputs_directory()))
            + f"/reconstruction_{lens_name}_{reconstruction_id}.h5"
        )


    def save_reconstruction_output(self, lens_name, reconstruction_id, result):
        """Save source reconstruction output to HDF5 file.

        :param lens_name: name of the lens
        :type lens_name: `str`
        :param reconstruction_id: identifier for reconstruction run
        :type reconstruction_id: `str`
        :param result: reconstruction result dictionary
        :type result: `dict`
        """
        save_file = self.get_reconstruction_output_file_path(lens_name, reconstruction_id)

        with h5py.File(save_file, "w") as f:
            # Store metadata as JSON attributes
            f.attrs["lens_model_id"] = result.get("lens_model_id", "")
            f.attrs["band_index"] = result.get("band_index", 0)
            f.attrs["optimal_lambda"] = result.get("optimal_lambda", 0.0)
            f.attrs["magnification"] = result.get("magnification", 0.0)
            f.attrs["background_rms"] = result.get("background_rms", 0.0)

            # Store source grid and regularization params as JSON
            f.attrs["source_grid_params"] = json.dumps(
                self.encode_numpy_arrays(result.get("source_grid_params", {}))
            )
            f.attrs["regularization_params"] = json.dumps(
                self.encode_numpy_arrays(result.get("regularization_params", {}))
            )
            f.attrs["kwargs_lens"] = json.dumps(
                self.encode_numpy_arrays(result.get("kwargs_lens", []))
            )

            # Store grid extents
            f.attrs["source_grid_extent"] = json.dumps(
                result.get("source_grid_extent", [])
            )
            f.attrs["image_grid_extent"] = json.dumps(
                result.get("image_grid_extent", [])
            )

            # Store arrays as datasets
            f.create_dataset(
                "source_pixel_values",
                data=result.get("source_pixel_values", np.array([])),
            )
            f.create_dataset(
                "source_image", data=result.get("source_image", np.array([]))
            )
            f.create_dataset(
                "lensed_image", data=result.get("lensed_image", np.array([]))
            )
            f.create_dataset(
                "convolved_image", data=result.get("convolved_image", np.array([]))
            )
            f.create_dataset("residual", data=result.get("residual", np.array([])))

            # Optionally store large matrices (can be disabled for space)
            if result.get("M_matrix") is not None:
                f.create_dataset("M_matrix", data=result["M_matrix"], compression="gzip")
            if result.get("b_vector") is not None:
                f.create_dataset("b_vector", data=result["b_vector"])
            if result.get("U_matrix") is not None:
                f.create_dataset("U_matrix", data=result["U_matrix"], compression="gzip")


    def load_reconstruction_output(self, lens_name, reconstruction_id):
        """Load source reconstruction output from HDF5 file.

        :param lens_name: lens name
        :type lens_name: `str`
        :param reconstruction_id: identifier for the reconstruction
        :type reconstruction_id: `str`
        :return: reconstruction result dictionary
        :rtype: `dict`
        """
        load_file = self.get_reconstruction_output_file_path(lens_name, reconstruction_id)

        with h5py.File(load_file, "r") as f:
            result = {
                "lens_model_id": str(f.attrs["lens_model_id"]),
                "band_index": int(f.attrs["band_index"]),
                "optimal_lambda": float(f.attrs["optimal_lambda"]),
                "magnification": float(f.attrs["magnification"]),
                "background_rms": float(f.attrs["background_rms"]),
                "source_grid_params": self.decode_numpy_arrays(
                    json.loads(str(f.attrs["source_grid_params"]))
                ),
                "regularization_params": self.decode_numpy_arrays(
                    json.loads(str(f.attrs["regularization_params"]))
                ),
                "kwargs_lens": self.decode_numpy_arrays(
                    json.loads(str(f.attrs["kwargs_lens"]))
                ),
                "source_grid_extent": json.loads(str(f.attrs["source_grid_extent"])),
                "image_grid_extent": json.loads(str(f.attrs["image_grid_extent"])),
                "source_pixel_values": f["source_pixel_values"][:],
                "source_image": f["source_image"][:],
                "lensed_image": f["lensed_image"][:],
                "convolved_image": f["convolved_image"][:],
                "residual": f["residual"][:],
            }

            # Load optional large matrices if present
            if "M_matrix" in f:
                result["M_matrix"] = f["M_matrix"][:]
            if "b_vector" in f:
                result["b_vector"] = f["b_vector"][:]
            if "U_matrix" in f:
                result["U_matrix"] = f["U_matrix"][:]

        return result

    @classmethod
    def encode_numpy_arrays(cls, obj):
        """Encode a list/dictionary containing numpy arrays through recursion for JSON
        serialization.

        :param obj: object
        :type obj:
        :return: object with `ndarray`s encoded as dictionaries
        :rtype:
        """
        if isinstance(obj, np.ndarray):
            return {"__ndarray__": obj.tolist(), "shape": obj.shape}
        elif isinstance(obj, list):
            encoded = []
            for element in obj:
                encoded.append(cls.encode_numpy_arrays(element))
            return encoded
        elif isinstance(obj, dict):
            encoded = {}
            for key, value in obj.items():
                encoded[key] = cls.encode_numpy_arrays(value)
            return encoded
        else:
            return obj

    @classmethod
    def decode_numpy_arrays(cls, obj):
        """Decode a list/dictionary containing encoded numpy arrays through recursion.

        :param obj: object with `ndarray`s encoded as dictionaries
        :type obj:
        :return: object with `ndarray`s as `numpy.ndarray`
        :rtype:
        """
        if isinstance(obj, dict):
            if "__ndarray__" in obj:
                return np.asarray(obj["__ndarray__"]).reshape(obj["shape"])
            else:
                decoded = {}
                for key, value in obj.items():
                    decoded[key] = cls.decode_numpy_arrays(value)
                return decoded
        elif isinstance(obj, list):
            decoded = []
            for element in obj:
                decoded.append(cls.decode_numpy_arrays(element))
            return decoded
        else:
            return obj

    def get_semantic_segmentation_file_path(self, lens_name, band):
        """Get the file path for the semantic segmentation data for `lens_name`.

        :param lens_name: lens name
        :type lens_name: `str`
        :param band: band name
        :type band: `str`
        :return: file path
        :rtype: `str`
        """
        return self.path2str(
            Path(self.get_outputs_directory())
            / f"semantic_segmentation_{lens_name}_{band}.npy"
        )

    def load_semantic_segmentation(self, lens_name, band):
        """Load semantic segmentation data from file.

        :param lens_name: lens name
        :type lens_name: `str`
        :param band: band name
        :type band: `str`
        :return: semantic segmentation array
        :rtype: `numpy.ndarray`
        """
        semantic_segmentation = np.load(
            self.get_semantic_segmentation_file_path(lens_name, band)
        )

        return semantic_segmentation

    def save_semantic_segmentation(self, lens_name, band, semantic_segmentation):
        """Save semantic segmentation data to file.

        :param lens_name: lens name
        :type lens_name: `str`
        :param band: band name
        :type band: `str`
        :param semantic_segmentation: semantic segmentation array
        :type semantic_segmentation: `numpy.ndarray`
        :return: None
        :rtype:
        """
        save_file = self.get_semantic_segmentation_file_path(lens_name, band)
        np.save(save_file, semantic_segmentation)

    def get_mask_file_path(self, lens_name, band):
        """Get the file path for the mask data for `lens_name`.

        :param lens_name: lens name
        :type lens_name: `str`
        :param band: band name
        :type band: `str`
        :return: file path
        :rtype: `str`
        """
        return self.path2str(
            self.get_settings_directory() / "masks" / f"mask_{lens_name}_{band}.npy"
        )

    def load_mask(self, lens_name, band):
        """Load mask data from file.

        :param lens_name: lens name
        :type lens_name: `str`
        :param band: band name
        :type band: `str`
        :return: mask array
        :rtype: `numpy.ndarray`
        """
        mask = np.load(self.get_mask_file_path(lens_name, band))

        return mask

    def save_mask(self, lens_name, band, mask):
        """Save mask data to file.

        :param lens_name: lens name
        :type lens_name: `str`
        :param band: band name
        :type band: `str`
        :param mask: mask array
        :type mask: `numpy.ndarray`
        :return: None
        :rtype:
        """
        save_file = self.get_mask_file_path(lens_name, band)
        np.save(save_file, mask)

    def get_trained_nn_model_file_path(self, source_type="galaxy"):
        """Get the file path for the trained model.

        :param lens_type: type of lens, 'galaxy' or 'quasar'
        :type lens_type: `str`
        :return: file path
        :rtype: `str`
        """
        assert source_type in ["galaxy", "quasar"]
        path = self.path2str(
            self._root_path
            / "trained_nn"
            / f"lensed_{source_type}_segmentation_model.h5"
        )

        # Check if the directory exists, create if not
        if not Path(path).parent.is_dir():
            Path(path).parent.mkdir()

        # Check if the file exists
        if not Path(path).is_file():
            # Download the model using gdown
            index = ["galaxy", "quasar"].index(source_type)
            file_id = [
                "1MAR2i5WlLlW_mAub3lbLLIlL6MPGXG8s",
                "1xO6Mniir3169H-7K5nThXR4lLvOUp6OQ",
            ][index]

            print("AI model not found in local storage. Downloading from the web...")
            gdown.download(
                id=file_id,
                output=path,
                quiet=False,
            )

        return path
