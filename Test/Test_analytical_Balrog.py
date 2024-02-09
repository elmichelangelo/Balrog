import unittest
from unittest.mock import patch, MagicMock
from Analytical_Balrog import data_preprocessing, apply_cuts
from Handler.helper_functions import open_all_balrog_dataset, calc_color
import pandas as pd
import numpy as np


class TestDataPreprocessing(unittest.TestCase):

    @patch("Analytical_Balrog.open_all_balrog_dataset")
    @patch("Analytical_Balrog.calc_color")
    @patch("Analytical_Balrog.apply_cuts")
    def test_data_preprocessing(self, patch_open_dataset, patch_calc_color, patch_apply_cuts):
        # Setting up mock dataframe
        mock_df = pd.DataFrame({
            "detected": [1, 1, 0, 1],
            "BDF_G_0": [2, 3, 1, 4],
            "BDF_G_1": [2, 3, 1, 4],
            "BDF_FLUX_DERED_CALIB_U": [2, 3, 1, 4],
            "BDF_FLUX_DERED_CALIB_G": [2, 3, 1, 4],
            "BDF_FLUX_DERED_CALIB_R": [2, 3, 1, 4],
            "BDF_FLUX_DERED_CALIB_I": [2, 3, 1, 4],
            "BDF_FLUX_DERED_CALIB_Z": [2, 3, 1, 4],
            "BDF_FLUX_DERED_CALIB_J": [2, 3, 1, 4],
            "BDF_FLUX_DERED_CALIB_H": [2, 3, 1, 4],
            "BDF_FLUX_DERED_CALIB_K": [2, 3, 1, 4],
            "BDF_MAG_DERED_CALIB_U": [2, 3, 1, 4],
            "BDF_MAG_DERED_CALIB_G": [2, 3, 1, 4],
            "BDF_MAG_DERED_CALIB_R": [2, 3, 1, 4],
            "BDF_MAG_DERED_CALIB_I": [2, 3, 1, 4],
            "BDF_MAG_DERED_CALIB_Z": [2, 3, 1, 4],
            "BDF_MAG_DERED_CALIB_J": [2, 3, 1, 4],
            "BDF_MAG_DERED_CALIB_H": [2, 3, 1, 4],
            "BDF_MAG_DERED_CALIB_K": [2, 3, 1, 4],
            "unsheared/flux_r": [2, 3, 1, 4],
            "unsheared/flux_i": [2, 3, 1, 4],
            "unsheared/flux_z": [2, 3, 1, 4],
            "ID": [1, 2, 3, 4]
            # add more columns as per your data
        })

        # Create a mock configuration
        mock_cfg = {
            "PATH_DATA": "/path/to/data",
            "FILENAME_MERGED_CAT": "filename_merged_cat.csv",
            "BDF_BINS": ["U", "G", "R", "I", "Z", "J", "H", "K"],
            "UNSHEARED_BINS": ["r", "i", "z"],
            "SOMPZ_COLS": ["col1", "col2"],
            "COVARIANCE_COLUMNS": ["col3", "col4"]
            # add more configuration values as per your data
        }

        # Patching methods
        patch_open_dataset.return_value = mock_df
        patch_calc_color.side_effect = lambda *args, **kwargs: mock_df
        patch_apply_cuts.return_value = mock_df

        result = data_preprocessing(mock_cfg)

        # Assert the preprocessing steps are correct
        patch_open_dataset.assert_called_with(f"{mock_cfg['PATH_DATA']}/{mock_cfg['FILENAME_MERGED_CAT']}")

        # Check if any null values still exist in dataframe
        self.assertEqual(result.isna().sum().sum(), 0)


if __name__ == '__main__':
    unittest.main()
