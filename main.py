from src.balrog_wide_field import create_balrog_subset

import os
import sys

if __name__ == '__main__':
    path = os.path.abspath(sys.path[0])
    no_samples = 10000  # int(3E6)  # int(250000)  # int(250000) # 20000 # int(8E6)
    print(f"{path}/Data/balrog_catalog_mcal_detect_deepfield_wo_cuts_26442133.pkl")
    create_balrog_subset(
        path_all_balrog_data=f"{path}/Data/balrog_cat_mcal_detect_df_wo_cuts_26442133.pkl",
        path_save=f"{path}/Output",
        name_save_file="balrog_mag_lupt_wo_cuts_w_fillna_gauss",
        only_detected=True,
        apply_unsheared_cuts=True,
        apply_bdf_cuts=False,
        apply_fill_na="Default",
        number_of_samples=no_samples,
        bdf_bins=["U", "G", "R", "I", "Z", "J", "H", "K"],
        unsheared_bins=["r", "i", "z"],
        protocol=2,
        plot_data=False
    )
