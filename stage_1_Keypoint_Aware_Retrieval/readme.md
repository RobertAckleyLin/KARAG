# KARAG: stage1 Keypoint-Aware Retrieval 

## Dataset from [Open X-Embodiment: Robotic Learning Datasets and RT-X Models](https://robotics-transformer-x.github.io/)
- You can also get the metadata of each dataset by searching on the [official tensorflow dataset](https://www.tensorflow.org/datasets/)
- Download path for the sub dataset

```
gs://gresearch/robotics/fractal20220817_data/0.1.0 has size 111.07 GiB
gs://gresearch/robotics/kuka/0.1.0 has size 778.02 GiB
gs://gresearch/robotics/bridge/0.1.0 has size 387.49 GiB
gs://gresearch/robotics/taco_play/0.1.0 has size 47.77 GiB
gs://gresearch/robotics/jaco_play/0.1.0 has size 9.24 GiB
gs://gresearch/robotics/berkeley_cable_routing/0.1.0 has size 4.67 GiB
gs://gresearch/robotics/roboturk/0.1.0 has size 45.39 GiB
gs://gresearch/robotics/nyu_door_opening_surprising_effectiveness/0.1.0 has size 7.12 GiB
gs://gresearch/robotics/viola/0.1.0 has size 10.40 GiB
gs://gresearch/robotics/berkeley_autolab_ur5/0.1.0 has size 76.39 GiB
gs://gresearch/robotics/toto/0.1.0 has size 127.66 GiB
gs://gresearch/robotics/language_table/0.0.1 has size 399.23 GiB
gs://gresearch/robotics/columbia_cairlab_pusht_real/0.1.0 has size 2.80 GiB
gs://gresearch/robotics/stanford_kuka_multimodal_dataset_converted_externally_to_rlds/0.1.0 has size 31.98 GiB
gs://gresearch/robotics/nyu_rot_dataset_converted_externally_to_rlds/0.1.0 has size 5.33 MiB
gs://gresearch/robotics/stanford_hydra_dataset_converted_externally_to_rlds/0.1.0 has size 72.48 GiB
gs://gresearch/robotics/austin_buds_dataset_converted_externally_to_rlds/0.1.0 has size 1.49 GiB
gs://gresearch/robotics/nyu_franka_play_dataset_converted_externally_to_rlds/0.1.0 has size 5.18 GiB
gs://gresearch/robotics/maniskill_dataset_converted_externally_to_rlds/0.1.0 has size 151.05 GiB
gs://gresearch/robotics/cmu_franka_exploration_dataset_converted_externally_to_rlds/0.1.0 has size 602.24 MiB
gs://gresearch/robotics/ucsd_kitchen_dataset_converted_externally_to_rlds/0.1.0 has size 1.33 GiB
gs://gresearch/robotics/ucsd_pick_and_place_dataset_converted_externally_to_rlds/0.1.0 has size 3.53 GiB
gs://gresearch/robotics/austin_sailor_dataset_converted_externally_to_rlds/0.1.0 has size 18.85 GiB
gs://gresearch/robotics/austin_sirius_dataset_converted_externally_to_rlds/0.1.0 has size 6.55 GiB
gs://gresearch/robotics/bc_z/0.1.0 has size 80.54 GiB
gs://gresearch/robotics/usc_cloth_sim_converted_externally_to_rlds/0.1.0 has size 254.52 MiB
gs://gresearch/robotics/utokyo_pr2_opening_fridge_converted_externally_to_rlds/0.1.0 has size 360.57 MiB
gs://gresearch/robotics/utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds/0.1.0 has size 829.37 MiB
gs://gresearch/robotics/utokyo_saytap_converted_externally_to_rlds/0.1.0 has size 55.34 MiB
gs://gresearch/robotics/utokyo_xarm_pick_and_place_converted_externally_to_rlds/0.1.0 has size 1.29 GiB
gs://gresearch/robotics/utokyo_xarm_bimanual_converted_externally_to_rlds/0.1.0 has size 138.44 MiB
gs://gresearch/robotics/robo_net/1.0.0 has size 799.91 GiB
gs://gresearch/robotics/berkeley_mvp_converted_externally_to_rlds/0.1.0 has size 12.34 GiB
gs://gresearch/robotics/berkeley_rpt_converted_externally_to_rlds/0.1.0 has size 40.64 GiB
gs://gresearch/robotics/kaist_nonprehensile_converted_externally_to_rlds/0.1.0 has size 11.71 GiB
gs://gresearch/robotics/stanford_mask_vit_converted_externally_to_rlds/0.1.0 has size 76.17 GiB
gs://gresearch/robotics/tokyo_u_lsmo_converted_externally_to_rlds/0.1.0 has size 335.71 MiB
gs://gresearch/robotics/dlr_sara_pour_converted_externally_to_rlds/0.1.0 has size 2.92 GiB
gs://gresearch/robotics/dlr_sara_grid_clamp_converted_externally_to_rlds/0.1.0 has size 1.65 GiB
gs://gresearch/robotics/dlr_edan_shared_control_converted_externally_to_rlds/0.1.0 has size 3.09 GiB
gs://gresearch/robotics/asu_table_top_converted_externally_to_rlds/0.1.0 has size 737.60 MiB
gs://gresearch/robotics/stanford_robocook_converted_externally_to_rlds/0.1.0 has size 124.62 GiB
gs://gresearch/robotics/eth_agent_affordances/0.1.0 has size 17.27 GiB
gs://gresearch/robotics/imperialcollege_sawyer_wrist_cam/0.1.0 has size 81.87 MiB
gs://gresearch/robotics/iamlab_cmu_pickup_insert_converted_externally_to_rlds/0.1.0 has size 50.29 GiB
gs://gresearch/robotics/uiuc_d3field/0.1.0 has size 15.82 GiB
gs://gresearch/robotics/utaustin_mutex/0.1.0 has size 20.79 GiB
gs://gresearch/robotics/berkeley_fanuc_manipulation/0.1.0 has size 8.85 GiB
gs://gresearch/robotics/cmu_play_fusion/0.1.0 has size 6.68 GiB
gs://gresearch/robotics/cmu_stretch/0.1.0 has size 728.06 MiB
gs://gresearch/robotics/berkeley_gnm_recon/0.1.0 has size 18.73 GiB
gs://gresearch/robotics/berkeley_gnm_cory_hall/0.1.0 has size 1.39 GiB
gs://gresearch/robotics/berkeley_gnm_sac_son/0.1.0 has size 7.00 GiB
```

## Acknowledgement

Our code is built on top of [RLBench](https://github.com/stepjam/RLBench) simulation environment and [Peract](https://github.com/peract/peract) codebase.