# Custom LIBERO Task: Separated Plate And Ramekin

This variant tests whether the bowl-between task was failing because the plate and ramekin were visually or physically too close.

- Language: `Pick up the black bowl next to the ramekin and place it between the plate and the ramekin`
- BDDL: `pick_black_bowl_next_to_ramekin_place_between_plate_ramekin_separated.bddl`
- Init states: `init_states/pick_black_bowl_next_to_ramekin_place_between_plate_ramekin_separated.pruned_init`
- Goal predicate: `(On akita_black_bowl_1 main_table_between_plate_ramekin_region)`

Region layout:

- Plate region: `x=[0.12, 0.14], y=[0.19, 0.21]`
- Ramekin region: `x=[-0.30, -0.28], y=[0.19, 0.21]`
- Bowl start region: `x=[-0.30, -0.28], y=[0.31, 0.33]`
- Between target region: `x=[-0.12, -0.04], y=[0.16, 0.24]`

The plate and ramekin centers are separated by roughly 42 cm along the table x-axis. The target region is centered around the midpoint and widened to roughly 8 cm x 8 cm.

Important: this variant requires its own generated init states. Reusing the native LIBERO-Spatial init states would overwrite these object placements during `env.set_init_state(...)`.
