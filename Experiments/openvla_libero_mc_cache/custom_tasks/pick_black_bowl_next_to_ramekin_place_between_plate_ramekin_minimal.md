# Custom LIBERO Task: Minimal Bowl Between

This variant removes task-irrelevant objects and fixtures from the separated bowl-between task.

- Language: `Pick up the black bowl next to the ramekin and place it between the plate and the ramekin`
- BDDL: `pick_black_bowl_next_to_ramekin_place_between_plate_ramekin_minimal.bddl`
- Init states: `init_states/pick_black_bowl_next_to_ramekin_place_between_plate_ramekin_minimal.pruned_init`
- Goal predicate: `(On akita_black_bowl_1 main_table_between_plate_ramekin_region)`

Kept entities:

- `main_table`
- `akita_black_bowl_1`
- `plate_1`
- `glazed_rim_porcelain_ramekin_1`

Removed entities:

- `akita_black_bowl_2`
- `cookies_1`
- `wooden_cabinet_1`
- `flat_stove_1`

Region layout:

- Plate region: `x=[0.12, 0.14], y=[0.19, 0.21]`
- Ramekin region: `x=[-0.30, -0.28], y=[0.19, 0.21]`
- Bowl start region: `x=[-0.30, -0.28], y=[0.31, 0.33]`
- Between target region: `x=[-0.12, -0.04], y=[0.16, 0.24]`

This BDDL must use freshly generated init states because the object set differs from native LIBERO-Spatial tasks.
