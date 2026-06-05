# Custom LIBERO Task: Put Bowl Between Plate And Ramekin

This custom task is built for the semantic-static VLA-Cache probe:

- Language: `Put the black bowl between the plate and the ramekin`
- BDDL: `put_black_bowl_between_plate_and_ramekin.bddl`
- Init states: `init_states/put_black_bowl_between_plate_and_ramekin_from_next_to_ramekin.pruned_init`
- Init-state source: native LIBERO-Spatial task `pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate`
- Goal predicate: `(On akita_black_bowl_1 main_table_between_plate_ramekin_region)`

The source init states intentionally place `akita_black_bowl_1` at `main_table_next_to_ramekin_region`, not at the target `main_table_between_plate_ramekin_region`. This avoids the invalid case where the rollout is successful immediately after reset.

LIBERO evaluates the final condition through the built-in `On` predicate. For a table site region target, `On(object, site_region)` is checked by testing whether the object's body position lies inside the MuJoCo box site that represents `main_table_between_plate_ramekin_region`.
