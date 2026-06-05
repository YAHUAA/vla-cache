# Custom LIBERO Task: Pick Bowl Next To Ramekin And Place Between

This is the widened version of the custom bowl-between probe.

- Language: `Pick up the black bowl next to the ramekin and place it between the plate and the ramekin`
- BDDL: `pick_black_bowl_next_to_ramekin_place_between_plate_ramekin_wide.bddl`
- Init states: `init_states/put_black_bowl_between_plate_and_ramekin_from_next_to_ramekin.pruned_init`
- Init-state source: native LIBERO-Spatial task `pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate`
- Goal predicate: `(On akita_black_bowl_1 main_table_between_plate_ramekin_region)`
- Target region: `x=[-0.08, -0.02], y=[0.17, 0.23]`

Compared with `put_black_bowl_between_plate_and_ramekin.bddl`, this variant makes two changes:

- The prompt is closer to the LIBERO training style by explicitly saying where the bowl starts and where it should be placed.
- The between region is widened from roughly 2cm x 2cm to roughly 6cm x 6cm.
