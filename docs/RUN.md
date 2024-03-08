# Run the code

1\) Extract 2D masks and first stage feature from RGB-D sequences:

```
sh scripts/grounding_2d.sh
```

2\) Generate 3D instances from 2D masks:

```
sh scripts/generate_3d_inst.sh
```

3\) Refine second stage feature from 3D instances:

```
sh scripts/refine_grounding_feat.sh
```

4\) Run interactive visualization (required Open3D):

...updating