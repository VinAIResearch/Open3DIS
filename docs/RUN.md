# Run the code

In bash files, there is --config option for choosing the Open3DIS configuration

1\) Extract 2D masks and first stage feature from RGB-D sequences:

```
sh scripts/grounding_2d.sh <config>
```

2\) Generate 3D instances from 2D masks:

```
sh scripts/generate_3d_inst.sh <config>
```

3\) Refine second stage feature from 3D instances:

```
sh scripts/refine_grounding_feat.sh <config>
```

After refine grounded features, re-run step 2 to finalize the 3D output masks

4\) Run interactive visualization (required Pyviz3D):

```
sh scripts/vis.sh
```

5\) Run evaluation:

We provided custom evaluation script
```
sh scripts/eval.sh
```

6\) Misc:

Promptable Segmentation
```
sh scripts/text_query.sh <config> <text_query>
```


Maskwise feature computation memory efficient
```
sh scripts/maskwise_vocab.sh
```

Class-agnostic evaluation
```
sh scripts/eval_classagnostic.sh
```

Reproduce OpenSUN3D challenge 3D instance retrieval
```
# Generating results
sh scripts/opensun3d.sh
# Submit results
python tools/submit_opensun3d.py 
``` 
Submit ScanNet++ 3D instance segmentation benchmark
```
# Step 1-4 with 'configs/scannetpp_benchmark_instance_test.yaml'
# Submit results
sh scripts/submit_scannetpp_benchmark.sh configs/scannetpp_benchmark_instance_test.yaml
```
[OpenYOLO3D](https://github.com/aminebdj/OpenYOLO3D) achieves remarkable results, we provide a reproducible version of OpenYOLO3D based on Open3DIS codebase
```
sh scripts/reproduce_openyolo3d.sh <config>
```