# SVPR-ReID🌱

SVPR-ReID: Semantic-Driven Visual Progressive Refinement for Aerial-Ground Person ReID: A Challenging Large-Scale Benchmark (AAAI 2026 Poster)

### Updates👨‍💻

We will update more detailed result (including dataset, training, verification) in the future

* [X] 2025.7.23: Build project page.
* [X] 2025.8.5: Add base code.
* [X] 2025.11.8: Add the CP2108 and the usage license (CP2108 is undergoing systematic collation and revision) .

### News🔉

20251008 - Our paper has been passed in phrase 1 for review ！

20251108 - Our paper has been accepted by AAAI'26 !

20251130 - Our paper is available on  arxiv !



---

## Dataset：CP2108💰

<img src=".\assets\dataset_motivation_v2.png"/>

### Hightlight💴

### Settings♻️

<img src=".\assets\dataset_characteristic.png"/>

<img src=".\assets\dataset_characteristic_attributes.png"/>

---

## Method：SVPR-ReID💡

<img src=".\assets\SVPR-ReID.png"/>

---

### Requirements🔏

### Setup

You need to “cd” the project dir ！！！

#### step1 Environments:

run the basic python environments as follows （following by CLIP-ReID）：

```c
sh setup.sh
```

#### step2 Datasets:



### Training & Testing🔧

For example, if you want to run CNN-based CLIP-ReID-baseline for the Market-1501, you need to modify the bottom of configs/person/cnn_base.yml to

```
DATASETS:
   NAMES: ('')
   ROOT_DIR: ('your_dataset_dir')
OUTPUT_DIR: 'your_output_dir'
```

Then run：

```python
CUDA_VISIBLE_DEVICES=3 python train_clipreid_xhao.py --config_file configs/CP2000/vit_clipreid_baseline_v100.yml
```



### Citation

