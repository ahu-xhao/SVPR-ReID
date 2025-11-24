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

- **CP2108:**
- **AGReID:**
- **AGReIDv2: **
- **CARGO:** 

Download the datasets  and then unzip them to `your_dataset_dir`.

### Training🔧

**Pretrained Models**

- **ViT-B**: [Baidu Pan](https://pan.baidu.com/s/1YE-24vSo5pv_wHOF-y4sfA) (Code: `vmfm`)
- **CLIP**: [Baidu Pan](https://pan.baidu.com/s/1YPhaL0YgpI-TQ_pSzXHRKw) (Code: `52fu`)

For example, if you want to run CLIP-ReID baseline for the CP2108, you need to download the pretrained weights of [vit-based CLIP-ReID]() modify the bottom of `configs/CP2108/vit_clipreid_baseline_v100.yml` to

```
DATASETS:
  ROOT_DIR: ('your_dataset_dir')
  NAMES: ("CP2000_ALL")
  TESTS: ("CP2000_ALL", "CP2000_AG", "CP2000_GA")
  VERSION: 100
OUTPUT_DIR: 'your_output_dir'
```

Then run：

```python
CUDA_VISIBLE_DEVICES=3 python train_clipreid_xhao.py --config_file configs/CP2108/vit_clipreid_baseline_v100.yml
```



### Evaluation🔧

For example, if you want to test ViT-based CLIP-ReID Baseline for CP2108 :

```
CUDA_VISIBLE_DEVICES=3 python test_clipreid_xhao.py --config_file configs/CP2108/vit_clipreid_baseline_v100.yml
```



### Acknowledgement

Codebase from [TransReID](https://github.com/damo-cv/TransReID), [CLIP](https://github.com/openai/CLIP), and [CoOp](https://github.com/KaiyangZhou/CoOp).



### Citation



