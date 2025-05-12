# Logit Separability-Driven Samples and Multiple Class-Related Words Selection for Advancing In-Context Learning

After preparing the training, validation, and testing datasets with the verbalizer, you can find them in the `data` folder or build your own datasets. Please apply for AMAN dataset usage approval before testing on it.

## Steps to Run

### 1. Obtain Training Dataset Vocabulary Logits
```bash
python llama_zsl_raw_label_train.py
```

### 2. Filter the Verbalizer Based on LLM Feedback
```bash
python llama_zsl_rawlabel_fileter_verbalzer.py
```
- Manually delete the `.csv` file with logit values `< 0` (**requirement1**). 
- If some labels are missing words after deletion, check `befiltered.csv` and add them back (**requirement2**).
- Rename the CSV file with `_movelogit.csv`.

Convert JSON to CSV for the verbalizer:
```bash
cd data_preprocessing
python convert_json_to_csv_verbalizer.py
```
Run the filtered verbalizer script:
```bash
python llama3_zsl_logits_filter_verbalizer_pointbiseria2.py
```

Filtered verbalizers are available in `data/filtered_verbalizer` for LLaMA-3.

### 3. Select Demonstration Sample (1-Shot Example)
Run both methods and choose the best:
```bash
python llama_zsl_logits_filter_demo_method1.py
python llama_zsl_logits_filter_demo_method2.py
```

Go to the `data_preprocessing` folder and execute:
```bash
python build_train_logit.py
python build_valid_logit.py
```

### 4. Class Name Check (Optional)
```bash
python llama_zsl_rawlabel_train_demo.py
```
> Note: For binary classification, we suggest skipping this step.

### 5. Multiple Label Addition Iteration
```bash
python llama_logits_labelselection_system_method1.py
python llama_logits_labelselection_system_method1_order.py
python llama_logits_labelselection_system_method2.py
python llama_logits_labelselection_system_method2_order.py
```

### 6. Evaluation
```bash
python llama3_logits_labelselection_result_micl.py
```

---

## SOTA (State-of-the-Art) Methods

For `selficl` and `topk` retrieved files, refer to [self-adaptive-ICL](https://github.com/Shark-NLP/self-adaptive-ICL):
- `retrieved2` is for selficl.
- `retrieved` is for topk.
- Or, navigate to the `selficl` folder and follow the README.
> Only applicable for LLaMA-2 and GPT-2 XL.

### Steps:
1. Build datasets with each label for both validation and training sets.
```bash
python prerank_myllama_splitlabel_test.py
```
2. Retrieve test labels:
```bash
python retriever_llama_withlabel_test.py
```
3. Retrieve validation labels:
```bash
python retriever_llama_withlabel_valid.py
```

### Multiple Label Addition Iteration for LLaMA-3
```bash
python llama3_logits_labelselection_system_dicl.py
python llama3_logits_labelselection_system_selficl.py
python llama3_logits_labelselection_system_topk.py
python llama3_logits_labelselection_system_vanilla.py
```

### Evaluation
```bash
python llama3_logits_labelselection_result_vanilla_mlabel.py
```
For **DICL, SelfICL, and Top-K**, run:
```bash
python llama3_logits_labelselection_result_sota_mlabel.py
```

---

## Citation
If you use this work, please cite our paper (Accepted at **NAACL 2025 Main**):

```bibtex
@inproceedings{zhu2025logit,
  title={Logit Separability-Driven Samples and Multiple Class-Related Words Selection for Advancing In-Context Learning},
  author={Zhu, Zixiao and Feng, Zijian and Zhou, Hanzhang and Qian, Junlang and Mao, Kezhi},
  booktitle={Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  pages={6739--6759},
  year={2025}
}
```

