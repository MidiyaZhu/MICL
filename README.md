# MICL
After having the training, validation and testing datasets with verbalizer, you can find us in folder 'data' or build your own datasets. Please apply for the AMAN dataset usage approval before testing on it.

1. run llama_zsl_raw_label_train.py to obtain the training dataset vocabulary logits.

2. run llama_zsl_rawlabel_fileter_verbalzer.py to filter the verbalizer semantically based on the LLM feedback. 
please mannually delete the .csv file with logit value<0 (requirement1) but if you find some labels missing words after deleting, go to 'befiltered.csv' and add them all (requirment2). renamed the csv file with end as '_movelogit.csv'
run convert_json_to_csv_verbalizer.py in data_preprocessing folder and obtained the csv file 
run llama3_zsl_logits_filter_verbalizer_pointbiseria2.py for filtered verbalizer.

You can just use them for llama3 in data/filtered_verbalzier folder.

3. select demonstration sample (1-shot for example) based on requirment1 or requirment2 (we run both and choose the best).
run llama_zsl_logits_filter_demo_method1.py
run llama_zsl_logits_filter_demo_method2.py

go to folder /data_preprocessing
then

run build_train_logit.py
run build_valid_logit.py

6. class name check (optional)
run llama_zsl_rawlabel_train_demo.py (for binary class, we suggest do not do it)

7. multiple label addition iteration
run llama_logits_labelselection_system_method1.py
run llama_logits_labelselection_system_method1_order.py
run llama_logits_labelselection_system_method2.py
run llama_logits_labelselection_system_method2_order.py

8. evaluation:
run llama3_logits_labelselection_result_micl


sota:

For selficl and topk retrieved files, please see the code https://github.com/Shark-NLP/self-adaptive-ICL, where 'retrieved2' is for selficl and 'retrieved' is for topk. or run folder 'selficl' with readme: (only for llama2 and gpt2xl)
'build your datasets with each label for both validation and train set.

run prerank_myllama_splitlabel_test.py for both your validation and test sets retrievedlabel.

then

run retriever_llama_withlabel_test.py for test retrieved2label

then 

run retriever_llama_withlabel_valid.py for valid retrieved2 label'


1. multiple label addition iteration

run llama3_logits_labelselection_system_dicl.py
run llama3_logits_labelselection_system_selficl.py
run llama3_logits_labelselection_system_topk.py
run llama3_logits_labelselection_system_vanilla.py


2. evaluation:
run llama3_logits_labelselection_result_vanilla_mlabel.py

dicl,selficl,topk
run llama3_logits_labelselection_result_sota_mlabel.py


Please cite our paper:

@article{zixiao2024micl,
  title={MICL: Improving In-Context Learning through Multiple-Label Words in Demonstration},
  author={Zixiao, Zhu and Zijian, Feng and Hanzhang, Zhou and Junlang, Qian and Kezhi, Mao},
  journal={arXiv preprint arXiv:2406.10908},
  year={2024}
}
