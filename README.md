## HINDbr: Heterogeneous Information Network Based Duplicate Bug Report Prediction

### Publication
HINDBR: Heterogeneous Information Network Based Duplicate Bug Report Prediction (ISSRE'2020).

```
@inproceedings{xiao2020hindbr,
  title={HINDBR: Heterogeneous Information Network Based Duplicate Bug Report Prediction},
  author={Xiao, Guanping and Du, Xiaoting and Sui, Yulei and Yue, Tao},
  booktitle={2020 IEEE 31st International Symposium on Software Reliability Engineering (ISSRE)},
  pages={195--206},
  year={2020},
  organization={IEEE}
}
```
---

### Installation and Tested Environment
Run the following command to install software dependencies (Python 3.6, tensorflow-gpu==1.15.0, Keras==2.3.1)
```sh
pip3 install -r requirements.txt
```

Note, HINDbr is developed and tested in a DELL Precision Tower with a 3.60 GHz Intel i9-9900K CPU, 32 GB memory, 512GB SSD and 2TB HDD storage, and an NVIDIA RTX 2080Ti GPU, running Ubuntu 18.04. We didn't test it in other hardware and software environments.

### Bug Data

Download the [bug reports](https://github.com/hindbr/BugData) and arrange the directory level as the following depiction. Otherwise, you need to change the directory setting from <ins>data/xmlfile_path.setting</ins>.
```sh
  AnyDir   
    |  
    |----BugData  
    |      |----eclipse  
    |      |----freedesktop  
    |      |----...  
    |----HINDbr 
    |      |----data
    |      |----...
```
  
---

### Bug Report HIN Construction
Download the bug data first. Then, run the command to generate bug report heterogeneous information network from bug reports. You can change the project's name in <ins>PROJECT</ins> from <ins>data_generation_br_hin.py</ins>.
```sh
python3 data_generation_br_hin.py
```
Generated bug report HINs are stored in <ins>data/bug_report_hin</ins>.

---

### Pretrained Embeddings
<b>HIN2Vec embedding</b>:  
Download the [hin_embedding](https://drive.google.com/drive/folders/1_3LeYmWu5lcRWdJICubu_vssrtTwRRgR?usp=sharing) that we have trained and put it into <ins>data/pretrained_embeddings/hin2vec/</ins>. Or, you can train it by the command.
```sh
python3 hin2vec_training.py
```

<b>Word2Vec embedding</b>:    
Download the [word_embedding](https://drive.google.com/drive/folders/1srUUWp1x_nYUF714NhBLmxmwyZF5raM4?usp=sharing) that we have trained and put it into <ins>data/pretrained_embeddings/word2vec/</ins>. Or, you can train it by the command.
```sh
python3 word2vec_training.py
```
  
---

### Bug Group Generation  
Run the command to generate bug groups. You can change the project's name in <ins>PROJECT</ins> from <ins>data_generation_bug_groups.py</ins>.
```sh
python3 data_generation_bug_groups.py
```
Generated bug groups are stored in <ins>data/bug_report_groups</ins>.

---
### Model Training and Test Data Generation
Run the command to generate model training and test data. You can change the project's name in <ins>PROJECT</ins> from <ins>data_generation_model_training.py</ins>.
```sh
python3 data_generation_model_training.py
```
Generated bug pairs are stored in <ins>data/model_training</ins>.

---

### Before-JIT and After-JIT Data Generation
Run the command to generate before-JIT and after-JIT pairs for the model evaluation. You can change the project's name in <ins>PROJECT</ins> from <ins>data_generation_before_after_jit.py</ins>.
```sh
python3 data_generation_before_after_jit.py
```
Generated before-JIT and after-JIT bug pairs are stored in <ins>data/before_after_jit_data</ins>.

---

### Model Training and Test 

Run the command to perform HINDbr and Baseline model training and test. You can change the setting of the training projects from <ins>run_hindbr_training.sh</ins> for HINDbr, or <ins>run_baseline_training.sh</ins> for Baseline model. 
```sh
bash run_hindbr_training.sh
```
```sh
bash run_baseline_training.sh
```
Trained models are stored in directory <ins>output/trained_model</ins> and the training history and test results are stored in <ins>output/training_history</ins>.

---

### Before-JIT and After-JIT Evaluation
Run the command to perform before-JIT and after-JIT evaluation. You can change the setting of the training projects from <ins>run_jit_evaluation.sh</ins>.
```sh
bash run_jit_evaluation.sh
```
Evaluation results are stored in <ins>output/training_history</ins>.
