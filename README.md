## A 2D Entity Pair Tagging Scheme for Relation Triplet Extraction
This repository contains the source code and datasets for the paper: **A 2D Entity Pair Tagging Scheme for Relation Triplet Extraction**.
## Framework
1. **The Directory: _data_**

   In this directory are the dataset used for our experiment

2. **The Directory: _pre_trained_bert_**

   This directory is used to store the files on which the pre-trained model(BERT) depends. The pre-trained BERT (bert-base-cased) will be downloaded automatically after running the code. Also, you can manually download the pre-trained BERT model and decompress it under this directory.

3. **The Package: _config_**

   The configuration of our experiment

4. **The Package: _data_loader_**

   Data pre-processing and loading

5. **The Package: _models_**

   The core model of our experiment
6. **The File: _utils.py_**

   Training and testing methods
7. **The File: _train.py_**

   Entry to start training

8. **The File: _test.py_**
   
   Entry to start testing
## Usage

1. **Environment**
   ```shell
   conda create -n your_env_name python=3.9
   conda activate your_env_name
   cd 2DEPT
   pip install -r requirements.txt
   ```

2. **Train the model (take NYT as an example)**

    Modify the second dim of `batch_triple_matrix` in `.data_loader/data_loader.py` to the number of relations, and run

    ```shell
    python train.py --dataset=NYT --batch_size=8 --rel_num=24 
    ```
    The model weights with the best performance on dev dataset will be stored in `.checkpoint/NYT/`

3. **Evaluate on the test set (take NYT as an example)**

   run 
    ```shell
    python test.py --dataset=NYT --rel_num=24
    ```

    The extracted results will be saved in `result/NYT`.


## **Acknowledgment**

I am deeply grateful for the inspiration of paper: **OneRel: Joint Entity and Relation Extraction with One Model in One Step**, Yu-Ming Shang, Heyan Huang and Xian-Ling Mao, AAAI-2022.
