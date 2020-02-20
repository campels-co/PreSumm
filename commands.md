# PreSumm commands

**Package Requirements**: torch==1.1.0 (but torch==1.2.0 will be used for CamemBERT) pytorch_transformers tensorboardX multiprocess pyrouge

Installing previous versions of PyTorch : [PyTorch](https://pytorch.org/get-started/previous-versions/)

```bash
conda create -n PreSumm python=3.6
conda activate PreSumm

# With CUDA
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
# Without CUDA
conda install pytorch==1.2.0 torchvision==0.4.0 cpuonly -c pytorch

pip install pytorch-transformers
conda install -c conda-forge tensorboardx
conda install -c conda-forge multiprocess
pip install pyrouge
```

**Updates Jan 22 2020**: Now you can **Summarize Raw Text Input!**. Swith to the dev branch, and use `-mode test_text` and use `-text_src $RAW_SRC.TXT` to input your text file.

* use `-test_from $PT_FILE$` to use your model checkpoint file.
* Format of the source text file:
  * For **abstractive summarization**, each line is a document.
  * If you want to do **extractive summarization**, please insert `[CLS] [SEP]` as your sentence boundaries.
* There are example input files in the [raw_data directory](https://github.com/nlpyang/PreSumm/tree/dev/raw_data)

## Abstractive summarization with trained model

CNN/DM Abstractive : [bertsumextabs_cnndm_final_model](https://drive.google.com/open?id=1-IKVCtc4Q-BdZpjXc4s70_fRsWnjtYLr)

```bash
cd src
# With CUDA
python train.py -task abs -mode test_text -test_from ../models/bertsumextabs_cnndm_final_model/model_step_148000.pt -text_src ../raw_data/temp.raw_src -result_path ../output/output_abs.txt -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50
# Without CUDA
python train.py -task abs -mode test_text -test_from ../models/bertsumextabs_cnndm_final_model/model_step_148000.pt -text_src ../raw_data/temp.raw_src -result_path ../output/output_abs.txt -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50
```

## Extractive summarization with trained model

CNN/DM BertExt : [bertext_cnndm_transformer](https://drive.google.com/open?id=1kKWoV0QCbeIuFt85beQgJ4v0lujaXobJ)

```bash
cd src
# With CUDA
python train.py -task ext -mode test_text -test_from ../models/bertext_cnndm_transformer/bertext_cnndm_transformer.pt -text_src ../raw_data/temp_ext.raw_src -result_path ../output/output_ext.txt -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50
# Without CUDA
python train.py -task ext -mode test_text -test_from ../models/bertext_cnndm_transformer/bertext_cnndm_transformer.pt -text_src ../raw_data/temp_ext.raw_src -result_path ../output/output_ext.txt -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50
```

## Tests with CamemBERT model

[CamemBERT](https://camembert-model.fr/)

**Package Requirements**: fairseq

```bash
conda install -c powerai fairseq
# or
pip install fairseq
```

```bash
cd src
# With CUDA
python train.py -task ext -mode test_text -test_from ../models/camembert.v0/model.pt -text_src ../raw_data/potter.raw_src -result_path ../output/output_potter.txt -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50
# Without CUDA
python train.py -task ext -mode test_text -test_from ../models/camembert.v0/model.pt -text_src ../raw_data/potter.raw_src -result_path ../output/output_potter.txt -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50
```

```
$ python train.py -task ext -mode test_text -test_from ../models/camembert.v0/model.pt -text_src ../raw_data/potter.raw_src -result_path ../output/output_potter.txt -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50

[2020-02-20 15:19:52,712 INFO] Loading checkpoint from ../models/camembert.v0/model.pt
Traceback (most recent call last):
  File "train.py", line 155, in <module>
    test_text_ext(args)
  File "/mnt/c/Users/arthu/Desktop/Git/PreSumm/src/train_extractive.py", line 253, in test_text_ext
    opt = vars(checkpoint['opt'])
KeyError: 'opt'
```