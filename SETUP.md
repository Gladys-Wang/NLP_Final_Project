Create a new folder called "embeddings" and download embeddings to "embeddings" from
http://evexdb.org/pmresources/vec-space-models/PubMed-w2v.bin

We suggest you create a virtualenv satisfying requirements.txt. You will need to install a very recent
PyTorch (1.1 or later):
```
conda create  -p ~/local/evidence_inference_venv
conda activate ~/local/evidence_inference_venv
# this may differ for your CUDA installation
conda install pytorch -c pytorch
pip install -r requirements.txt
python -m spacy download en
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz
```


You may reproduce the baseline pipeline via:
```
conda activate ~/local/evidence_inference_venv
python evidence_inference/models/pipeline.py --output_dir outputs/ --params params/bert_pipeline_ev2.0.json
```

You may reproduce the pipeline with LoRA via:
```
conda activate ~/local/evidence_inference_venv
python evidence_inference/models/pipeline_lora.py --output_dir outputs/ --params params/bert_pipeline_ev2.0.json
```

