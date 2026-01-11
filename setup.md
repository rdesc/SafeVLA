# SafeVLA setup on Mila cluster with singularity

Clone repo
```bash
git clone https://github.com/rdesc/SafeVLA
cd SafeVLA
```

Copy singularity image
```bash
cp /network/scratch/d/deschaer/SafeVLA/data/models/safevla.sif .
```

Set env variables
```bash
export CODE_PATH=path_to_repo_SafeVLA/  # edit this!!
export DATA_PATH=/network/scratch/d/deschaer/SafeVLA/data
```

Launch singularity module
```bash
module load singularity

Start singularity container
```bash
singularity shell --nv \
  --bind ${CODE_PATH}:/root/SafeVLA \
  --bind ${DATA_PATH}:/root/data \
  --bind ${HF_HOME}:/root/huggingface \
  safevla.sif
```

Go to safevla root inside repo
```bash
cd /root/SafeVLA 
```

Test eval script
```bash
bash /root/SafeVLA/scripts/eval.sh --task_type pickup --ckpt_path /root/data/models/spoc_IL/model.ckpt
```

Test train script
```bash
bash /root/SafeVLA/scripts/train.sh --task_type fetch --il_ckpt_path /root/data/models/spoc_IL/model.ckpt --output_dir /root/data/models/ --dataset_dir /root/data/test01
```


