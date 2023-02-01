# Baselines for the UDASE task of the CHiME-7 challenge

We pre-train a supervised Sudo rm- rf [1] teacher on some out-of-domain data (e.g. Libri1to3mix) and try to adapt a student model with the RemixIT [2] method on the unlabeled CHiME-5 data.

**Fully-supervised Sudo rm -rf out-of-domain teacher**

- The supervised Sudo rm -rf model has been trained on the out-of-domain (OOD) Libri3mix data using the available isolated clean speech and noise signals, where the proportion of 1-speaker, 2-speaker, and 3-speaker mixtures is set to 0.5, 0.25, and 0.25, respectively. 
- The trained model has an encoder/decoder with 512 basis, 41 filter taps, a hop-size of 20 time-samples, and a depth of U = 8 U-ConvBlocks. 
- We use as a loss function the negative scale invariant signal-to-noise ratio (SI-SNR) with equal weights on the speaker mix and the noise component.
- We let the model train with an initial learning rate of 0.001 for 80 epochs while decreasing it to a third of its value every 15 epochs.

**Self-supervised RemixIT's student**

- The RemixIT network uses the pre-trained OOD supervised teacher and initializes exactly the same network as a student from the same checkpoint. 

- In a nutshell, RemixIT uses the mixtures from the CHiME-5 data in the following way: 
    
    - 1) it feeds a batch of CHiME-5 mixtures in the frozen teacher to get some estimated speech and estimated noise waveforms;
    - 2) it permutes the teacher's estimated noise waveforms across the batch dimension; 
    - 3) it synthesizes new bootstrapped mixtures by adding the initial speech teacher's estimates with the permuted teacher's noise estimates; 
    - 4) it trains the student model using as pseudo-targets the teacher's estimates. 

    We use as a loss function the negative scale invariant signal-to-noise ratio (SI-SNR) with equal weights on both speech and noise pseudo-targets provided by the teacher network. 
    
    Although multiple teacher update protocols can be used, we have chosen to use an exponential moving average update [3] with a teacher momentum of $\gamma=0.99$, which can potentially provide the student model with even higher quality estimates than the initial OOD pre-trained teacher model.
    An exponential movin teacher weight update (notice that for $\gamma=0$ the update protocol reduces to a sequentially updated teacher):
    $$\theta\_{\mathcal{T}}^{(j+1)} = \gamma \theta\_{\mathcal{T}}^{(j)} + (1 - \gamma) \theta\_{\mathcal{S}}^{(j)}$$
    
    We train the student models with the EMA teacher protocol update with an initial learning rate of 0.0003 and decreasing it to a third of its value every 10 epochs. We pick the student model chekpoint with the highest scoring mean overall mos as computed by the DNS-MOS (35 epochs) ```remixit_chime_adapted_student_besmos_ep35.pt```. When training with the CHiME-5 data automatically annotated with Brouhaha's VAD (potentially all training mixtures would contain at least one active speaker), we choose the checkpoint with the highest performing mean BAK_MOS (85 epochs) as computed by DNS-MOS on the dev set ```remixit_chime_adapted_student_bestbak_ep85_using_vad.pt```. 
    
    A final student where we update the teacher only every 10 epochs and set $\gamma=0.$ (which is essentially the same as a sequentially updated teacher protocol) is also provided in ```remixit_chime_adapted_student_static_teacher_ep_33.py``` which is chosen based on the highest performing model (33 epochs), in terms of SI-SNR, on the Libri1to3CHiME data with 1 speaker active. We train this student modelwith an initial learning rate of 0.0001 and decreasing it to a third of its value every 10 epochs.

## Table of contents

- [Datasets Generation](#datasets-generation)
- [Repo and paths Configurations](#repo-and-paths-configurations)
- [How to train the supervised teacher](#how-to-train-the-supervised-teacher)
- [How to adapt the RemixIT student](#how-to-adapt-the-remixit-student)
- [How to load a pretrained checkpoint](#how-to-load-a-pretrained-checkpoint)
- [Instructions for performance evaluation](#instructions-for-performance-evaluation)
- [Baseline performance](#baseline-performance)
- [References](#references)

## Datasets generation
Two datasets are required for generation, namely, Libri3mix and CHiME-5.

For the generation of Libri3Mix (450GB) one can follow the instructions [here](https://github.com/JorisCos/LibriMix) or just follow this:
```shell
cd {path_to_generate_Libri3mix}
git clone https://github.com/JorisCos/LibriMix
cd LibriMix 
conda create --name librimix # optional
conda activate librimix # optional
pip install -r requirements.txt
conda install -c conda-forge sox # for linux
# conda install -c groakat sox # for windows
./generate_librimix.sh storage_dir 
```

For the generation of the CHiME-5 data (25GB) follow the instructions [here](https://github.com/UDASE-CHiME2023/CHiME-5) or just follow these steps (this step requires the existence of CHiME-5 data (167GB) under some path, [apply-and-get-CHiME5-here](https://chimechallenge.github.io/chime6/download.html)):
```shell
cd {path_to_generate_CHiME_processed_data}
# clone data repository
git clone https://github.com/UDASE-CHiME2023/CHiME-5.git
cd CHiME-5

# create CHiME conda environment
conda env create -f environment.yml
conda activate CHiME

# Run the appropriate scripts to create the training, dev and eval datasets
python create_audio_segments.py {insert_path_of_CHiME5_data} json_files {insert_path_of_processed_10s_CHiME5_data} --train_10s

# Create the training data with VAD annotations - might somewhat help with the adaptation
python create_audio_segments.py {insert_path_of_CHiME5_data} json_files {insert_path_of_processed_10s_CHiME5_data} --train_10s --train_vad --train_only
```


## Repo and paths configurations
Set the paths for the aforementioned datasets and include the path of this repo.

```shell
git clone https://github.com/UDASE-CHiME2023/baseline.git
export PYTHONPATH={the path that you stored the github repo}:$PYTHONPATH
cd baseline
conda create --name baseline python # optional
conda activate baseline # optional
sudo apt-get update # if RuntimeError: Unsupported compiler -- at least C++11 support is needed!
sudo apt-get install build-essential -y # if RuntimeError: Unsupported compiler -- at least C++11 support is needed!
python -m pip install --user -r requirements.txt
```

You should change the following in ```__config__.py```:
```shell
LIBRI3MIX_ROOT_PATH = '{inset_path_to_Libri3mix}'
CHiME_ROOT_PATH = '{insert_path_of_processed_10s_CHiME5_data}'
LIBRICHiME_ROOT_PATH = '{insert_path_of_processed_LibriCHiME5_data}'

API_KEY = 'your_comet_ml_API_key'
```

To get ```your_comet_ml_API_key```, you can follow the instructions [here](https://www.comet.com/docs/v2/guides/getting-started/quickstart/).

## How to train the supervised teacher
Running the out-of-domain supervised teacher with SI-SNR loss is as easy as: 
```shell
cd {the path that you stored the github repo}/baseline
python -Wignore run_sup_ood_pretrain.py --train libri1to3mix --val libri1to3mix libri1to3chime --test libri1to3mix \
-fs 16000 --enc_kernel_size 81 --num_blocks 8 --out_channels 256 --divide_lr_by 3. \
--upsampling_depth 7 --patience 15  -tags supervised_ood_teacher --n_epochs 81 \
--project_name uchime_baseline_v3 --clip_grad_norm 5.0 --save_models_every 10 --audio_timelength 8.0 \
--p_single_speaker 0.5 --min_or_max min --max_num_sources 2 \
--checkpoint_storage_path {insert_path_to_save_models} --log_audio --apply_mixture_consistency \
--n_jobs 12 -cad 2 3 -bs 24
```

Don't forget to set _n_jobs_ to the number of CPUs to use, _cad_ to the cuda ids to be used and _bs_ to the batch size used w.r.t. your system. Also you need to set the _checkpoint_storage_path_ to a valid path.

## How to adapt the RemixIT student
If you want to adapt your model to the CHiME-5 data you can use as a warm-up checkpoint the previous teacher model and perform RemixIT using the CHiME-5 mixture dataset (in order to use the annotated with VAD data just simple use the *--use_vad* at the end of the followin command): 
```shell
cd {the path that you stored the github repo}/baseline
python -Wignore run_remixit.py --train chime --val chime libri1to3chime --test libri1to3mix \
-fs 16000 --enc_kernel_size 81 --num_blocks 8 --out_channels 256 --divide_lr_by 3. \
--student_depth_growth 1 --n_epochs_teacher_update 1 --teacher_momentum 0.99 \
--upsampling_depth 7 --patience 10 --learning_rate 0.0003 -tags remixit student allData \
--n_epochs 100 --project_name uchime_baseline_v3 --clip_grad_norm 5.0 --audio_timelength 8.0 \
--min_or_max min --max_num_sources 2 --save_models_every 1 --initialize_student_from_checkpoint \
--checkpoint_storage_path /home/thymios/UCHIME_checkpoints \
--warmup_checkpoint ../pretrained_checkpoints/libri1to3mix_supervised_teacher_w_mixconsist.pt \
--checkpoint_storage_path {insert_path_to_save_models} --log_audio --apply_mixture_consistency \
--n_jobs 12 -cad 2 3 -bs 24
```

## How to load a pretrained checkpoint
```python
import baseline.utils.mixture_consistency as mixture_consistency
import baseline.models.improved_sudormrf as improved_sudormrf

model = improved_sudormrf.SuDORMRF(
        out_channels=256,
        in_channels=512,
        num_blocks=8,
        upsampling_depth=7,
        enc_kernel_size=81,
        enc_num_basis=512,
        num_sources=2,
    )
# You can load the state_dict as here:
model.load_state_dict(torch.load('.../unsup_speech_enh_adaptation/pretrained_checkpoints/remixit_chime_adapted_student_bestbak_ep85_using_vad.pt'))
model = torch.nn.DataParallel(model).cuda()

# Scale the input mixture, perform inference and apply mixture consistency
input_mix = input_mix.unsqueeze(1).cuda() 
# input_mix.shape = (batch, 1, time_samples)
input_mix_std = input_mix.std(-1, keepdim=True)
input_mix_mean = input_mix.mean(-1, keepdim=True)
input_mix = (input_mix - input_mix_mean) / (input_mix_std + 1e-9)

estimates = model(input_mix)
estimates = mixture_consistency.apply(estimates, input_mix)
```

## Instructions for performance evaluation

Coming soon.

## Baseline performance

Coming soon.

### Reverberant LibriCHiME-5 dataset

|                                                      | SI-SDR (dB) | OVR_MOS | BAK_MOS | SIG_MOS |
| ---------------------------------------------------- | ----------- | ------- | ------- | ------- |
| unprocessed                                          |             |         |         |         |
| Sudo rm -rf (fully-supervised out-of-domain teacher) |             |         |         |         |
| RemixIT (self-supervised student)                    |             |         |         |         |
| RemixIT (self-supervised student) using VAD          |             |         |         |         |

### Single-speaker segments of the CHiME-5 dataset

|                        Mean                          | OVR_MOS | BAK_MOS | SIG_MOS |
| ---------------------------------------------------- | ------- | ------- | ------- |
| unprocessed                                          |         |         |         |
| Sudo rm -rf (fully-supervised out-of-domain teacher) |         |         |         |
| RemixIT (self-supervised student)                    |         |         |         |
| RemixIT (self-supervised student) using VAD          |         |         |         |


## References

Initial repo: https://github.com/etzinis/unsup_speech_enh_adaptation/

[1] Tzinis, E., Wang, Z., Jiang, X., and Smaragdis, P., “Compute and memory efficient universal sound source separation.” In Journal of Signal Processing Systems, vol. 9, no. 2, pp. 245–259, 2022, Springer.

[2] Tzinis, E., Adi, Y., Ithapu, V. K., Xu, B., Smaragdis, P., and Kumar, A., “RemixIT: Continual Self-Training of Speech Enhancement Models via Bootstrapped Remixing.” In IEEE Journal of Selected Topics in Signal Processing, vol. 16, no. 6, pp. 1329–1341, 2022, IEEE.
