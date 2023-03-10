# Baselines for the UDASE task of the CHiME-7 challenge

We pre-train a supervised Sudo rm- rf [1,2] teacher on some out-of-domain data (e.g. Libri1to3mix) and try to adapt a student model with the RemixIT [3] method on the unlabeled CHiME-5 data.

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
    
    **[Checkpoints](pretrained_checkpoints):**
    - Fully-supervised out-of-domain teacher model: ```libri1to3mix_supervised_teacher_w_mixconsist.pt```. 
    - We train the student models with the EMA teacher protocol update with an initial learning rate of 0.0003 and decreasing it to a third of its value every 10 epochs. We pick the student model chekpoint with the highest mean overall MOS as computed by DNS-MOS on the dev set ```remixit_chime_adapted_student.pt```. 
    - When training with the CHiME-5 data automatically annotated with Brouhaha's VAD (potentially all training mixtures would contain at least one active speaker), we choose the checkpoint with the highest mean overall MOS as computed by DNS-MOS on the dev set ```remixit_chime_adapted_student_using_vad.pt```. 

## Table of contents

- [Baselines for the UDASE task of the CHiME-7 challenge](#baselines-for-the-udase-task-of-the-chime-7-challenge)
  - [Table of contents](#table-of-contents)
  - [Datasets generation](#datasets-generation)
  - [Repo and paths configurations](#repo-and-paths-configurations)
  - [How to train the supervised teacher](#how-to-train-the-supervised-teacher)
  - [How to adapt the RemixIT student](#how-to-adapt-the-remixit-student)
  - [How to load a pretrained checkpoint](#how-to-load-a-pretrained-checkpoint)
  - [Instructions for performance evaluation](#instructions-for-performance-evaluation)
  - [Baseline performance](#baseline-performance)
    - [Reverberant LibriCHiME-5 dataset](#reverberant-librichime-5-dataset)
    - [Single-speaker segments of the CHiME-5 dataset](#single-speaker-segments-of-the-chime-5-dataset)
  - [References](#references)

## Datasets generation
Two datasets are required for generation, namely, Libri3mix and CHiME-5.

* **LibriMix:** For the generation of Libri3Mix one can follow the instructions [here](https://github.com/JorisCos/LibriMix) or just follow this:
```shell
cd {path_to_generate_Libri3mix}
git clone https://github.com/JorisCos/LibriMix
cd LibriMix 
pip install -r requirements.txt
conda install -c conda-forge sox # for linux
# conda install -c groakat sox # for windows
chmod +x generate_librimix.sh
./generate_librimix.sh storage_dir 
```
**Note:** If you have limited space, you can modify ```generate_librimix.sh``` to generate only Libri3Mix (not Libri2Mix) with ```freqs=16k``` and ```modes=min```.

* **CHiME-5:** For the generation of the CHiME-5 data follow the instructions [here](https://github.com/UDASE-CHiME2023/CHiME-5) or just follow these steps (this step requires the existence of CHiME-5 data under some path, [apply-and-get-CHiME5-here](https://chimechallenge.github.io/chime6/download.html)):
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

* **LibriCHiME-5:** For the generation of the reverberant LibriCHiME-5 data follow the instructions [here](https://github.com/UDASE-CHiME2023/reverberant-LibriCHiME-5).

## Repo and paths configurations
Set the paths for the aforementioned datasets and include the path of this repo.

```shell
git clone https://github.com/UDASE-CHiME2023/baseline.git
export PYTHONPATH={the path that you stored the github repo}:$PYTHONPATH
cd baseline
python -m pip install --user -r requirements.txt
vim __config__.py
```

You should change the following in ```__config__.py```:
```shell
LIBRI3MIX_ROOT_PATH = '{inset_path_to_Libri3mix}'
CHiME_ROOT_PATH = '{insert_path_of_processed_CHiME5_data}'
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
--warmup_checkpoint ../pretrained_checkpoints/libri1to3mix_supervised_teacher_w_mixconsist.pt \
--checkpoint_storage_path {insert_path_to_save_models} --log_audio --apply_mixture_consistency \
--n_jobs 12 -cad 2 3 -bs 24
```

## How to load a pretrained checkpoint
```python
import torch
import torchaudio
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
model.load_state_dict(torch.load('.../pretrained_checkpoints/remixit_chime_adapted_student_using_vad.pt'))
model = torch.nn.DataParallel(model).cuda()

# Scale the input mixture, perform inference and apply mixture consistency
input_mix, _ = torchaudio.load('audio_file.wav') # audio file should be mono channel
input_mix = input_mix.unsqueeze(1).cuda() 
# input_mix.shape = (batch, 1, time_samples)
input_mix_std = input_mix.std(-1, keepdim=True)
input_mix_mean = input_mix.mean(-1, keepdim=True)
input_mix = (input_mix - input_mix_mean) / (input_mix_std + 1e-9)

estimates = model(input_mix)
estimates = mixture_consistency.apply(estimates, input_mix)
```

## Instructions for performance evaluation

**Participants are asked to normalize their signals to -30 LUFS before computing the DNS-MOS performance scores. The same normalization should be applied to the submitted audio signals** (see [Submission](https://www.chimechallenge.org/current/task2/submission) section of the UDASE task website). 

The motivation for this normalization is that DNS-MOS (especially the SIG and BAK scores) is very sensitive to a change of the input signal loudness. This sensitivity to the overall signal loudness would make it difficult to compare different systems without a common normalization procedure. 

Regarding the listening tests, we do not want to evaluate the overall gain of the submitted systems, which is the reason why we also ask participants to normalize the submitted signals.

The value of -30 LUFS for the normalized loudness was chosen to avoid clipping of most of the unprocessed mixture signals in the CHiME-5 dataset. In the dev set, less than 2% of the unprocessed mixtures clip after loudness normalization to -30 LUFS. In the eval set, none of the unprocessed mixtures will clip after loudness normalization to -30 LUFS. Clipping of the CHiME-5 mixtures seems to be mostly due to friction-like noise caused by manipulations/movements of the in-ear binaural microphone worn by the participants of the CHiME-5 dinner parties.

We suggest to use [Pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) for loudness normalization to -30 LUFS. An example code is given below.

```python 
import soundfile as sf
import pyloudnorm as pyln

x, sr = sf.read("test.wav") # load audio

# peak normalize to an arbitrary value, e.g. 0.7
# this might be necessary before computing the loudness, to avoid -inf
x = x/np.max(np.abs(x))*0.7 

# measure the loudness 
meter = pyln.Meter(sr) # create loudness meter
loudness = meter.integrated_loudness(x)

# loudness normalize to -30 LUFS
x_norm = pyln.normalize.loudness(x, loudness, -30.0)
```

An evaluation script for computing the DNS-MOS and SI-SDR metrics is available at `./baseline/utils/final_evaluation.py`.



## Baseline performance

The average SI-SDR values (in dB) over the dev set of LibriCHiME-5 (1-3 speakers), as well as the DNS-MOS values over the dev set of CHiME-5 (1 speaker), for the unprocessed inputs, pretrained teacher and students models, are given in the following tables.

### Reverberant LibriCHiME-5 dataset

| Mean                                                 | SI-SDR (dB) |
| ---------------------------------------------------- | ----------- |
| unprocessed                                          | 6.57        |
| Sudo rm -rf (fully-supervised out-of-domain teacher) | 8.23        |
| RemixIT (self-supervised student)                    | 9.46        |
| RemixIT (self-supervised student) using VAD          | **9.83**    |

### Single-speaker segments of the CHiME-5 dataset

| Mean                                                 | OVR-MOS  | BAK-MOS  | SIG-MOS  |
| ---------------------------------------------------- | -------- | -------- | -------- |
| unprocessed                                          | 3.03     | 3.04     | **3.64** |
| Sudo rm -rf (fully-supervised out-of-domain teacher) | 3.08     | 3.79     | 3.48     |
| RemixIT (self-supervised student)                    | 3.07     | 3.84     | 3.43     |
| RemixIT (self-supervised student) using VAD          | **3.09** | **3.85** | 3.46     |


## References

[1] Tzinis, E., Wang, Z., & Smaragdis, P. (2020, September). Sudo rm-rf: Efficient networks for universal audio source separation. In 2020 IEEE 30th International Workshop on Machine Learning for Signal Processing (MLSP). <https://arxiv.org/abs/2007.06833>

[2] Tzinis, E., Wang, Z., Jiang, X., and Smaragdis, P., Compute and memory efficient universal sound source separation. In Journal of Signal Processing Systems, vol. 9, no. 2, pp. 245???259, 2022, Springer. <https://arxiv.org/pdf/2103.02644.pdf>

[3] Tzinis, E., Adi, Y., Ithapu, V. K., Xu, B., Smaragdis, P., & Kumar, A. (October, 2022). RemixIT: Continual self-training of speech enhancement models via bootstrapped remixing. In IEEE Journal of Selected Topics in Signal Processing. <https://arxiv.org/abs/2202.08862>
