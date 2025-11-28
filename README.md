# BSMamba2

Official PyTorch implementation of BSMamba2 model for music source separation.

This repository is the official implementation of our paper. We would like to acknowledge the following repositories for their valuable contributions:

- [lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer): Helpful for BSRoformer model code
- [amanteur/BandSplitRNN-PyTorch](https://github.com/amanteur/BandSplitRNN-PyTorch): Helpful for training code
- [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training): Helpful for dataset and augmentation

## Model Architecture

![BSMamba2 Architecture](assets/architecture.png)

## Pretrained Weights

| Model | cSDR | uSDR | Link |
|-------|------|------|------|
| bsroformer | 10.47 | 10.29 | [Link](https://drive.google.com/file/d/1xAjVjCIp71gqjjgCJYjU1yaMLp39pHEm/view?usp=sharing) |
| bsmamba2 | 11.03 | 10.70 | [Link](https://drive.google.com/file/d/1W-IX8I5B2g-8JA5Zaf-HkUul0MpOEBaZ/view?usp=sharing) |

## Installation

```bash
pip install -r requirements.txt
```

### Other Requirements

The following requirements are needed (from [state-spaces/mamba](https://github.com/state-spaces/mamba)):

- Linux
- NVIDIA GPU
- PyTorch 1.12+
- CUDA 11.6+

## Dataset Setup

### 0. Setting

1. Download [MUSDB18 HQ](https://zenodo.org/record/3338373)

2. Create a `valid` folder and move the following songs from the `train` directory to the `valid` directory:
   - 'Actions - One Minute Smile'
   - 'Clara Berry And Wooldog - Waltz For My Victims'
   - 'Johnny Lokke - Promises & Lies'
   - 'Patrick Talbot - A Reason To Leave'
   - 'Triviul - Angelsaint'
   - 'Alexander Ross - Goodbye Bolero'
   - 'Fergessen - Nos Palpitants'
   - 'Leaf - Summerghost'
   - 'Skelpolu - Human Mistakes'
   - 'Young Griffo - Pennies'
   - 'ANiMAL - Rockshow'
   - 'James May - On The Line'
   - 'Meaxic - Take A Step'
   - 'Traffic Experiment - Sirens'

   The final directory structure should look like:
   ```
   MUSDB18HQ/  (directory name can be different)
   ├── test/
   ├── train/
   └── valid/
   ```

3. Set the environment variable:
   ```bash
   export MUSDB_DIR={MUSDB18HQ path}
   ```

4. If you want to use pretrained weights, download the desired model from the table above and place it in `src/logs/`.

5. Prepare the dataset by running SAD (Source Activity Detection) process on train/valid/test:
   ```bash
   cd src
   python prepare_dataset.py -i {dataset path} -o {output path} --subset {train/valid/test}
   ```

6. Configure the dataset paths in `src/config`:
   - Set `file_dir` to the path where the actual audio files are located in the train dataset
   - Set `txt_dir` to the parent directory of the files created in step 5
   
   **Note:** If you are using pretrained weights, you must manually update the `file_dir` and `txt_dir` in the downloaded model's `tb_logs/hparams.yaml` file.

## Usage

**Note:** All commands should be executed from the `src` directory. Please change to the `src` directory before running any commands:

```bash
cd src
```

### 1. Training

Configure the config files (if you want to reproduce the results, download the pretrained weights and match the config settings).

```bash
cd src
python train.py
```

### 2. Evaluation

```bash
cd src
python evaluate.py --run-dir logs/{some_log}/vocals/{some_date} --duration {duration}
```

### 3. Inference

```bash
cd src
python inference.py -i {input_audio_path} -c logs/{some_log}/vocals/{some_date}/weights/{some_weight}.ckpt
```


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

