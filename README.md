# VoiceConvertor

## Spectrogram creation and sampling
For spectrogram creation the `WINDOW_SIZE` of 256 was used which is defined in `Util/Eva_config_consts`. One `WINDOW_SIZE`
corresponds to one spectrum. By consecutive shifting the window and transforming it using FT a spectrogram is obtained. Notice
that the window can be shifted by different `WINDOW_STEP` which affects how many spectrums are in a spectrogram. In other words,
spectrums have some some percentage of overlap. 75% overlap has shown the best performance in spectrogram inversion, therefore the
`WINDOW_STEP` was chosen to be 64.

## Data set creation
## Model architecture

**EVA-NET:**

| Layer     | Layer output size |
|-----------|-------------------|
| INPUT     | 512x11x2          |
| CONV3-64  | 512x11x64         |
| CONV3-64  | 512x11x64         |
| POOL 2x1  | 256x11x64         |
| CONV3-128 | 256x11x128        |
| CONV3-128 | 256x11x128        |
| POOL 2x1  | 128x11x128        |
| CONV3-256 | 128x11x256        |
| CONV3-256 | 128x11x256        |
| POOL 2x1  | 64x11x256         |
|           |                   |
| FC        | 2048x1x1          |
| FC        | 2048x1x1          |
| FC        | Speaker / Phoneme |

The CONV3-64 stays for convolutional layer with 3x3 filter size and 64 is the number of filters in this layer. The POOL 2x1
stands for the pooling filters of size 2x1.

## Training
## Results
## References

1. [Very deep Multilingual convolutional neural network for LVCSR](https://arxiv.org/pdf/1509.08967v2.pdf)
2. [Phone Recognition on the TIMIT Database] (http://cdn.intechopen.com/pdfs/15948/InTech-Phoneme_recognition_on_the_timit_database.pdf)