from tfcompat.hparam import HParams

# NOTE: If you want full control for model architecture. please take a look
# at the code and change whatever you want. Some hyper parameters are hardcoded

dataset_base_dir = "/mnt/datasets/processed/disentIntel"
out_dir = "out"

# Default hyperparameters:
hparams = HParams(
    # Content Encoder (Encoder_7)
    freq=8,                         # downsample factor (tune)
    dim_neck=8,                     # BLSTM dim
    dim_enc=512,                    # Conv dim (tune)
    # Rythm Encoder (Encoder_t)
    freq_2=8,
    dim_neck_2=1,
    dim_enc_2=128,
    # Pitch Encoder (Encoder_6)
    freq_3=8,
    dim_neck_3=32,
    dim_enc_3=256,
    dim_freq=80,
    dim_spk_emb=14212,              # 82 (original), 3860 (de), 14212 (en)
    dim_f0=257,
    dim_dec=512,
    len_raw=128,
    chs_grp=16,
    # interp (Random Resampling via linear interpolation)
    min_len_seg=19,
    max_len_seg=32,
    min_len_seq=64,
    max_len_seq=128,
    max_len_pad=1024,               # 192 (original), 672 (de), 576 (en)
                                    # has to be: % 32 == 0
    # data loader
    common_data_dir=f"{dataset_base_dir}/CommonSpeakerSplit",
    common_speaker_dir=f"{dataset_base_dir}/CommonSpeakerSplit/wavs",
    common_root_dir=f"{dataset_base_dir}/CommonSpeakerSplit/spmel",
    common_feat_dir=f"{dataset_base_dir}/CommonSpeakerSplit/raptf0",
    uaspeech_sagi_data_dir=f"{dataset_base_dir}/UASPEECH_SAGI",
    uaspeech_sagi_audio_dir=f"{dataset_base_dir}/UASPEECH_SAGI/audio",
    uaspeech_sagi_word_file=f"{dataset_base_dir}/UASPEECH_SAGI/transcription_lut.csv",
    uaspeech_custom_3_data_dir=f"{dataset_base_dir}/UASPEECH_Custom_3",
    uaspeech_custom_3_audio_dir=f"{dataset_base_dir}/UASPEECH_Custom_3/audio",
    uaspeech_custom_3_inference_dir=f"{dataset_base_dir}/UASPEECH_Custom_3/inference",
    weights="run/models/174000-G.ckpt",
    batch_size=16,
    mode="train",
    shuffle=True,
    num_workers=0,
    samplier=8,
)


def hparams_debug_string():
    values = hparams.values()
    hp = ["  %s: %s" % (name, values[name]) for name in values]
    return "Hyperparameters:\n" + "\n".join(hp)
