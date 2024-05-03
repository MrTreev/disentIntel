import os
import torch
import pickle
import numpy as np
from torch.utils.data.sampler import Sampler

torch.multiprocessing.set_sharing_strategy("file_system")


class Utterances(torch.utils.data.Dataset):
    """Dataset class for the Utterances dataset."""

    def __init__(self, root_dir, feat_dir, mode):
        """Initialize and preprocess the Utterances dataset."""
        self.root_dir = root_dir
        self.feat_dir = feat_dir
        self.mode = mode
        self.step = 20
        self.split = 0

        # metaname = os.path.join(self.root_dir, "train.pkl")
        metaname = os.path.join(self.root_dir[:-6], "train.pkl")
        meta = pickle.load(open(metaname, "rb"))

        meta = list(meta)
        dataset = list(
            len(meta) * [None]
        )
        for i in range(0, len(meta), self.step):
            self.load_data(meta[i : i + self.step], dataset, i, mode)

        dataset = list(dataset)
        # very importtant to do dataset = list(dataset)
        if mode == "train":
            self.train_dataset = list(dataset)
            self.num_tokens = len(self.train_dataset)
        elif mode == "test":
            self.test_dataset = list(dataset)
            self.num_tokens = len(self.test_dataset)
        else:
            raise ValueError

        print("Finished loading {} dataset...".format(mode))

    def load_data(self, submeta, dataset, idx_offset, mode):
        for k, sbmt in enumerate(submeta):
            uttrs = len(sbmt) * [None]
            # fill in speaker id and embedding
            uttrs[0] = sbmt[0]
            uttrs[1] = sbmt[1]
            # fill in data
            sp_tmp = np.load(os.path.join(self.root_dir, sbmt[2]))
            f0_tmp = np.load(os.path.join(self.feat_dir, sbmt[2]))
            if self.mode == "train":
                sp_tmp = sp_tmp[self.split :, :]
                f0_tmp = f0_tmp[self.split :]
            elif self.mode == "test":
                sp_tmp = sp_tmp[: self.split, :]
                f0_tmp = f0_tmp[: self.split]
            else:
                raise ValueError
            try:
                uttrs[2] = (sp_tmp, f0_tmp)
                dataset[idx_offset + k] = uttrs
            except Exception as e:
                print(e)

    def __getitem__(self, index):
        dataset = self.train_dataset if self.mode == "train" else self.test_dataset

        list_uttrs = dataset[index]
        emb_org = list_uttrs[1]

        melsp, f0_org = list_uttrs[2]

        # this is a tuple for batch[i]
        return melsp, emb_org, f0_org

    def __len__(self):
        """Return the number of spkrs."""
        return self.num_tokens


class MyCollator(object):
    def __init__(self, hparams):
        self.min_len_seq = hparams.min_len_seq
        self.max_len_seq = hparams.max_len_seq
        self.max_len_pad = hparams.max_len_pad

    def __call__(self, batch):
        # batch[i] is a tuple of __getitem__ outputs
        # full_melspec:
        #       melspec of the entire audio file ~15min ?
        #       probably not but just a preselected segment
        #       (see other code and try to find it)
        # speaker_embedding:
        #       speaker embedding
        # pitch_contour:
        #       pitch contour
        new_batch = []
        for token in batch:
            (
                full_melspec,
                speaker_embedding,
                pitch_contour,
            ) = token
            len_crop = np.random.randint(
                # 64-129, two ints
                # 1.5s ~ 3s
                self.min_len_seq, self.max_len_seq + 1,
            )
            left = np.random.randint(
                0, len(full_melspec) - len_crop, size=2
            )

            cropped_melspec = full_melspec[left : left + len_crop, :]
            pitch_contour = pitch_contour[left : left + len_crop]

            cropped_melspec = np.clip(cropped_melspec, 0, 1)

            cropped_melspec_padded = np.pad(
                cropped_melspec,
                ((0, self.max_len_pad - cropped_melspec.shape[0]), (0, 0)),
                "constant"
            )
            pitch_contour_padded = np.pad(
                pitch_contour[:, np.newaxis],
                ((0, self.max_len_pad - pitch_contour.shape[0]), (0, 0)),
                "constant",
                constant_values=-1e10,
            )

            new_batch.append((
                cropped_melspec_padded,
                speaker_embedding,
                pitch_contour_padded,
                len_crop
            ))

        batch = new_batch

        (
            cropped_melspec,
            speaker_embedding,
            pitch_contour,
            crop_length
        ) = zip(*batch)

        melsp = torch.from_numpy(
            # Stack along batch-dimension e.g. (16, 192, 80)
            # (batch-size, 192 16ms frames, 80 frequency bins) ?
            np.stack(cropped_melspec, axis=0)
        )
        spk_emb = torch.from_numpy(
            # Stack along batch-dimension e.g. (16, 82)
            # (batch-size, 82 one-hot encoded speaker embedding vector) ?
            np.stack(speaker_embedding, axis=0)
        )
        pitch = torch.from_numpy(
            # Stack along batch-dimension e.g. (16, 192, 1)
            # (batch-size, 192 10ms frames, 1 pitch value)
            np.stack(pitch_contour, axis=0)
        )
        len_org = torch.from_numpy(
            # stack along batch-dimension e.g. (16)
            np.stack(crop_length, axis=0)
        )

        return melsp, spk_emb, pitch, len_org


class MultiSampler(Sampler):
    """Samples elements more than once in a single pass through the data."""

    def __init__(self, num_samples, n_repeats, shuffle=False):
        self.num_samples = num_samples
        self.n_repeats = n_repeats
        self.shuffle = shuffle

    def gen_sample_array(self):
        self.sample_idx_array = torch.arange(
            self.num_samples, dtype=torch.int64
        ).repeat(self.n_repeats)
        if self.shuffle:
            self.sample_idx_array = self.sample_idx_array[
                torch.randperm(len(self.sample_idx_array))
            ]
        return self.sample_idx_array

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.sample_idx_array)


def get_loader(hparams) -> torch.utils.data.DataLoader:
    """Build and return a data loader."""

    def worker_init_fn():
        np.random.seed((torch.initial_seed()) % (2**32))

    dataset = Utterances(
        hparams.common_root_dir, hparams.common_feat_dir, hparams.mode
    )
    my_collator = MyCollator(hparams)
    # sampler = MultiSampler(
    #     len(dataset),
    #     hparams.samplier,
    #     shuffle=hparams.shuffle
    # )
    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=hparams.batch_size,
        sampler=sampler,
        num_workers=hparams.num_workers,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        collate_fn=my_collator,
    )
    return data_loader
