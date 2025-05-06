import os
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import random


class ESC50Dataset(Dataset):
    def __init__(self, root_dir, folds, mode='train', target_length=216, augment=True, out_dim=2):
        """
        ESC-50 Dataset类

        参数：
            root_dir: 数据集根目录（包含audio/和meta/esc50.csv）
            folds: 选择的折数列表（1-5）
            mode: 模式（train/val/test）
            target_length: 目标频谱图时间维度长度
            augment: 是否应用数据增强
        """
        self.root_dir = root_dir
        self.folds = folds
        self.mode = mode
        self.target_length = target_length
        self.augment = augment
        self.sr = 22050  # 目标采样率
        self.out_dim = out_dim

        # 加载并过滤元数据
        self.metadata = pd.read_csv(os.path.join(root_dir, 'meta', 'esc50.csv'))
        self.metadata = self.metadata[self.metadata['fold'].isin(folds)]
        self.filenames = self.metadata['filename'].tolist()
        self.labels = self.metadata['target'].values

        # 音频处理参数
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mels = 128

        # 梅尔频谱图转换器
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        self.db_transform = T.AmplitudeToDB()

        # 数据增强参数
        self.time_mask = 15  # 时间遮蔽最大长度
        self.freq_mask = 10  # 频率遮蔽最大长度

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # 加载音频文件
        audio_path = os.path.join(self.root_dir, 'audio', self.filenames[idx])
        waveform, sr = torchaudio.load(audio_path)

        # 预处理流程
        waveform = self._preprocess_audio(waveform, sr)
        mel_spec = self._create_mel_spectrogram(waveform)
        mel_spec = self._postprocess_spectrogram(mel_spec)

        return mel_spec, self.labels[idx]

    def _preprocess_audio(self, waveform, orig_sr):
        """音频预处理流程"""
        # 重采样
        if orig_sr != self.sr:
            resampler = T.Resample(orig_sr, self.sr)
            waveform = resampler(waveform)

        # 转换为单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 数据增强（仅训练模式）
        if self.augment and self.mode == 'train':
            waveform = self._audio_augmentation(waveform)

        return waveform

    def _create_mel_spectrogram(self, waveform):
        """生成梅尔频谱图"""
        mel_spec = self.mel_transform(waveform)
        mel_db = self.db_transform(mel_spec)
        return mel_db

    def _postprocess_spectrogram(self, spec):
        """频谱图后处理"""
        # 调整时间维度
        spec = self._adjust_time_dimension(spec)

        # 数据增强（仅训练模式）
        if self.augment and self.mode == 'train':
            spec = self._spectrogram_augmentation(spec)

        # 标准化
        spec = (spec - spec.mean()) / (spec.std() + 1e-9)
        if self.out_dim == 2:
            spec = spec.squeeze(0)
        return spec

    def _audio_augmentation(self, waveform):
        """音频时域增强"""
        # 随机时移（最大1秒）
        if random.random() < 0.7:
            shift = random.randint(-self.sr, self.sr)
            waveform = torch.roll(waveform, shifts=shift, dims=1)
            if shift > 0:
                waveform[:, :max(0, shift)] = 0
            else:
                waveform[:, max(0, -shift):] = 0

        # 添加高斯噪声
        if random.random() < 0.5:
            snr = random.uniform(15, 30)
            noise = torch.randn_like(waveform) * 10 ** (-snr / 20)
            waveform += noise

        # 随机增益
        if random.random() < 0.5:
            gain = random.uniform(-6, 6)
            waveform = waveform * (10 ** (gain / 20))

        return waveform

    def _spectrogram_augmentation(self, spec):
        """频谱图增强"""
        # 时间遮蔽
        if random.random() < 0.5:
            spec = T.TimeMasking(time_mask_param=self.time_mask)(spec)

        # 频率遮蔽
        if random.random() < 0.5:
            spec = T.FrequencyMasking(freq_mask_param=self.freq_mask)(spec)

        return spec

    def _adjust_time_dimension(self, spec):
        """调整时间维度到固定长度"""
        current_length = spec.shape[2]
        if current_length > self.target_length:
            # 随机/中心裁剪
            start = (current_length - self.target_length) // 2
            if self.mode == 'train' and self.augment:
                start = random.randint(0, current_length - self.target_length)
            spec = spec[:, :, start:start + self.target_length]
        elif current_length < self.target_length:
            # 时间轴填充
            pad = self.target_length - current_length
            spec = torch.nn.functional.pad(spec, (0, pad), mode='constant')
        return spec


if __name__ == '__main__':
    dataset = ESC50Dataset(root_dir='../data/ESC-50', folds=[1], mode='train', target_length=216, augment=False)
