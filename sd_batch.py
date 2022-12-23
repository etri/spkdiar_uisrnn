#!/usr/bin/env python

import os
import sys
import argparse

import torch
import numpy as np

from utils import load_json
from nnet import Nnet
#from trainer import get_logger

#from kaldi_python_io import Reader, ScriptReader

import librosa
import soundfile as sf
from scipy.io import wavfile

import uisrnn

#logger = get_logger(__name__)

sr = 16000
nfft = 512
window = 0.025
hop = 0.01
nmels = 40
mel_basis = librosa.filters.mel(sr, n_fft=nfft, n_mels=nmels)

dvector_args, model_args, training_args, inference_args = uisrnn.parse_arguments()
print(' '.join(sys.argv)+'\n')
os.makedirs(os.path.dirname(dvector_args.dvector_log_file), exist_ok=True)
with open(dvector_args.dvector_log_file, 'w') as ofp:
    print(' '.join(sys.argv)+'\n', file=ofp)
os.makedirs(os.path.dirname(training_args.uisrnn_log_file), exist_ok=True)
with open(training_args.uisrnn_log_file, 'w') as ofp:
    print(' '.join(sys.argv)+'\n', file=ofp)

class NnetComputer(object):
    """
    Compute output of networks
    """
    def __init__(self, cpt_dir, gpuid, dwin, dhop):
        # chunk size when inference
        loader_conf = load_json(cpt_dir, "loader.json")
        #self.chunk_size = sum(loader_conf["chunk_size"]) // 2
        self.chunk_size = dwin
        self.hop = dhop
        #logger.info("Using chunk size {:d}".format(self.chunk_size))
        # GPU or CPU
        self.device = f"cuda:{gpuid}" if gpuid >= 0 else "cpu"
        # load nnet
        nnet = self._load_nnet(cpt_dir)
        self.nnet = nnet.to(self.device)
    def _load_nnet(self, cpt_dir):
        # nnet config
        nnet_conf = load_json(cpt_dir, "mdl.json")
        nnet = Nnet(**nnet_conf)
        cpt_fname = os.path.join(cpt_dir, "best.pt.tar")
        cpt = torch.load(cpt_fname, map_location="cpu")
        nnet.load_state_dict(cpt["model_state_dict"])
        #logger.info("Load checkpoint from {}, epoch {:d}".format(cpt_fname, cpt["epoch"]))
        nnet.eval()
        return nnet
    def _make_chunk(self, feats):
        T, F = feats.shape
        # step: half chunk
        S = self.hop
        N = (T - self.chunk_size) // S + 1
        if N <= 0:
            return feats
        elif N == 1:
            return feats[:self.chunk_size]
        else:
            chunks = torch.zeros([N, self.chunk_size, F],
                                 device=feats.device,
                                 dtype=feats.dtype)
            for n in range(N):
                chunks[n] = feats[n * S:n * S + self.chunk_size]
            return chunks
    def compute(self, feats):
        feats = torch.tensor(feats, device=self.device)
        with torch.no_grad():
            chunks = self._make_chunk(feats)  # N x C x F
            dvector = self.nnet(chunks)  # N x D
            #dvector = torch.mean(dvector, dim=0).detach()
            dvector = dvector.detach()
            return dvector.cpu().numpy()

def count_speaker(spk):
    speaker_count = []
    count=1
    if len(spk)>1:
        for i in range(1, len(spk)):
            if spk[i-1] == spk[i]:
                count += 1
            else :
                speaker_count.append([spk[i-1], count])
                count = 1
        speaker_count.append([spk[i], count])
    else:
        speaker_count.append([spk[0], 1])
    return speaker_count

def run():
    model = uisrnn.UISRNN(model_args)
    model.load(model_args.model_name)
    computer = NnetComputer(dvector_args.checkpoint, dvector_args.gpu, dvector_args.dwin, dvector_args.dhop)
    with open(dvector_args.dvector_log_file, 'a') as ofp:
        print(dvector_args.vad_file, end=' ', file=ofp)
    _, data = wavfile.read(dvector_args.wav_file)
    if data.dtype == 'int16':
        nbit = 16
    elif data.dtype == 'int32':
        nbit = 32
    max_nbit = float(2**(nbit-1))
    data = data / max_nbit
    segment = []
    seg = []
    times = []
    with open(dvector_args.vad_file) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            items = line.split()
            vad = items[2]
            if vad == "1":
                st = int(float(items[0])*100)
                et = int(float(items[1])*100)
                seg = np.asarray(data[round(st/100*sr):round(et/100*sr)])
                S = librosa.core.stft(y=seg, n_fft=nfft, win_length=round(window*sr), hop_length=round(hop*sr))
                S = np.abs(S)**2
                S = np.log10(np.dot(mel_basis, S) + 1e-6)
                feats = S.T[:-1]
                if len(feats) < dvector_args.dwin:
                    continue
                dvector = computer.compute(feats)
                dvector = dvector.astype(np.float64)
                segment.append(dvector)
                #if dur < min_len:
                #    continue
                times.append((st, et, dvector.shape[0]))
                #print(f'{st} {et} {et-st} {dvector.shape[0]}')

    segment = np.concatenate(segment, axis=0)
    print(segment.shape)
    with open(dvector_args.dvector_log_file, 'a') as ofp:
        print(len(segment), file=ofp)

    predicted_cluster_id = model.predict(segment, inference_args)

    os.makedirs(os.path.dirname(dvector_args.out_file), exist_ok=True)
    with open(dvector_args.out_file, 'w') as ofp:
        nseg = 0
        for t in times:
            spk = predicted_cluster_id[nseg:nseg+t[2]]
            spk_count = count_speaker(spk)
            nseg += t[2]
            #print(t[0], t[1], t[2], spk)
            #print(spk_count, len(spk_count))
            count = 0
            if len(spk_count) == 1:
                st = t[0]
                et = t[1]
                #print(f'    ({st} {et} {spk[0]}) {len(spk)} {et-st}')
                print(f'{st} {et} {spk[0]}', file=ofp)
            else:
                for i, s in enumerate(spk_count):
                    if i == 0:
                        st = t[0]
                        et = t[0] + dvector_args.dwin//2 + dvector_args.dhop * s[1]
                    elif i == len(spk_count)-1:
                        st = t[0] + dvector_args.dwin//2 + dvector_args.dhop * count
                        et = t[1]
                    else:
                        st = t[0] + dvector_args.dwin//2 + dvector_args.dhop * count
                        et = st + dvector_args.dhop * s[1]
                    #print(f'    {i} ({st} {et} {s[0]}) {s[1]} {et-st}')
                    print(f'{st} {et} {s[0]}', file=ofp)
                    count += s[1]

    print(f'{len(predicted_cluster_id)} segments, {len(set(predicted_cluster_id))} speakers')
    #print('Predicted labels:')
    #print(predicted_cluster_id)
    #print('-' * 80)
    with open(training_args.uisrnn_log_file, 'a') as ofp:
        print('Predicted labels:', file=ofp)
        print(predicted_cluster_id, file=ofp)
        print(f'{len(set(predicted_cluster_id))} speakers', file=ofp)
        print('-' * 80, file=ofp)

if __name__ == "__main__":
    run()
    print('Done')
