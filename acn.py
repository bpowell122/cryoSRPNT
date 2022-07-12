'''
Corrupt particle images with structural noise --> CTF --> digital/shot noise
| 02/2021:Written by Ellen Zhong, Emily Navarrete, and Joey Davis
|  07/2022: Refactored to include tilt series and run faster with less memory usage by Barrett Powell
'''

import argparse
import numpy as np
import os
import time
import matplotlib.pyplot as plt

import torch
from torch.utils import data
from torch.utils.data import DataLoader

from cryodrgn.ctf import compute_ctf
from cryodrgn import mrc
from cryodrgn import utils

try:
    from memory_profiler import profile
except:
    pass

log = utils.log
vlog = utils.vlog


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('particles', type=os.path.abspath, help='Input MRC stack (.mrcs)')
    parser.add_argument('outstack', type=os.path.abspath, help='Output .mrcs particle stack')
    parser.add_argument('--ctf', type=os.path.abspath, help='Get all ctf parameters from a cryodrgn-format ctf pkl')
    parser.add_argument('--snr1', default=1.4, type=float, help='Intermediate SNR for pre-CTF application of structural noise')
    parser.add_argument('--std1', type=float, help='Override --snr1 with gaussian noise stdev. Set to 0 for no structural noise')
    parser.add_argument('--snr2', default=0.05, type=float, help='Final SNR after post-CTF application of shot noise')
    parser.add_argument('--std2', type=float, help='Override --snr2 with gaussian noise stdev. Set to 0 for no shot noise')
    parser.add_argument('--out-pkl', type=os.path.abspath, help='Optional output pkl for ctf params')
    parser.add_argument('--invert', default=True, help='Invert the image data sign. Default is to invert, which is common/correct for most EM processing')
    parser.add_argument('--normalize', action='store_true', help='Normalize output particle stack to have a mean 0 and std 1')

    group = parser.add_argument_group('Define CTF parameters at command line')
    group.add_argument('--Apix', type=float, help='Pixel size (A/pix)')
    group.add_argument('--dfu', default=15000, type=float, help='Defocus U (Å)')
    group.add_argument('--dfv', default=15000, type=float, help='Defocus V (Å)')
    group.add_argument('--ang', default=0, type=float, help='Astigmatism angle (deg)')
    group.add_argument('--kv', default=300, type=float, help='Microscope voltage (kV)')
    group.add_argument('--cs', default=2, type=float, help='Spherical aberration (mm)')
    group.add_argument('--wgh', default=0.1, type=float, help='Amplitude constrast ratio')
    group.add_argument('--ps', default=0, type=float, help='Phase shift (deg)')
    group.add_argument('--b-factor', default=None, type=float, help='B factor for Gaussian envelope (Å^2)')
    group.add_argument('--df-std', default=None, type=float, help='Jiggle defocus per image with this stdev')
    group.add_argument('--no-astigmatism', action='store_true', help='Keep dfu and dfv the same per particle if sampling with --df-std')

    group = parser.add_argument_group('Tilt series exclusive parameters')
    group.add_argument('--tilt-series', type=os.path.abspath, help='Path to file (.txt) specifying full tilt series x-axis stage-tilt scheme in degrees. '
                                                                   'Real-space particles will be weighted by cos(tilt) between structural noise and CTF')
    group.add_argument('--dose', default=None, type=float, help='Dose in e- / A2 / tilt. '
                                                               'Fourier-space particles will be weighted by exposure-dependent amplitude attenuation before structural noise')

    group = parser.add_argument_group('Optional additional arguments')
    group.add_argument('--seed', type=int, help='Random seed for sampling defocus values')
    group.add_argument('-v','--verbose',action='store_true',help='Increases verbosity')
    group.add_argument('-b', type=int, default=500, help='Minibatch size')
    group.add_argument('--out-png', type=os.path.abspath, help='Path to save montage of first 9 projections')

    return parser


def calculate_dose_weights(ntilts, D, pixel_size, dose_per_A2_per_tilt, voltage):
    '''
    code adapted from Grigorieff lab summovie_1.0.2/src/core/electron_dose.f90
    see also Grant and Grigorieff, eLife (2015) DOI: 10.7554/eLife.06980
    assumes even-sized FFT (i.e. non-ht-symmetrized, DC component is bottom-right of central 4 px)
    '''
    cumulative_doses = dose_per_A2_per_tilt * np.arange(1, ntilts+1)
    dose_weights = np.zeros((ntilts, D, D))
    fourier_pixel_sizes = 1.0 / (np.array([D, D]))  # in units of 1/px
    box_center_indices = np.array([D, D]) // 2
    critical_dose_at_dc = 0.001 * (2 ** 31) # shorthand way to ensure dc component is always weighted ~1
    voltage_scaling_factor = 1.0 if voltage == 300 else 0.8  # 1.0 for 300kV, 0.8 for 200kV microscopes

    for k, dose_at_end_of_tilt in enumerate(cumulative_doses):

        for j in range(D):
            y = ((j - box_center_indices[1]) * fourier_pixel_sizes[1])

            for i in range(D):
                x = ((i - box_center_indices[0]) * fourier_pixel_sizes[0])

                if ((i, j) == box_center_indices).all():
                    spatial_frequency_critical_dose = critical_dose_at_dc
                else:
                    spatial_frequency = np.sqrt(x ** 2 + y ** 2) / pixel_size  # units of 1/A
                    spatial_frequency_critical_dose = (0.24499 * spatial_frequency ** (
                        -1.6649) + 2.8141) * voltage_scaling_factor  # eq 3 from DOI: 10.7554/eLife.06980

                dose_weights[k, j, i] = np.exp((-0.5 * dose_at_end_of_tilt) / spatial_frequency_critical_dose)  # eq 5 from DOI: 10.7554/eLife.06980

    assert dose_weights.min() >= 0.0
    assert dose_weights.max() <= 1.0
    return dose_weights


class ImageDataset(data.Dataset):
    '''
    Quick dataset class to shovel particles and corresponding CTF params into pytorch dataloader
    Benefit = faster computations on GPU, using pytorch dataloader framework
    '''
    def __init__(self, particles, ctf_params, ntilts=1):
        D = particles.shape[-1]
        if ntilts > 1:
            particles = particles.reshape(-1, ntilts, D, D)
            ctf_params = ctf_params.reshape(-1, ntilts, 9)
        self.particles = particles
        self.ctf_params = ctf_params
        self.N = particles.shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return index, self.particles[index], self.ctf_params[index]


def plot_projections(out_png, imgs):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10,10))
    axes = axes.ravel()
    for i in range(min(len(imgs),9)):
        axes[i].imshow(imgs[i])
    plt.savefig(out_png)


# @profile
def main(args):
    vlog(args)

    # set seed
    if args.seed is not None:
        np.random.seed(args.seed)

    # configure CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    log(f'Use cuda {use_cuda}')
    if use_cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    # load particles
    log(f'Loading particles from {args.particles} ...')
    particles = mrc.parse_mrc(args.particles)[0].astype(np.float32, copy=False)
    assert particles.shape[-1] == particles.shape[-2], 'Images must be square'
    Nimg, D, D = particles.shape
    log(f'Loaded {Nimg} {D}x{D} particles')

    # normalize the particles to 0-mean (not worrying about unit stdev yet)
    log('Normalizing input stack to 0-mean...')
    particles /= np.mean(particles)

    # calculate std of clean particle stack subset
    log('Calculating standard deviation of input stack...')
    subset = np.random.choice(np.arange(Nimg), min(10000, Nimg // 10), replace=False)  # subset bc np.std makes full copy of ptcl array internally and this can cause memory problems for large stacks
    std = np.std(particles[subset])
    log(f'Input stack standard deviation: {std}')

    # calculate std1
    assert args.snr1 > 0, '--snr1 must be positive'
    if args.std1 is None:
        snr1 = args.snr1
        std1 = std/np.sqrt(snr1)
    else:
        std1 = args.std1
        snr1 = (std / std1) ** 2
    if std1 > 0:
        log(f'Will add s1 (structural noise) with stdev {std1} targeting SNR {snr1}')
    else:
        log(f'Will not add s1 (structural noise); std1 <= 0')

    # calculate std2
    assert args.snr2 > 0, '--snr2 must be positive'
    if args.std2 is None:
        # cascading of noise processes according to Frank and Al-Ali (1975) & Baxter (2009)
        # args.std2 is overall SNR; here we need SNR for shot noise alone
        snr2 = (1 + 1/args.snr1) / (1/args.snr2 - 1/args.snr1)
        std2 = std/np.sqrt(snr2)
    else:
        std2 = args.std2
        snr2 = (std / std2) ** 2
    if std2 > 0:
        log(f'Will add s2 (shot noise) with stdev {std2} targeting SNR {snr2}')
    else:
        log(f'Will not add s2 (shot noise); std2 <= 0')

    # calculate overall final snr
    if (std1 <= 0) and (std2 <= 0):
        pass
    elif (std1 <= 0) and (std2 > 0):
        log(f'Final SNR: {snr2}')
    elif (std1 > 0) and (std2 <= 0):
        log(f'Final SNR: {snr1}')
    else:
        log(f'Final SNR: {(snr1 * snr2 / (1 + snr1 + snr2))}')  # rearranged cascading noise processes from above

    # load CTF from pkl or prepare ctf_params array from args
    if args.ctf is not None:
        log(f'Loading CTF parameters from {args.ctf}')
        ctf_params = utils.load_pkl(args.ctf)
        assert ctf_params.shape[0] == Nimg, f'CTF pkl file contains data for {ctf_params.shape[0]} particles but dataset has {Nimg} particles'
    else:
        log('CTF pkl file not specified; using CTF parameters specified at command line')
        assert args.Apix is not None, '--Apix must be specified if ctf.pkl is not provided'
        ctf_params = np.zeros((1, 9))
        ctf_params[0,0] = D
        ctf_params[0,1] = args.Apix
        ctf_params[0,2] = args.dfu
        ctf_params[0,3] = args.dfv
        ctf_params[0,4] = args.ang
        ctf_params[0,5] = args.kv
        ctf_params[0,6] = args.cs
        ctf_params[0,7] = args.wgh
        ctf_params[0,8] = args.ps
        ctf_params = np.tile(ctf_params, (Nimg, 1))
    if args.df_std is not None:
        log(f'Jiggling defocus values by stdev {args.df_std}')
        df_mean = np.mean(ctf_params[0,2:4])
        df_std = np.random.normal(df_mean, args.df_std, Nimg)
        if args.no_astigmatism:
            assert args.dfv == args.dfu, "--dfu and --dfv must be the same"
            ctf_params[:,2] += df_std
            ctf_params[:,3] += df_std
        else:
            ctf_params[:,2] += df_std
            ctf_params[:,3] += np.random.normal(df_mean, args.df_std, Nimg)



    # prepare frequency lattice
    freqs = np.arange(-D/2, D/2) / (ctf_params[0,1] * D)
    x0, x1 = np.meshgrid(freqs, freqs)
    freqs = np.stack([x0.ravel(), x1.ravel()], axis=1)

    # calculate tilt and dose weighting matrices, if provided
    if args.tilt_series:
        tilt_series = np.loadtxt(args.tilt_series, dtype=np.float32)
        ntilts = len(tilt_series)
        log(f'Loaded tilt scheme from {args.tilt_series} with {len(tilt_series)} tilts: {tilt_series}')

        log('Using tilt scheme to calculate further attenuation of SNR due to increasing tilt')
        tilt_weights = np.cos(tilt_series * np.pi / 180).reshape(1, ntilts, 1, 1)

        if args.dose is not None:
            log(f'Using dose {args.dose}e-/A2/tilt and tilt scheme to calculate exposure dependent amplitude attenuation of each spatial frequency')
            assert len(set(ctf_params[:,1])) == 1, 'Found multiple pixel sizes in ctf_params; this is not currently supported for dose weighting'
            assert len(set(ctf_params[:,5])) == 1, 'Found multiple voltages in ctf_params; this is not currently supported for dose weighting'
            dose_weights = calculate_dose_weights(ntilts, D, ctf_params[0,1], args.dose, ctf_params[0,5])
            plot_projections('dose_weights.png', dose_weights[:9])
        else:
            dose_weights = np.ones((ntilts, D, D))
    else:
        ntilts = 1

    # instantiate dataset and dataloader
    particle_dataset = ImageDataset(particles, ctf_params, ntilts=ntilts)

    # convert key variables to tensors for pytorch-based computations
    freqs = torch.from_numpy(freqs).to(device)
    if args.tilt_series:
        dose_weights = torch.from_numpy(dose_weights).to(device)
        tilt_weights = torch.from_numpy(tilt_weights).to(device)

    ### do all processing in dataloader context
    log('Done all configuration steps; starting processing now!')
    t1 = time.time()
    data_generator = DataLoader(particle_dataset, batch_size = args.b, shuffle = False)
    for i, (batch_idx, batch_ptcls, batch_ctf_params) in enumerate(data_generator):
        vlog(f'Corrupting particles {(i+1) * args.b}/{particle_dataset.N}')

        # FFT particle stack, apply dose weights, IFFT stack
        if args.tilt_series:
            batch_ptcls = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(batch_ptcls, dim=(-1, -2))), dim=(-1, -2))
            batch_ptcls = batch_ptcls.view(-1, ntilts, D, D) * dose_weights.view(1, ntilts, D, D)
            batch_ptcls = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(batch_ptcls, dim=(-1, -2))),dim=(-1, -2))

        # apply structural noise std1
        batch_ptcls += torch.normal(mean=0, std=std1, size=batch_ptcls.shape)

        # apply tilt weighting
        if args.tilt_series:
            batch_ptcls *= tilt_weights
            batch_ptcls = batch_ptcls.view(-1, D, D)
            batch_ctf_params = batch_ctf_params.view(-1, 9)

        # FFT stack, apply CTF, IFFT stack
        batch_ptcls = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(batch_ptcls, dim=(-1, -2))), dim=(-1, -2))
        ctf_weights = compute_ctf(freqs, *torch.split(batch_ctf_params[:, 2:], 1, dim=1), bfactor=args.b_factor).view(-1, D, D)
        batch_ptcls *= ctf_weights
        batch_ptcls = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(batch_ptcls, dim=(-1, -2))), dim=(-1, -2))

        # IFFT stack and apply shot noise std2
        batch_ptcls += torch.normal(mean=0, std=std2, size=batch_ptcls.shape)

        # invert if requested
        if args.invert:
            batch_ptcls *= -1

        # return particle to input array; update array in-place
        if args.tilt_series:
            batch_ptcls = batch_ptcls.view(-1, ntilts, D, D)

        particle_dataset.particles[batch_idx.cpu().numpy()] = batch_ptcls.real.cpu().numpy()


    ### post-dataloader final steps
    t2 = time.time()
    log(f'Added CTF and noise to {Nimg} images in {t2 - t1}s ({(t2 - t1) / (Nimg)}s per image)')

    # normalize particle stack via subset sampling
    if args.normalize:
        log('Calculating standard deviation of output stack...')
        particles -= np.mean(particles[subset])
        particles /= np.std(particles[subset])
        log(f'Normalized output stack mean and stdev to 0, 1')

    # undo any remaining tilt_series particle reshaping
    particles = particles.reshape(-1, D, D)

    # save particles.mrcs
    log(f'Writing image stack to {args.outstack}')
    header = mrc.MRCHeader.make_default_header(particles, Apix=ctf_params[0,1], is_vol=False)
    with open(args.outstack, 'wb') as f:
        header.write(f)
        particles.tofile(f)  # this syntax avoids cryodrgn.mrc.write()'s call to .tobytes() which copies the array in memory

    # save ctf.pkl
    if args.out_pkl:
        log(f'Writing ctf parameters to {args.out_pkl}')
        utils.save_pkl(ctf_params, args.out_pkl)

    if args.out_png:
        log(f'Saving {args.out_png}')
        plot_projections(args.out_png, particles[:9])

    log('Done!')


if __name__ == '__main__':
    main(parse_args().parse_args())
