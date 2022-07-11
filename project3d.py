'''
|  Generate projections of a 3D volume
|  02/2021: Written by Ellen Zhong, Emily Navarret, and Joey Davis
|  07/2022: Refactored to include tilt series and run faster with less memory usage by Barrett Powell
'''

import argparse
import numpy as np
import os
import time

import torch
import torch.nn.functional as F
import torch.utils.data as data

from cryodrgn import utils
from cryodrgn import mrc
from cryodrgn import lie_tools, so3_grid

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

try:
    from memory_profiler import profile
except:
    pass

log = utils.log
vlog = utils.vlog

def parse_args(parser):
    parser.add_argument('mrc', type=os.path.abspath, help='Input volume')
    parser.add_argument('outstack', type=os.path.abspath, help='Output projection stack (.mrcs)')
    parser.add_argument('outpose', type=os.path.abspath, help='Output poses (.pkl)')

    group = parser.add_argument_group('Required, mutually exclusive, projection schemes')
    group = group.add_mutually_exclusive_group(required=True)
    group.add_argument('--in-pose', type=os.path.abspath, help='Explicitly provide input poses (cryodrgn .pkl format)')
    group.add_argument('--healpy-grid', type=int, help='Resolution level at which to uniformly sample a sphere (equivalent to healpy log_2(NSIDE)')
    group.add_argument('--so3-random', type=int, help='Number of projections to randomly sample from SO3')

    group = parser.add_argument_group('Optional pose sampling arguments')
    group.add_argument('--t-extent', type=float, default=0, help='Extent of image translation in pixels, defining upper bound for random sampling from uniform distribution')
    group.add_argument('--stage-tilt', type=float, help='Right-handed x-axis stage tilt offset in degrees (simulate stage-tilt SPA collection)')
    group.add_argument('--tilt-series', type=os.path.abspath, help='Path to file (.txt) specifying full tilt series x-axis stage-tilt scheme in degrees')

    group = parser.add_argument_group('Optional additional arguments')
    group.add_argument('--is-mask', action='store_true', help='Takes max value along z instead of integrating along z, to create mask images from mask volumes')
    group.add_argument('--out-png', type=os.path.abspath, help='Path to save montage of first 9 projections')
    group.add_argument('-b', type=int, default=100, help='Minibatch size (default: %(default)s)')
    group.add_argument('--seed', type=int, help='Random seed')
    group.add_argument('-v','--verbose',action='store_true',help='Increases verbosity')
    return parser


class Projector:
    def __init__(self, vol, is_mask=False, stage_tilt=None, tilt_series=None, device=None):
        nz, ny, nx = vol.shape
        assert nz==ny==nx, 'Volume must be cubic'
        x2, x1, x0 = np.meshgrid(np.linspace(-1, 1, nz, endpoint=True), 
                             np.linspace(-1, 1, ny, endpoint=True),
                             np.linspace(-1, 1, nx, endpoint=True),
                             indexing='ij')

        lattice = np.stack([x0.ravel(), x1.ravel(), x2.ravel()],1).astype(np.float32)
        self.lattice = torch.from_numpy(lattice)

        self.vol = torch.from_numpy(vol.astype(np.float32))
        self.vol = self.vol.unsqueeze(0)
        self.vol = self.vol.unsqueeze(0)

        self.nz = nz
        self.ny = ny
        self.nx = nx

        # FT is not symmetric around origin
        D = nz
        c = 2/(D-1)*(D/2) -1 
        self.center = torch.tensor([c,c,c]) # pixel coordinate for vol[D/2,D/2,D/2]

        if stage_tilt is not None:
            assert stage_tilt.shape == (3,3)
            stage_tilt = torch.from_numpy(stage_tilt)
        self.tilt = stage_tilt

        if tilt_series:
            tilt_series = np.loadtxt(args.tilt_series, dtype=np.float32)
            tilt_series_matrices = np.zeros((tilt_series.size, 3, 3), dtype=np.float32)

            for i, tilt in enumerate(tilt_series):
                tilt_series_matrices[i] = utils.xrot(tilt)

            self.tilts_matrices = torch.from_numpy(tilt_series_matrices).to(device)
            log(f'Loaded tilt scheme from {args.tilt_series} with {len(tilt_series)} tilts: {tilt_series}')

        self.tilts = tilt_series
        self.is_mask = is_mask

    def rotate(self, rot):
        '''
        rot: B x 3 x 3 rotation matrix
        lattice: D^3 x 3
        tilts_matrices: ntilts x 3 x 3 rotation matrices
        '''
        B = rot.size(0)
        grid = self.lattice @ rot # B x D^3 x 3
        grid = grid.view(-1, self.nz, self.ny, self.nx, 3)
        offset = self.center - grid[:,int(self.nz/2),int(self.ny/2),int(self.nx/2)]
        grid += offset[:,None,None,None,:]
        grid = grid.view(1, -1, self.ny, self.nx, 3)
        vol = F.grid_sample(self.vol, grid, align_corners=False)
        vol = vol.view(B,self.nz,self.ny,self.nx)
        return vol

    def project(self, rot):
        if self.is_mask:
            return self.rotate(rot).max(dim=1)[0]
        else:
            return self.rotate(rot).sum(dim=1)

    def translate(self, image, tran):
        B, D, D = image.shape
        ft = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(image)))  # following standard torch/numpy fft convention
        mag = torch.abs(ft)
        phase = torch.angle(ft)

        freqs_x = torch.linspace(-D/2, D/2, D)
        freqs_x = freqs_x.unsqueeze(0).unsqueeze(0)
        freqs_x = freqs_x.expand(B, D, D)   # increase frequencies horizontally for phase shifting in x
        freqs_y = freqs_x.clone().transpose(-1,-2)  # increase frequencies vertically for phase shifting in y

        freqs_x = freqs_x * tran[:, 0].view(B,1,1)  # positive translation shifts images left
        freqs_y = freqs_y * tran[:, 1].view(B,1,1)  # positive translation shifts images up

        phase_shift = 2 * np.pi * (freqs_x + freqs_y)
        phase = (phase + phase_shift) % (2 * np.pi)

        out = torch.polar(mag, phase)
        out = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(out))).real   # following standard torch/numpy fft convention

        return out

   
class Poses(data.Dataset):
    def __init__(self, rots, trans):
        self.rots = rots
        self.trans = trans
        self.N = rots.shape[0]
        assert self.rots.shape == (self.N,3,3)
        assert self.trans.shape == (self.N,2)
    def __len__(self):
        return self.N
    def __getitem__(self, index):
        return self.rots[index], self.trans[index]


def plot_projections(out_png, imgs):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10,10))
    axes = axes.ravel()
    for i in range(min(len(imgs),9)):
        axes[i].imshow(imgs[i])
    plt.savefig(out_png)


def mkbasedir(out):
    if not os.path.exists(os.path.dirname(out)):
        os.makedirs(os.path.dirname(out))


def warnexists(out):
    if os.path.exists(out):
        log(f'Warning: {out} already exists. Will overwrite at the end of this script. [CTRL]+[C] to cancel.')


def main(args):
    vlog(args)
    for out in (args.outstack, args.out_png, args.outpose):
        if not out: continue
        mkbasedir(out)
        warnexists(out)

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    log(f'Use cuda {use_cuda}')
    if use_cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    t1 = time.time()    
    vol, _ = mrc.parse_mrc(args.mrc)
    D = vol.shape[-1]
    assert D % 2 == 0
    log(f'Loaded {vol.shape} volume')

    if args.stage_tilt:
        theta = args.stage_tilt*np.pi/180
        args.stage_tilt = np.array([[1., 0.,             0.           ],
                                    [0., np.cos(theta), -np.sin(theta)],
                                    [0., np.sin(theta),  np.cos(theta)]]).astype(np.float32)

    # initialize projector
    projector = Projector(vol, args.is_mask, args.stage_tilt, args.tilt_series, device=device)
    if use_cuda:
        projector.lattice = projector.lattice.to(device)
        projector.vol = projector.vol.to(device)

    # generate rotation matrices
    if args.healpy_grid is not None:
        quats = so3_grid.grid_SO3(args.healpy_grid)
        rots = lie_tools.quaternions_to_SO3(torch.from_numpy(quats)).to(device)
        log(f'Generating {rots.shape[0]} rotations at resolution level {args.healpy_grid}')
    elif args.so3_random is not None:
        rots = lie_tools.random_SO3(args.so3_random).to(device)
        log(f'Generating {rots.shape[0]} random rotations')
    elif args.in_pose is not None:
        poses = utils.load_pkl(args.in_pose)
        assert type(poses) == tuple, '--in-pose .pkl file must have both rotations and translations!'
        rots = torch.from_numpy(poses[0].astype(np.float32)).to(device)
        log(f'Generating {rots.shape[0]} rotations from {args.in_pose}')
    else:
        raise RuntimeError

    # expand rotation matrices to account for tilt scheme if specified
    if projector.tilt is not None:
        log('Composing rotations with stage tilt')
        rots = projector.tilt @ rots
    if projector.tilts is not None:
        log('Composing rotations with stage tilt series')
        # expands to `lattice @ tilts_matrices @ rots` rotations
        # .view ordering keeps sequential tilts of same particle adjacent in out.mrcs and outpose.pkl
        rots = (projector.tilts_matrices @ rots.unsqueeze(1)).view(-1, 3, 3)

    # generate translation matrices
    if args.in_pose is not None:
        log('Generating translations from input poses')
        assert args.t_extent == 0, 'Only one of --in-pose and --t-extent can be specified'
        poses = utils.load_pkl(args.in_pose).astype(np.float32)
        assert type(poses) == tuple, '--in-pose .pkl file must have both rotations and translations!'
        trans = poses[1]
        assert trans.max() < 1, 'translations from .pkl file must be expressed in boxsize fraction'
    elif args.t_extent != 0:
        assert args.t_extent > 0, '--t-extent must have a non-negative value'
        assert args.t_extent < projector.nx, '--t-extent cannot be larger than the projection boxsize'
        log(f'Generating translations between +/- {args.t_extent} pixels')
        trans = (np.random.rand(rots.shape[0], 2,) * 2 * args.t_extent - args.t_extent).astype(np.float32)
        trans /= D # convert to boxsize fraction
    else:
        log('No translations specified; will not shift images')
        trans = np.zeros((rots.shape[0], 2), dtype=np.float32)

    # construct poses dataset
    poses = Poses(rots, trans)

    # apply rotations and project 3D to 2D
    log('Processing...')
    pose_iterator = data.DataLoader(poses, batch_size=args.b, shuffle=False)
    out_imgs = np.zeros((len(rots), D, D), dtype=np.float32)
    for i, (rot, tran) in enumerate(pose_iterator):
        vlog(f'Projecting {(i+1) * args.b}/{poses.N}')
        projection = projector.project(rot)
        projection = projector.translate(projection, tran)
        out_imgs[i * args.b: (i+1) * args.b] = projection.cpu().numpy()

    t2 = time.time()
    log(f'Projected {poses.N} images in {t2-t1}s ({(t2-t1) / (poses.N)}s per image)')

    log(f'Saving {args.outstack}')
    mrc.write(args.outstack, out_imgs)

    log(f'Saving {args.outpose}')
    utils.save_pkl((poses.rots, poses.trans), args.outpose)

    if args.out_png:
        log(f'Saving {args.out_png}')
        plot_projections(args.out_png, out_imgs[:9])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parse_args(parser).parse_args()
    utils._verbose = args.verbose
    main(args)
