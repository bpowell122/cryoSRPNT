'''
|  Generate projections of a 3D volume
|  02/2021: Written by Ellen Zhong, Emily Navarret, and Joey Davis
|  07/2022: Updated to include tilt series by Barrett Powell
'''

import argparse
import numpy as np
import os
import time
import pickle
from scipy.ndimage.fourier import fourier_shift

import torch
import torch.nn.functional as F
import torch.utils.data as data

from cryodrgn import utils
from cryodrgn import mrc
from cryodrgn import lie_tools, so3_grid

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from memory_profiler import profile

log = utils.log
vlog = utils.vlog

def parse_args(parser):
    parser.add_argument('mrc', help='Input volume')
    parser.add_argument('outstack', type=os.path.abspath, help='Output projection stack (.mrcs)')
    parser.add_argument('outpose', type=os.path.abspath, help='Output poses (.pkl)')

    group = parser.add_argument_group('Required, mutually exclusive, projection schemes')
    group = group.add_mutually_exclusive_group(required=True)
    group.add_argument('--in-pose', type=os.path.abspath, help='Explicitly provide input poses (cryodrgn .pkl format)')
    group.add_argument('--healpy-grid', type=int, help='Resolution level at which to uniformly sample a sphere (equivalent to healpy log_2(NSIDE)')
    group.add_argument('--so3-random', type=int, help='Number of projections to randomly sample from SO3')

    group = parser.add_argument_group('Pose sampling arguments')
    group.add_argument('--t-extent', type=float, default=0, help='Extent of image translation in pixels')
    group.add_argument('--stage-tilt', type=float, help='Right-handed x-axis stage tilt offset in degrees (simulate stage-tilt SPA collection)')
    group.add_argument('--tilt-series', type=os.path.abspath, help='Path to file (.txt) specifying full tilt series x-axis stage-tilt scheme in degrees')

    group = parser.add_argument_group('Additional arguments')
    group.add_argument('--is-mask', action='store_true', help='Takes max value along z instead of integrating along z, to create mask images from mask volumes')
    group.add_argument('--out-png', type=os.path.abspath, help='Path to save montage of first 9 projections')
    group.add_argument('-b', type=int, default=100, help='Minibatch size (default: %(default)s)')
    group.add_argument('--seed', type=int, help='Random seed')
    group.add_argument('-v','--verbose',action='store_true',help='Increases verbosity')
    # group.add_argument('--chunk', type=int, default=1e6, help='Chunksize (in # of images) to split particle stack when saving')
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
        rot: B x 3 x 3
        lattice: D^3 x 3
        tilts_matrices: ntilts x 3 x 3
        '''
        B = rot.size(0)
        if self.tilt is not None:
            rot = self.tilt @ rot
        if self.tilts is not None:
            rot = (self.tilts_matrices @ rot.unsqueeze(1)).view(-1,3,3)
            B = rot.size(0) # batchsize is now args.b * len(self.tilts)
        grid = self.lattice @ rot # B x D^3 x 3
        grid = grid.view(-1, self.nz, self.ny, self.nx, 3)
        offset = self.center - grid[:,int(self.nz/2),int(self.ny/2),int(self.nx/2)]
        grid += offset[:,None,None,None,:]
        grid = grid.view(1, -1, self.ny, self.nx, 3)
        vol = F.grid_sample(self.vol, grid, align_corners=False)
        vol = vol.view(B,self.nz,self.ny,self.nx)
        return vol, rot

    def project(self, rot):
        vols, rots = self.rotate(rot)
        if self.is_mask:
            return vols.max(dim=1)[0], rots
        else:
            return vols.sum(dim=1), rots
        # else:
        #     imgs = torch.empty((rot.shape[0], self.tiltseries_matrices.shape[0], self.ny, self.nx)).to(0)
        #     for i, tilt in enumerate(self.tiltseries_matrices):
        #         assert tilt.shape == (3,3), print(tilt.shape)
        #         imgs[:,i,:,:] = self.rotate(rot @ tilt).sum(dim=1)
        #     return imgs.view(-1, imgs.shape[-1], imgs.shape[-1])

   
class Poses(data.Dataset):
    def __init__(self, pose_pkl, device=None):
        poses = utils.load_pkl(pose_pkl)
        assert type(poses) == tuple, '--in-pose .pkl file must have both rotations and translations!'

        self.rots = torch.from_numpy(poses[0].astype(np.float32)).to(device)
        self.trans = poses[1].astype(np.float32)
        self.N = len(poses[0])
        assert self.rots.shape == (self.N,3,3)
        assert self.trans.shape == (self.N,2)
        assert self.trans.max() < 1
    def __len__(self):
        return self.N
    def __getitem__(self, index):
        return self.rots[index]


class RandomRot(data.Dataset):
    def __init__(self, N, device=None):
        self.N = N
        self.rots = lie_tools.random_SO3(N).to(device)
    def __len__(self):
        return self.N
    def __getitem__(self, index):
        return self.rots[index]


class GridRot(data.Dataset):
    def __init__(self, resol, device=None):
        quats = so3_grid.grid_SO3(resol)
        self.rots = lie_tools.quaternions_to_SO3(torch.from_numpy(quats)).to(device)
        self.N = len(self.rots)
    def __len__(self):
        return self.N
    def __getitem__(self, index):
        return self.rots[index]


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
        log('Warning: {} already exists. Overwriting.'.format(out))


# TODO update to torch.fft (https://github.com/numpy/numpy/issues/13442#issuecomment-489015370)
def translate_img(img, t):
    '''
    img: BxYxX real space image
    t: Bx2 shift in pixels
    '''
    ff = np.fft.fft2(np.fft.fftshift(img))
    ff = fourier_shift(ff, t)
    return np.fft.fftshift(np.fft.ifft2(ff)).real


# @profile
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
        rots = GridRot(args.healpy_grid, device=device)
        log(f'Generating {len(rots)} rotations at resolution level {args.healpy_grid}')
    elif args.in_pose is not None:
        rots = Poses(args.in_pose, device=device)
        log(f'Generating {len(rots)} rotations from {args.in_pose}')
    else:
        rots = RandomRot(args.so3_random, device=device)
        log(f'Generating {len(rots)} random rotations')

    # apply rotations and project 3D to 2D
    log('Rotating and projecting...')
    rot_iterator = data.DataLoader(rots, batch_size=args.b, shuffle=False)
    ntilts = projector.tilts.size if projector.tilts is not None else 1
    out_imgs = np.zeros((len(rots) * ntilts, projector.nx, projector.nx), dtype=np.float32)
    out_rots = np.zeros((len(rots) * ntilts, 3, 3), dtype=np.float32)
    for i, rot in enumerate(rot_iterator):
        vlog(f'Projecting {(i+1) * args.b * ntilts}/{len(rots) * ntilts}')
        projection, rot = projector.project(rot)

        out_imgs[i * args.b * ntilts : (i+1) * args.b * ntilts] = projection.cpu().numpy()
        out_rots[i * args.b * ntilts : (i+1) * args.b * ntilts] = rot.cpu().numpy()

    # if projector.tilt_series:
    #     log('Projecting tiltseries...')
    #     ntilts = projector.tilts.size
    #     imgs = [] #np.empty((iterator.__len__() * ntilts, vol.shape[0], vol.shape[0]))
    #     final_rots = np.empty((rot_iterator.__len__() * ntilts, 3, 3))
    #     for i, rot in enumerate(rot_iterator):
    #         vlog('Projecting {}/{}'.format((i+1)*len(rot), args.N))
    #         for j, tilt in enumerate(projector.tiltseries_matrices):
    #             pose = rot @ tilt.view(1, 3, 3)
    #             assert pose.shape == (1, 3, 3), print(pose.shape)
    #             projections = projector.project(pose)
    #             projections = projections.cpu().numpy()
    #             imgs.append(projections)
    #             final_rots[ntilts*i + j] = pose.view(3, 3).cpu().numpy()
    # else:
    #     imgs = [] #np.empty((iterator.__len__(), vol.shape[0], vol.shape[0]))
    #     for i, rot in enumerate(rot_iterator):
    #         vlog('Projecting {}/{}'.format((i+1)*len(rot), args.N))
    #         projections = projector.project(rot)
    #         projections = projections.cpu().numpy()
    #         imgs.append(projections)
    #     final_rots = rots

    t2 = time.time()
    log(f'Rotated and projected {rots.N * ntilts} images in {t2-t1}s ({(t2-t1) / (rots.N * ntilts)}s per image)')
    # imgs = np.vstack(imgs).astype(np.float32)

    # generate translation matrices
    if args.in_pose is not None:
        assert args.t_extent == 0, 'Only one of --in-pose and --t-extent can be specified'
        log('Generating translations from input poses')
        trans = rots.trans * D # convert from fraction of boxsize to pixels
        trans = -trans[:,::-1] # convention for scipy
    elif args.t_extent != 0:
        assert args.t_extent > 0, '--t-extent must have a non-negative value'
        assert args.t_extent < out_imgs.shape[-1], '--t-extent cannot be larger than the projection boxsize'
        log(f'Generating translations between +/- {args.t_extent} pixels')
        trans = np.random.rand(out_imgs.shape[0], 2) * 2 * args.t_extent - args.t_extent
    else:
        log('No translations specified; will not shift images')
        trans = None

    # apply translations
    if trans is not None:
        log('Translating...')
        for img in range(out_imgs.shape[0]):
            if img % 1000 == 0:
                vlog(f'Translated {img} / {out_imgs.shape[0]}')
            out_imgs[img] = translate_img(out_imgs[img], trans[img])

        # chunksize = args.chunk if args.chunk is not None else out_imgs.shape[0]
        # nchunks = int(np.ceil(out_imgs.shape[0] / chunksize))
        # if nchunks == 1:
        #     out_mrcs = args.o
        # else:
        #     out_mrcs = [f'.{i}'.join(os.path.splitext(args.o)) for i in range(nchunks)]
        #
        # for i in range(nchunks):
        #     log(f'Translating chunk {i} of {nchunks}')
        #     slices = np.arange(i * args.chunk, (i + 1) * args.chunk)
        #     chunk_imgs = out_imgs[i * args.chunk:(i + 1) * args.chunk]
        #     chunk_trans = trans[i * args.chunk:(i + 1) * args.chunk]
        #     chunk_translated = np.asarray([translate_img(img, t) for img,t in zip(chunk_imgs, chunk_trans)], dtype=np.float32)
        #     log(chunk_translated.shape)
        #     log(f'Saving {out_mrcs[i]}')
        #     mrc.write(out_mrcs[i], chunk_translated, is_vol=False)
        #
        # if nchunks > 1:
        #     out_mrcs_basenames = [os.path.basename(x) for x in out_mrcs]
        #     out_txt = f'{os.path.splitext(args.o)[0]}.txt'
        #     log(f'Saving {out_txt}')
        #     with open(out_txt, 'w') as f:
        #         f.write('\n'.join(out_mrcs_basenames))
        # # imgs = np.asarray([translate_img(img, t) for img,t in zip(imgs,trans)], dtype=np.float32)
        # # convention: we want the first column to be x shift and second column to be y shift
        # # reverse columns since current implementation of translate_img uses scipy's
        # # fourier_shift, which is flipped the other way
        # # convention: save the translation that centers the image

        trans = -trans[:,::-1]  # undo scipy convention
        trans /= D  # convert translation from pixel to fraction

        t3 = time.time()
        log(f'Translated {out_imgs.shape[0]} images in {t3-t2}s ({(t3-t2) / (out_imgs.shape[0])}s per image)')

    log(f'Saving {args.outstack}')
    mrc.write(args.outstack, out_imgs)

    # log('Saving {}'.format(args.o))
    # mrc.write(args.o,imgs.astype(np.float32))
    log(f'Saving {args.outpose}')
    utils.save_pkl((out_rots, trans), args.outpose)

    if args.out_png:
        log(f'Saving {args.out_png}')
        plot_projections(args.out_png, out_imgs[:9])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parse_args(parser).parse_args()
    utils._verbose = args.verbose
    main(args)
