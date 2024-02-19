module unload cuda

module load cuda113

conda create --name cocoatorch python=3.7 --no-default-packages --quiet --yes && \
pip install --ignore-installed --use-deprecated=legacy-resolver --no-cache-dir \
  setuptools==57.5.0 && \
conda install -n cocoatorch --quiet --yes  \
  'conda-forge::libgcc-ng=10.3.0' \
  'conda-forge::libstdcxx-ng=10.3.0' \
  'conda-forge::libgfortran-ng=10.3.0' \
  'conda-forge::gxx_linux-64=10.3.0' \
  'conda-forge::gcc_linux-64=10.3.0' \
  'conda-forge::gfortran_linux-64=10.3.0' \
  'conda-forge::openmpi=4.1.1' \
  'conda-forge::sysroot_linux-64=2.17' \
  'conda-forge::git=2.33.1' \
  'conda-forge::git-lfs=3.0.2' \
  'conda-forge::hdf5=1.10.6' \
  'conda-forge::git-lfs=3.0.2' \
  'conda-forge::cmake=3.21.3' \
  'conda-forge::boost=1.76.0' \
  'conda-forge::gsl=2.7' \
  'conda-forge::fftw=3.3.10' \
  'conda-forge::cfitsio=4.0.0' \
  'conda-forge::openblas=0.3.18' \
  'conda-forge::lapack=3.9.0' \
  'conda-forge::armadillo=10.7.3'\
  'conda-forge::expat=2.4.1' \
  'conda-forge::cython=0.29.24' \
  'conda-forge::numpy=1.21.4' \
  'conda-forge::scipy=1.7.2' \
  'conda-forge::pandas=1.3.4' \
  'conda-forge::mpi4py=3.1.2' \
  'conda-forge::matplotlib=3.5.0' \
  'conda-forge::astropy=4.3.1' \
  'conda-forge::freeimage' \
  'conda-forge::libtiff' \
  'conda-forge::libpng=1.6.37' \
  'conda-forge::imageio=2.9.0' \
  'conda-forge::numba=0.48.0' \
  'conda-forge::notebook=6.1.1' \
  'conda-forge::scikit-image=0.16.2' \
  'conda-forge::torchinfo' && \
  conda activate cocoatorch && \
  pip install --ignore-installed --use-deprecated=legacy-resolver --no-cache-dir \
    imutils  \
    pyfits \
    seaborn  \
    emcee \
    iminuit \
    matplotlib \
    jupyterlab \
    h5py \
    tqdm && \
  pip install --ignore-installed --no-cache-dir --use-deprecated=legacy-resolver \
    torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 torchsummary --extra-index-url https://download.pytorch.org/whl/cu113
  