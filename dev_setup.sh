cd ..
git clone git@github.com:565353780/camera-control.git
git clone --depth 1 https://github.com/camenduru/simple-knn.git
git clone --depth 1 https://github.com/rahul-goel/fused-ssim.git

conda install cmake conda-forge::gmp conda-forge::cgal -y

pip install ninja

pip3 install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124

pip install open3d trimesh scikit-image opencv-python \
  plyfile tqdm

cd camera-control
./dev_setup.sh

cd ../simple-knn
python setup.py install

cd ../fused-ssim
python setup.py install

cd ../milo/submodules/diff-gaussian-rasterization_ms
python setup.py install

cd ../diff-gaussian-rasterization
python setup.py install

cd ../diff-gaussian-rasterization_gof
python setup.py install

cd ../tetra_triangulation
cmake .
make -j
pip install .
