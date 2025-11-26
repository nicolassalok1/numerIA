$boostRoot="D:/Programmes/Miniconda3/envs/lgbm-gpu/Library"
$boostInc="$boostRoot/include"; $boostLib="$boostRoot/lib"
$nvcc="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/nvcc.exe"

Remove-Item -Recurse -Force .\lightgbm_gpu_build\build -ErrorAction SilentlyContinue

cmake -S lightgbm_gpu_build -B lightgbm_gpu_build\build -G "Visual Studio 17 2022" -A x64 -T "cuda=12.9" `
  -DUSE_GPU=ON -DUSE_CUDA=ON -DUSE_OPENCL=OFF `
  -DBoost_USE_STATIC_LIBS=OFF `
  -DBOOST_ROOT="$boostRoot" -DBOOST_INCLUDEDIR="$boostInc" -DBOOST_LIBRARYDIR="$boostLib" `
  -DCMAKE_CUDA_COMPILER="$nvcc"

cmake --build lightgbm_gpu_build\build --config Release



"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"