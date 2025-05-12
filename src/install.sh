#  export CUDA_HOME=/usr/local/cuda-11.8/
conda create -n mace python=3.10

conda env config vars set CUDA_HOME="/usr/local/cuda-11.8/"
 # echo $CUDA_HOME
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install diffusers==0.22.0 transformers==4.46.2 huggingface_hub==0.25.2
pip install accelerate openai omegaconf opencv-python


# cd Grounded-Segment-Anything
python -m pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
pip install "numpy<2"
