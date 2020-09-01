# CleanPlateBlender
 
This Add-on for Blender 2.8x is utilizing Deep Learning methods to remove objects from videos. The technique to do this is based mainly on the following repository: https://github.com/nbei/Deep-Flow-Guided-Video-Inpainting. I've modified it so it can be used more easily. The changes are available in my [fork](https://github.com/lukas-blecher/Deep-Flow-Guided-Video-Inpainting).

If you don't have a Nvidia GPU in your system or your if are willing to sacrifice quality for faster computation check out the [onion branch](https://github.com/lukas-blecher/CleanPlateBlender/tree/onion) of this repository where I integrated another method for video inpainting into Blender. 

## Usage
Create a mask for the object you want to remove in the `Clip Editor`. This Add-on is compatible with my [AutoMask](https://github.com/lukas-blecher/AutoMask) (WIP) Blender Add-on that uses Deep Learning to track a mask in a video. 

* Mask: Select the mask you want to remove
* Downscale: A downscaling factor from the original video size. Usefull if you run out of CUDA memory.
* Enlarge Mask: If your mask is not covering the object completley this parameter enlarges the mask during computation.
* File Format: The image format of the output images.
* Output Directory: Where to save the output images.
* Threshold: If you are not satisfied with the results try varying this parameter.

## Installation
The Add-on is heavily dependent on python libraries. You can either install the dependencies to the Blender python or if python is already installed on the system and the python version is compatible with the Blender python version you can also install the dependencies on your system. In the latter case you need to replace `#r'PYTHON_PATH'` with the path to your python site packages in the `__init__.py` file.
1. Download the repository as `.zip` file
2. Add the `zip` file as Add-on in Blender (**Do not yet activate it since the dependencies are missing**)
3. Go to the Blender Add-on directory and install the python packages. On Windows `pip` is already shipped with the Blender python. for other operating systems you first need to get `pip` installed.
```
pip install -r requirements.txt
```
1. The cupy installation is also a little bit tricky. If you have a [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) installed, choose the binary version of that Toolkit. Otherwise, you need to install a [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) first. Choose the hightest version compatible with your GPU. 
2. Install PyTorch (>= 1.3.0). Follow the official [instructions](https://pytorch.org/get-started/locally/).
3. Download the pretrained models and save them into `weights`. Follow these [instructions](https://github.com/lukas-blecher/CleanPlateBlender/blob/master/weights/README.md).
4. Now you can activate the Add-on in Blender


## License
The software is for educaitonal and academic research purpose only [[1](https://github.com/JiahuiYu/generative_inpainting#license), [2](https://github.com/twhui/LiteFlowNet/blob/master/LICENSE)].


## Acknowledgements

[Deep-Flow-Guided-Video-Inpainting](https://github.com/nbei/Deep-Flow-Guided-Video-Inpainting), [LiteFlowNet original](https://github.com/twhui/LiteFlowNet), [LiteFlowNet pytorch](https://github.com/sniklaus/pytorch-liteflownet), [DeepFill](https://github.com/JiahuiYu/generative_inpainting), [Softmax splatting](https://github.com/sniklaus/softmax-splatting)