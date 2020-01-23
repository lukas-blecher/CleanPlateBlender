# CleanPlateBlender
 
This Add-on for Blender 2.8x is utilizing Deep Learning methods to remove objects from videos. The technique to do this is based mainly on the following repository: https://github.com/seoungwugoh/opn-demo. 

This branch is for you if you don't have a Nvidia GPU in your system or your if are willing to sacrifice quality for faster computation. The [master branch](https://github.com/lukas-blecher/CleanPlateBlender/tree/master) of this repository uses another method for video inpainting. 

## Usage
Create a mask for the object you want to remove in the `Clip Editor`. This Add-on is compatible with my [AutoMask](https://github.com/lukas-blecher/AutoMask) Blender Add-on that uses Deep Learning to track a mask in a video. 

* Mask: Select the mask you want to remove
* Downscale: A downscaling factor from the original video size. Usefull if you run out of CUDA memory.
* Memorize Every: Memorize every nth frame beforehand.
* File Format: The image format of the output images.
* Output Directory: Where to save the output images.

## Installation
The Add-on is heavily dependent on python libraries. You can either install the dependencies to the Blender python or if python is already installed on the system and the python version is compatible with the Blender python version you can also install the dependencies on your system. In the latter case you need to replace `#r'PYTHON_PATH'` with the path to your python site packages in the `__init__.py` file.
1. Download the repository as `.zip` file
2. Add the `zip` file as Add-on in Blender (**Do not yet activate it since the dependencies are missing**)
3. Go to the Blender Add-on directory and install the python packages
```
pip install -r requirements.txt
```
4. Download the pretrained models and save them into `weights`. Follow these [instructions](https://github.com/lukas-blecher/CleanPlateBlender/blob/onion/weights/info.md).
5. Now you can activate the Add-on in Blender



## License
This software is for non-commercial use only [[1](https://github.com/seoungwugoh/opn-demo#--terms-of-use)].


## Acknowledgements

[Onion-Peel Networks for Deep Video Completion](https://github.com/seoungwugoh/opn-demo)
