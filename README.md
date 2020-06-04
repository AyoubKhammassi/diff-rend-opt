# Differentiable Rendering Optimization
This repository is a collection of image processing techniques used in conjunction with the renderer to optimlize differentiable rendering. Our renderer of choice is **Mitsuba 2**, and even though our work is built on top of this renderer, the techniques that we’ll present can be applied to any differentiable renderer and they’re not implementation dependent.

## How to setup Mitsuba2?
1. Clone the project
2. Install [Miniconda 3](https://docs.conda.io/en/latest/miniconda.html)
3. Instal [Cuda 10.2](https://developer.nvidia.com/cuda-downloads)

## How to setup the project for coding?
1. Open an Anaconda Prompt (Miniconda3) (*Just type anaconda in windows search bar*)
2. cd  c:/path/where/you/cloned/the/repo/
3. You need to set up some environment variables before you can you use Mitsuba's Python API, you can do that by executing .\tools\mitsuba2\setPath.bat
4. Run code . to lanuch Visual Studio Code
5. Use those fingers and start typing scripts :p 

**NB :** Make sure to install the Python extension for VS Code, and choose the Minoconda's Python as your interpreter of choice to be used for debugging.


