## Description

NNEDI3 is an intra-field only deinterlacer. It takes a frame, throws away one field, and then interpolates the missing pixels using only information from the remaining field. It has same rate and double rate modes. NNEDI3 is also very good for enlarging images by powers of 2.

This is [a port of the VapourSynth plugin NNEDI3CL](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-NNEDI3CL).

[NNEDI3CL_rpow2.avsi](https://github.com/Asd-g/AviSynthPlus-NNEDI3CL/blob/main/NNEDI3CL_rpow2.avsi) is provided - a wrapper of NNEDI3CL for enlarging images by powers of 2.

### Requirements:

- AviSynth+ r3688 (can be downloaded from [here](https://gitlab.com/uvz/AviSynthPlus-Builds) until official release is uploaded) or later

- Microsoft VisualC++ Redistributable Package 2022 (can be downloaded from [here](https://github.com/abbodi1406/vcredist/releases))

### Usage:

The file `nnedi3_weights.bin` is required. It must be located in the same folder as NNEDI3CL.

```
NNEDI3CL(clip input, int "field", bool "dh", bool "dw", int[] "planes", int "nsize", int "nns", int "qual", int "etype", int "pscrn", int "device", bool "list_device", bool "info", bool "st", bool "luma")
```

### Parameters:

- input\
    A clip to process.\
    It must be in 8..32-bit planar format.

- field\
    Controls the mode of operation (double vs same rate) and which field is kept.\
    -2: Double rate (alternates each frame), `_FieldBased` frame property order or if `_FieldBased` is `0`, or missing - AviSynth internal order.\
    -1: Same rate, `_FieldBased` frame property order or if `_FieldBased` is `0`, or missing - AviSynth internal order.\
    0: Same rate, keep bottom field.\
    1: Same rate, keep top field.\
    2: Double rate (alternates each frame), starts with bottom.\
    3: Double rate (alternates each frame), starts with top.\
    Default: -1.

- dh\
    Doubles the height of the input.\
    Each line of the input is copied to every other line of the output and the missing lines are interpolated.\
    If field=0, the input is copied to the odd lines of the output.\
    If field=1, the input is copied to the even lines of the output.\
    field must be set to either 0 or 1 when using dh=true.\
    Default: False.

- dw\
    Doubles the width of the input.\
    field must be set to either 0 or 1 when using dw=true.\
    Default: False.

- planes\
    Sets which planes will be processed.\
    Planes that are not processed will contain uninitialized memory.\
    Default: [0, 1, 2, 3].

- nsize\
    Sets the size of the local neighborhood around each pixel (x_diameter x y_diameter) that is used by the predictor neural network.\
    For image enlargement it is recommended to use 0 or 4. Larger y_diameter settings will result in sharper output. For deinterlacing larger x_diameter settings will allow connecting lines of smaller slope.\
    However, what setting to use really depends on the amount of aliasing (lost information) in the source. If the source was heavily low-pass filtered before interlacing then aliasing will be low and a large x_diameter setting wont be needed, and vice versa.\
    0: 8x6\
    1: 16x6\
    2: 32x6\
    3: 48x6\
    4: 8x4\
    5: 16x4\
    6: 32x4\
    Default: 6.

- nns\
    Sets the number of neurons in the predictor neural network.\
    0 is fastest.\
    4 is slowest, but should give the best quality.\
    This is a quality vs speed option; however, differences are usually small. The difference in speed will become larger as `qual` is increased.\
    0: 16\
    1: 32\
    2: 64\
    3: 128\
    4: 256\
    Default: 1.

- qual\
    Controls the number of different neural network predictions that are blended together to compute the final output value.\
    Each neural network was trained on a different set of training data.\
    Blending the results of these different networks improves generalization to unseen data.\
    Possible values are 1 or 2.\
    Essentially this is a quality vs speed option. Larger values will result in more processing time, but should give better results.\
    However, the difference is usually pretty small. I would recommend using `qual>1` for things like single image enlargement.\
    Default: 1.

- etype\
    Controls which set of weights to use in the predictor nn.\
    0: weights trained to minimize absolute error\
    1: weights trained to minimize squared error\
    Default: 0.

- pscrn\
    Controls whether or not the prescreener neural network is used to decide which pixels should be processed by the predictor neural network and which can be handled by simple cubic interpolation.\
    The prescreener is trained to know whether cubic interpolation will be sufficient for a pixel or whether it should be predicted by the predictor nn.\
    The computational complexity of the prescreener nn is much less than that of the predictor nn.\
    Since most pixels can be handled by cubic interpolation, using the prescreener generally results in much faster processing.\
    The prescreener is pretty accurate, so the difference between using it and not using it is almost always unnoticeable.\
    The new prescreener is faster than the old one, and it also causes more pixels to be handled by cubic interpolation.\
    1: old prescreener\
    2: new prescreener (unavailable with float input)\
    Default: 2.

- device\
    Sets target OpenCL device.\
    Use list_device to get the index of the available devices.\
    By default the default device is selected.

- list_device\
    Whether to draw the devices list on the frame.

- info\
    Whether to draw the OpenCL-related info on the frame.

- st\
    Whether to read the data always in single thread mode even if `prefetch()` is used.\
    In some cases using `NNEDI3CL` and `prefetch()` could cause very high cpu usage or crash. In these cases `st=true` could help without forcing `MT_SERIALIZED` mode.\
    Default: Auto determined by device properties.

- luma\
    Whether the format of the output video is Y when only luma plane is processed.\
    It has effect only for YUV clips.\
    Default: False.

### Building:

- Requires `Boost` and `OpenCL`.

- Windows\
    Use solution files.

- Linux
    ```
    Requirements:
        - Git
        - C++17 compiler
        - CMake >= 3.16
    ```
    ```
    git clone https://github.com/Asd-g/AviSynthPlus-NNEDI3CL && \
    cd AviSynthPlus-NNEDI3CL && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j$(nproc) && \
    sudo make install
    ```
