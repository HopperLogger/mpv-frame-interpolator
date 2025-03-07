<div align="center">
  <img alt="logo" height="200px" src="https://github.com/HopperLogger/HopperRender/assets/121826818/faa253f0-3276-4404-aa4a-bb9c2b35056c">
</div>

# HopperRender
A fork of mpv-player/mpv, enhanced with real-time frame interpolation using OpenCL. This integration allows for smoother video playback by generating intermediate frames between existing ones, leveraging the GPU for efficient, high-performance processing.
The goal is to achieve pretty decent frame interpolation with a variety of user customizable settings.
The filter can be easily used with [SMPlayer](https://github.com/smplayer-dev/smplayer).
> Please keep in mind that this project is still in ongoing development and there are very likely some bugs depending on the environment you're running and the setting you use. The interpolation quality is also not perfect yet, but pretty decent most of the time, especially for 24 fps -> 60 fps conversion.

## Contribution
Apart from the meson.build script and a few modifications to some mpv source files, the main HopperRender code lives in `mpv/video/filter/HopperRender/`

## Features
- Realtime frame interpolation of any source framerate to your monitors refresh-rate
- Compatible with HDR video
- Comes with an AppIndicator that shows the current stats and allows several output modes, and settings to be selected
- Warps frames in both directions and blends them for the smoothest experience
- HSV Flow visualization lets you see the calculated movements of objects in a scene
- Automatically adjusts internal settings to match the PC's performance _(currently deactivated)_
- Automatically detects the source frame rate (as well as playback speed) and disables interpolation if not needed
- Automatic scene change detection to prevent them from being interpolated _(currently deactivated)_

## How to get started?
This filter uses the OpenCL API and can be compiled on any system that has the OpenCL driver and header files installed.
The compilation is the same as for the [mpv](https://github.com/mpv-player/mpv) base. You can either manually build [mpv](https://github.com/mpv-player/mpv) or use the automated [mpv-build](https://github.com/mpv-player/mpv-build) script and replace the mpv folder with this repo.

### Compilation
1. Make sure you have all the dependencies needed to compile mpv. You can check for missing libraries in the meson build log, or checkout the Troubleshooting section.
2. Install the OpenCL driver for your GPU `opencl-nvidia`, `opencl-amd` or `intel-opencl-runtime`
3. Install the OpenCL headers `opencl-headers`
4. Clone the mpv-build repository `git clone https://github.com/mpv-player/mpv-build.git`
5. Change to the build directory `cd mpv-build`
6. Run the automated build script to do a proper compilation of ffmpeg and mpv `./rebuild -j 8`
7. Clone the HopperRender repository `git clone https://github.com/HopperLogger/mpv-frame-interpolator.git`
8. Delete the original mpv directory ``rm -r -f mpv``
9. Rename the new mpv directory `mv mpv-frame-interpolator mpv`
10. Run the build again, but this time compiling mpv with HopperRender `./build -j 8`
11. Install the final build `sudo ./install`

### Testing and SMPlayer integration
- To use it in your terminal, run: `/usr/local/bin/mpv /path/to/video --vf=HopperRender`
- To use it in [SMPlayer](https://www.smplayer.info/), install SMPlayer via your distro's software manager or download it from the website.
- In the SMPlayer preferences, set the path to the multimedia engine to '/usr/local/bin/mpv'
- Goto Advanced->MPlayer/mpv and enter '--vf=HopperRender' in the options field.

That's it! You can now play a video with SMPlayer and HopperRender will interpolate it to your monitor's native refresh-rate.

## Troubleshooting
- If the compilation fails, it might be helpful to install the normal `mpv` package to ensure its dependencies are installed.
- If the playback doesn't start, the python AppIndicator is most likely the cause. Make sure the `gobject-introspection` package and python library `pygobject` is installed. If it still doesn't work, the AppIndicator can be disabled by modifiyng the flag in `mpv-build/mpv/video/filter/HopperRender/config.h` and then recompiling.
- Make sure not to move, rename, or delete the mpv-build folder and clone it to your home folder, as the filter depends on the AppIndicator Python script and OpenCL kernels to be at a constant path.
- The following are a few libraries recommended for proper mpv and ffmpeg compilation:
```
[PACKAGE-INSTALLER]  build-essential pkg-config \
                     yasm nasm libx264-dev libx265-dev \
                     libfdk-aac-dev libvpx-dev libopus-dev \
                     libnuma-dev libass-dev libfreetype6-dev \
                     libasound2-dev libavcodec-dev libavfilter-dev \
                     libavformat-dev libavutil-dev libdrm-dev \
                     libegl1-mesa-dev libjack-jackd2-dev libjpeg-dev \
                     liblcms2-dev liblua5.2-dev libpulse-dev \
                     librubberband-dev libsdl2-dev libswresample-dev \
                     libswscale-dev libuchardet-dev libva-dev \
                     libvdpau-dev libwayland-dev libx11-dev \
                     libxext-dev libxinerama-dev libxkbcommon-dev \
                     libxrandr-dev libxv-dev zlib1g-dev \
                     libbluray-dev libdvdnav-dev libdvdread-dev \
                     libgbm-dev libjpeg-dev libmp3lame-dev \
                     libopus-dev libplacebo-dev libshaderc-dev \
                     libsmbclient-dev libvulkan-dev libx264-dev \
                     libx265-dev libxcb-shm0-dev libxcb-xfixes0-dev \
                     libxvidcore-dev libzimg-dev spirv-cross
```
- If you're having trouble with getting nvdec hardware decoding to run, try the following:

### Ubuntu/Debian:
```
sudo apt install libffnvcodec-dev
git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
cd nv-codec-headers
sudo make install
``` 
### Arch Linux:
```
sudo pacman -S ffnvcodec-headers
git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
cd nv-codec-headers
sudo make install
```

## Settings
You can access the filter status and settings when playing back a video with HopperRender by right clicking on the HopperRender icon in the panel _(tested on the Cinnamon Desktop Enviornment)_.

- You can activate and deactivate the interpolation
- You can select which type of frame output you want to see:
    - _Warped Frame 1 -> 2: Shows just the warping from the previous to the current frame_
    - _Warped Frame 2 -> 1: Shows just the warping from the current to the previous frame_
    - _Blended Frame: Blends both warp directions together_
    - _HSV Flow: Visualizes the optical flow as a color representation, where the color indicates the direction of movement_

    <div align="center">
    <img alt="color-circle" height="200px" src="https://github.com/HopperLogger/HopperRender/assets/121826818/b025d4ce-cfa2-4702-b184-2c09f4254246">
    </div>

    - _Grey Flow: Visualizes the optical flow as a black and white representation, where the brightness indicates the magnitude of movement_
    - _Blurred Frames: Outputs the blurred source frames_
    - _Side-by-side 1: Shows the difference between no interpolation on the left, and interpolation on the right (split in the middle)_
    - _Side-by-side 2: Shows the difference between no interpolation on the left, and interpolation on the right (scaled down side by side)_
    - _Tearing Test_: Runs a white bar across the screen. Allows you to check for stuttering or tearing.
- You can select a shader that changes the dynamic range of the video
- You can select the calculation resolution used to calculate the optical flow _(this does not affect the output resolution!)_
- In the status section, you can see the current state of HopperRender, the number of calculation steps that are currently performed, the source framerate, the frame and calculation resolutions, as well as much more technical details

## How it works
> Note: The following is a very brief overview of how the filter works. It is not a complete, or 100% accurate description of the interpolation process. Refer to the source code for more details.

- To prevent the algorithm from focusing on pixel level details or compression artifacts, we first (depending on the user setting) blur the frames internally to use for the optical flow calculation
- HopperRender uses an offset array that shifts the frame according to the values contained in it
- The offset array has 5 layers that contain different shifts that can be 'tried out' at the same time to find the best one
- The first step involves setting the 5 layers to a horizontal shift of -2, -1, 0, 1, and 2
- Then, the first frame is shifted accordingly and we subtract the y-channel difference of the shifted frame to the unshifted next frame
- We then reduce all the absolute pixel deltas to one value and find out which layer (i.e. which horizontal shift) contains the lowest value and therefore difference
- Depending on the resulting layer index, we can either move on to the same procedure for the vertical movement, or continue moving in the negative or positive x direction
- We repeat this process until we are certain we found the best offset, or are out of calculation steps
- After having found the best offset for the entire frame, we decrease our window size to a quarter the size and continue the search again for every individual window starting at the previous position
- Depending on the user setting, we do this until we get to the individual pixel level
- Finally, we flip the offset array to give us not just the ideal shift to warp frame 1 to frame 2, but also to warp frame 2 to frame 1
- We then blur both offset arrays depending on the user settings to get a more smooth warp
- Then we use these offset arrays to generate intermediate frames by multiplying the offset values by certain scalars
- We add a bit of artifact removal for the pixels that weren't ideally moved and blend the warped frames from both directions together

## Acknowledgements

This project is a fork of [mpv](https://github.com/mpv-player/mpv).
