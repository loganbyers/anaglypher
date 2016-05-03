# anaglypher
Python script for making anaglyph-3D maps from a DEM.

## Description
This program takes an image and a digital elevation model (DEM) that are
co-registered and creates a red-cyan anaglyph image by simulating stereo
imagery using the elevation model. The resulting anaglyph image retains the
geolocation, extent, and bit-depth of the input image, so it remains suitable for
use in GIS programs.

### Preprocessing
The image and DEM must be co-registered, so that they have the same dimensions.
Adding such a preprocessing step is a planned addition in the future.

## Requirements
* [gdal with python bindings](http://www.gdal.org/)
* [numpy](http://www.numpy.org/)

## Usage
The software is usable as a script from the command line, which presents the simplest interface.

### Options
* `--dem, -d`: elevation dataset filename
* `--input, -i`: image to anaglyph
* `--output, -o`: anaglyph image to save
* `--altitude`: elevation/altitude of the observer
* `--separation, --sep, -s`: eye-separation distance of the observer
* `--nadir, -n`: ratio between the left and right eye that points nadir to the ground
* `--plane, -p`: elevation/altitude to place the image-plane
* `--hillshade, -hill`: make a hillshade of the DEM and process it into an anaglyph
* `--lookup, --lut`: speed up the processing by pre-computing shift values using integer approximations of elevation values
* `--nointerp`: speed up the processing by not interpolating pixel brightness, but using whole-pixel shifts

### Examples
**Process a satellite image**
```bash
$ python anaglypher.py --dem elevation.tif --input image.tif --output output.tif \
  --plane 0.0 --altitude 3000 --separation 100
```

**Process a satellite image and make an anaglyph hillshade**
```bash
$ python anaglypher.py --dem elevation.tif --input image.tif --output output.tif \
  --plane 0.0 --altitude 3000 --separation 100 --hillshade
```

**Make image pop-out with low image plane**
```bash
$ python anaglypher.py --dem elevation.tif --input image.tif --output output.tif \
  --plane -2000 --altitude 3000 --separation 100
```

**Make image recessed with high image plane**
```bash
$ python anaglypher.py --dem elevation.tif --input image.tif --output output.tif \
  --plane 2000 --altitude 3000 --separation 100
```

**Use a lookup table to speed computations**
```bash
$ python anaglypher.py --dem elevation.tif --input image.tif --output output.tif \
  --plane 0.0 --altitude 3000 --separation 100 --lookup
```

**Don't use interpolation to speed computations**
```bash
$ python anaglypher.py --dem elevation.tif --input image.tif --output output.tif \
  --plane 0.0 --altitude 3000 --separation 100 --nointerp
```

### Parameterization
Playing with the parameterization will affect the resulting anaglyph map.
The ratio between human height and pupillary distance is around 30.
Decreasing this ratio will simulate higher vertical exaggeration.

As shown above, setting the image plane below the lowest elevation will cause
the anaglyph image to pop-up.
Setting the image plane above the highest elevation will create an anaglyph
image that is recessed.
