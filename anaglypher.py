"Make anaglyph images from a raster and a DEM."

from argparse import ArgumentParser
from subprocess import call
from sys import stdout
import numpy as np
try:
    import gdal
except ImportError:
    from osgeo import gdal


def hillshade(source, target, zfactor=1, az=315, alt=45):
    """
    Make a hillshade image from a DEM using GDAL binaries.

    Parameters
    ----------
    dem: str
        DEM file to use.
    target: str
        Output file to save hillshade.
    zfactor: float, optional
        Z-multiplier factor for elevation exageration.
    az: float, optional
        Sun azimuth to simulate.
    alt: float, optional
        Sun zenith, practically.
    """
    call(['gdaldem', 'hillshade', source, target,
          '-z', str(zfactor), '-az', str(az), '-alt', str(alt)])


class Anaglypher(object):
    """Class for creating anaglyph 3D images."""

    def __init__(self):
        """Make the anaglypher."""
        self.dem_filename = None
        self.dem_ds = None
        self.dem_rb = None
        self.dem_nodata = None
        self.image_filename = None
        self.image_band = None
        self.img_ds = None
        self.img_rb = None

        self.observer_altitude = None
        self.eye_spacing = None
        self.nadir = None
        self.azimuth_up = None
        self.map_plane_altitude = None
        self.left_array = None
        self.right_array = None
        self.use_lookup = None
        self.no_interpolation = None
        self.nodata_val = None
        pass

    def set_dem(self, dem):
        """Open the DEM file."""
        self.dem_filename = dem
        self.dem_ds = gdal.Open(self.dem_filename, gdal.GA_ReadOnly)
        if self.dem_ds is None:
            stdout.write('ERROR: DEM file <{d}> invalid\n'.format(d=dem))
            return
        self.dem_rb = self.dem_ds.GetRasterBand(1)
        self.dem_nodata = self.dem_rb.GetNoDataValue()

    def set_input_image(self, image, band=1):
        """
        Open the image to anaglyph.

        Parameters
        ----------
        image: str
            Image to use.
        band: int, optional
            Band of the image to use.
        """
        self.image_filename = image
        self.image_band = band
        self.img_ds = gdal.Open(self.image_filename, gdal.GA_ReadOnly)
        if self.dem_ds is None:
            stdout.write('ERROR: image file <{i}> invalid\n'.format(i=image))
            return
        self.img_rb = self.img_ds.GetRasterBand(band)

    def set_observer(self, altitude=None, eye_spacing=None, nadir=0.5):
        """
        Set the properties of the observer.

        Parameters
        ----------
        altitude: float
            Altitude of observer.
        eye_spacing: float
            Distance between observer's eyes.
        nadir: float, optional
            Ratio between left and right eye that is nadir to ground.

        """
        self.observer_altitude = np.float(altitude)
        self.eye_spacing = np.float(eye_spacing)
        if nadir < 0:
            self.nadir = 0
        elif nadir > 1:
            self.nadir = 1
        else:
            self.nadir = nadir

    def set_map_plane_altitude(self, altitude):
        """Set the map plane altitude."""
        self.map_plane_altitude = altitude

    def _shift_from_elevation(self, elevation, left):
        """
        Compute the pixel shift from an elevation.

        Parameters
        ----------
        elevation: float
            Elevation value.
        left: bool
            Whether computing the left eye (True) or right eye (False).

        Returns
        -------
        shift: float
            Horizontal displacement.
        """
        if left:
            diff = elevation - self.map_plane_altitude
            return self.nadir * self.eye_spacing * \
                diff / (self.observer_altitude - elevation)
        else:
            diff = self.map_plane_altitude - elevation
            return (1 - self.nadir) * self.eye_spacing * \
                diff / (self.observer_altitude - elevation)

    def build_lookup_table(self):
        """
        Build a lookup table for pixel shifts from integer elevation values.
        """
        dem = self.dem_rb.ReadAsArray()
        dem_max = int(np.ceil(np.amax(dem)))
        dem_min = int(np.floor(np.amin(dem)))
        elevation_values = range(dem_min, dem_max + 1)
        self.lookup_left = {e: self._shift_from_elevation(e, True) for e in elevation_values}
        self.lookup_right = {e: self._shift_from_elevation(e, False) for e in elevation_values}

    def lookup_shift(self, elevation, left=True):
        """
        Get elevation shift from a precomputed table.

        Parameters
        ----------
        elevation: float
            Elevation to approximate
        left: bool, optional
            Whether left eye is being used (True, default) or right eye.
        """
        if left:
            return self.lookup_left[int(elevation)]
        else:
            return self.lookup_right[int(elevation)]

    def execute(self):
        """Execute the anaglyph computation process."""
        # Get the data as numpy arrays
        img = self.img_rb.ReadAsArray()
        dem = self.dem_rb.ReadAsArray()
        pixel_res = self.dem_ds.GetGeoTransform()[1]
        if not img.shape == dem.shape:
            stdout.write('ERROR: Mismatch in raster shapes\n')
            return
        # Get nodata value
        try:
            self.nodata_val = np.iinfo(img.dtype).max
        except:
            try:
                self.nodata_val = np.finfo(img.dtype).max
            except:
                self.nodata_val = 0

        # Make left and right eye arrays
        self.left_array = np.ndarray(img.shape, dtype=img.dtype)
        self.left_array.fill(self.nodata_val)
        self.right_array = np.ndarray(img.shape, dtype=img.dtype)
        self.right_array.fill(self.nodata_val)

        # if using simple pixel movement without interpolation
        if self.no_interpolation:
            # walk through every pixel in input image
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    pixel_brightness = img[i, j]
                    # get shifts
                    if self.use_lookup:
                        shift_left = self.lookup_shift(dem[i, j], True)
                        shift_right = self.lookup_shift(dem[i, j], False)
                    else:
                        shift_left = self._shift_from_elevation(dem[i, j], True)
                        shift_right = self._shift_from_elevation(dem[i, j], False)
                    # set pixel shift to integer value
                    shift_left_pix = int(np.round(shift_left / pixel_res))
                    shift_right_pix = int(np.round(shift_right / pixel_res))
                    # set result image value if within image bounds
                    if (j+shift_left_pix >= 0) and (j+shift_left_pix < img.shape[1]):
                        self.left_array[i, j+shift_left_pix] = pixel_brightness
                    if (j+shift_right_pix >= 0) and (j+shift_right_pix < img.shape[1]):
                        self.right_array[i, j+shift_right_pix] = pixel_brightness
                stdout.write("\r" + str(i))
                stdout.flush()
            self.execute_bool = True
            return self.left_array, self.right_array

        # if interpolating pixel values (higher quality, longer computations)
        elif not self.no_interpolation:
            # walk through each row
            for i in range(img.shape[0]):
                # make array for each row to store values
                row_left = np.zeros((img.shape[1], 2))
                row_right = np.zeros((img.shape[1], 2))
                # walk through each pixel
                for j in range(img.shape[1]):
                    pixel_brightness = img[i, j]
                    # get shifts
                    if self.use_lookup:
                        shift_left = self.lookup_shift(dem[i, j], True)
                        shift_right = self.lookup_shift(dem[i, j], False)
                    else:
                        shift_left = self._shift_from_elevation(dem[i, j], True)
                        shift_right = self._shift_from_elevation(dem[i, j], False)
                    # get pixel centerpoint
                    shift_left_pix = shift_left / pixel_res
                    shift_right_pix = shift_right / pixel_res
                    # get portion of migrated pixel in each pixel bin
                    remain_left = np.remainder(shift_left_pix, 1)
                    remain_right = np.remainder(shift_right_pix, 1)
                    # get pixel bins to put data into
                    ceil_left = int(np.ceil(shift_left_pix))
                    floor_left = int(np.floor(shift_left_pix))
                    ceil_right = int(np.ceil(shift_right_pix))
                    floor_right = int(np.floor(shift_right_pix))

                    # put the pixel brighness into each pixel bin
                    if dem[i, j] > self.map_plane_altitude:
                        # for each of 4 possible bins, ensure within image bounds
                        # store the pixel brightness weighted by amount in bin
                        if (j+ceil_left >= 0) and (j+ceil_left < img.shape[1]):
                            row_left[j+ceil_left, 0] += pixel_brightness * remain_left
                            row_left[j+ceil_left, 1] += remain_left
                        if (j+floor_left >= 0) and (j+floor_left < img.shape[1]):
                            row_left[j+floor_left, 0] += pixel_brightness * (1 - remain_left)
                            row_left[j+floor_left, 1] += 1 - remain_left
                        if (j+ceil_right >= 0) and (j+ceil_right < img.shape[1]):
                            row_right[j+ceil_right, 0] += pixel_brightness * (1 - remain_right)
                            row_right[j+ceil_right, 1] += 1 - remain_right
                        if (j+floor_right >= 0) and (j+floor_right < img.shape[1]):
                            row_right[j+floor_right, 0] += pixel_brightness * remain_right
                            row_right[j+floor_right, 1] += remain_right
                    else:
                        # for each of 4 possible bins, ensure within image bounds
                        # store the pixel brightness weighted by amount in bin
                        if (j+ceil_left >= 0) and (j+ceil_left < img.shape[1]):
                            row_left[j+ceil_left, 0] += pixel_brightness * (1 - remain_left)
                            row_left[j+ceil_left, 1] += 1-remain_left
                        if (j+floor_left >= 0) and (j+floor_left < img.shape[1]):
                            row_left[j+floor_left, 0] += pixel_brightness * remain_left
                            row_left[j+floor_left, 1] += remain_left
                        if (j+ceil_right >= 0) and (j+ceil_right < img.shape[1]):
                            row_right[j+ceil_right, 0] += pixel_brightness * remain_right
                            row_right[j+ceil_right, 1] += remain_right
                        if (j+floor_right >= 0) and (j+floor_right < img.shape[1]):
                            row_right[j+floor_right, 0] += pixel_brightness * (1 - remain_right)
                            row_right[j+floor_right, 1] += 1 - remain_right
                # compute the weighted average for each pixel bin
                self.left_array[i, :] = row_left[:, 0] / row_left[:, 1]
                self.right_array[i, :] = row_right[:, 0] / row_right[:, 1]
                stdout.write("\r" + str(i))
                stdout.flush()

            self.execute_bool = True
            return self.left_array, self.right_array

    def save_to_gtiff(self, fname, left, right):
        """
        Save the result image to a geotiff.

        Parameters
        ----------
        fname: str
            Filename for the anaglyph image.
        left: array
            Left eye values.
        right: array
            Right eye values.
        """
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(fname, self.img_ds.RasterXSize,
                           self.img_ds.RasterYSize, 3, self.img_rb.DataType)
        ds.SetProjection(self.img_ds.GetProjection())
        ds.SetGeoTransform(self.img_ds.GetGeoTransform())
        # fill out the individual bands
        ds.GetRasterBand(1).SetNoDataValue(self.nodata_val)
        ds.GetRasterBand(2).SetNoDataValue(self.nodata_val)
        ds.GetRasterBand(3).SetNoDataValue(self.nodata_val)
        ds.GetRasterBand(1).WriteArray(left)
        ds.GetRasterBand(2).WriteArray(right)
        ds.GetRasterBand(3).WriteArray(right)
        # destroy the dataset object
        ds = None
        stdout.write(fname + '\n')


if __name__ == '__main__':
    parser = ArgumentParser(description='Make an anaglyph-3D image.')
    parser.add_argument('--dem', '-d', metavar='dem', type=str, required=True,
                        help='digital elevation model (DEM) file')
    parser.add_argument('--input', '-i', metavar='input', type=str,
                        required=True, help='image file to anaglyph-3D')
    parser.add_argument('--output', '-o', metavar='output', type=str,
                        required=True, help='output file to write')
    parser.add_argument('--altitude', '--alt', '-a', metavar='altitude',
                        required=True, type=float, help="observer's altitude")
    parser.add_argument('--separation', '--sep', '-s', metavar='separation',
                        required=True, type=float, help="distance between eyes")
    parser.add_argument('--nadir', '-n', type=float, default=0.5,
                        help='normalized distance between '
                             'left eye (0) and right eye (1) to '
                             'be assumed lying over the pixel; default 0.5')
    parser.add_argument('--plane', '-p', metavar='plane', type=float,
                        required=True, default=0.0,
                        help='altitude where there is no color shifting; '
                             'default 0.0')
    parser.add_argument('--hillshade', '--hill', action='store_true',
                        help='make a stereo hillshade if flag is set')
    parser.add_argument('--lookup', '--lut', action='store_true',
                        help='use a lookup table to improve speed')
    parser.add_argument('--nointerp', action='store_true', help='do not interpolate')
    args = parser.parse_args()
    stdout.write(str(args) + '\n')

    ana = Anaglypher()
    ana.set_observer(altitude=args.altitude, eye_spacing=args.separation,
                     nadir=args.nadir)
    ana.set_map_plane_altitude(args.plane)
    ana.set_dem(args.dem)
    if args.lookup:
        ana.use_lookup = True
        ana.build_lookup_table()
    if args.nointerp:
        ana.no_interpolation = True
    if args.hillshade:
        hillshade(args.dem, args.output + '.hillshade.tif')
        ana.set_input_image(args.output + '.hillshade.tif')
        left, right = ana.execute()
        ana.save_to_gtiff(args.output + '.hillshade.tif', left, right)
    ana.set_input_image(args.input)
    left, right = ana.execute()
    ana.save_to_gtiff(args.output, left, right)
