import os.path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal

KML_TEMPLATE = '<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:atom="http://www.w3.org/2005/Atom" xmlns:gx="http://www.google.com/kml/ext/2.2">' \
               '<Folder>' \
               '<name></name>' \
               '<GroundOverlay>' \
               '<name>velocity</name>' \
               '<Icon>' \
               '<href>%IMG%</href>' \
               '</Icon>' \
               '<altitudeMode>clampToGround</altitudeMode>' \
               '<LatLonBox>' \
               '<north>%NORTH%</north>' \
               '<east>%EAST%</east>' \
               '<south>%SOUTH%</south>' \
               '<west>%WEST%</west>' \
               '</LatLonBox>' \
               '</GroundOverlay>' \
               '<ScreenOverlay>' \
               '<name>colorbar</name>' \
               '<Icon>' \
               '<href>%LEGEND%</href>' \
               '<viewBoundScale>0.75</viewBoundScale>' \
               '</Icon>' \
               '<overlayXY x="0" y="0" xunits="fraction" yunits="fraction"/>' \
               '<screenXY x="0" y="0" xunits="fraction" yunits="fraction"/>' \
               '<size x="0" y="250" xunits="pixel" yunits="pixel"/>' \
               '<rotation>0</rotation>' \
               '<visibility>1</visibility>' \
               '<open>0</open>' \
               '</ScreenOverlay>' \
               '</Folder>' \
               '</kml>'


def main(input_file, output_file):

    colors = np.array([[215, 25, 28], [253, 174, 97], [255, 255, 191], [171, 221, 164], [43, 131, 186]])

    input_ds = gdal.Open(input_file)
    geoTransform = input_ds.GetGeoTransform()
    east = geoTransform[0]
    north = geoTransform[3]
    west = east + geoTransform[1] * input_ds.RasterXSize
    south = north + geoTransform[5] * input_ds.RasterYSize

    input_data = input_ds.GetRasterBand(1).ReadAsArray()

    driver = gdal.GetDriverByName("GTiff")
    (m, n) = np.shape(input_data)
    (min_val, max_val) = [np.nanmin(input_data), np.nanmax(input_data)]
    intervals = np.arange(min_val, max_val, (max_val - min_val) / 5)
    intervals = np.append(intervals, max_val)

    # Create rgba tiff
    rgba_tiff_file = os.path.dirname(output_file) + os.path.sep + "img.tif"
    rgba_tiff_ds = driver.Create(rgba_tiff_file, n, m, 4)
    rgba_tiff_ds.SetGeoTransform(geoTransform)
    rgba_tiff_ds.SetProjection(input_ds.GetProjection())

    rband_data = np.ndarray((m, n), dtype=np.uint16)
    gband_data = np.ndarray((m, n), dtype=np.uint16)
    bband_data = np.ndarray((m, n), dtype=np.uint16)
    alphaband_data = np.ndarray((m, n), dtype=np.uint16)

    # Write rgba channels
    for r in range(m):
        for c in range(n):
            if not np.isnan(input_data[r, c]):
                idx = np.where(intervals >= input_data[r, c])[0][0] - 1
                rband_data[r, c] = colors[idx][0]
                gband_data[r, c] = colors[idx][1]
                bband_data[r, c] = colors[idx][2]
                alphaband_data[r, c] = 255

    rgba_tiff_ds.GetRasterBand(1).WriteArray(rband_data)
    rgba_tiff_ds.GetRasterBand(2).WriteArray(gband_data)
    rgba_tiff_ds.GetRasterBand(3).WriteArray(bband_data)
    rgba_tiff_ds.GetRasterBand(4).WriteArray(alphaband_data)
    rgba_tiff_ds.FlushCache()

    # Convert tiff to png
    rgba_png_file = os.path.dirname(rgba_tiff_file) + os.path.sep + "img.png"
    options_list = [
        '-ot Byte',
        '-of PNG'
    ]
    options_string = " ".join(options_list)
    gdal.Translate(
        rgba_png_file,
        rgba_tiff_file,
        options=options_string
    )

    # Create legend
    labels = ["{} - {}".format(np.round(intervals[i], 2), np.round(intervals[i + 1], 2)) for i in
              range(len(intervals) - 1)]
    marker_color = [tuple(colors[i] / 255) for i in range(len(colors))]

    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    handles = [f("s", marker_color[i]) for i in range(len(marker_color))]
    legend = plt.legend(handles, labels, title="Velocity (mm/year)", loc=3, framealpha=1,
                        frameon=True)
    legend_png_file = os.path.dirname(rgba_png_file) + os.path.sep + "legend.png"
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array([-5, -5, 5, 5])))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(legend_png_file, dpi="figure", bbox_inches=bbox)

    # Create kml file
    template = KML_TEMPLATE.replace("%IMG%", os.path.basename(rgba_png_file))\
        .replace("%LEGEND%", os.path.basename(legend_png_file))\
        .replace("%NORTH%", str(north)) \
        .replace("%SOUTH%", str(south)) \
        .replace("%EAST%", str(east))\
        .replace("%WEST%", str(west))

    kml_file = open(output_file, "w+")
    kml_file.write(template)
    kml_file.close()


if __name__ == "__main__":
    [input_file, output_file] = sys.argv[1:]

    main(input_file, output_file)
