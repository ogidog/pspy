import sys, os
import shapefile
from datetime import date
from scipy.io import loadmat

def mat2shp(*args):
    ts_plot_file = args[1]
    ps_plot_file = args[2]
    outputDir = args[3]

    ts_v_d = loadmat(outputDir + os.path.sep + ts_plot_file)
    v_d = loadmat(outputDir + os.path.sep + ps_plot_file)

    ph_disp = v_d["ph_disp"].flatten()
    ph_mm = ts_v_d["ph_mm"]
    lonlat = ts_v_d["lonlat"]
    days = ts_v_d["day"].flatten()
    ids = ts_v_d["ref_ps"].flatten()

    w = shapefile.Writer(outputDir + os.path.sep + "ts_v-d", shapeType = 1)
    w.field("id", "N")
    w.field("vel", "N", decimal = 10)
    for day in days:
        col_name = "D" + date.fromordinal(day - 366).strftime("%Y%m%d")
        w.field(col_name, "N", decimal = 10)

    for id in ids:
        record_values_list = [id, ph_disp[id - 1]]
        record_values_list.extend(ph_mm[id - 1])
        w.record(*record_values_list)
        w.point(lonlat[id - 1][0], lonlat[id - 1][1])

    w.close()

###############################################################################
if __name__ == "__main__":
    # For testing
    #args = ['', 'ps_plot_ts_v-d.mat', 'ps_plot_v-d.mat', 'C:\\Users\\Ryzen\\Documents\\PYTHON\\stampsexport_tmp']
    args = sys.argv
    mat2shp(*args)