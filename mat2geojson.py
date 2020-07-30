import sys
import shapefile
from datetime import date

from scipy.io import loadmat


def main(args):
    disp = args[1]

    if disp == "v-d":
        ts_v_d = loadmat("ps_plot_ts_" + disp + ".mat")
        v_d = loadmat("ps_plot_" + disp + ".mat")

        ph_disp = v_d["ph_disp"].flatten()
        ph_mm = ts_v_d["ph_mm"]
        lonlat = ts_v_d["lonlat"]
        days = ts_v_d["day"].flatten()
        ids = ts_v_d["ref_ps"].flatten()

        w = shapefile.Writer("ts_v-d", shapeType=1)
        w.field("id", "N")
        w.field("vel", "N", decimal=10)
        for day in days:
            col_name = "D" + date.fromordinal(day - 366).strftime("%Y%m%d")
            w.field(col_name, "N", decimal=10)

        for id in ids:
            record_values_list = [id, ph_disp[id - 1]]
            record_values_list.extend(ph_mm[id - 1])
            w.record(*record_values_list)
            w.point(lonlat[id-1][0],lonlat[id-1][1])

        w.close()



if __name__ == "__main__":
    args = sys.argv
    main(args)
    sys.exit(0)
