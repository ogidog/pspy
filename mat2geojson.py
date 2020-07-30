import sys
import shapefile
from datetime import date

from scipy.io import loadmat


def main(args):
    disp = args[1]

    if disp == "v-d":
        ts_v_d = loadmat("ps_plot_ts_" + disp + ".mat")
        v_d = loadmat("ps_plot_" + disp + ".mat")

        ph_disp = v_d["ph_disp"]
        ph_mm = ts_v_d["ph_mm"]
        lonlat = ts_v_d["lonlat"]
        days = ts_v_d["day"].flatten()

        w = shapefile.Writer("ts_v-d", shapeType=1)
        w.field("id", "N")
        w.field("VEL", "N", decimal=10)
        for day in days:
            col_name = "D" + date.fromordinal(day - 366).strftime("%Y%m%d")
            w.field(col_name, "N", decimal=10)

        w.close()

    print(ts_v_d)


if __name__ == "__main__":
    args = sys.argv
    main(args)
    sys.exit(0)
