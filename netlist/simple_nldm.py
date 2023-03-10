# stores several LUTs from a simple cell library
# for demonstration of simultaneously solving net and cell
# delays.

class LUT:
    def __init__(self, xs, ys, vs):
        self.xs = xs
        self.ys = ys
        self.vs = vs
        assert len(self.vs) == len(self.xs) * len(self.ys)

    def query(self, x, y):
        raise NotImplementedError

inv_x1_az_cell_rise = LUT(
    xs=[0.00,1.00,2.00,4.00,8.00,16.00,32.00],
    ys=[5.00,30.00,50.00,80.00,140.00,200.00,300.00,500.00],
    vs=[
        9.376, 14.576, 18.136, 22.088, 27.856, 32.352, 38.568, 48.992,
        13.544, 18.744, 22.88, 27.96, 35.32, 40.944, 48.52, 60.664,
        17.704, 22.904, 27.064, 32.992, 41.784, 48.456, 57.336, 71.2,
        26.04, 31.24, 35.4, 41.64, 52.84, 61.408, 72.68, 89.872,
        42.704, 47.904, 52.064, 58.304, 70.784, 82.472, 97.92, 121.136,
        76.04, 81.24, 85.4, 91.64, 104.12, 116.6, 137.272, 170.648,
        142.704, 147.904, 152.064, 158.304, 170.784, 183.264, 204.064, 245.664
    ]
)

inv_x1_az_rise_trans = LUT(
    xs=[0.00,1.00,2.00,4.00,8.00,16.00,32.00],
    ys=[5.00,30.00,50.00,80.00,140.00,200.00,300.00,500.00],
    vs=[
        10, 10.976, 13.104, 16.08, 20.136, 22.92, 26.36, 31.864,
        15, 15.36, 16.92, 20.224, 25.72, 29.648, 34.384, 41.048,
        20, 20.072, 21.128, 23.928, 30.376, 35.272, 41.328, 49.488,
        30, 30, 30.256, 32.08, 38.128, 44.616, 52.912, 64.456,
        50, 50, 50, 50.32, 54.008, 59.808, 71.024, 88.184,
        90, 90, 90, 90, 90.448, 93.336, 101.536, 123.584,
        170, 170, 170, 170, 170, 170, 172.12, 185.672
    ]
)

inv_x1_cap_a = 1.00
