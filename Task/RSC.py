import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.special import spherical_jn, spherical_yn


class RCSSphere:
    def __init__(self, diameter):
        self.r = diameter / 2
        self.c = 3e8

    def rcs(self, freq):
        lam = self.c / freq
        k = 2 * np.pi / lam
        x = k * self.r

        N = 50
        s = 0

        for n in range(1, N):
            jn = spherical_jn(n, x)
            jn_1 = spherical_jn(n - 1, x)

            yn = spherical_yn(n, x)
            yn_1 = spherical_yn(n - 1, x)

            hn = jn + 1j * yn
            hn_1 = jn_1 + 1j * yn_1

            an = jn / hn
            bn = (x * jn_1 - n * jn) / (x * hn_1 - n * hn)

            s += (-1)**n * (n + 0.5) * (bn - an)

        sigma = (lam**2 / np.pi) * abs(s)**2
        return sigma


class JSONWriter:
    def write(self, filename, freq, lam, rcs):

        with open(filename, "w") as f:
            f.write('{\n')
            f.write('    "data": [\n')

            for i in range(len(freq)):
                line = (
                    f'        {{"freq": {freq[i]}, '
                    f'"lambda": {lam[i]}, '
                    f'"rcs": {rcs[i]}}}'
                )

                if i != len(freq) - 1:
                    line += ','

                f.write(line + '\n')

            f.write('    ]\n')
            f.write('}\n')


class PlotResult:
    def __init__(self, filename):
        self.filename = filename

    def run(self):

        with open(self.filename) as f:
            params = json.load(f)["data"]["10"]

        D = float(params["D"])
        fmin = float(params["fmin"])
        fmax = float(params["fmax"])

        sphere = RCSSphere(D)

        freq = np.linspace(fmin, fmax, 1000)
        lam = 3e8 / freq
        rcs = np.array([sphere.rcs(f) for f in freq])

        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot()
        plt.plot(freq/1e9, rcs)
        plt.xlabel("Частота, ГГц")
        plt.ylabel("ЭПР, м²")
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.5f'))
        plt.title("ЭПР идеально проводящей сферы")
        plt.grid()
        plt.show()

class TextResult:
    def __init__(self, filename):
        self.filename = filename

    def run(self):

        with open(self.filename) as f:
            params = json.load(f)["data"]["10"]

        D = float(params["D"])
        fmin = float(params["fmin"])
        fmax = float(params["fmax"])

        sphere = RCSSphere(D)

        freq = np.linspace(fmin, fmax, 1000)
        lam = 3e8 / freq
        rcs = np.array([sphere.rcs(f) for f in freq])

        writer = JSONWriter()
        writer.write("result.json", freq, lam, rcs)

if __name__ == "__main__":
    PlotResult("task_rcs_02.json").run()
    TextResult("task_rcs_02.json").run()
