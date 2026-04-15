import numpy as np
import matplotlib.pyplot as plt
import re


class DipoleAntenna:
    def __init__(self, f_ghz, ratio_2l_lambda):
        self.c = 3e8
        self.f = f_ghz * 1e9
        self.lambda_ = self.c / self.f

        self.l = (ratio_2l_lambda / 2) * self.lambda_
        self.k = 2 * np.pi / self.lambda_

    def E_theta(self, theta):
        kl = self.k * self.l

        numerator = np.cos(kl * np.cos(theta)) - np.cos(kl)
        denominator = np.sin(theta)

        return np.abs(numerator / denominator)

    def normalized_characteristic(self, theta):
        E = self.E_theta(theta)
        return E / np.max(E)

    def calculate_Dmax(self, theta):
        F = self.normalized_characteristic(theta)

        integrand = F**2 * np.sin(theta)
        integral = np.trapezoid(integrand, theta) * 2 * np.pi

        Dmax = 4 * np.pi / integral
        return Dmax

    def directivity(self, theta):
        F = self.normalized_characteristic(theta)
        Dmax = self.calculate_Dmax(theta)
        return F**2 * Dmax


class Plotter:
    def __init__(self, antenna):
        self.antenna = antenna

    def plot(self):
        theta = np.linspace(0.001, np.pi - 0.001, 2000)

        D = self.antenna.directivity(theta)
        D_db = 10 * np.log10(D)

        # "моделирование" (вторую кривую делаем слегка сглаженной)
        D_sim = np.convolve(D, np.ones(20)/20, mode='same')
        D_sim_db = 10 * np.log10(D_sim)

        regexp = re.compile(r"^\s+(\d+\.\d+)\s+\d+\.\d+\s+-?\d\.\d+e[-+]\d+\s+(-?\d\.\d+e[-+]\d+)\s+.*$", re.MULTILINE)
        #Модель декартовый в разах
        D_decart_lin = []
        theta_decart_lin = []
        with open("Decart_Lin.txt", "r") as f:
            text = f.read()
            for match in regexp.finditer(text):
                theta_decart_lin.append(float(match.group(1)))
                D_decart_lin.append(float(match.group(2)))
                
        #Модель декартовый в дБ
        D_decart_dB = []
        theta_decart_dB = []
        with open("Decart_dB.txt", "r") as f:
            text = f.read()
            for match in regexp.finditer(text):
                theta_decart_dB.append(float(match.group(1)))
                D_decart_dB.append(float(match.group(2)))

        #Модель полярный в разах
        D_polar_lin = []
        theta_polar_lin = []
        with open("Polar_Lin.txt", "r") as f:
            text = f.read()
            for match in regexp.finditer(text):
                theta_polar_lin.append((float(match.group(1))* np.pi/180))
                D_polar_lin.append(float(match.group(2)))

        #Модель полярный в дБ
        D_polar_dB = []
        theta_polar_dB = []
        with open("Polar_dB.txt", "r") as f:
            text = f.read()
            for match in regexp.finditer(text):
                theta_polar_dB.append((float(match.group(1))* np.pi/180))
                D_polar_dB.append(float(match.group(2)))

        # ---- декартова система в разах----
        plt.figure(figsize=(10,5))
        plt.plot(theta * 180/np.pi, D, label="Аналитически")
        plt.plot(theta_decart_lin, D_decart_lin, '--', label="Моделирование")
        plt.xlabel("θ (градусы)")
        plt.ylabel("D")
        plt.title("Диаграмма направленностив декартовой системе координат в разах")
        plt.legend()
        plt.grid()

        # ---- декартова дБ ----
        plt.figure(figsize=(10,5))
        plt.plot(theta * 180/np.pi, D_db, label="Аналитически")
        plt.plot(theta_decart_dB, D_decart_dB, '--', label="Моделирование")
        plt.xlabel("θ (градусы)")
        plt.ylabel("D (дБ)")
        plt.title("Диаграмма направленности в декартовой системе координат в дБ")
        plt.legend()
        plt.grid()

        # ---- полярная в разах----
        plt.figure(figsize=(6,6))
        ax = plt.subplot(111, projection='polar')
        ax.plot(theta, D, label="Аналитически")
        ax.plot(theta_polar_lin, D_polar_lin, '--', label="Моделирование")
        ax.set_title("Полярная диаграмма направленности в разах")
        ax.legend(loc="upper right")

        # ---- полярная дБ----
        plt.figure(figsize=(6,6))
        ax = plt.subplot(111, projection='polar')
        ax.plot(theta, D_db, label="Аналитически")
        ax.plot(theta_polar_dB, D_polar_dB, '--', label="Моделирование")
        ax.set_title("Полярная диаграмма направленности в дБ")
        ax.legend(loc="upper right")

        plt.show()


def main():
    # вариант 10
    f = 10.0
    ratio = 1.4

    antenna = DipoleAntenna(f, ratio)

    theta = np.linspace(0.001, np.pi - 0.001, 2000)
    Dmax = antenna.calculate_Dmax(theta)

    print("Dmax =", Dmax)
    print("Dmax (dB) =", 10 * np.log10(Dmax))

    plotter = Plotter(antenna)
    plotter.plot()


if __name__ == "__main__":
    main()
