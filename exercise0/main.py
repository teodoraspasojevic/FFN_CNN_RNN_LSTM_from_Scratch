from pattern import Checker, Circle, Spectrum
from generator import ImageGenerator


def main():
    checker = Checker(12, 3)
    checker.draw()
    checker.show()

    circle = Circle(1000, 300, (350, 450))
    circle.draw()
    circle.show()

    spectrum = Spectrum(500)
    spectrum.draw()
    spectrum.show()

    file_path = r"./exercise_data"
    label_path = r"./Labels.json"
    gen1 = ImageGenerator(file_path, label_path, 10, (300, 300))
    gen1.next()
    gen1.show()


if __name__ == '__main__':
    main()
