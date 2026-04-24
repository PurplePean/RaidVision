from monitorcontrol import get_monitors
import time

monitor = get_monitors()[0]

with monitor:
    print("Current brightness:", monitor.get_luminance())
    print("Current contrast:", monitor.get_contrast())

    print("Setting brightness to 80 and contrast to 70")
    monitor.set_luminance(80)
    monitor.set_contrast(70)

    time.sleep(3)

    print("Resetting brightness to 50 and contrast to 50")
    monitor.set_luminance(50)
    monitor.set_contrast(50)