import ctypes


def apply_gamma_ramp(gamma=1.0, brightness=0, contrast=1.0):
    hdc = ctypes.windll.user32.GetDC(0)

    ramp_type = (ctypes.c_ushort * 256) * 3
    ramp = ramp_type()

    for i in range(256):
        normalized = i / 255.0

        adjusted = normalized * contrast + (brightness / 100.0)
        adjusted = max(0.0, min(1.0, adjusted))

        adjusted = adjusted ** (1.0 / gamma)

        value = int(adjusted * 65535)
        value = max(0, min(65535, value))

        ramp[0][i] = value
        ramp[1][i] = value
        ramp[2][i] = value

    success = ctypes.windll.gdi32.SetDeviceGammaRamp(hdc, ctypes.byref(ramp))
    ctypes.windll.user32.ReleaseDC(0, hdc)

    return bool(success)


def reset_gamma():
    return apply_gamma_ramp(gamma=1.0, brightness=0, contrast=1.0)


def test_gamma():
    print("Testing gamma ramp...")
    print("Applying aggressive setting for 3 seconds.")

    success = apply_gamma_ramp(gamma=1.5, brightness=10, contrast=1.1)
    print("Applied:", success)

    input("Press Enter to reset display...")

    reset_success = reset_gamma()
    print("Reset:", reset_success)


if __name__ == "__main__":
    test_gamma()