import numpy as np

def unwise_to_rgb(
    imgs, scale1=1.0, scale2=1.0, arcsinh=1.0 / 20.0, mn=-20.0, mx=10000.0, w1weight=9.0
):
    img = imgs[0]
    H, W = img.shape
    w1, w2 = imgs
    rgb = np.zeros((H, W, 3), np.uint8)
    img1 = w1 / scale1
    img2 = w2 / scale2

    if arcsinh is not None:

        def nlmap(x):
            return np.arcsinh(x * arcsinh) / np.sqrt(arcsinh)

        # intensity -- weight W1 more
        bright = (w1weight * img1 + img2) / (w1weight + 1.0)
        I = nlmap(bright)

        # color -- abs here prevents weird effects when, eg, W1>0 and W2<0.
        mean = np.maximum(1e-6, (np.abs(img1) + np.abs(img2)) / 2.0)
        img1 = np.abs(img1) / mean * I
        img2 = np.abs(img2) / mean * I

        mn = nlmap(mn)
        mx = nlmap(mx)

    img1 = (img1 - mn) / (mx - mn)
    img2 = (img2 - mn) / (mx - mn)

    rgb[:, :, 2] = (np.clip(img1, 0.0, 1.0) * 255).astype(np.uint8)
    rgb[:, :, 0] = (np.clip(img2, 0.0, 1.0) * 255).astype(np.uint8)
    rgb[:, :, 1] = rgb[:, :, 0] / 2 + rgb[:, :, 2] / 2

    return rgb
    
def flux_to_rgb(imgs, bands, scales=None, m=0.02, Q=20, alpha=1.0, p=1.0):

    # default value for SDSS
    rgbscales = {
        "u": (2, 1.5),  # 1.0,
        "g": (2, 2.5),
        "r": (1, 1.5),
        "i": (0, 1.0),
        "z": (0, 0.4),  # 0.3
    }
    if scales is not None:
        rgbscales.update(scales)

    I = 0
    for img, band in zip(imgs, bands):
        plane, scale = rgbscales[band]
        img = np.maximum(0, img * scale + m)
        I = I + img
    I /= len(bands)

    fI = np.arcsinh(alpha * Q * I) / np.sqrt(Q ** p)
    I += (I == 0.0) * 1e-6
    H, W = I.shape
    rgb = np.zeros((H, W, 3), np.float32)
    for img, band in zip(imgs, bands):
        plane, scale = rgbscales[band]
        rgb[:, :, plane] = (img * scale + m) * fI / I

    rgb = np.clip(rgb, 0, 1)

    return rgb

def make_rgb(img, survey):
    if survey not in ["ls_grz", "unwise_w1w2", "sdss_gri"]:
        raise ValueError("Survey not supported")
    if survey == "ls_grz":
        rgb = flux_to_rgb(
            img, bands="grz", scales=dict(g=(2, 6.0), r=(1, 3.4), z=(0, 2.2)), m=0.03
        )
    elif survey == "unwise_w1w2":
        rgb = unwise_to_rgb(
            img,
            scale1=1.0,
            scale2=1.0,
            arcsinh=1.0 / 20.0,
            mn=-20.0,
            mx=10000.0,
            w1weight=9.0,
        )
    elif survey == "sdss_gri":
        rgb = flux_to_rgb(
            img,
            bands="gri",
            scales=dict(
                u=(2, 1.5),
                g=(2, 2.8),
                r=(1, 1.4),
                i=(0, 1.1),
                z=(0, 0.4),
            ),
            m=0.0,
            Q=20,
            alpha=0.8,
            p=0.72,
        )
    return rgb
