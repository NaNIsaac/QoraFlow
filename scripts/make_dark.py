import numpy as np
import matplotlib.pyplot as plt

from QoraFlow.config import ConfigManager
from QoraFlow.integrator.loader import PalXFELLoader
from QoraFlow.logger import Logger, setup_logger

logger: Logger = setup_logger()
ConfigManager.initialize(r"config.yaml")
config = ConfigManager.load_config()

def main():
    raw_dark_dir = config.path.raw_dark_dir
    image_list = []
    for file in raw_dark_dir.rglob("*.h5"):
        logger.info(f"Load {file}")
        loader = PalXFELLoader(file)
        data = loader.get_data()
        images = data["images"]

        if images.ndim == 2:
            images = images[None, ...]
        image_list.append(images)

    dark_imgs = np.concatenate(image_list, axis=0)
    dark_img = np.mean(dark_imgs, axis=0)

    logger.info(f"Stack shape: {dark_imgs.shape}")
    logger.info(f"Dark image shape: {dark_img.shape}")
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), layout="constrained")

    ax.set_title("Mean Dark Image")
    im = ax.imshow(np.log1p(dark_img))

    cbar = fig.colorbar(im, ax=ax, location='right', aspect=50)
    cbar.set_label('Pixel Value (log scale)')
    plt.show()

    file = config.path.save_dark_dir
    np.save(file, dark_imgs)

    logger.info(f"File saved at {file}")
    
    return

if __name__ == "__main__":
    main()