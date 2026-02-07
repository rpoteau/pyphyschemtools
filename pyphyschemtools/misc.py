import os
from PIL import Image
from qrcode import QRCode
from IPython.display import display, Image as IPImage

class QRCodeGenerator:
    """
    A utility class to generate QR codes with embedded logos for pyphyschemtools.
    Initialize the generator with specific QR parameters.

    Args:
        box_size (int): The size of each box in the QR code grid.
        border (int): The thickness of the white border.
        version (int): The complexity of the QR code (1 to 40).
    """
    
    def __init__(self, box_size=40, border=1, version=4):
        self.box_size = box_size
        self.border = border
        self.version = version

    def generate(self, url, logo_path, fill_color="#1a525f", back_color="white",
                 save_path=None, logo_ratio=4, width=250, flatten=False):

        """
        Creates a QR code with a centered logo, saves it to disk, and displays it.

        If save_path is not provided, the image is saved in the logo's directory 
        with a ``qrcode_`` prefix.

        Args:
            url (str): The target URL for the QR code.
            logo_path (str): Path to the logo image file.
            fill_color (str): Hex code or name for the QR code color.
            back_color (str): Background color.
            save_path (str, optional): Custom path to save the PNG. Defaults to None.
            logo_ratio (float): Ratio of logo size relative to QR code size (default 1/4).
            width (int): Display width for Jupyter Notebook rendering.
            flatten (bool): If True, removes transparency by placing the logo on a 
                solid white background. Defaults to False.

        Returns:
            PIL.Image: The generated QR code image object.
            
        """
        # 1. Initialize and build the QR Code
        qr = QRCode(version=self.version, box_size=self.box_size, border=self.border)
        qr.add_data(url)
        qr.make(fit=True)

        # 2. Create base image
        img = qr.make_image(fill_color=fill_color, back_color=back_color).convert("RGBA")

        # 3. Handle Logo embedding and Path generation
        if os.path.exists(logo_path):
            logo = Image.open(logo_path).convert("RGBA")

            # --- Optional: Remove Alpha Channel (Flattening) ---
            if flatten:
                # Create a solid white background matching the logo's current size
                background = Image.new("RGBA", logo.size, (255, 255, 255, 255))
                # Composite the logo over the white background
                logo = Image.alpha_composite(background, logo).convert("RGB")
                
            # Calculate aspect ratio
            original_width, original_height = logo.size
            aspect_ratio = original_width / original_height
            
            # Target size for the larger dimension
            target_max_dim = int(img.size[0] / logo_ratio)
            
            if original_width > original_height:
                # Landscape or wide logo
                new_width = target_max_dim
                new_height = int(target_max_dim / aspect_ratio)
            else:
                # Portrait or tall logo
                new_height = target_max_dim
                new_width = int(target_max_dim * aspect_ratio)
            
            # Resize while maintaining aspect ratio
            logo = logo.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Center position
            img_w, img_h = img.size
            logo_w, logo_h = logo.size
            pos = ((img_w - logo_w) // 2, (img_h - logo_h) // 2)

            # Paste logo using alpha channel as mask
            if logo.mode == 'RGBA':
                img.paste(logo, pos, mask=logo.split()[3])
            else:
                img.paste(logo, pos)

            # Define automatic save path if none provided
            if save_path is None:
                directory = os.path.dirname(logo_path)
                filename = os.path.basename(logo_path)
                save_path = os.path.join(directory, f"qrcode_{filename}")
        else:
            if save_path is None:
                save_path = "qrcode_no_logo.png"
            print(f"Warning: Logo path '{logo_path}' not found. Generating QR without logo.")

        # 4. Save to disk and display in Jupyter
        img.save(save_path)
        display(IPImage(filename=save_path, width=width))

        return img
