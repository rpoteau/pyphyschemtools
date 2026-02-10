# Miscellaneous tools

**`misc.py` module**

The `misc` module contains general-purpose helper tools that support the broader ecosystem of `pyphyschemtools`. This includes project branding.

So far, it only contains the `QRcodeGenerator` class

## Branded QR Code Generation

The `QRCodeGenerator` class allows you to create customized QR codes with an embedded logo. This is particularly useful for linking physical lab documents or posters to digital documentation.

Here is how to generate a QR code pointing to the project repository with the official logo centered:

```python
from pyphyschemtools.tools4AS.misc import QRCodeGenerator

gen = QRCodeGenerator()
# This will save to 'images/qrcode_Logo.png' and display it automatically
gen.generate(
    url="https://github.com/rpoteau/pyPhysChem",
    logo_path="images/Logo.png",
    fill_color="#1a525f", #default hex code or name for the QR code color
    back_color="white", #default name of the background color
    logo_ratio=4, #default ratio of logo size relative to QR code size
    flatten=True # removes transparency by placing the logo on a solid white background. Defaults to False.
)