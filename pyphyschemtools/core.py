############################################################
#          Text Utilities
############################################################
from .visualID_Eng import fg, bg, hl

def centerTitle(content=None):
    '''
    centers and renders as HTML a text in the notebook
    font size = 16px, background color = dark grey, foreground color = white
    '''
    from IPython.display import display, HTML
    display(HTML(f"<div style='text-align:center; font-weight: bold; font-size:16px;background-color: #343132;color: #ffffff'>{content}</div>"))
    
    
def centertxt(content=None,font='sans', size=12,weight="normal",bgc="#000000",fgc="#ffffff"):
    '''
    centers and renders as HTML a text in the notebook
    
    input: 
    - content = the text to render (default: None)
    - font = font family (default: 'sans', values allowed =  'sans-serif' | 'serif' | 'monospace' | 'cursive' | 'fantasy' | ...)
    - size = font size (default: 12)
    - weight = font weight (default: 'normal', values allowed = 'normal' | 'bold' | 'bolder' | 'lighter' | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 )
    - bgc = background color (name or hex code, default = '#ffffff')
    - fgc = foreground color (name or hex code, default = '#000000')
    '''
    from IPython.display import display, HTML
    display(HTML(f"<div style='text-align:center; font-family: {font}; font-weight: {weight}; font-size:{size}px;background-color: {bgc};color: {fgc}'>{content}</div>"))


def smart_trim(img):
    """
    Determines the bounding box of the meaningful content in an image.
    
    This function automatically detects if the image has transparency (Alpha channel).
    If it does, it calculates the bounding box based on non-transparent pixels.
    If the image is opaque, it assumes a white background and calculates the 
    bounding box by detecting differences from a pure white canvas.

    Args:
        img (PIL.Image.Image): The source image object.

    Returns:
        tuple: A 4-tuple (left, upper, right, lower) defining the crop box, 
               or None if the image is uniform/empty.
    """
    import sys
    from pathlib import Path
    from PIL import Image, ImageOps

    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        return img.getbbox()
    else:
        bg = Image.new("RGB", img.size, (255, 255, 255))
        diff = ImageOps.difference(img.convert("RGB"), bg)
        return diff.getbbox()

def crop_images(input_files, process_folder=False):
    """
    Trims whitespace or transparency from image files and saves the results.
    
    If process_folder is True, input_files is treated as a directory path, 
    and all images within (excluding those ending in -C) are processed.
    Otherwise, input_files is treated as a single file path or a list of paths.
    
    The function preserves original image metadata (DPI, ICC profiles).

    Args:
        input_files (str, Path, or list): File path(s) or a directory path.
        process_folder (bool): If True, treats input_files as a directory to crawl.

    Returns:
        None: Prints status messages to the console for each file processed.
    """
    import sys
    from pathlib import Path
    from PIL import Image, ImageOps

    files_to_process = []

    if process_folder:
        folder_path = Path(input_files)
        if folder_path.is_dir():
            # On récupère png, jpg, jpeg (insensible à la casse)
            # On exclut les fichiers finissant déjà par -C
            extensions = ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG')
            for ext in extensions:
                for f in folder_path.glob(ext):
                    if not f.stem.endswith('-C'):
                        files_to_process.append(f)
        else:
            print(f"❌ Error: {input_files} is not a valid directory.")
            return
    else:
        # Logique existante pour fichier unique ou liste
        if isinstance(input_files, (str, Path)):
            files_to_process = [Path(input_files)]
        else:
            files_to_process = [Path(f) for f in input_files]

    for path in files_to_process:
        if not path.exists() or path.is_dir():
            continue

        output_path = path.with_name(f"{path.stem}-C{path.suffix}")

        try:
            with Image.open(path) as img:
                info = img.info.copy()
                bbox = smart_trim(img)
                
                if bbox:
                    img_trimmed = img.crop(bbox)
                    img_trimmed.save(output_path, **info)
                    print(f"✅ Saved: {output_path.name}")
                else:
                    print(f"⚠️ Skipping {path.name}: No content detected.")

        except Exception as e:
            print(f"❌ Error processing {path.name}: {e}")