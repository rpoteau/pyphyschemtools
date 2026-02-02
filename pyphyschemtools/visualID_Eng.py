import os, sys, platform
import datetime, time
#from IPython.core.display import display,Image,Markdown,HTML
from IPython.display import display,Image,Markdown,HTML
from urllib.request import urlopen

__author__ = "Romuald POTEAU"
__maintainer__ =  "Romuald POTEAU"
__email__ = "romuald.poteau@univ-tlse3.fr"
__status__ = "Development"

# Get the absolute path to this file's directory within the package
_PKG_PATH = os.path.dirname(__file__)

_start_time   = None
_end_time     = None
_chrono_start = None
_chrono_stop  = None

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   OFF = '\033[0m'

class fg:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    LIGHTGRAY = "\033[37m"
    DARKGRAY = "\033[90m"    
    BLACK = '\033[30m'
    WHITE = "\033[38;5;231m"
    OFF = '\033[0m'
class hl:
    BLINK = "\033[5m"
    blink = "\033[25m" #reset blink
    BOLD = '\033[1m'
    bold = "\033[21m" #reset bold
    UNDERL = '\033[4m'
    underl = "\033[24m" #reset underline
    ITALIC = "\033[3m"
    italic = "\033[23m"
    OFF = '\033[0m'
class bg:
    DARKRED = "\033[38;5;231;48;5;52m"
    DARKREDB = "\033[38;5;231;48;5;52;1m"
    LIGHTRED = "\033[48;5;217m"
    LIGHTREDB = "\033[48;5;217;1m"
    LIGHTYELLOW = "\033[48;5;228m"
    LIGHTYELLOWB = "\033[48;5;228;1m"
    LIGHTGREEN = "\033[48;5;156m"
    LIGHTGREENB = "\033[48;5;156;1m"
    LIGHTBLUE = "\033[48;5;117m"
    LIGHTBLUEB = "\033[48;5;117;1m"
    OFF = "\033[0m"


def apply_css_style():
    """
    Explicitly reads and applies the visualID CSS stylesheet 
    from the package resources.
    """
    css_path = os.path.join(_PKG_PATH, "resources", "css", "visualID.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            styles = f.read()
        # Some CSS files might not have the <style> tag if they are raw CSS
        if not styles.strip().startswith("<style>"):
            styles = f"<style>{styles}</style>"
        display(HTML(styles))
    else:
        print(f"[Warning] CSS file not found at {css_path}")

def init(which=None):
    """
    Initializes the notebook environment: applies CSS, 
    displays the banner, and shows hostname/time.
    """
    global _start_time
    _start_time = datetime.datetime.now()

    # 1. Call the explicit CSS function
    apply_css_style()
    
    # 2. Display the banner
    if which == "Research":
        banner = "pyPC_LPCNO_Banner.svg"
    elif which== "t4pPC":
        banner = "tools4pyPC_banner.svg"
    else:
        banner = "pyPhysChemBanner.svg"
    banner_path = os.path.join(_PKG_PATH, "resources", "svg", banner)
    
    if os.path.exists(banner_path):
        with open(banner_path, "r") as f:
            svg_data = f.read()
        display(HTML(f'<div style="text-align: center;">{svg_data}</div>'))
    
    # 3. Environment Info
    now = datetime.datetime.now().strftime("%A %d %B %Y, %H:%M:%S")
    display(Markdown(f"**Environment initialized:** {now} on {platform.node()}"))

def display_md(text):
    display(Markdown(text))
    
def hdelay(sec):
    return str(datetime.timedelta(seconds=int(sec)))    
    
# Return human delay like 01:14:28 543ms
# delay can be timedelta or seconds
def hdelay_ms(delay):
    if type(delay) is not datetime.timedelta:
        delay=datetime.timedelta(seconds=delay)
    sec = delay.total_seconds()
    hh = sec // 3600
    mm = (sec // 60) - (hh * 60)
    ss = sec - hh*3600 - mm*60
    ms = (sec - int(sec))*1000
    return f'{hh:02.0f}:{mm:02.0f}:{ss:02.0f} {ms:03.0f}ms'

def chrono_start():
    global _chrono_start, _chrono_stop
    _chrono_start=time.time()

# return delay in seconds or in humain format
def chrono_stop(hdelay=False):
    global _chrono_start, _chrono_stop
    _chrono_stop = time.time()
    sec = _chrono_stop - _chrono_start
    if hdelay : return hdelay_ms(sec)
    return sec

def chrono_show():
    print('\nDuration : ', hdelay_ms(time.time() - _chrono_start))

def end():
    """
    Terminates the notebook session: displays duration, 
    end time, and the termination logo from package resources.
    """
    global _start_time, _end_time
    
    # 1. Calcul du temps
    _end_time = datetime.datetime.now()
    end_str = _end_time.strftime("%A %d %B %Y, %H:%M:%S")
    
    # Calcul de la durée si _start_time existe
    if _start_time:
        duration = hdelay_ms(_end_time - _start_time)
    else:
        duration = "Unknown (init() was not called)"

    # 2. Affichage des infos de fin
    md = f'**End at:** {end_str}  \n'
    md += f'**Duration:** {duration}'

    # 3. Affichage du logo de fin (depuis le package)
    # On suppose que le logo s'appelle logoEnd.svg et est dans resources/svg/
    logo_path = os.path.join(_PKG_PATH, "resources", "svg", "logoEnd.svg")
    
    if os.path.exists(logo_path):
        with open(logo_path, "r") as f:
            svg_data = f.read()
        display(HTML(f'<div style="text-align: center; width: 800px; margin: auto;">{svg_data}</div>'))
    else:
        # Fallback si le logo est manquant
        print(f"[Warning] End logo not found at {logo_path}")
    display_md(md)

