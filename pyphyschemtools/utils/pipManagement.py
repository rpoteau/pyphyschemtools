#!/usr/bin/env python3
# coding: utf-8
import os
import sys
from datetime import datetime, timezone
import subprocess
from pathlib import Path
import string
import platform

pMversion = "20260304"

################################################################################################

String = "\n\033[91m\033[1mWhich command?\033[0m\n\
\033[1m# Output installed packages in requirements format\033[0m\n\
1. pip freeze > requirements.txt\n\
\033[1m# Output installed packages in requirements format, but with no version number\033[0m\n\
101. pip freeze | sed 's/==.*$/''/' > requirements.txt\n\
\033[1m# verify installed packages have compatible dependencies\033[0m\n\
2. pip check\n\
\033[0m# List installed packages, including editables; possibility to list only packages that are not dependencies of installed packages\033[0m\n\
\033[32m3. pip list\033[0m\n\
4. pip list --not-required\n\
\033[1m# display the installed python packages in form of a dependency tree\033[0m\n\
5. pipdeptree\n\
\033[32m55. pipdeptree --reverse --packages package-name\033[0m\n\
\033[1m# pip-review is a convenience wrapper around pip. It can list available updates by deferring to pip list --outdated. It can also automatically or interactively install available updates for you by deferring to pip install\033[0m\n\
6. pip list --outdated\n\
7. pip list --outdated --not-required\n\
8. pip-review\n\
\033[1m# pip-check gives a quick overview of all installed packages and their update status. Under the hood it calls ``pip list --outdated --format=columns``\033[0m\n\
9. pip-check --hide-unchange\n\
\033[32m10. pip-check --hide-unchanged --show-updated\033[0m\n\
\033[32m11. pip-check --hide-unchanged --not-required --show-update\033[0m\n\
\033[1m# install a new package\033[0m\n\
\033[32m15. pip install package-name\033[0m\n\
\033[1m# uninstall a package\033[0m\n\
\033[32m16. pip uninstall package-name\033[0m\n\
\033[1m# upgrade packages\033[0m\n\
\033[32m20. pip install -r requirements.txt --upgrade\033[0m\n\
21. pip install --upgrade package_name(s)\n\
22. pip-review --local --interactive\n\
\n\
\033[1m# restore an environment\033[0m\n\
\033[32m23. pip uninstall $(pip freeze); pip install -r requirements.txt\033[0m\n\
\033[1m# list current environments\033[0m\n\
40. virtualenv New_Env_Name; activate New_Env_Name; pip install -r requirements.txt \n\
\033[1m# Make a new empty virtual environment\033[0m\n\
41. virtualenv New_Env_Name \n\
\n\
\033[1m# list pip cache contents\033[0m\n\
50. pip cache list \n\
\033[1m# purge the pip cache\033[0m\n\
51. pip cache purge \n\
\n\
\033[1m# Delete a virtual environment\033[0m\n\
\033[91m\033[1mD. rm -Rf Env_Name\033[0m\n\
\n\
\033[91m\033[1mx. Exit\033[0m"


def run(python_folder=None):

    if len(sys.argv) > 1 and sys.argv[1].lower() in ["-v", "--version"]:
        print(f"pipManagement version {pMversion}")
        return # Exit the function immediately

    folder = python_folder or os.getenv("PIP_MGMT_FOLDER", "Python3")
    p_home = Path.home() / folder

    print(os.getenv("PIP_MGMT_FOLDER"))
    
    if not p_home.exists():
        print(f"\033[91mDirectory {p_home} not found.\033[0m")
        print(f"Check your PIP_MGMT_FOLDER variable or ensure {folder} exists in your Home directory.")
        # You could even add an input here to ask the user to fix it!
        sys.exit()

    current_os = platform.system()
    print(f"pipManagement run on an {current_os} system")
    if current_os == "Windows":
        bin_folder = "Scripts"
        def get_act(p, e): return f"'{p}/{e}/Scripts/activate' && "
    else:
        # Mac (Darwin) and Linux use 'bin'
        bin_folder = "bin"
        def get_act(p, e): return f"source '{p}/{e}/bin/activate';\n"

    choice = ""
    alphabet = string.ascii_lowercase

    print("-------------------------------------------------------------------------------------------")
    print(f"\033[34;1mpipManagement v.{pMversion}\033[0m")
    print()
    print(f"Detected Python Home (PIP_MGMT_FOLDER variable): {p_home}")
    print("-------------------------------------------------------------------------------------------")
    
    while choice != "x":
        # 1. Automatic scanning at each turn to stay up to date
        detected_venvs = []
        if p_home.exists() and p_home.is_dir():
            for folder in p_home.iterdir():
                if folder.is_dir() and (folder / bin_folder / "activate").exists():
                    detected_venvs.append(folder.name)
        else:
            # This is the "Else" part: what happens if the folder is missing
            print(f"\033[91mError: The directory {p_home} does not exist.\033[0m")
            sys.exit()
        
        # Sort lowercase first
        detected_venvs.sort(key=str.lower)
        
        # Map to alphabet
        venv = {alphabet[i]: name for i, name in enumerate(detected_venvs) if i < len(alphabet)}
        
        # 2. Display the detected environments clearly
        listenv = ". ".join([f"{k}: {v}" for k, v in venv.items()])
        print("\n" + "="*40)
        print(f"DETECTED ENVIRONMENTS in {p_home}:")
        for key, name in venv.items():
            print(f"  \033[1m{key}\033[0m: {name}")
        print("="*40)

        print(f'{String}')
        print()
        now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    
        envNN = ""
        env = ""
        choice = input("enter a number (or x to terminate): ")
        if choice == "x":
            sys.exit()
        while envNN not in venv and choice not in ["30","40","41","50","51"]:
            envNN = input(f"Which virtualenv environment? {listenv}\nChoice: ")

            if envNN == "x":
                break  # Exit the environment selection loop
            if envNN in venv:
                env = venv[envNN]
            else:
                print(f"\033[31mSelect a letter from the list above or enter x to exit this menu.\033[0m")

        if envNN == "x":
            continue
    
        pyAct = get_act(p_home, env) if env else ""
        
        pipCom = ""
        if choice == "1":
            pipCom = f"{pyAct}cd '{p_home}';\npip freeze > '{now}requirements_{env}.txt'"
    
        if choice == "101":
            pipCom = f"{pyAct}cd '{p_home}';\npip freeze | sed 's/==.*$/''/' > '{now}requirements_{env}_noVersionNumber.txt'"
    
        if choice == "2":
            pipCom = f"{pyAct}pip check"
    
        if choice == "3":
            pipCom = f"{pyAct}pip list"
    
        if choice == "4":
            pipCom = f"{pyAct}pip list --not-required"
    
        if choice == "5":
            pipCom = f"{pyAct}pipdeptree"
    
        if choice == "55":
            package_name = input("copy/paste the name of the package that other packages depend on: ")
            pipCom = f"{pyAct}pipdeptree --reverse --packages {package_name}"
    
        if choice == "6":
            pipCom = f"{pyAct}pip list --outdated"
    
        if choice == "7":
            pipCom = f"{pyAct}pip list --outdated --not-required"
    
        if choice == "8":
            pipCom = f"{pyAct}pip-review"
    
        if choice == "9":
            pipCom = f"{pyAct}pip-check --hide-unchanged"
    
        if choice == "10":
            pipCom = f"{pyAct}pip-check --hide-unchanged --show-update"
    
        if choice == "11":
            pipCom = f"{pyAct}pip-check --hide-unchanged --not-required --show-update"
    
        if choice == "15":
            package_names = input("copy/paste the name of the package(s) you want to install (x = cancel operation): ")
            if package_names != "x":
                reqBefore = f"{now}requirements_{env}_BeforeNewInstall.txt"
                pipCom = f"{pyAct}pip freeze > '{p_home}/{now}requirements_{env}_BeforeNewInstall.txt';\n"
                pipCom += f"pip install {package_names} ;\n"
                pipCom += f"pip freeze > '{p_home}/{now}requirements_{env}_AfterNewInstall.txt'"
            else:
                pipCom = "echo operation cancelled"
    
        if choice == "16":
            package_names = input("copy/paste the name of the package(s) you want to uninstall (x = cancel operation): ")
            if package_names != "x":
                pipCom = f"{pyAct}pip freeze > '{p_home}/{now}requirements_{env}_BeforeUninstall.txt';\n"
                pipCom += f"pip uninstall {package_names} -y ;\n"
                pipCom += f"pip freeze > '{p_home}/{now}requirements_{env}_AfterUninstall.txt'"
            else:
                pipCom = "echo operation cancelled"
    
        if choice == "20":
            print(f"Checking in {p_home}...")
            systemCom = f"cd '{p_home}';\nls -lrt *req*.txt | grep {env}"
            subprocess.run(systemCom, shell=True)
            reqtxt = input("copy/paste the reference requirements/txt file: ")
            reqtxtTmp = reqtxt + "_tmp"
            pipCom = f"{pyAct}pip freeze > '{p_home}/{now}requirements_{env}_BeforeUpdate.txt';\n"
            os.chdir(p_home)
            print (os.getcwd())
            finp = open(reqtxt, "rt")
            fout = open(reqtxtTmp, "wt")
            for line in finp:
            	fout.write(line.replace('==', '>='))
            print(f"{reqtxtTmp} created after {reqtxt} with '==' replaced with '>='")
            finp.close()
            fout.close()
            pipCom += f"pip install -r '{p_home}/{reqtxtTmp}' --upgrade;\n"
            pipCom += f"pip freeze > '{p_home}/{now}requirements_{env}_AfterUpdate.txt'"
    
        if choice == "21":
            package_names = input("copy/paste the name of the package(s) you want to update (x = cancel operation): ")
            if package_names != "x":
                reqBefore = f"{now}requirements_{env}_BeforeNewInstall.txt"
                pipCom = f"{pyAct}pip freeze > '{p_home}/{now}requirements_{env}_BeforeUpdate.txt';\n"
                pipCom += f"pip install --upgrade {package_names} ;\n"
                pipCom += f"pip freeze > '{p_home}/{now}requirements_{env}_AfterUpdate.txt'"
            else:
                pipCom = "echo operation cancelled"
    
        if choice == "22":
            pipCom = f"{pyAct}pip freeze > '{p_home}/{now}requirements_{env}_BeforeUpdate.txt';\n"
            pipCom += f"pip-review --local --interactive;\n"
            pipCom += f"pip freeze > '{p_home}/{now}requirements_{env}_AfterUpdate.txt'"
    
        if choice == "23":
            systemCom = "cd " + p_home + ";\n ls -lrt *req*.txt |grep " + env
            subprocess.run(systemCom, shell=True)
            print(f"\nFiles found in {p_home} (sorted by date):")
            reqtxt = input("copy/paste the reference requirements/txt file you want to restore: ")
            pipCom = f"{pyAct}pip freeze > '{p_home}/{now}requirements_{env}_BeforeRestoration.txt';\n"
            pipCom += f"pip uninstall $(pip freeze) -y;\n"
            pipCom += f"pip install -r '{p_home}/{reqtxt}';\n"
            pipCom += f"pip freeze > '{p_home}/{now}requirements_{env}_AfterRestoration.txt'"
    
        if choice == "30":
            pipCom = f"cd '{p_home}';\nls -d */"
            print()
            print(f"virtualenv defined in the present script: {listenv}")
            print()
    
        if choice == "40":
            NewEnvTxt = input("Name of the new environment: ")
            systemCom = "cd " + p_home + ";\nvirtualenv " + NewEnvTxt + ";\n ls -lrt *req*.txt"
            subprocess.run(systemCom, shell=True)
            ReqTxt = input(f"\033[33mcopy/paste from the previous list the reference requirements.txt file, i.e. the environment you want to clone: \033[0m")
            pyAct = f"source '{p_home}/{NewEnvTxt}/bin/activate';\n"
            pipCom = f"{pyAct} pip install -r '{p_home}/{ReqTxt}';\n"
    
        if choice == "41":
            NewEnvTxt = input("Name of the new environment: ")
            systemCom = f"cd '{p_home}'; virtualenv {NewEnvTxt}"
            subprocess.run(systemCom, shell=True)
            print()
            print(f"\033[33mActivate the environment with: source {p_home}/{NewEnvTxt}/bin/activate\033[0m")
    
        if choice == "50":
            pipCom = "pip cache list"
    
        if choice == "51":
            pipCom = "pip cache purge"
    
        if choice == "D":
            YorN = input(f"\033[91m\033[1mAre you sure that you want to delete {env} (Y/N)? \033[0m ")
            if YorN == "Y":
                systemCom = f"rm -Rf '{p_home}/{env}'"
                subprocess.run(systemCom, shell=True)
                print(f"{p_home}/{env} deleted")
    
        print(f"\033[94m\033[1m{pipCom}\033[0m")
    
        if choice in ["6","7","8","9","10","11"]:
           print(f"\033[31mWait...\033[0m")
        # Ici on exécute la commande, quel que soit le choix (1, 2, 15, 20...)
        subprocess.run(pipCom, shell=True)
    
        if choice in ["20"]:
            print(f"Now deleting {reqtxtTmp}")
            os.remove(reqtxtTmp)

        if choice in ["15"]:
            print(f"\033[31mCheck carefully the previous installation messages. In case of any doubt, run command \033[31;1m#2\033[0m\033[31m (pip check)\033[0m")
        if choice in ["40"]:

            print(f"\033[31mDon't forget to add \033[31;1m{NewEnvTxt}\033[0m\033[31m in the venv variable of the present pipManagement.py script\033[0m")

        if choice in ["15","16","20","21","23"]:
            if choice in ["15","21"]:
                print(f"\033[31mCheck carefully the previous installation messages, as well as the foregoing \033[31;1mpip check command\033[0m")
            print(f"\033[31;1mNow checking if dependencies are still OK\033[0m (expecting 'No broken requirements found)'")
            pipCom = f"{pyAct}pip check"
            subprocess.run(pipCom, shell=True)
            if choice in ["15","21"]:
                print(f"\033[31mIf you want to revert to the previous configuration, run command \033[31;1m#23\033[0m\033[31m (pip install -r requirements.txt), with the \033[31;1m{reqBefore}\033[0m\033[31m file")
        choice = input(f"\n\033[32mPress Enter to continue or x to exit...\033[0m")

if __name__ == "__main__":
    run()
