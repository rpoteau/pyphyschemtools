# Python environmennt management

**`pipManagement` tool**

The `pipManagement` tool is a cross-platform CLI utility designed to simplify the maintenance of Python virtual environments specifically for physical chemistry workflows.

## Key Features

* **Auto-Detection**: Automatically scans your `PIP_MGMT_FOLDER` (defaulting to `~/Python3`) for valid virtual environments. No manual editing of the script is required when adding new environments.
* **Cross-Platform**: Works natively on **Linux**, **macOS** (Darwin), and **Windows** by detecting the correct activation paths (`bin/` vs `Scripts/`).
* **Dependency Safety**: Automatically runs `pip check` after installations or removals to ensure your scientific stack remains consistent and free of version conflicts.
* **Alphabetical Mapping**: Automatically maps detected environments to a simple letter-based selection menu (a, b, c...) sorted case-insensitively.

---

## Usage

Once `pyphyschemtools` is installed, the command is universal across all operating systems. Open your terminal (Terminal on Mac/Linux, or PowerShell/CMD on Windows) and type:

```bash
pipManagement
```

---

## Configuration: Setting up PIP_MGMT_FOLDER

By default, the tool looks for a folder named `Python3` located directly in your User Home directory. If your virtual environments are stored in a different location, you must define the `PIP_MGMT_FOLDER` environment variable.



The tool resolves the final path by combining your **Home Directory** with the value of this variable:
* **Linux/macOS**: `/home/username/` + `PIP_MGMT_FOLDER`
* **Windows**: `C:\Users\username\` + `PIP_MGMT_FOLDER`


### 🐧 Linux & 🍎 macOS (Zsh/Bash)
On Unix-based systems, you define the variable in your shell configuration file so it is loaded every time you open a terminal.

1. **Open your configuration file** (most modern Macs use `~/.zshrc`, while Linux usually uses `~/.bashrc`):
    ```bash
    nano ~/.zshrc  #mac
    nano ~/.bashrc #linux
    ```
2. **Add the following line at the end of the file**:
    ```bash
    export PIP_MGMT_FOLDER="MyCustomFolder"
    ```
3. **Save and Exit**: Press Ctrl+O, then Enter to save, and Ctrl+X to exit.
4. Refresh your current shell to apply the changes immediately:
    ``` bash
    source ~/.zshrc #mac
    source ~/.bashrc #linux
    ```

### 🪟 Windows (PowerShell / CMD)
Windows users can set this globally through the System Properties to ensure it works in both PowerShell and Command Prompt.

1. Press the Windows Key and type env.
2. Select Edit the system environment variables.
3. In the window that appears, click the Environment Variables button at the bottom right.
4. Under the User variables section (the top half), click New....
5. Fill in the details:
    - Variable name: PIP_MGMT_FOLDER
    - Variable value: MyCustomFolder (e.g., Documents\PythonEnvs)
6. Click OK on all windows and restart any open terminal windows for the change to take effect.

🛠️ **Troubleshooting on Windows**
If the command pipManagement is not recognized after installation, ensure that the Python Scripts folder is in your system PATH. Alternatively, you can always launch the tool manually via Python:
```powershell
python -m pyphyschemtools.utils.pipManagement
```