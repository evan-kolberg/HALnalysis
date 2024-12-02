import os
import shutil
import nbformat
import subprocess
from nbconvert import PythonExporter
from typing import List, Tuple, Dict
from tqdm import tqdm


def create_chip_folder_structure(plots_path: str, chip_name: str, div_days: List[int], wells: List[int]) -> str:
    chip_folder = os.path.join(plots_path, chip_name)
    os.makedirs(chip_folder, exist_ok=True)
    
    for div_day in div_days:
        div_folder = os.path.join(chip_folder, f"DIV{div_day}")
        os.makedirs(div_folder, exist_ok=True)
        
        for well in wells:
            well_folder = os.path.join(div_folder, f"WELL{well}")
            os.makedirs(well_folder, exist_ok=True)
    
    return chip_folder


def update_notebook_paths(notebook: nbformat.NotebookNode, replacements: List[Tuple[str, str, int]]) -> None:
    for new_data_path, new_plts_path, new_well_no in replacements:
        new_data_path = os.path.normpath(new_data_path).replace('\\', '/')
        new_plts_path = os.path.normpath(new_plts_path).replace('\\', '/')
        
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                lines = cell.source.split('\n')
                new_lines = []
                for line in lines:
                    if line.startswith("data_path ="):
                        new_lines.append(f"data_path = '{new_data_path}'")
                    elif line.startswith("plts_path ="):
                        new_lines.append(f"plts_path = '{new_plts_path}'")
                    elif line.startswith("well_no ="):
                        new_lines.append(f"well_no = {new_well_no}")
                    elif "plt.show()" not in line:
                        new_lines.append(line)
                cell.source = '\n'.join(new_lines)


def process_notebooks(input_path: str, replacements: List[Tuple[str, str, int, int]], temp_dir: str, chip_name: str, skip_cells: List[int] = []) -> None:
    with open(input_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)

    os.makedirs(temp_dir, exist_ok=True)

    for _, (new_data_path, new_plts_path, new_well_no, div_day) in tqdm(enumerate(replacements, start=1), total=len(replacements)):
        updated_notebook = notebook.copy()
        update_notebook_paths(updated_notebook, [(new_data_path, new_plts_path, new_well_no)])
        
        output_path = os.path.join(temp_dir, f'{chip_name}_DIV{div_day}_WELL{new_well_no}.ipynb')
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(updated_notebook, f)

        updated_notebook.cells = [cell for idx, cell in enumerate(updated_notebook.cells) if idx + 1 not in skip_cells]
        exporter = PythonExporter()
        script, _ = exporter.from_notebook_node(updated_notebook)
        
        script_path = os.path.join(temp_dir, f'{chip_name}_DIV{div_day}_WELL{new_well_no}.py')
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script)

        process = subprocess.Popen(['python', script_path], env=os.environ.copy(), creationflags=subprocess.CREATE_NEW_CONSOLE)
        process.wait()


if __name__ == "__main__":
    
    # if you want to skip to certain div/well files, just exit the current external console window
    # ----------------------------
    input_path = "main.ipynb"
    chip_name = "M07480"
    div_h5_paths: Dict[int, str] = {
        28: "C:\\Users\\evank\\OneDrive\\Documents\\HALnalysis_summer_2024_data\\data\\M07480\\DIV28.h5",
        33: "C:\\Users\\evank\\OneDrive\\Documents\\HALnalysis_summer_2024_data\\data\\M07480\\DIV33.h5",
        35: "C:\\Users\\evank\\OneDrive\\Documents\\HALnalysis_summer_2024_data\\data\\M07480\\DIV35.h5",
        41: "C:\\Users\\evank\\OneDrive\\Documents\\HALnalysis_summer_2024_data\\data\\M07480\\DIV41.h5",
        47: "C:\\Users\\evank\\OneDrive\\Documents\\HALnalysis_summer_2024_data\\data\\M07480\\DIV47.h5",
    }
    plots_path = "C:/Users/evank/OneDrive/Documents/HALnalysis_summer_2024_data/plots"
    wells = [0, 1, 2, 3, 4, 5] # 0 indexed, not 1 (because of the machine)
    skip_cells = [] # 1 indexed, not 0 (because of enumeration)
    analysis_package_dir = "D:/programming/HALnalysis/analysis_package"
    temp_dir = os.path.join(os.getcwd(), "TEMP")
    # ----------------------------



    div_days = list(div_h5_paths.keys())

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    os.makedirs(temp_dir, exist_ok=True)

    chip_folder = create_chip_folder_structure(plots_path, chip_name, div_days, wells)

    replacements: List[Tuple[str, str, int, int]] = []
    for div_day in div_days:
        for well in wells:
            data_path = div_h5_paths[div_day]
            plts_path = os.path.join(chip_folder, f"DIV{div_day}", f"WELL{well}")
            replacements.append((data_path, plts_path, well, div_day))

    shutil.copytree(analysis_package_dir, os.path.join(temp_dir, "analysis_package"))

    process_notebooks(input_path, replacements, temp_dir, chip_name, skip_cells)



