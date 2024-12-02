import argparse
import nbformat
from nbconvert import PythonExporter
import os

# in terminal, run: python ipynb_to_py.py data_testing.ipynb
# Conversion successful: data_testing.ipynb -> data_testing.p


def convert_ipynb_to_py(ipynb_file, py_file):

    try:
        with open(ipynb_file, 'r', encoding='utf-8') as f:
            notebook_content = nbformat.read(f, as_version=4)
        
        python_exporter = PythonExporter()
        python_script, _ = python_exporter.from_notebook_node(notebook_content)
        
        with open(py_file, 'w', encoding='utf-8') as f:
            f.write(python_script)
        
        print(f"Conversion successful: {ipynb_file} -> {py_file}")

    except Exception as e:
        print(f"Error during conversion: {e}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert Jupyter Notebook (.ipynb) to Python script (.py)")
    parser.add_argument("input_file", help="Path to the input .ipynb file")
    
    args = parser.parse_args()
    
    input_file = args.input_file
    output_file = os.path.splitext(input_file)[0] + '.py'
    
    convert_ipynb_to_py(input_file, output_file)


    