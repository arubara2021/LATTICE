import os
from pathlib import Path

def combine_python_scripts(source_dir: str, output_file: str):
    base_path = Path(source_dir)
    out_path = Path(output_file)
    
    # Ensure the source directory exists
    if not base_path.exists() or not base_path.is_dir():
        print(f"Error: The directory {base_path} does not exist.")
        return

    # Open the output file in write mode
    with out_path.open('w', encoding='utf-8') as outfile:
        # Recursively search for all .py files in the directory
        python_files = list(base_path.rglob('*.py'))
        
        if not python_files:
            print(f"No .py files found in {base_path}.")
            return
            
        for py_file in python_files:
            # Write a clear separator and the relative file name
            outfile.write(f"\n{'='*50}\n")
            outfile.write(f"FILE: {py_file.relative_to(base_path)}\n")
            outfile.write(f"{'='*50}\n\n")
            
            # Read and append the file contents
            try:
                with py_file.open('r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
            except Exception as e:
                outfile.write(f"# Error reading {py_file.name}: {e}\n")
                
    print(f"Successfully combined {len(python_files)} Python files into {out_path}")

if __name__ == "__main__":
    # Your target directory
    TARGET_DIR = r"C:\Users\arun0\Videos\MLE\ONNX\formula2onnx"
    
    # Output file saved in the ONNX parent folder
    OUTPUT_TXT = r"C:\Users\arun0\Videos\MLE\ONNX\combined_formula2onnx_code.txt"
    
    combine_python_scripts(TARGET_DIR, OUTPUT_TXT)