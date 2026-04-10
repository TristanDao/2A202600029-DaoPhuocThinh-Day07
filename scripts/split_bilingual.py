import os
import sys
from pathlib import Path

# Add root directory to path to allow importing from src
sys.path.append(os.getcwd())

from src.utils import split_vi_en

def process_data_folder(input_dir: str, output_base: str):
    input_path = Path(input_dir)
    output_path = Path(output_base)
    
    if not input_path.exists():
        print(f"Error: Input directory {input_dir} does not exist.")
        return

    # Find all .md or .txt files recursively
    files = list(input_path.glob("**/*.md")) + list(input_path.glob("**/*.txt"))
    
    if not files:
        print(f"No .md or .txt files found in {input_dir}")
        return

    print(f"Found {len(files)} files. Starting split...")
    
    processed_count = 0
    for file_path in files:
        try:
            # Maintain subfolder structure relative to input_dir
            relative_path = file_path.relative_to(input_path)
            
            content = file_path.read_text(encoding="utf-8")
            en_text, vi_text = split_vi_en(content)
            
            # Save English version
            if en_text:
                en_file = output_path / "en" / relative_path
                en_file.parent.mkdir(parents=True, exist_ok=True)
                en_file.write_text(en_text, encoding="utf-8")
                
            # Save Vietnamese version
            if vi_text:
                vi_file = output_path / "vi" / relative_path
                vi_file.parent.mkdir(parents=True, exist_ok=True)
                vi_file.write_text(vi_text, encoding="utf-8")
                
            processed_count += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"Successfully processed {processed_count} files.")
    print(f"Balanced parts stored in: \n - {output_path}/en\n - {output_path}/vi")

if __name__ == "__main__":
    # You can customize these paths as needed
    input_directory = "data/products"
    output_directory = "data/ready"
    process_data_folder(input_directory, output_directory)
