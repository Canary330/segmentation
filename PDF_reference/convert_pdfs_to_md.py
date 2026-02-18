import os
import fitz
from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph

def extract_images_from_pdf(doc, output_folder, base_name):
    """Extract images from PDF and save them"""
    image_list = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list_in_page = page.get_images()
        
        for img_index, img in enumerate(image_list_in_page):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_name = f"{base_name}_page{page_num + 1}_img{img_index + 1}.{image_ext}"
            image_path = os.path.join(output_folder, image_name)
            
            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)
            
            image_list.append((page_num + 1, image_name))
    
    return image_list

def extract_images_from_docx(doc_path, output_folder, base_name):
    """Extract images from Word document"""
    doc = Document(doc_path)
    image_list = []
    
    for rel in doc.part.rels.values():
        if "image" in rel.target_ref:
            img_index = len(image_list) + 1
            image_ext = rel.target_ref.split('.')[-1]
            image_name = f"{base_name}_img{img_index}.{image_ext}"
            image_path = os.path.join(output_folder, image_name)
            
            with open(image_path, "wb") as image_file:
                image_file.write(rel.target_part.blob)
            
            image_list.append(image_name)
    
    return image_list

def convert_docx_to_md(docx_path, md_path, images_folder, base_name):
    """Convert Word document to Markdown"""
    doc = Document(docx_path)
    markdown_content = []
    
    # Extract images
    images = extract_images_from_docx(docx_path, images_folder, base_name)
    img_counter = 0
    
    for element in doc.element.body:
        if isinstance(element, CT_P):
            para = Paragraph(element, doc)
            text = para.text.strip()
            
            # Check for headings
            if para.style.name.startswith('Heading'):
                level = para.style.name.replace('Heading ', '')
                if level.isdigit():
                    markdown_content.append(f"{'#' * int(level)} {text}\n")
                else:
                    markdown_content.append(f"{text}\n")
            elif text:
                markdown_content.append(f"{text}\n")
                
        elif isinstance(element, CT_Tbl):
            table = Table(element, doc)
            markdown_content.append("\n")
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                markdown_content.append("| " + " | ".join(cells) + " |")
            markdown_content.append("\n")
    
    # Add image references at the end
    if images:
        markdown_content.append("\n## 图片\n")
        for img_name in images:
            markdown_content.append(f"![{img_name}](images/{img_name})\n")
    
    with open(md_path, 'w', encoding='utf-8') as md_file:
        md_file.write('\n'.join(markdown_content))

def convert_pdf_to_md(pdf_path, md_path, images_folder, base_name):
    """Convert PDF to Markdown with images"""
    doc = fitz.open(pdf_path)
    markdown_content = []
    
    # Extract images
    images = extract_images_from_pdf(doc, images_folder, base_name)
    
    # Extract text from each page
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        markdown_content.append(f"## Page {page_num + 1}\n\n{text}\n")
        
        # Add image references for this page
        page_images = [img for img in images if img[0] == page_num + 1]
        if page_images:
            markdown_content.append("\n### 图片\n")
            for _, img_name in page_images:
                markdown_content.append(f"![{img_name}](images/{img_name})\n")
    
    doc.close()
    
    with open(md_path, 'w', encoding='utf-8') as md_file:
        md_file.write('\n'.join(markdown_content))

def convert_documents_to_md(folder_paths):
    for folder in folder_paths:
        if not os.path.exists(folder):
            print(f"Folder {folder} does not exist.")
            continue

        # Create md_files and images subfolders
        md_folder = os.path.join(folder, 'md_files')
        images_folder = os.path.join(md_folder, 'images')
        os.makedirs(md_folder, exist_ok=True)
        os.makedirs(images_folder, exist_ok=True)

        for file_name in os.listdir(folder):
            base_name = os.path.splitext(file_name)[0]
            md_file_name = base_name + '.md'
            md_path = os.path.join(md_folder, md_file_name)
            
            if file_name.endswith('.pdf'):
                file_path = os.path.join(folder, file_name)
                try:
                    convert_pdf_to_md(file_path, md_path, images_folder, base_name)
                    print(f"✓ Converted PDF: {file_name} -> md_files/{md_file_name}")
                except Exception as e:
                    print(f"✗ Failed to convert PDF {file_name}: {e}")
                    
            elif file_name.endswith('.docx') or file_name.endswith('.doc'):
                file_path = os.path.join(folder, file_name)
                try:
                    convert_docx_to_md(file_path, md_path, images_folder, base_name)
                    print(f"✓ Converted Word: {file_name} -> md_files/{md_file_name}")
                except Exception as e:
                    print(f"✗ Failed to convert Word {file_name}: {e}")

def extract_md_filenames(folder_paths, output_file):
    """Extract Markdown filenames from folders and save to a text file."""
    all_filenames = []

    for folder in sorted(folder_paths, key=lambda x: int(x)):
        md_folder = os.path.join(folder, 'md_files')
        if not os.path.exists(md_folder):
            print(f"Folder {md_folder} does not exist.")
            continue

        filenames = [f for f in os.listdir(md_folder) if f.endswith('.md')]
        all_filenames.extend(f"{folder}/{filename}" for filename in filenames)

    with open(output_file, 'w', encoding='utf-8') as f:
        for filename in all_filenames:
            f.write(filename + '\n')

if __name__ == "__main__":
    folders = ["1", "2", "3"]
    convert_documents_to_md(folders)

    # Extract Markdown filenames and save to a text file
    output_txt_file = "md_filenames.txt"
    extract_md_filenames(folders, output_txt_file)
    print(f"✓ Extracted Markdown filenames to {output_txt_file}")