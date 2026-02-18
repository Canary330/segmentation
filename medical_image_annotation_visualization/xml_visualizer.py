# Usage: Run this script using Python: python medical_image_annotation_visualization/xml_visualizer.py
# It allows previewing XML annotations from the '标注' folder overlaid on images from 'FOCUS-dataset',
# along with source annotations (masks and ellipses) from the dataset itself.

import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tkinter as tk
from tkinter import ttk
import cv2

# Source Dataset Constants (from zenodo_visualize.py)
SOURCE_COLOR_DICT = {
    1: (0, 255, 0),      # cardiac - 绿
    2: (0, 180, 255),    # thorax - 蓝
    3: (255, 165, 0),    # chamber1 - 橙
    4: (255, 0, 0),      # chamber2 - 红
    5: (255, 255, 0),    # 
}

SOURCE_LABEL_MAP = {
    1: "Cardiac",
    2: "Thorax",
    3: "Chamber1",
    4: "Chamber2",
    5: "Chamber3"
}

def read_ellipse(txt_path):
    res = []
    if not os.path.exists(txt_path):
        return res
    with open(txt_path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 6:
                continue
            cx, cy, a, b, angle = map(float, p[:5])
            label = p[5]
            res.append((cx, cy, a, b, angle, label))
    return res

class XMLVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Image Annotation Visualizer")
        self.root.geometry("1200x800")

        # Define paths relative to this script
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.xml_dir = os.path.join(self.script_dir, "..", "标注")
        self.dataset_dir = os.path.join(self.script_dir, "..", "FOCUS-dataset")

        self.subsets = ["testing", "training", "validation"]
        self.current_xml_data = {}
        self.label_colors = {}
        self.all_image_names = []
        self.show_annotations = tk.BooleanVar(value=True)

        # Try to load a font for labels
        try:
            # Common font paths on MacOS/Linux/Windows
            font_paths = ["/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "arial.ttf"]
            self.font = None
            for fp in font_paths:
                if os.path.exists(fp):
                    self.font = ImageFont.truetype(fp, 14) 
                    break
            if not self.font:
                self.font = ImageFont.load_default()
        except:
            self.font = ImageFont.load_default()

        self.setup_ui()
        self.load_subset("testing")

    def setup_ui(self):
        # Top Control Panel
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(control_frame, text="Subset:").pack(side=tk.LEFT)
        self.subset_var = tk.StringVar()
        self.subset_combo = ttk.Combobox(control_frame, textvariable=self.subset_var, values=self.subsets, state="readonly")
        self.subset_combo.set("testing")
        self.subset_combo.pack(side=tk.LEFT, padx=5)
        self.subset_combo.bind("<<ComboboxSelected>>", lambda e: self.load_subset(self.subset_var.get()))

        # Toggle Button in Top Right
        self.toggle_btn = ttk.Checkbutton(control_frame, text="Show Annotations", variable=self.show_annotations, command=self.refresh_image)
        self.toggle_btn.pack(side=tk.RIGHT)

        # Main Content area
        main_frame = ttk.Frame(self.root)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left side - Listbox for images with Scrollbar and Search
        list_frame = ttk.Frame(main_frame, padding="10")
        list_frame.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(list_frame, text="Search Image:").pack(side=tk.TOP, anchor=tk.W)
        search_frame = ttk.Frame(list_frame)
        search_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.search_entry.bind("<KeyRelease>", lambda e: self.filter_list())
        
        self.clear_btn = ttk.Button(search_frame, text="X", width=3, command=self.clear_search)
        self.clear_btn.pack(side=tk.LEFT)

        ttk.Label(list_frame, text="Images:").pack(side=tk.TOP, anchor=tk.W, pady=(10, 0))
        
        list_inner_frame = ttk.Frame(list_frame)
        list_inner_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.image_listbox = tk.Listbox(list_inner_frame, width=30)
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_inner_frame, orient=tk.VERTICAL, command=self.image_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_listbox.config(yscrollcommand=scrollbar.set)
        
        self.image_listbox.bind("<<ListboxSelect>>", self.on_image_select)

        # Right side - Canvas for image display
        self.canvas_frame = ttk.Frame(main_frame, padding="10")
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, background="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def load_subset(self, subset):
        # 1. Clear previous data
        self.current_xml_data = {}
        self.all_image_names = []
        self.label_colors = {}

        # 2. Get all images from filesystem
        img_dir = os.path.join(self.dataset_dir, subset, "images")
        if os.path.exists(img_dir):
            for f in os.listdir(img_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.all_image_names.append(f)
        self.all_image_names.sort()

        # 3. Load XML annotations if exists
        xml_path = os.path.join(self.xml_dir, f"{subset}.xml")
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for label in root.findall(".//label"):
                name = label.find("name").text
                color = label.find("color").text
                self.label_colors[name] = color

            for img_tag in root.findall("image"):
                img_name = img_tag.get("name")
                annotations = []
                
                # We skip 'box' elements as requested (no rectangles)
                
                for mask in img_tag.findall("mask"):
                    annotations.append({
                        "type": "mask",
                        "label": mask.get("label"),
                        "rle": [int(x) for x in mask.get("rle").split(",")],
                        "left": int(mask.get("left")),
                        "top": int(mask.get("top")),
                        "width": int(mask.get("width")),
                        "height": int(mask.get("height"))
                    })

                for poly in img_tag.findall("polygon"):
                    points_str = poly.get("points")
                    pts = []
                    for pair in points_str.split(";"):
                        if not pair: continue
                        x, y = map(float, pair.split(","))
                        pts.append((x, y))
                    annotations.append({"type": "polygon", "label": poly.get("label"), "points": pts})

                self.current_xml_data[img_name] = {
                    "width": int(img_tag.get("width")),
                    "height": int(img_tag.get("height")),
                    "annotations": annotations
                }

        self.filter_list()

    def clear_search(self):
        self.search_var.set("")
        self.filter_list()

    def filter_list(self):
        search_term = self.search_var.get().lower()
        self.image_listbox.delete(0, tk.END)
        for name in self.all_image_names:
            if search_term in name.lower():
                self.image_listbox.insert(tk.END, name)

    def decode_rle(self, rle, width, height):
        mask = np.zeros(width * height, dtype=np.uint8)
        current_pos = 0
        val = 0
        for count in rle:
            mask[current_pos : current_pos + count] = val
            current_pos += count
            val = 1 - val
        return mask.reshape((height, width))

    def refresh_image(self):
        self.on_image_select(None)

    def draw_text_with_bg(self, draw, position, text):
        # Always use black text as requested
        text_color = (0, 0, 0, 255)
        if not self.font:
            draw.text(position, text, fill=text_color)
            return
        
        try:
            bbox = draw.textbbox(position, text, font=self.font)
            # Add small padding to the box for better visibility
            padded_bbox = (bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2)
            # Opaque white background for maximum contrast
            draw.rectangle(padded_bbox, fill=(255, 255, 255, 255))
            draw.text(position, text, fill=text_color, font=self.font)
        except:
            draw.text(position, text, fill=text_color, font=self.font)

    def on_image_select(self, event):
        selection = self.image_listbox.curselection()
        if not selection:
            return
        
        img_name = self.image_listbox.get(selection[0])
        subset = self.subset_var.get()
        base = os.path.splitext(img_name)[0]
        
        img_path = os.path.join(self.dataset_dir, subset, "images", img_name)
        if not os.path.exists(img_path):
            return

        # Load image
        img_pil = Image.open(img_path).convert("RGBA")
        
        # If toggle is OFF, just show original image
        if not self.show_annotations.get():
            combined = img_pil
        else:
            img_np = np.array(img_pil.convert("RGB"))
            
            # 1. Overlay Source Dataset Mask
            mask_path = os.path.join(self.dataset_dir, subset, "annfiles_mask", f"{base}_mask.png")
            if os.path.exists(mask_path):
                mask_src = Image.open(mask_path).convert("L")
                mask_src = mask_src.resize(img_pil.size, Image.NEAREST)
                mask_np = np.array(mask_src)
                
                color_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
                for cls, color in SOURCE_COLOR_DICT.items():
                    color_mask[mask_np == cls] = color
                
                overlay_np = cv2.addWeighted(img_np, 0.65, color_mask, 0.35, 0)
                img_pil = Image.fromarray(overlay_np).convert("RGBA")
            
            # Prepare overlay for shapes and text
            overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            labels_to_draw = []

            # 2. Draw Source Ellipses (Circles)
            ell_path = os.path.join(self.dataset_dir, subset, "annfiles_ellipse", f"{base}.txt")
            for cx, cy, a, b, angle, label in read_ellipse(ell_path):
                box_coords = [cx - a, cy - b, cx + a, cy + b]
                draw.ellipse(box_coords, outline=(0, 255, 0, 255), width=2)
                labels_to_draw.append(((cx, cy), f"SRC:{label}"))

            # 3. Draw XML Annotations
            if img_name in self.current_xml_data:
                xml_data = self.current_xml_data[img_name]
                for ann in xml_data["annotations"]:
                    color_hex = self.label_colors.get(ann["label"], "#FFFFFF")
                    r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
                    
                    if ann["type"] == "mask":
                        m_np = self.decode_rle(ann["rle"], ann["width"], ann["height"])
                        m_img = Image.new("L", (ann["width"], ann["height"]), 0)
                        m_img.putdata(m_np.flatten() * 255)
                        col_m = Image.new("RGBA", (ann["width"], ann["height"]), (r, g, b, 128))
                        overlay.paste(col_m, (ann["left"], ann["top"]), m_img)
                        labels_to_draw.append(((ann["left"], ann["top"]), f"XML:{ann['label']}"))

                    elif ann["type"] == "polygon":
                        pts = ann["points"]
                        draw.polygon(pts, outline=(r, g, b, 255), fill=(r, g, b, 64))
                        labels_to_draw.append(((pts[0][0], pts[0][1] - 20), f"XML:{ann['label']}"))

            # Finally, draw all labels on top of everything
            for pos, txt in labels_to_draw:
                self.draw_text_with_bg(draw, pos, txt)

            # Combine
            combined = Image.alpha_composite(img_pil, overlay)
        
        # Resize to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width > 1 and canvas_height > 1:
            ratio = min(canvas_width / combined.width, canvas_height / combined.height)
            new_size = (int(combined.width * ratio), int(combined.height * ratio))
            combined = combined.resize(new_size, Image.Resampling.LANCZOS)

        self.tk_img = ImageTk.PhotoImage(combined)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=self.tk_img, anchor=tk.CENTER)

if __name__ == "__main__":
    root = tk.Tk()
    app = XMLVisualizer(root)
    root.mainloop()
