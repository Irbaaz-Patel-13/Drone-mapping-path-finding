import cv2
import numpy as np
import os
import csv
from tkinter import *
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

# Ensure the output directory exists
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Global Variables
input_image_path = ""
blur_ksize = 3
canny_threshold1 = 100
canny_threshold2 = 200
dilate_iterations = 1
erode_iterations = 1
brightness = 100
contrast = 100
threshold_value = 127

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing and Path Planning")
        self.root.geometry("1200x800")
        self.root.configure(bg="#e6f7ff")  # Light blue background

        self.original_image = None
        self.processed_image = None
        self.image_loaded = False  # Prevent processing before parameter changes

        self.use_gaussian_blur = BooleanVar(value=False)
        self.use_median_filter = BooleanVar(value=False)

        self.create_scrollable_ui()

    # Scrollable UI Setup
    def create_scrollable_ui(self):
        main_frame = Frame(self.root, bg="#e6f7ff")
        main_frame.pack(fill="both", expand=True)

        self.canvas = Canvas(main_frame, bg="#e6f7ff")
        scrollbar = Scrollbar(main_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.scrollable_frame = Frame(self.canvas, bg="#e6f7ff")
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.create_ui_elements()

    # Main Interface Elements
    def create_ui_elements(self):
        title_label = Label(self.scrollable_frame, text="Image Processing and Path Planning",
                            font=("Helvetica", 18, "bold"),
                            bg="#e6f7ff", fg="#003366")  # Dark blue text
        title_label.grid(row=0, column=0, columnspan=2, pady=20)

        select_image_btn = Button(self.scrollable_frame, text="Select Image", font=("Helvetica", 14),
                                  command=self.select_image, bg="#4CAF50", fg="white")  # Green button
        select_image_btn.grid(row=1, column=0, columnspan=2, pady=10)

        self.image_frame = LabelFrame(self.scrollable_frame, text="Live Preview", font=("Helvetica", 12, "bold"),
                                      padx=10, pady=10,
                                      bg="#e6f7ff", fg="#003366")  # Dark blue text
        self.image_frame.grid(row=2, column=0, rowspan=8, padx=10, pady=10, sticky="n")

        self.image_label = Label(self.image_frame)
        self.image_label.pack()

        self.create_parameter_section()

        process_btn = Button(self.scrollable_frame, text="Process and Save Image", font=("Helvetica", 14),
                             command=self.save_processed_image, bg="#2196F3", fg="white")  # Blue button
        process_btn.grid(row=10, column=0, columnspan=2, pady=10)

        # Tuning Sequence Label
        tuning_sequence_label = Label(self.scrollable_frame, text="Tuning Sequence for Parameters:",
                                       font=("Helvetica", 12, "bold"), bg="#e6f7ff", fg="#003366")
        tuning_sequence_label.grid(row=11, column=0, columnspan=2, pady=10)

        tuning_sequence_text = """
        1. Image Preprocessing and Enhancement
        2. Feature Detection and Contour Filtering
        3. Path Planning and Optimization
        4. Performance Metrics and Graphs
        """
        tuning_sequence = Label(self.scrollable_frame, text=tuning_sequence_text, bg="#e6f7ff", fg="#003366",
                                justify="left", font=("Helvetica", 10))
        tuning_sequence.grid(row=12, column=0, columnspan=2, pady=10)

    # Parameter Adjustment Section
    def create_parameter_section(self):
        Label(self.scrollable_frame, text="1. Gaussian Blur:", bg="#e6f7ff", fg="#003366").grid(row=2, column=1, sticky="w")
        self.blur_slider = Scale(self.scrollable_frame, from_=1, to=15, orient=HORIZONTAL, bg="#e6e6ff", fg="#003366", command=self.update_preview)
        self.blur_slider.set(blur_ksize)
        self.blur_slider.grid(row=3, column=1, padx=20, sticky="we")

        Label(self.scrollable_frame, text="2. Canny Edge Detection Threshold 1:", bg="#e6f7ff", fg="#003366").grid(row=4, column=1, sticky="w")
        self.canny1_slider = Scale(self.scrollable_frame, from_=0, to=255, orient=HORIZONTAL, bg="#e6e6ff", fg="#003366", command=self.update_preview)
        self.canny1_slider.set(canny_threshold1)
        self.canny1_slider.grid(row=5, column=1, padx=20, sticky="we")

        Label(self.scrollable_frame, text="3. Canny Edge Detection Threshold 2:", bg="#e6f7ff", fg="#003366").grid(row=6, column=1, sticky="w")
        self.canny2_slider = Scale(self.scrollable_frame, from_=0, to=255, orient=HORIZONTAL, bg="#e6e6ff", fg="#003366", command=self.update_preview)
        self.canny2_slider.set(canny_threshold2)
        self.canny2_slider.grid(row=7, column=1, padx=20, sticky="we")

        Label(self.scrollable_frame, text="4. Dilation Iterations:", bg="#e6f7ff", fg="#003366").grid(row=8, column=1, sticky="w")
        self.dilate_slider = Scale(self.scrollable_frame, from_=0, to=5, orient=HORIZONTAL, bg="#e6e6ff", fg="#003366", command=self.update_preview)
        self.dilate_slider.set(dilate_iterations)
        self.dilate_slider.grid(row=9, column=1, padx=20, sticky="we")

        Label(self.scrollable_frame, text="5. Erosion Iterations:", bg="#e6f7ff", fg="#003366").grid(row=10, column=1, sticky="w")
        self.erode_slider = Scale(self.scrollable_frame, from_=0, to=5, orient=HORIZONTAL, bg="#e6f7ff", fg="#003366", command=self.update_preview)
        self.erode_slider.set(erode_iterations)
        self.erode_slider.grid(row=11, column=1, padx=20, sticky="we")

        Label(self.scrollable_frame, text="6. Brightness:", bg="#e6f7ff", fg="#003366").grid(row=12, column=1, sticky="w")
        self.brightness_slider = Scale(self.scrollable_frame, from_=0, to=200, orient=HORIZONTAL, bg="#e6e6ff", fg="#003366", command=self.update_preview)
        self.brightness_slider.set(brightness)
        self.brightness_slider.grid(row=13, column=1, padx=20, sticky="we")

        Label(self.scrollable_frame, text="7. Contrast:", bg="#e6f7ff", fg="#003366").grid(row=14, column=1, sticky="w")
        self.contrast_slider = Scale(self.scrollable_frame, from_=0, to=200, orient=HORIZONTAL, bg="#e6e6ff", fg="#003366", command=self.update_preview)
        self.contrast_slider.set(contrast)
        self.contrast_slider.grid(row=15, column=1, padx=20, sticky="we")

        Label(self.scrollable_frame, text="8. Threshold Value:", bg="#e6f7ff", fg="#003366").grid(row=16, column=1, sticky="w")
        self.threshold_slider = Scale(self.scrollable_frame, from_=0, to=255, orient=HORIZONTAL, bg="#e6f7ff", fg="#003366", command=self.update_preview)
        self.threshold_slider.set(threshold_value)
        self.threshold_slider.grid(row=17, column=1, padx=20, sticky="we")

        Label(self.scrollable_frame, text="9. Sharpening:", bg="#e6f7ff", fg="#003366").grid(row=18, column=1, sticky="w")
        self.sharpness_slider = Scale(self.scrollable_frame, from_=0, to=10, orient=HORIZONTAL, bg="#e6f7ff", fg="#003366", command=self.update_preview)
        self.sharpness_slider.set(0)
        self.sharpness_slider.grid(row=19, column=1, padx=20, sticky="we")

        Label(self.scrollable_frame, text="10. Histogram Equalization:", bg="#e6f7ff", fg="#003366").grid(row=20, column=1, sticky="w")
        self.equalize_slider = Scale(self.scrollable_frame, from_=0, to=1, orient=HORIZONTAL, bg="#e6f7ff", fg="#003366", command=self.update_preview)
        self.equalize_slider.set(0)
        self.equalize_slider.grid(row=21, column=1, padx=20, sticky="we")

        Label(self.scrollable_frame, text="11. Rotation Angle:", bg="#e6f7ff", fg="#003366").grid(row=22, column=1, sticky="w")
        self.rotation_slider = Scale(self.scrollable_frame, from_=0, to=360, orient=HORIZONTAL, sbg="#e6f7ff", fg="#003366", command=self.update_preview)
        self.rotation_slider.set(0)
        self.rotation_slider.grid(row=23, column=1, padx=20, sticky="we")

        Label(self.scrollable_frame, text="12. Edge Detection Type:", bg="#e6f7ff", fg="#003366").grid(row=24, column=1, sticky="w")
        self.edge_type_option = ttk.Combobox(self.scrollable_frame, values=["Canny", "Sobel", "Laplacian"],
                                             state="readonly")
        self.edge_type_option.set("Canny")
        self.edge_type_option.grid(row=25, column=1, padx=20, sticky="we")

        # Noise Reduction Options
        Label(self.scrollable_frame, text="Noise Reduction:", bg="#e6f7ff", fg="#003366").grid(row=26, column=1, sticky="w")
        Checkbutton(self.scrollable_frame, text="Use Gaussian Blur", variable=self.use_gaussian_blur, bg="#e6f7ff", fg="#003366", command=self.update_preview).grid(row=27, column=1, sticky="w", padx=20)
        Checkbutton(self.scrollable_frame, text="Use Median Filter", variable=self.use_median_filter, bg="#e6f7ff", fg="#003366", command=self.update_preview).grid(row=28, column=1, sticky="w", padx=20)

    def select_image(self):
        global input_image_path
        input_image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.gif"), ("All Files", "*.*")]
        )
        if input_image_path:
            self.load_and_display_image()
        else:
            messagebox.showwarning("Warning", "No image selected.")

    def load_and_display_image(self):
        self.original_image = cv2.imread(input_image_path)
        if self.original_image is not None:
            self.image_loaded = True
            self.update_preview()
        else:
            messagebox.showerror("Error", "Failed to load image. Please ensure the file is a valid image.")

    def process_image(self):
        image = self.original_image.copy()
        blur_value = self.blur_slider.get()
        if blur_value % 2 == 0:
            blur_value += 1
        blur_value = max(1, blur_value)

        if self.use_gaussian_blur.get():
            image = cv2.GaussianBlur(image, (blur_value, blur_value), 0)
        if self.use_median_filter.get():
            image = cv2.medianBlur(image, blur_value)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        edge_type = self.edge_type_option.get()
        if edge_type == "Canny":
            edges = cv2.Canny(gray, self.canny1_slider.get(), self.canny2_slider.get())
        elif edge_type == "Sobel":
            edges = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5) + cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            edges = np.uint8(np.clip(edges, 0, 255))
        elif edge_type == "Laplacian":
            edges = cv2.Laplacian(gray, cv2.CV_64F)
            edges = np.uint8(np.clip(edges, 0, 255))

        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=self.dilate_slider.get())
        edges = cv2.erode(edges, kernel, iterations=self.erode_slider.get())

        _, thresh = cv2.threshold(gray, self.threshold_slider.get(), 255, cv2.THRESH_BINARY)

        thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        final_image = cv2.addWeighted(thresh_color, self.brightness_slider.get() / 100, edges_color,
                                      self.contrast_slider.get() / 100, 0)

        sharpness_value = self.sharpness_slider.get()
        if sharpness_value > 0:
            kernel_sharp = np.array([[0, -1, 0], [-1, 5 + sharpness_value, -1], [0, -1, 0]])
            final_image = cv2.filter2D(final_image, -1, kernel_sharp)

        if self.equalize_slider.get() == 1:
            final_image = cv2.equalizeHist(cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY))

        angle = self.rotation_slider.get()
        if angle != 0:
            (h, w) = final_image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            final_image = cv2.warpAffine(final_image, M, (w, h))

        return final_image

    def update_preview(self, event=None):
        if self.image_loaded:
            self.processed_image = self.process_image()
            self.display_image(self.processed_image)

    def display_image(self, image):
        preview_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preview_image = Image.fromarray(preview_image)
        preview_image = preview_image.resize((800, 800), Image.LANCZOS)
        preview_photo = ImageTk.PhotoImage(preview_image)
        self.image_label.configure(image=preview_photo)
        self.image_label.image = preview_photo

    def save_processed_image(self):
        if self.processed_image is not None:
            save_path = os.path.join(output_dir, "processed_image.png")
            cv2.imwrite(save_path, self.processed_image)
            self.save_parameters()
            messagebox.showinfo("Success", f"Image and parameters saved at {save_path}")
        else:
            messagebox.showerror("Error", "No processed image to save.")

    def save_parameters(self):
        params = {
            "blur_ksize": self.blur_slider.get(),
            "canny_threshold1": self.canny1_slider.get(),
            "canny_threshold2": self.canny2_slider.get(),
            "dilate_iterations": self.dilate_slider.get(),
            "erode_iterations": self.erode_slider.get(),
            "brightness": self.brightness_slider.get(),
            "contrast": self.contrast_slider.get(),
            "threshold_value": self.threshold_slider.get(),
            "sharpness": self.sharpness_slider.get(),
            "histogram_equalization": self.equalize_slider.get(),
            "rotation_angle": self.rotation_slider.get(),
            "edge_detection_type": self.edge_type_option.get(),
            "use_gaussian_blur": self.use_gaussian_blur.get(),
            "use_median_filter": self.use_median_filter.get()
        }
        with open(os.path.join(output_dir, "parameters.csv"), mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=params.keys())
            writer.writeheader()
            writer.writerow(params)

# Main Program Execution
root = Tk()
app = ImageProcessingApp(root)
root.mainloop()