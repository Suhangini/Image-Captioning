INTRODUCTION:
IMAGE CAPTIONING - An AI System that takes an input image and generate a short, natural language sentence describing its content.
This project performs Image Captioning using BLIP's processor and conditional generation model to produce the natural language description for uploaded images. 
Images are loaded and preprocessed with Pillow (PIL), displayed in colab using cv2_imshow where needed and the passed to BlipProcessor + BlipForConditionalGeneration for caption generation. 

HOW IT WORKS? :
Load image: open with PIL and converted to RGB for consistent processing across inputs.
Display in Colab: Visualize inputs or results using cv2_imshow, which is OpenCV- friendly display function for colab notebook.
Conditional captioning: tokenie the image with an optional promt via. BlipProcessor, then generate a caption with BlipForConditionalGeneration and decode to text.
