# Painting the Periphery: Comparing Generative Video Models in Outpainting for Cropless Video Stabilisation

### COMPSCI 760 Group Research Project

### Members:
- Sreelakshmi Gireesh (sgir748)
- Eric Heo (sheo291)
- Sean McConnachie (smcc417)
- Karo Rasool (kras712)
- Annie Rajesh Vadakara (araj949)

---

### Summary  
**Painting the Periphery: Comparing Generative Video Models in Outpainting for Cropless Video Stabilisation** explores how state-of-the-art generative diffusion models can restore cropped regions in stabilised video footage. Traditional video stabilisers achieve smoothness by cropping the frame, which sacrifices both field of view (FOV) and resolution. This project investigates whether modern generative video models can “outpaint” the missing borders, allowing stable videos to retain their original FOV without quality loss.  

The project compares four models - **WAN-14B**, **WAN-1.3B**, **DiffuEraser**, and **Stable Diffusion** - on their ability to produce realistic and temporally consistent video outpainting. Evaluation is conducted using **Fréchet Video Distance (FVD)**, **Field-of-View Ratio (FOV)**, and **CLIP Similarity** metrics, alongside manual qualitative analysis.

---

### Description / Motivation  
Video stabilisation is a crucial step in filmmaking, mobile videography, and post-production, yet existing stabilisers rely on cropping to smooth motion - inevitably shrinking the usable frame and reducing resolution. This limitation inspired the exploration of a “cropless” alternative: leveraging **generative diffusion models** capable of **inpainting**, i.e., filling in missing regions based on learned visual context.  

The project’s central research question asks:  
> *How do recent diffusion-based video models compare in peripheral inpainting quality and temporal consistency for stabilised videos?*  

By combining traditional motion correction with generative outpainting, the study bridges video stabilisation and AI-driven content generation. Its findings demonstrate that diffusion-based video models, particularly WAN-14B, can extend stabilised footage convincingly while maintaining realism and motion continuity - paving the way for future applications in **film restoration**, **AI-assisted editing**, and **consumer-level video enhancement**.

---

### [Data Accessible Here](https://drive.google.com/drive/folders/14q4m4xqqDUvaDrRR9CRhgiuYBWzbQ4_L?usp=sharing)

### Project Structure

```
├── README.md
├── lib.py              # Shared code
├── Preprocessing       # Code for data filtering and mask generation
├── Diffueraser         # Video model for evaluation
├── stableDiff          # Video model for evaluation
├── wan                 # Video model for evaluation
├── Evaluation          # Code for FOV, FVD and CLIP metrics
└── EvaluationOutputs   # .csv files with results from Evaluation
```
