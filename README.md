# microbleed-detection
Automated Detection of Cerebral Microbleeds (CMBs) on MR images using Knowledge Distillation Framework

Code for implementation of automated tool for CMB detection using knowledge detection.

### Preprint link: https://www.medrxiv.org/content/10.1101/2021.11.15.21266376v1.full.pdf

This is a beta release of the code for CMB detection. Any issues please contact: vaanathi@iisc.ac.in.

#### Software versions used for truenet:

Python 3.5.2

PyTorch 1.2.0

#### Method:
<img
src="images/main_architecture_final.png"
alt="Candidate detection and discrimination steps for CMB detection."
/>

For the initial CMB candidate detection, in addition to the intensity characteristics, we use the radial symmetry property of CMBs as input along with the preprocessed input image. We used a combination of weighted cross-entropy (CE) and Dice loss functions. In candidate discrimination step, we use a student-teacher framework for classifying true CMB candidates from FPs. The teacher model uses a multi-tasking architecture consisting of three parts: 

(1) feature extractor (Tf ) 
(2) voxel-wise CMB segmentor (Ts) 
(3) patch-level CMB classifier (Tc)

The student model consists of the feature extractor and patch-level classifier parts (Tf + Tc) of the teacher model. We trained the student model in an offline manner using response-based knowledge distillation (KD). 

#### Scripts

Data preprocessing
Step 1: Candidate detection: Training (supervised)
Step 2: Teacher model training 
Step 2: Student model training
Step 3: Candidate postprocessing

#### Input formats and time taken:
Currently nifti files supported and any single modality (T2* GRE/SWI/QSM) is sufficient. Similar file types supported by preprocessing codes too.

Currently the implementation takes <5mins/scan for detecting CMBs.


Also refer: https://www.frontiersin.org/articles/10.3389/fninf.2021.777828/full

#### Citation:

@article{sundaresan2021automated,
  title={Automated Detection of Cerebral Microbleeds on MR images using Knowledge Distillation Framework},
  author={Sundaresan, Vaanathi and Arthofer, Christoph and Zamboni, Giovanna and Murchison, Andrew G and Dineen, Robert A and Rothwell, Peter M and Auer, Dorothee P and Wang, Chaoyue and Miller, Karla L and Tendler, Benjamin C and others},
  journal={medRxiv},
  pages={2021--11},
  year={2021},
  publisher={Cold Spring Harbor Laboratory Press}
}
