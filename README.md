# Meta-learning In-Context Enables Training-Free Cross Subject Brain Decoding
### Abstract
Visual decoding from brain signals is a key challenge at the intersection of computer vision and neuroscience, requiring methods that bridge neural representations and computational models of vision. A field-wide goal is to achieve generalizable, cross-subject models. A major obstacle towards this goal is the substantial variability in neural representations across individuals, which has so far required training bespoke models or fine-tuning separately for each subject. To address this challenge, we introduce a meta-optimized approach for semantic visual decoding from fMRI that ***generalizes to novel subjects without any fine-tuning***. By simply conditioning on a small set of image-brain activation examples from the new individual, our model rapidly infers their unique neural encoding patterns to facilitate robust and efficient visual decoding. Our approach is explicitly optimized for in-context learning of the new subject's encoding model and performs decoding by hierarchical inference, inverting the encoder. First, for multiple brain regions, we estimate the per-voxel visual response encoder parameters by constructing a context over multiple stimuli and responses. Second, we construct a context consisting of encoder parameters and response values over multiple voxels to perform aggregated functional inversion. We demonstrate strong cross-subject and cross-scanner generalization across diverse visual backbones without retraining or fine-tuning. Moreover, our approach requires neither anatomical alignment nor stimulus overlap. This work is a critical step towards a generalizable foundation model for non-invasive brain decoding.

### Codebase
* Requirements (in addition to usual python stack)
  * Numpy
  * h5py
  * nibabel (install from pip)
  * pycortex (install from pip) -- only for brain 3D visualization
  * PyTorch 2.0.1 or above
  * Hugginface Diffusers (and Transformers and accelerate)

Notes:

1. We trained a decoder, BrainCodec for the higher visual cortex decoding.

2. In fMRI terminology, decoder means a function that predicts seen stimuli from brain activations. (network takes in brain activations, and predicts an image).

3. Model checkpoint is provided on Google Drive for 1,2,5 as training subjects, 7 as unseen subject [here](https://drive.google.com/file/d/1hoj2BYgKYs_s70-hy2knMCM6ap3GOl0D/view?usp=drive_link).

The organization of the code is as such:

```
|- models 
  |- models.py (our braincodec model implementation)
|- dataset 
  |- multisubj_dataset.py (this is where we store the dataloader for NSD subject)
|- utils
  |- helper.py (some useful tools to smooth training and testing)
|- finetuning.py (finetuning script, by default use 1,2,5 as training subjects, 7 as unseen subject)
|- pretraining.py (pretraining script)

```


