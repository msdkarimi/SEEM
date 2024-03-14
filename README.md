## 1. Methodology(architecture layers) <br>
    SEEM model Generally consists of two main parts:

    - Language 
    - Vision

### 1-1 Language
    
- The language architecture comprises a transformer model that employs CLIP as its tokenizer. Its primary function is to calculate embeddings for sentences related to grounding tasks and determine the similarity between a provided visual embedding (such as a combination of image features and grounding sentence) and the embeddings of each class. These class embeddings could represent either the class name or a sentence created through prompt engineering, mirroring the similarity computation process found in CLIP.<br>
- In original SEEM language encoder is kept frozen.


### 1-2 Vision 

- The vision component comprises three modules: the vision backbone, the pixel decoder, and the SEEM decoder.


#### 1-2-1) Vision backbone: 

- The FocalNet module, a Feature Pyramid Network, generates multiscale features of the input image. 
- In original SEEM this module is kept frozen. 

#### 1-2-2) Pixel-Decoder:

- The main role of this component is to combine the multiscale features derived from the input image by the backbone and produce a feature termed as the mask feature. Subsequently, both this mask feature and the output from the backbone, which consists of multiscale features, are forwarded to the SEEM decoder. 
- In original SEEM this module is kept frozen.

#### 1-2-3) SEEM-Decoder:

- SEEM-Decoder is generally a transformer, which generally takes three type of inputs to perform the segmentation task:<br>

  - mask feature
  - multiscale feature
  - embedding of grounding sentences
  
- The SEEM transformer comprises four primary components: cross-attention, self-attention, MLP, and prediction head.
- In original SEEM this module is trainable.

## 2. Fine-Tuning
- To fine-tune the SEEM model, we employed adapters.
- During this fine-tuning phase, we maintained the SEEM-Decoder in a frozen state, except for the layerNorms. As a result, the only trainable components included the adapters and layerNorms within the SEEM-Decoder.
- Specifically, we placed one adapter after cross-attention, another following self-attention, and a third after the MLP.
- Additionally, within the prediction head, we introduced an adapter running parallel to the mask_embedding, which functions as an MLP.

## 3. Dataset
    
- Firstly, it's essential to assign a color to each category and indicate whether it represents a "thing" or "stuff." "Thing" refers to categories that can be confined within a closed area, such as a ball or a human, while "stuff" denotes categories that cannot be confined within a closed area, such as the sky
- We need to use panoptic-api. By means of this API we can easily convert color of each class to a unique number, this number will be used later in classification part of model to understand if pixel is classified correctly or not
- Another benefit of the panoptic API is that if we have both panoptic ground truth images and a panoptic ground truth JSON file, we can easily create semantic segmentation ground truth images via this API.

- To create dataset for training the model, we generally need four kind of data:
  
  - Panoptic segmentation ground truth images
  - The panoptic JSON file simply outlines the segments-info for each image in the training dataset. These segments-info specify the categories and IDs obtained through the panoptic API, there is not any polygons.
  - Semantic segmentation ground truth images
  - The grounding ground truth JSON file is where we define a sentence or sentences for each segmentation(polygon). Later in the SEEM model, these sentences will be encoded into embeddings using a language encoder. These embeddings will then be combined with vision features to create a unified representation.
  - To produce the panoptic ground truth images accurately, we must identify not just the regions of interest but also include other pixel classes. This entails incorporating generic classes like "wall" or "ceiling" into the specification.

## 4. Train

- Inotder to train the model we need to download pretrained weight of original SEEM
- pretrained weights should be located in `datasets/xdecoder_data/pretrained` 

  


