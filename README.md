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

## 2 Fine-Tuning
- To fine-tune the SEEM model, we employed adapters.
- During this fine-tuning phase, we maintained the SEEM-Decoder in a frozen state, except for the layerNorms. As a result, the only trainable components included the adapters and layerNorms within the SEEM-Decoder.
- Specifically, we placed one adapter after cross-attention, another following self-attention, and a third after the MLP.
- Additionally, within the prediction head, we introduced an adapter running parallel to the mask_embedding, which functions as an MLP.

