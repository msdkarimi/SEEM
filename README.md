# 1. Methodology(architecture layers) <br>
    SEEM model Generally consists of two main parts:
    
    - Vision  
    - Language

    Language:
        
        The language architecture comprises a transformer model that employs CLIP as its tokenizer. Its primary function is to calculate embeddings for sentences related to grounding tasks and determine the similarity between a provided visual embedding (such as a combination of image features and grounding sentence) and the embeddings of each class. These class embeddings could represent either the class name or a sentence created through prompt engineering, mirroring the similarity computation process found in CLIP.
    

- sdfsdfsdf
