## File Directory

This directory contains auxiliary files used during training and inference.

### Dictionary Files
- **english_words.txt**
- **vn_words.txt**

These `.txt` files contain dictionaries of words used during training.  
Each line corresponds to a word that can be sampled as training text.

If you want to train the model for another language, simply replace or modify these files with a dictionary corresponding to the target language.

### Font File
- **unifont.pickle**

This file contains the font information used to render text images.  
It serves as the initial font template for generating the **query samples**.

The font file provides a consistent visual style when synthesizing the first query images during training or inference.