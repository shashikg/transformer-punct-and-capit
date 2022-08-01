# Taken from https://github.com/xashru/punctuation-restoration
# Slightly modified to suit my needs

import numpy as np

def augment_none(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style, **kwargs):
    """
    apply no augmentation
    """
    x_aug.append(x[i])
    y_aug.append(y[i])
    y_mask_aug.append(y_mask[i])


def augment_substitute(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style, **kwargs):
    """
    replace a token with a random token or the unknown token
    """
    if kwargs['sub_style'] == 'rand':
        x_aug.append(np.random.randint(tokenizer.vocab_size))
    else:
        x_aug.append(token_style['UNK'])
        
    y_aug.append(y[i])
    y_mask_aug.append(y_mask[i])


def augment_insert(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style, **kwargs):
    """
    insert the unknown token before this token
    """
    x_aug.append(token_style['UNK'])
    y_aug.append(0)
    y_mask_aug.append(1)
    x_aug.append(x[i])
    y_aug.append(y[i])
    y_mask_aug.append(y_mask[i])

def augment_delete(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style, **kwargs):
    """
    remove this token i..e, not add in augmented tokens
    """
    return


def augment_all(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style, **kwargs):
    """
    apply substitution with alpha_sub probability, deletion with alpha_del probability and insertion with
    1-(alpha_sub+alpha_sub) probability
    """
    
    r = np.random.rand()
    if r < kwargs['alpha_sub']:
        augment_substitute(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style, **kwargs)
    elif r < kwargs['alpha_sub'] + kwargs['alpha_del']:
        augment_delete(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style, **kwargs)
    else:
        augment_insert(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style, **kwargs)

# supported augmentation techniques
AUGMENTATIONS = {
    'none': augment_none,
    'substitute': augment_substitute,
    'insert': augment_insert,
    'delete': augment_delete,
    'all': augment_all
}
