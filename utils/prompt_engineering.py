import numpy as np


def get_prompt_templates():
    prompt_templates = [
        '{}.',
        'A photo of the extensive {}.',
        'A blurry photo of a {}.',
        'A close-up photo of the {} with visible damage.',
        'A photo capturing the aftermath of {}.',
        'A clear photo showing the {} damage.',
        'A snapshot of the {} in poor condition.',
        'An image showcasing the extent of {}.',
        'A zoomed-in photo revealing the {} damage.',
        'A snapshot of the {} with evident damage.',
        'An image depicting the severe {}.',
        'A photo revealing the cause of {}.',
        'A snapshot capturing the onset of {}.',
        'An image illustrating the effects of {}.',
        'A close-up of the {} with water damage.',
        'A photo showing the progression of {}.',
        'An up-close view of the {}.',
        'A photo of the {} damage.',
        'An image showing {} deterioration.',
        'A close-up of the {}.',
        'A snapshot of {} damage.',
        'An image revealing {}.',
        'A clear picture of the {}.',
        'A photo capturing {}.',
        'An image depicting {} damage.',
        'A close-up shot of the {}.',
        'A snapshot revealing {} deterioration.',
        'A photo of wet {}.',
        'An image showing {} from water.',
        'A close-up of {} damage.',
        'A snapshot of {} affected by moisture.',
        'An image revealing water-damaged {}.',
        'A clear picture of {} soaked with water.',
        'A photo capturing the wet {}.',
        'An image depicting {} damage from dampness.',
        'A close-up shot of {} affected by water.',
        'A snapshot revealing the damp {}.',
        'A snapshot of {} soaked with water.',
        'An image revealing water-drenched {}.',
        'A clear picture of {} affected by water.',
        'A photo capturing the wetness of {}.',
        'An image depicting {} damage from water.',
        'A close-up shot of {} dampened by water.',
        'A snapshot revealing the wet {}.'
    ]
    prompt_templates_seem = [
        '{}.',
        'a photo of a {}.',
        'a bad photo of a {}.',
        'a photo of many {}.',
        'a sculpture of a {}.',
        'a photo of the hard to see {}.',
        'a low resolution photo of the {}.',
        'a rendering of a {}.',
        'graffiti of a {}.',
        'a bad photo of the {}.',
        'a cropped photo of the {}.',
        'a tattoo of a {}.',
        'the embroidered {}.',
        'a photo of a hard to see {}.',
        'a bright photo of a {}.',
        'a photo of a clean {}.',
        'a photo of a dirty {}.',
        'a dark photo of the {}.',
        'a drawing of a {}.',
        'a photo of my {}.',
        'the plastic {}.',
        'a photo of the cool {}.',
        'a close-up photo of a {}.',
        'a black and white photo of the {}.',
        'a painting of the {}.',
        'a painting of a {}.',
        'a pixelated photo of the {}.',
        'a sculpture of the {}.',
        'a bright photo of the {}.',
        'a cropped photo of a {}.',
        'a plastic {}.',
        'a photo of the dirty {}.',
        'a jpeg corrupted photo of a {}.',
        'a blurry photo of the {}.',
        'a photo of the {}.',
        'a good photo of the {}.',
        'a rendering of the {}.',
        'a {} in a video game.',
        'a photo of one {}.',
        'a doodle of a {}.',
        'a close-up photo of the {}.',
        'the origami {}.',
        'the {} in a video game.',
        'a sketch of a {}.',
        'a doodle of the {}.',
        'a origami {}.',
        'a low resolution photo of a {}.',
        'the toy {}.',
        'a rendition of the {}.',
        'a photo of the clean {}.',
        'a photo of a large {}.',
        'a rendition of a {}.',
        'a photo of a nice {}.',
        'a photo of a weird {}.',
        'a blurry photo of a {}.',
        'a cartoon {}.',
        'art of a {}.',
        'a sketch of the {}.',
        'a embroidered {}.',
        'a pixelated photo of a {}.',
        'itap of the {}.',
        'a jpeg corrupted photo of the {}.',
        'a good photo of a {}.',
        'a plushie {}.',
        'a photo of the nice {}.',
        'a photo of the small {}.',
        'a photo of the weird {}.',
        'the cartoon {}.',
        'art of the {}.',
        'a drawing of the {}.',
        'a photo of the large {}.',
        'a black and white photo of a {}.',
        'the plushie {}.',
        'a dark photo of a {}.',
        'itap of a {}.',
        'graffiti of the {}.',
        'a toy {}.',
        'itap of my {}.',
        'a photo of a cool {}.',
        'a photo of a small {}.',
        'a tattoo of the {}.',
    ]
    return prompt_templates

def prompt_engineering(classnames, topk=1, suffix='.'):
    prompt_templates = get_prompt_templates()
    temp_idx = np.random.randint(min(len(prompt_templates), topk))

    if isinstance(classnames, list):
        classname = random.choice(classnames)
    else:
        classname = classnames

    return prompt_templates[temp_idx].replace('.', suffix).format(classname.replace(',', '').replace('+', ' '))