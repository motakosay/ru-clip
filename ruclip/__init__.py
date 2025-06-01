# -*- coding: utf-8 -*-
import os

from huggingface_hub import hf_hub_download

from . import model, processor, predictor
from .model import CLIP
from .processor import RuCLIPProcessor
from .predictor import Predictor

MODELS = {
    'ruclip-vit-base-patch32-224': dict(
        repo_id='ai-forever/ruclip-vit-base-patch32-224',
        filenames=[
            'bpe.model', 'config.json', 'pytorch_model.bin'
        ]
    ),
    'ruclip-vit-base-patch16-224': dict(
        repo_id='ai-forever/ruclip-vit-base-patch16-224',
        filenames=[
            'bpe.model', 'config.json', 'pytorch_model.bin'
        ]
    ),
    'ruclip-vit-large-patch14-224': dict(
        repo_id='ai-forever/ruclip-vit-large-patch14-224',
        filenames=[
            'bpe.model', 'config.json', 'pytorch_model.bin'
        ]
    ),
    'ruclip-vit-large-patch14-336': dict(
        repo_id='ai-forever/ruclip-vit-large-patch14-336',
        filenames=[
            'bpe.model', 'config.json', 'pytorch_model.bin'
        ]
    ),
    'ruclip-vit-base-patch32-384': dict(
        repo_id='ai-forever/ruclip-vit-base-patch16-384',
        filenames=[
            'bpe.model', 'config.json', 'pytorch_model.bin'
        ]
    ),
    'ruclip-vit-base-patch16-384': dict(
        repo_id='ai-forever/ruclip-vit-base-patch16-384',
        filenames=[
            'bpe.model', 'config.json', 'pytorch_model.bin'
        ]
    ),
}


def load(name, device='cpu', cache_dir='/tmp/ruclip', use_auth_token=None):
    """Load a ruCLIP model"""
    assert name in MODELS, f'All models: {MODELS.keys()}'
    config = MODELS[name]
    repo_id = config['repo_id']
    cache_dir = os.path.join(cache_dir, name)

    for filename in config['filenames']:
        # ✅ Correct usage without invalid URL
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            force_filename=filename,
            local_files_only=True,
            use_auth_token=use_auth_token
        )

    clip = CLIP.from_pretrained(cache_dir).eval().to(device)
    clip_processor = RuCLIPProcessor.from_pretrained(cache_dir)
    return clip, clip_processor


__all__ = ['processor', 'model', 'predictor', 'CLIP', 'RuCLIPProcessor', 'Predictor', 'MODELS', 'load']
__version__ = '0.0.2'
