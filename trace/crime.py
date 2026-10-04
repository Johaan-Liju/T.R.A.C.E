"""Frozen video features and attention pooling for video-level crime classification."""
from pathlib import Path

import numpy as np
import torch

from trace.crime_data import CLASSES, file_hash
from trace.violence import RECIPE, encode, load_encoder

FORMAT = 'trace-crime-attention-v1'
LIMITATIONS = [
    'Research baseline with whole-video labels; no validated event localization.',
    'Sparse sampled windows can miss brief events; attention is not an event probability.',
    'Class scores are not calibrated probabilities or evidence of unlawful intent.',
    'New-camera accuracy, live speed and false alerts per hour are unmeasured.',
]


class CrimeHead(torch.nn.Module):
    """One bag per video: learn window importance from video labels (attention MIL)."""
    def __init__(self):
        super().__init__()
        self.attention = torch.nn.Sequential(torch.nn.Linear(512, 128), torch.nn.Tanh(),
                                             torch.nn.Linear(128, 1))
        self.classifier = torch.nn.Linear(512, len(CLASSES))

    def forward(self, features):
        attention = self.attention(features).squeeze(-1).softmax(dim=-1)
        pooled = (features * attention.unsqueeze(-1)).sum(dim=-2)
        return self.classifier(pooled), attention


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def metrics(labels, predictions):
    matrix = np.zeros((3, 3), dtype=int)
    for actual, predicted in zip(labels, predictions):
        matrix[int(actual), int(predicted)] += 1
    per_class = {}
    for i, name in enumerate(CLASSES):
        precision = float(matrix[i, i] / max(1, matrix[:, i].sum()))
        recall = float(matrix[i, i] / max(1, matrix[i].sum()))
        per_class[name] = dict(precision=precision, recall=recall,
                              f1=2 * precision * recall / max(1e-12, precision + recall),
                              support=int(matrix[i].sum()))
    return dict(accuracy=float(matrix.trace() / max(1, matrix.sum())),
                macro_f1=float(np.mean([r['f1'] for r in per_class.values()])),
                balanced_accuracy=float(np.mean([r['recall'] for r in per_class.values()])),
                per_class=per_class, confusion_matrix=matrix.tolist(),
                matrix_order=CLASSES, matrix_axes='rows=actual, columns=predicted')


def load_checkpoint(path, encoder):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if (checkpoint.get('format') != FORMAT or checkpoint.get('classes') != CLASSES
            or checkpoint.get('recipe') != RECIPE):
        raise ValueError('Not a compatible TRACE three-class crime checkpoint.')
    if checkpoint['encoder_sha256'] != file_hash(encoder):
        raise ValueError('Encoder weights differ from the training checkpoint.')
    head = CrimeHead()
    head.load_state_dict(checkpoint['head'])
    return head.eval(), checkpoint


def predict(source, checkpoint, encoder, device='cpu'):
    head, saved = load_checkpoint(checkpoint, encoder)
    features, metadata = encode(load_encoder(encoder, device), source, saved['clips'], device)
    with torch.inference_mode():
        logits, attention = head(((features - saved['mean']) / saved['std']).unsqueeze(0))
        scores = logits.softmax(-1)[0].tolist()
    for window, weight in zip(metadata['sampled_windows'], attention[0].tolist()):
        window['attention_weight'] = weight
    return dict(video=str(source), prediction=CLASSES[int(np.argmax(scores))],
                class_scores=dict(zip(CLASSES, scores)), **metadata, limitations=LIMITATIONS)
