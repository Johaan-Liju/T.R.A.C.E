import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from trace.crime import CrimeHead, load_checkpoint, metrics, predict
from trace.crime_data import CLASSES, file_hash, prepare, read_manifest, relative_video, video_path
from trace.train_crime import extract_features, fit


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class CrimeDataTests(unittest.TestCase):
    def make_dataset(self, root):
        train, test = [], []
        for label, name in enumerate(CLASSES):
            (root / name).mkdir()
            for index in range(6):
                path = root / name / f'{index}.mp4'
                path.write_bytes(f'{name}-{index}'.encode())
                row = {'video_path': f'{name}\\{index}.mp4', 'label': str(label)}
                (test if index == 5 else train).append(row)
        write_csv(root / 'train.csv', train)
        write_csv(root / 'test.csv', test)
        return train, test

    def test_import_portable_paths_preserves_test_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, test = self.make_dataset(root)
            (root / 'burglary/copy.mp4').write_bytes((root / 'burglary/0.mp4').read_bytes())
            train.append({'video_path': 'burglary/copy.mp4', 'label': '0'})
            write_csv(root / 'train.csv', train)
            rows = prepare(root / 'train.csv', root / 'test.csv', root, root / 'prepared')
            self.assertEqual(len(rows), 18)
            self.assertEqual({r['path'] for r in rows if r['split'] == 'test'},
                             {r['video_path'].replace('\\', '/') for r in test})
            self.assertEqual(len([r for r in rows if r['split'] == 'val']), 3)
            self.assertEqual(len(read_manifest(root / 'prepared/manifest.csv')), 18)
            with self.assertRaisesRegex(ValueError, 'Output exists'):
                prepare(root / 'train.csv', root / 'test.csv', root, root / 'prepared')

    def test_cross_split_duplicate_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_dataset(root)
            (root / 'burglary/5.mp4').write_bytes((root / 'burglary/0.mp4').read_bytes())
            with self.assertRaisesRegex(ValueError, 'crosses train/test'):
                prepare(root / 'train.csv', root / 'test.csv', root, root / 'prepared')
            self.assertFalse((root / 'prepared').exists())

    def test_recording_groups_stay_together_and_cross_test_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, test = self.make_dataset(root)
            groups = [{'video_path': r['video_path'],
                       'group': r['video_path'].split('\\')[0] + '/' + ('pair' if r['video_path'].endswith(('0.mp4', '1.mp4')) else r['video_path'])}
                      for r in train + test]
            write_csv(root / 'groups.csv', groups)
            rows = prepare(root / 'train.csv', root / 'test.csv', root, root / 'prepared', root / 'groups.csv')
            for name in CLASSES:
                self.assertEqual(len({r['split'] for r in rows if r['group'] == name + '/pair'}), 1)
            groups[-3]['group'] = groups[0]['group']  # burglary test shares a training recording
            write_csv(root / 'groups.csv', groups)
            with self.assertRaisesRegex(ValueError, 'crosses supplied train/test'):
                prepare(root / 'train.csv', root / 'test.csv', root, root / 'other', root / 'groups.csv')

    def test_paths_and_label_mapping_rejected(self):
        for path in ('../outside.mp4', 'C:\\outside.mp4', '/tmp/a.mp4', 'normal/../a.mp4'):
            with self.assertRaises(ValueError):
                relative_video(path)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, _ = self.make_dataset(root)
            with self.assertRaisesRegex(ValueError, 'Missing video'):
                video_path(root, 'normal/missing.mp4')
            train[0]['label'] = '1'
            write_csv(root / 'train.csv', train)
            with self.assertRaisesRegex(ValueError, 'Folder and label disagree'):
                prepare(root / 'train.csv', root / 'test.csv', root, root / 'prepared')


class TinyEncoder(torch.nn.Module):
    """Cheap test encoder; preserves actual video decoding and feature flow."""
    def forward(self, tensor):
        values = tensor.mean(dim=(2, 3, 4))
        return values.repeat(1, 171)[:, :512]


class CrimeTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self):
        generator = torch.Generator().manual_seed(11)
        features = torch.randn(18, 4, 512, generator=generator) * .05
        labels = torch.tensor([0, 1, 2] * 6)
        for index, label in enumerate(labels):
            features[index, 2, int(label) * 10:int(label) * 10 + 10] += 4
        return features, labels, ['train'] * 9 + ['val'] * 6 + ['test'] * 3

    def test_metrics_expose_majority_baseline(self):
        result = metrics([0] * 100 + [1] * 950 + [2] * 50, [1] * 1100)
        self.assertAlmostEqual(result['accuracy'], 950 / 1100)
        self.assertEqual(result['per_class']['burglary']['recall'], 0.)
        self.assertEqual(result['per_class']['vandalism']['recall'], 0.)
        self.assertAlmostEqual(result['balanced_accuracy'], 1 / 3)

    def test_resume_matches_uninterrupted_and_test_does_not_select_model(self):
        features, labels, splits = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            full, resumed, changed = [Path(tmp) / name for name in ('full', 'resumed', 'changed')]
            for directory in (full, resumed, changed):
                directory.mkdir()
            kwargs = dict(metadata={'clips': 4, 'encoder_sha256': 'test'}, patience=50)
            report, _ = fit(features, labels, splits, full, epochs=8, **kwargs)
            fit(features, labels, splits, resumed, epochs=3, **kwargs)
            resumed_report, _ = fit(features, labels, splits, resumed, epochs=8, resume=True, **kwargs)
            self.assertEqual(report, resumed_report)
            a = torch.load(full / 'best.pt', weights_only=True)
            b = torch.load(resumed / 'best.pt', weights_only=True)
            for key in a['head']:
                torch.testing.assert_close(a['head'][key], b['head'][key], rtol=0, atol=0)
            altered = features.clone()
            altered[-3:] = 999
            fit(altered, labels, splits, changed, epochs=8, evaluate_test=True, **kwargs)
            c = torch.load(changed / 'best.pt', weights_only=True)
            for key in a['head']:
                torch.testing.assert_close(a['head'][key], c['head'][key], rtol=0, atol=0)
            self.assertNotIn('test', report['metrics'])
            torch.testing.assert_close(a['mean'], features[:9].mean(dim=(0, 1)))

    def test_decode_cache_train_predict_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for split in ('train', 'val', 'test'):
                for label, name in enumerate(CLASSES):
                    (root / name).mkdir(exist_ok=True)
                    path = root / name / f'{split}.avi'
                    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 15, (64, 48))
                    self.assertTrue(writer.isOpened())
                    for frame_id in range(4):
                        frame = np.zeros((48, 64, 3), dtype=np.uint8)
                        frame[:, :, label] = 80 + frame_id * 20
                        writer.write(frame)
                    writer.release()
                    rows.append(dict(path=path.relative_to(root).as_posix(), sha256=file_hash(path), label=str(label), split=split))
            encoder = root / 'encoder.pth'
            encoder.write_bytes(b'test-only weights')
            with patch('trace.train_crime.load_encoder', return_value=TinyEncoder()):
                features = extract_features(rows, root, encoder, root / 'cache', 2, 'cpu')
            with patch('trace.train_crime.load_encoder', side_effect=AssertionError('Should use cache')):
                cached = extract_features(rows, root, encoder, root / 'cache', 2, 'cpu')
            torch.testing.assert_close(features, cached)
            output = root / 'run'
            output.mkdir()
            _, scores = fit(features, torch.tensor([int(r['label']) for r in rows]),
                            [r['split'] for r in rows], output,
                            {'encoder_sha256': file_hash(encoder), 'clips': 2}, epochs=3, evaluate_test=True)
            with patch('trace.crime.load_encoder', return_value=TinyEncoder()):
                result = predict(root / rows[-3]['path'], output / 'best.pt', encoder)
            np.testing.assert_allclose(list(result['class_scores'].values()), scores['test'][0], atol=1e-6)
            self.assertAlmostEqual(sum(w['attention_weight'] for w in result['sampled_windows']), 1, places=5)
            encoder.write_bytes(b'wrong encoder')
            with self.assertRaisesRegex(ValueError, 'Encoder weights differ'):
                load_checkpoint(output / 'best.pt', encoder)


if __name__ == '__main__':
    unittest.main()
