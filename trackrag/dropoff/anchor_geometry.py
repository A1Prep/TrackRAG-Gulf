"""Learn destination displacement and extent from *training* landmark/region pairs."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np

from .contracts import validate_box


def terms(text):
    words = re.findall(r"[a-z]+", text.lower())
    return set(words) | {" ".join(pair) for pair in zip(words, words[1:])}


def features(box, size, words, vocabulary):
    width, height = size
    x1, y1, x2, y2 = np.array(box) / [width, height, width, height]
    scale = np.array([max(x2-x1, .06), max(y2-y1, .07)])
    base = np.array([(x1+x2)/2, y2])
    geometry = [1., base[0]-.5, base[1]-.65, x2-x1, y2-y1]
    language = [float(t in words) for t in vocabulary]
    return np.array(geometry + language), base, scale


class AnchorGeometry:
    def __init__(self, data):
        if data.get("format") != "anchor-relative-regression-v1":
            raise ValueError("Unknown destination checkpoint format")
        self.vocabulary = data["vocabulary"]
        self.coefficients = np.asarray(data["coefficients"], dtype=float)
        if self.coefficients.shape != (5 + len(self.vocabulary), 6) or not np.isfinite(self.coefficients).all():
            raise ValueError("Invalid destination regression coefficients")
        self.bank = data.get("retrieved_offsets")
        self.pair_regression = data.get("pair_regression")
        if self.bank is not None:
            self.ids = self.bank["case_ids"]
            self.retrieval_vocab = self.bank["vocabulary"]
            self.idf = np.asarray(self.bank["idf"])
            n, v = len(self.ids), len(self.retrieval_vocab)
            self.text_vectors = np.zeros((n,v))
            for i, indexes in enumerate(self.bank["term_indexes"]):
                self.text_vectors[i,indexes] = self.idf[indexes]
            self.text_vectors /= np.maximum(np.linalg.norm(self.text_vectors,axis=1,keepdims=True),1.e-10)
            self.locations = np.asarray(self.bank["landmark_features"])
            self.offsets = np.asarray(self.bank["relative_points"])
            if (not n or len(set(self.ids)) != n or self.idf.shape != (v,) or len(self.bank["term_indexes"]) != n
                    or self.locations.shape != (n,4) or self.offsets.shape != (n,2) or
                    not all(np.isfinite(m).all() for m in (self.text_vectors,self.locations,self.offsets))):
                raise ValueError("Invalid retrieved destination checkpoint")

    @classmethod
    def load(cls, path, expected_sha=None):
        payload = Path(path).read_bytes()
        digest = sha256(payload).hexdigest()
        if expected_sha is not None and expected_sha != digest:
            raise ValueError("Destination checkpoint SHA256 differs from configuration")
        model = cls(json.loads(payload))
        model.sha256 = digest
        return model

    @staticmethod
    def fit(examples, use_text=True):
        """Fit six robust relative-coordinate outputs. Labels never enter predict()."""
        counts = Counter(t for e in examples for t in terms(e["text"]))
        vocab = sorted(t for t, n in counts.items() if n >= 5) if use_text else []
        x, y = [], []
        for e in examples:
            f, base, scale = features(e["anchor"], e["size"], terms(e["text"]), vocab)
            p = np.array(e["point"])/e["size"]
            b = np.array(e["box"]) / np.tile(e["size"], 2)
            x.append(f)
            y.append(np.r_[(p-base)/scale, np.log(np.maximum(p-b[:2], .003)/scale),
                           np.log(np.maximum(b[2:]-p, .003)/scale)])
        x, y = np.array(x), np.array(y)
        penalty = np.diag([.001, 3., 3., 3., 3.] + [18.]*len(vocab))
        columns = []
        for target in y.T:
            weights = np.ones(len(target))
            for _ in range(5):
                weighted = x.T*weights
                coef = np.linalg.solve(weighted@x + penalty, weighted@target)
                residual = np.abs(x@coef - target)
                weights = np.minimum(1., max(.15, 1.5*np.median(residual))/np.maximum(residual, 1.e-8))
            columns.append(coef)
        return {"format": "anchor-relative-regression-v1", "vocabulary": vocab,
                "coefficients": np.column_stack(columns).tolist(), "training_count": len(y)}

    def predict(self, boxes, size, text):
        point, box, _ = self.predict_with_evidence(boxes,size,text)
        return point, box

    @staticmethod
    def fit_retrieval(examples, fitted):
        counts = Counter(t for e in examples for t in terms(e["text"]))
        vocabulary = sorted(t for t,n in counts.items() if n >= 2)
        term_id = {word:i for i,word in enumerate(vocabulary)}
        bank = {"case_ids":[], "vocabulary":vocabulary, "term_indexes":[], "landmark_features":[],
                "relative_points":[], "idf":[np.log(1+len(examples)/counts[t]) for t in vocabulary]}
        for e in examples:
            _, base, scale = features(e["anchor"],e["size"],set(),[])
            bank["case_ids"].append(e["id"])
            bank["term_indexes"].append(sorted(term_id[t] for t in terms(e["text"]) if t in term_id))
            bank["landmark_features"].append([*base,*np.log(scale)])
            bank["relative_points"].append(((np.asarray(e["point"])/e["size"]-base)/scale).tolist())
        return {**fitted,"retrieved_offsets":bank}

    def _retrieve(self, text, base, scale):
        tokens = terms(text)
        query = np.array([t in tokens for t in self.retrieval_vocab], dtype=float)*self.idf
        query /= max(np.linalg.norm(query),1.e-10)
        anchor = np.r_[base,np.log(scale)]
        distance = np.sum(((self.locations-anchor)/[.35,.2,1,1])**2,axis=1)
        scores = 4*(self.text_vectors@query)-.5*distance
        top = np.argsort(-scores,kind="stable")[:35]
        w = np.exp(scores[top]-scores[top].max())
        w /= w.sum()
        offsets = self.offsets[top]
        delta = (offsets[:,None,:]-offsets[None,:,:])/[.65,.4]
        similarity = np.exp(-.5*np.sum(delta**2,axis=2))
        mode_w = similarity[(similarity@w).argmax()]*w
        mode_w /= mode_w.sum()
        receipt = [{"training_case":self.ids[i],"weight":float(a)} for i,a in zip(top,mode_w)]
        return mode_w@offsets, receipt

    def predict_with_evidence(self, boxes, size, text):
        if len(boxes) not in (1, 2):
            raise ValueError("Expected one or two landmark boxes")
        if len(boxes) == 2 and self.pair_regression is not None:
            from .pair_geometry import predict
            pair = predict(self.pair_regression,boxes,size)
            if pair is not None:
                return *pair, []
        boxes = np.array([validate_box(b, *size) for b in boxes])
        envelope = [*boxes[:, :2].min(axis=0), *boxes[:, 2:].max(axis=0)]
        x, base, scale = features(envelope, size, terms(text), self.vocabulary)
        values = x@self.coefficients
        point = np.clip(base+scale*values[:2], 0, 1-1/np.array(size))
        distances = np.tile(scale, 2)*np.exp(np.clip(values[2:], -6, 4))
        box = np.clip(np.r_[point, point]+distances*[-1, -1, 1, 1], 0, 1)
        retrieved = []
        if self.bank is not None:
            offset, retrieved = self._retrieve(text,base,scale)
            new_point = np.clip(base+scale*offset,0,1-1/np.asarray(size))
            box = np.clip(box+np.tile(new_point-point,2),0,1)
            point = new_point
        return (point*size).tolist(), (box*np.tile(size, 2)).tolist(), retrieved
