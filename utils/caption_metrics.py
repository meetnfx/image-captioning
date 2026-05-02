"""
Image captioning metrics: BLEU-1..4, METEOR, CIDEr, and ROUGE-L (pycocoevalcap).

References use simple whitespace + lowercasing (no Stanford CoreNLP).
METEOR needs Java + meteor-1.5.jar shipped with pycocoevalcap; if that fails,
METEOR is reported as NaN.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple, Union

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge


def _patch_meteor_destructor_safe() -> None:
    """Avoid AttributeError in Meteor.__del__ when __init__ fails before self.lock is set."""
    try:
        from pycocoevalcap.meteor import meteor as meteor_mod

        cls = meteor_mod.Meteor
        if getattr(cls, "_del_patch_applied", False):
            return
        orig = cls.__del__

        def safe_del(self):
            if not hasattr(self, "lock"):
                return
            try:
                orig(self)
            except Exception:
                pass

        cls.__del__ = safe_del  # type: ignore[method-assign]
        cls._del_patch_applied = True  # type: ignore[attr-defined]
    except Exception:
        pass


def simple_tokenize(s: str) -> str:
    return " ".join(str(s).lower().replace("\n", " ").strip().split())


def build_gts_and_res(
    references_by_image_id: Mapping[int, Sequence[str]],
    predictions: Sequence[Mapping[str, Union[int, str]]],
) -> Tuple[Dict[int, List[str]], Dict[int, List[str]]]:
    gts: Dict[int, List[str]] = {}
    res: Dict[int, List[str]] = {}
    for row in predictions:
        iid = int(row["image_id"])
        if iid not in references_by_image_id:
            raise KeyError(f"image_id {iid} missing from reference split")
        gts[iid] = [simple_tokenize(t) for t in references_by_image_id[iid]]
        res[iid] = [simple_tokenize(str(row["caption"]))]
    if set(gts.keys()) != set(res.keys()):
        raise ValueError("gts and res keys must match exactly")
    return gts, res


def compute_caption_metrics(
    references_by_image_id: Mapping[int, Sequence[str]],
    predictions: Sequence[Mapping[str, Union[int, str]]],
    include_rouge: bool = True,
) -> Dict[str, float]:
    gts, res = build_gts_and_res(references_by_image_id, predictions)

    out: Dict[str, float] = {}

    bleu_scorer = Bleu(4)
    bleu_scores, _ = bleu_scorer.compute_score(gts, res, verbose=0)
    for name, val in zip(["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"], bleu_scores):
        out[name] = float(val)

    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, res)
    out["CIDEr"] = float(cider_score)

    if include_rouge:
        rouge_scorer = Rouge()
        rouge_score, _ = rouge_scorer.compute_score(gts, res)
        out["ROUGE_L"] = float(rouge_score)

    try:
        from pycocoevalcap.meteor.meteor import Meteor

        _patch_meteor_destructor_safe()
        meteor = Meteor()
        meteor_score, _ = meteor.compute_score(gts, res)
        out["METEOR"] = float(meteor_score)
    except Exception:
        out["METEOR"] = float("nan")
    # Do not `del meteor`: triggers buggy Meteor.__del__; GC is enough.

    return out
