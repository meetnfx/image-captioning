"""
Image captioning metrics: BLEU-1..4, METEOR, CIDEr, and ROUGE-L (pycocoevalcap).

References use simple whitespace + lowercasing (no Stanford CoreNLP).
METEOR needs Java + meteor-1.5.jar shipped with pycocoevalcap; if that fails,
METEOR is reported as NaN.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge


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

    meteor: Optional[Any] = None
    try:
        from pycocoevalcap.meteor.meteor import Meteor

        meteor = Meteor()
        meteor_score, _ = meteor.compute_score(gts, res)
        out["METEOR"] = float(meteor_score)
    except Exception:
        out["METEOR"] = float("nan")
    finally:
        if meteor is not None:
            try:
                del meteor
            except Exception:
                pass

    return out
