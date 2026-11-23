"""Compare horizontal image motion using only the observations in cited frames."""
import re

from trackrag.evidence_time import referenced_track_ids, requested_frame_ids
from trackrag.retrieval import significant_image_displacement


_VERB = r'(?:mov(?:e[sd]?|ing)|shift(?:s|ed|ing)?|drift(?:s|ed|ing)?|travel(?:s|l?ed|l?ing)?)'
_DIRECTION = r'(left|right)(?:wards?)?\b'
_CLAIM = re.compile(rf'\b{_VERB}\s+(?:(?:slightly|steadily|horizontally)\s+)?'
                    rf'(?:(?:from\s+)?(?:the\s+)?(?:left|right)(?: side)?[\s–-]+(?:towards?|to)[\s–-]+|(?:(?:towards?|to)\s+)?)'
                    rf'(?:the\s+)?(?:(?:image|screen)[ -])?{_DIRECTION}', re.I)
_SHORT = re.compile(rf'(?:(?:towards?|to)\s+)?(?:the\s+)?(?:(?:image|screen)[ -])?{_DIRECTION}[.!]?', re.I)


def asks_horizontal_motion(text):
    return bool(re.search(rf'\b(?:{_VERB}|movement|motion|direction)\b', text, re.I)
                and re.search(r'\b(?:left(?:wards?)?|right(?:wards?)?|horizontal(?:ly)?|which (?:direction|side)|what direction)\b', text, re.I))


def horizontal_motion(items, track_id, frames=()):
    """Get centre positions for one identity on the original image, in observation order."""
    positions = {}
    bounds = (min(frames), max(frames)) if len(frames) >= 2 else None
    for item in items:
        for image in item['images']:
            frame = image['frame_id']
            if bounds and not bounds[0] <= frame <= bounds[1]:
                continue
            size = image.get('privacy', {}).get('original_size_wh')
            if not size:
                continue
            for row in image.get('observations', ()):
                if row['track_id'] == track_id:
                    left, _, right, _ = row['bbox_xyxy']
                    centre = (left + right) / (2 * size[0])
                    if positions.setdefault(frame, centre) != centre:
                        raise ValueError('Cited observations disagree on a track position in the same frame')
    ordered = sorted(positions)
    if len(ordered) < 2:
        return None
    xs = [positions[frame] for frame in ordered]
    delta = xs[-1] - xs[0]
    if not significant_image_displacement(delta):
        return None
    if any((current - prior) * delta < 0 and significant_image_displacement(current - prior)
           for prior, current in zip(xs, xs[1:])):
        return None
    return {'track_id': track_id, 'frames': ordered, 'centre_x_fraction': [round(x, 5) for x in xs],
            'image_direction': 'right' if delta > 0 else 'left'}


def verify_horizontal_claim(question, answer, cited):
    """Check a direct, unqualified image-direction claim about one identifiable track."""
    if not asks_horizontal_motion(question):
        return
    if re.search(r"\b(?:not|never|may|might|could|uncertain|relative|turn(?:s|ed|ing)?|compared|initially|then)\b", answer, re.I):
        return
    claimed = {match[1].lower() for match in _CLAIM.finditer(answer)}
    short = _SHORT.fullmatch(answer.strip())
    if short:
        claimed.add(short[1].lower())
    tracks = set(referenced_track_ids(question)) | set(referenced_track_ids(answer))
    if not tracks and len(cited) == 1 and cited[0]['interval']['track_id'] is not None:
        tracks = {cited[0]['interval']['track_id']}
    if len(claimed) != 1 or len(tracks) != 1:
        return
    frames = set(requested_frame_ids(question)) | set(requested_frame_ids(answer))
    motion = horizontal_motion(cited, next(iter(tracks)), frames)
    if motion and motion['image_direction'] not in claimed:
        raise ValueError("The answer's movement direction contradicts the cited track positions.")
