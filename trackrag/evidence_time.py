"""Public text references shared by evidence export and answer validation."""
import re


def _references(text,kind):
    values=[]
    pattern=rf'\b{kind}s?\s*(\d+(?:\s*(?:,\s*(?:and\s*)?|and|to|through|[-–/;])\s*(?:{kind}s?\s*)?\d+)*)'
    for match in re.finditer(pattern,text,re.I):
        for position,number in enumerate(re.finditer(r'\d+',match.group(1))):
            after=text[match.start(1)+number.end():]
            if position and re.match(r'\s*(?:kilomet(?:er|re)s?|km/?h|mph|met(?:er|re)s?|seconds?|minutes?|hours?)\b',after,re.I):
                break
            values.append(int(number.group()))
    return sorted(set(values))


def requested_frame_ids(text):
    return _references(text,'frame')


def referenced_track_ids(text):
    return _references(text,'track')
