"""Public text references shared by evidence export and answer validation."""
import re


def _references(text,kind):
    values=[]
    qualifier = r'(?:ids?|numbers?|nos?\.?)'
    label = rf'{kind}s?(?:[\s_-]*{qualifier})?'
    if kind == 'track':
        label = rf"(?:{label}|tracker(?:['’]s)?[\s_-]+{qualifier}|tracked\s+(?:as|under|with)\s+(?:(?:track\s+)?ids?|(?=#)))"
    marker = r'\s*[:=]?\s*#?\s*'
    separator = r'(?:,\s*(?:(?:and|or)\s+)?|\b(?:and|or|to|through)\b|[-–/;])'
    integer = r'\d+(?!\d|\.\d)'
    next_number = rf'\s*{separator}\s*(?:(?:{label}|{qualifier}){marker}|{marker}){integer}'
    pattern = rf'\b{label}{marker}({integer}(?:{next_number})*)'
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
