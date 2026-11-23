"""Match passenger wording to visible attributes of independently verified curbside points."""
from itertools import permutations
import math
from numbers import Real
import re
import unicodedata

from . import bay_intent


RELATION = re.compile(r"\b(between|in front of|outside of|outside|next to|alongside|beside|by|near|at)\b")
SIDE = re.compile(r"\b(?:(?:on|to) (?:the |our |my |your )?(left|right)(?:[- ]hand)?(?: side)?|"
                  r"(left|right)(?:[- ]hand)? side|(?:the )?(left|right)(?= (?:curb|kerb|space|spot|gap)))"
                  r"(?: of (?:the )?(?:road|street|image|photo|frame))?\b")
UNSUPPORTED = re.compile(r"\b(?:no|never|without|other than|apart from|away from|across|opposite|behind|past|"
                         r"before|after|beyond|upstream|downstream|under|until|leftmost|rightmost|nearest|closest|"
                         r"farthest|furthest|first|second|third|last|left of|right of|leftward|rightward|ahead|"
                         r"painted|marked|parallel|perpendicular|angled|safe|safely|legal|legally|allowed|reserved|"
                         r"wide enough|for free|free of charge|free parking|(?:parking|it) is free|paid|metered|"
                         r"permit|bus (?:stop|lane)|taxi rank|oncoming|moving|northbound|southbound|eastbound|"
                         r"westbound|(?:north|south|east|west) (?:side|of)|to the (?:north|south|east|west)|"
                         r"accommodate|either|or|unless|if(?! possible\b))\b")
DESTINATION = re.compile(r"\b(?:drop|stop|park|pull|let (?:me|us) out|get out|leave (?:me|us)|gap|"
                         r"pullout|space|spot|curbside|curb|kerb|bay|opening|layby)\b")
FILLER = set(("a an the this that those these some any i we you me us my our can could would will may please "
              "kindly just want like get be out off drop leave let take use show find choose stop park parking pull "
              "over up in into to on "
              "at open empty free unoccupied unmarked clear available vacant curbside curb kerb edge roadside space "
              "spot gap opening area pullout layby bay there here one of if possible thanks side it is for").split())
ARTICLES = {"a", "an", "the", "this", "that", "those", "these"}
SURFACES = {"gravel","paved","asphalt","pavers","cobblestone","dirt","sand","concrete"}
OBJECTS = {"palm","tree","bush","hedge","shrub","bin","fence","gate","wall","divider","sign","lamp"}
UNCERTAIN = re.compile(r"\b(?:unknown|unreadable|uncertain|unsure|unclear|none|no|not|unidentifiable|"
                       r"possibly|maybe|perhaps|might|could be|cannot|or)\b|\?",re.I)
SEPARATED = re.compile(r"\b(?:across (?:from|(?:(?:the|a|main) )*(?:road|street|divider|barrier))|"
                       r"opposite (?:the |this |our )?(?:side|point|mark|road|street)|"
                       r"(?:other|far) side of (?:the )?(?:road|street|divider|barrier)|"
                       r"separated (?:from|by)|behind (?:a |the )?(?:divider|barrier))\b",re.I)
ATTACHMENT = re.compile(r"\b(?:and|with|beside|alongside|next to|near|under(?:neath)?|beneath|below|above|"
                        r"over|on(?: top of)?|in front of|behind|by|showing|displaying|carrying|sporting|"
                        r"has|have|having|bearing)\b|\s+[-–—]\s+")
TYPES = {"vehicle","car","sedan","suv","hatchback","coupe","taxi","van","minivan","bus","truck","pickup",
         "flatbed","motorcycle"}
SUBTYPES = {"vehicle":TYPES, "car":{"car","sedan","suv","hatchback","coupe","taxi"},
            "truck":{"truck","pickup","flatbed"}, "van":{"van","minivan"}}
COLORS = bay_intent.COLORS | {"dark","navy","charcoal","maroon","cream","purple","pink"}
DARK = {"black","dark","navy","charcoal"}
SIZES = {"small","large"}
ALIASES = {**bay_intent.ALIASES, "automobiles":"car","autos":"car","saloon":"sedan","jeep":"suv","hatch":"hatchback",
           "blackish":"dark","darkish":"dark","trashcan":"bin","dustbin":"bin","bins":"bin","dumpsters":"dumpster",
           "storefront":"shop","storefronts":"shop","boutique":"shop","shrubbery":"shrub","bush":"shrub",
           "bushes":"shrub","kerb":"curb",
           "cafeterias":"cafeteria","marts":"mart","markets":"market","lawns":"lawn","little":"small",
           "compact":"small","mini":"small","big":"large","buses":"bus"}
VEHICLE_FILLER = ARTICLES | {"two","both","one","ones"}


def _normal(text):
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'").replace("grey", "gray")
    for pattern, replacement in (
        (r"\b(i|we)'d\b",r"\1 would"), (r"\b(?:pull[ -]?in|pull[ -]?out|lay[ -]?by)\b","pullout"),
        (r"\bpick[ -]?ups?(?: trucks?)?\b","pickup"), (r"\bflat[ -]?beds?(?: trucks?)?\b","flatbed"),
        (r"\bsports?[- ]utility vehicles?\b","suv"), (r"\bhatch[- ]backs?\b","hatchback"),
        (r"\brecycling\s+(?:bins?|cans?|containers?)\b","recycling bin"),
        (r"\b(?:rubbish|garbage|trash|litter|waste)\s+(?:bins?|cans?|containers?)\b","bin"),
        (r"\btree[- ]lined\b","row of trees"), (r"\b(?:curb|kerb)[- ]side\b","curbside"),
        (r"\bunpainted\b","unmarked"),
        (r"\bdropped off\b","drop off"), (r"\bright (?=between|by|beside|next to)",""),
        (r"[.!?]\s*(?:(?:can|could|would) you )?(?:drop|leave) (?:me|us)(?: off)? there\s*[.!?]*$",""),
        (r"\b(?:would (?:be fine|work)|works|is fine)(?: for (?:me|us))?[.!?]*\s*$",""),
        (r"\b(?:if possible|please|for me|for us)\s*[.!?]*$",""),
    ):
        text = re.sub(pattern,replacement,text)
    return text


def _terms(text):
    words = [ALIASES.get(w,w) for w in re.findall(r"[^\W_]+",text)]
    return [w[:-1] if w.endswith("s") and w[:-1] in TYPES else w for w in words]


def _vehicle(phrase, *, requested, inherited=None):
    words = set(_terms(phrase))
    kinds = words & TYPES
    kinds = {k for k in kinds if not any(other != k and other in SUBTYPES.get(k,{k}) for other in kinds)}
    colors, sizes = words & COLORS, words & SIZES
    if requested and words - VEHICLE_FILLER - TYPES - COLORS - SIZES - {"parked"}:
        return None
    if len(kinds)>1 or len(sizes)>1 or (requested and (len(colors-{"dark"})>1 or {"dark","white"}<=colors)):
        return None
    kind = next(iter(kinds),inherited)
    return (kind, colors, sizes, "parked" in words) if kind else None


def _noun_colors(words, nouns):
    """Only link an explicit colour to its noun, excluding colours belonging to features or other objects."""
    modifiers = COLORS|SIZES|{"light","bright","metallic","pale","parked","colored","coloured","plastic",
                              "metal","steel","wooden","concrete","round","tall","public","recycling","waste"}
    positions = [i for i,w in enumerate(words) if w in nouns]
    direct_colors = set()
    for i,w in enumerate(words):
        if w not in COLORS:
            continue
        before = any(i<j and set(words[i+1:j])<=modifiers for j in positions)
        after = (set(words[i+1:])<={"color","colour"} and
                 any(j<i and words[j+1:i] in ([],["is"],["in"],["painted"],["body","is"]) for j in positions))
        if before or after:
            direct_colors.add(w)
    return direct_colors


def _vehicle_match(spec, raw):
    """False means contradicted, None means an observed attribute is too coarse."""
    if not isinstance(raw,str) or UNCERTAIN.search(raw):
        return False
    phrase = ATTACHMENT.split(_normal(raw),maxsplit=1)[0]
    seen = _vehicle(phrase,requested=False)
    if not seen:
        return False
    kind, colors, sizes, parked = spec
    seen_kind, _, seen_sizes, seen_parked = seen
    seen_colors = _noun_colors(_terms(phrase),TYPES)
    if seen_kind in SUBTYPES.get(kind,{kind}):
        certain = True
    elif kind in SUBTYPES.get(seen_kind,{seen_kind}):
        certain = False
    else:
        return False
    for color in colors:
        if not seen_colors or (seen_colors=={"dark"} and color in DARK):
            certain = False
        elif color=="dark" and seen_colors & DARK:
            continue
        elif color not in seen_colors:
            return False
    if colors and len(seen_colors-{"dark"})>1:
        certain = False
    if sizes and sizes != seen_sizes:
        if seen_sizes:
            return False
        certain = False
    if parked and not seen_parked:
        certain = False
    return True if certain else None


def _pair(phrase):
    parts = re.split(r"\s+and\s+|\s*&\s*",phrase)
    if len(parts)==1 and re.search(r"\b(?:two|both|cars|suvs|vans|trucks|vehicles|hatchbacks|sedans)\b",phrase):
        spec = _vehicle(phrase,requested=True)
        return (spec,spec) if spec else None
    if len(parts)!=2:
        return None
    specs = [_vehicle(p,requested=True) for p in parts]
    for i, spec in enumerate(specs):
        if spec is None and specs[1-i] is not None:
            specs[i] = _vehicle(parts[i],requested=True,inherited=specs[1-i][0])
    return tuple(specs) if all(specs) else None


def _landmark(terms, descriptions, fields, *, equivalent_dividers=False):
    for key in fields:
        raw = descriptions.get(key)
        if not isinstance(raw,str) or SEPARATED.search(raw):
            continue
        for clause in re.split(r"[|;,.]",raw):
            if not clause.strip() or UNCERTAIN.search(clause):
                continue
            phrase = _normal(clause)
            if "and" in terms:
                sequence = [w for w in _terms(phrase) if w not in ARTICLES]
                if any(tuple(sequence[i:i+len(terms)])==tuple(terms) for i in range(len(sequence))):
                    return True
                continue
            for item in ATTACHMENT.split(phrase):
                tokens = _terms(item)
                words = set(tokens)
                nouns = set(terms)-COLORS
                if "palm" in words:
                    words.add("tree")
                    if "tree" in nouns:
                        nouns.add("palm")
                if "dumpster" in words:
                    words.add("bin")
                    if "bin" in nouns:
                        nouns.add("dumpster")
                if equivalent_dividers and ("median" in words or {"traffic","island"}<=words):
                    words.add("divider")
                if nouns and set(terms)<=words and (set(terms)&COLORS)<=_noun_colors(tokens,nouns):
                    return True
    return False


def observed_landmark(terms, descriptions, fields=("frontage","roadside")):
    """A height adjective may be unobserved on an identified object. Names and colours must match."""
    if _landmark(terms,descriptions,fields,equivalent_dividers=True):
        return True
    extent = {"low","short","tall","high"}
    core = set(terms)-extent
    if (core == set(terms) or not core & {"divider","wall","fence","hedge","shrub","tree","bin"}
            or not _landmark(core,descriptions,fields,equivalent_dividers=True)):
        return False
    opposite = {"tall","high"} if set(terms)&{"low","short"} else {"low","short"}
    if any(_landmark(core|{height},descriptions,fields,equivalent_dividers=True) for height in opposite):
        return False
    return None


def measured_frontages(text, parsed_groups):
    """Recover literal place names such as 'Park Place' when parking words also occur inside the name."""
    text = SIDE.sub(" ",_normal(text))
    relatives = list(bay_intent.RELATION.finditer(text))
    literal = []
    for index,relation in enumerate(relatives):
        phrase = text[relation.end():relatives[index+1].start() if index+1<len(relatives) else len(text)]
        phrase = re.sub(r"\b(?:in|into)\s+(?:(?:the|that|this|an?|open|empty)\s+)*(?:gap|space|spot|opening)\s*$",
                        "",phrase)
        phrase = re.sub(r"\b(?:and|please|thanks)\s*[,.!?]*$","",phrase)
        terms = tuple(dict.fromkeys(w for w in _terms(phrase) if w not in ARTICLES))
        if terms and not set(terms)&TYPES:
            fields = ("frontage",) if relation.group() in {"outside","in front of"} else ("frontage","roadside")
            literal.append((terms,fields))
    return [next(((terms,fields) for terms,fields in literal if set(base)<=set(terms)),
                 (base,("frontage","roadside"))) for base in parsed_groups]


def _requirements(text):
    sides = {next(s for s in m.groups() if s) for m in SIDE.finditer(text)}
    text = SIDE.sub(" ",text)
    if len(sides)>1 or re.search(r"\b(?:left|right)\b",text):
        return None
    parts = RELATION.split(text)
    prefix = set(_terms(parts[0])) - FILLER
    if prefix - SURFACES:
        return None
    places = [(prefix,("roadside",))] if prefix else []
    vehicles = ()
    relations = parts[1::2]
    for index,(relation,phrase) in enumerate(zip(relations,parts[2::2])):
        if index+1<len(relations) and relations[index+1]=="between":
            phrase = re.sub(r"\b(?:in|into|at)\s+(?:(?:the|that|this|an?|open|empty)\s+)*"
                            r"(?:gap|space|spot|opening)\s*$","",phrase)
        phrase = re.sub(r"\b(?:and(?: also)?|please|thanks)\s*[,.!?]*$","",phrase).strip(" ,.!?")
        terms = tuple(dict.fromkeys(w for w in _terms(phrase) if w not in ARTICLES))
        if relation=="at" and not set(terms)-FILLER:
            continue
        if not terms:
            return None
        if relation=="between":
            if vehicles or not (new_pair := _pair(phrase)):
                return None
            vehicles = new_pair
        elif set(terms) & TYPES:
            spec = _vehicle(phrase,requested=True)
            if vehicles or not spec or relation in {"in front of","outside","outside of"}:
                return None
            vehicles = (spec,)
        else:
            fields = ("frontage",) if relation in {"in front of","outside","outside of"} else ("frontage","roadside")
            clauses = re.split(r"\s+and\s+",phrase)
            if len(clauses)>1 and all(set(_terms(item)) & OBJECTS for item in clauses):
                places.extend((tuple(w for w in _terms(item) if w not in ARTICLES),fields) for item in clauses)
            else:
                places.append((terms,fields))
    if not (vehicles or places or sides or re.search(r"\b(?:space|spot|gap|opening|pullout|bay|curbside|layby)\b",text)):
        return None
    return sides,vehicles,places


def _numeric_pair(values):
    return (isinstance(values,(list,tuple)) and len(values)==2
            and all(isinstance(n,Real) and not isinstance(n,bool) and math.isfinite(n) for n in values))


def _spoken_form(text, expanded):
    from .curb_utterance import destination_clause,opposing_image_sides
    normalized = destination_clause(text,expanded=expanded)
    return opposing_image_sides(normalized) if expanded else (normalized,None)


def references_individual_sides(text):
    return _spoken_form(text,True)[1] is not None


def indexed_reference(text, *, expanded=False):
    """Whether the complete utterance asks for a vehicle gap or a specifically anchored open pull-in."""
    if not isinstance(text,str) or not bay_intent.permitted_destination(text):
        return False
    text = _normal(_spoken_form(text,expanded)[0])
    if not DESTINATION.search(text) or UNSUPPORTED.search(text):
        return False
    request = _requirements(text)
    if request is None:
        return False
    _,vehicles,landmarks = request
    return len(vehicles)==2 or bool(landmarks and re.search(r"\b(?:gap|pullout|opening|gravel|unpaved|layby)\b",text))


def _signed_directions(option, fields, directions, frame_size):
    crops=option.get("vehicle_crops")
    if not isinstance(crops,list) or len(crops)!=2:
        return False
    w,h=frame_size
    rows={}
    seen=set()
    for field,item in zip(("nearer_vehicle","farther_vehicle"),crops):
        if not isinstance(item,(list,tuple)) or len(item)!=2:
            return False
        track,box=item
        if (type(track) is not int or track in seen or not isinstance(box,(list,tuple)) or len(box)!=4
                or not _numeric_pair(box[:2]) or not _numeric_pair(box[2:]) or
                not (0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h)):
            return False
        seen.add(track)
        rows[field]=box
    x=option["point"][0]
    return all(x<rows[field][0]-1 if side=="left" else x>rows[field][2]+1
               for field,side in zip(fields,directions))


def match_points(text, options, frame_size, *, native_scalars=True, conversational=False, expanded=False):
    """Return the original options explicitly supported, then those suitable only for clarification."""
    if not isinstance(text,str) or not text.strip():
        return [],[]
    directions=None
    if conversational:
        if not bay_intent.permitted_destination(text):
            return [],[]
        text,directions = _spoken_form(text,expanded)
    text = _normal(text)
    if not DESTINATION.search(text) or bay_intent.UNSUPPORTED.search(text) or UNSUPPORTED.search(text):
        return [],[]
    requirements = _requirements(text)
    if isinstance(frame_size,dict):
        frame_size = (frame_size.get("width"),frame_size.get("height"))
    if requirements is None or not _numeric_pair(frame_size) or min(frame_size)<=0:
        return [],[]
    sides, vehicles, places = requirements
    w,h = frame_size
    unmarked = bool(re.search(r"\bunmarked\b",text))
    supported, possible, seen = [],[],set()
    for option in options:
        if not isinstance(option,dict):
            continue
        point, obs = option.get("point"), option.get("descriptions")
        if not isinstance(obs,dict) or not _numeric_pair(point) or not (0<=point[0]<w and 0<=point[1]<h):
            continue
        if not native_scalars and any(type(n) not in (int,float) for n in point):
            continue
        side = "left" if point[0]<.47*w else "right" if point[0]>.53*w else None
        if (sides and side and side not in sides) or not all(_landmark(terms,obs,fields) for terms,fields in places):
            continue
        ambiguous = bool(sides and side is None)
        if unmarked:
            evidence = option.get("evidence") or {}
            if evidence.get("painted_boundaries") or _landmark(("marked",),obs,("roadside",)):
                continue
            if not _landmark(("unmarked",),obs,("roadside",)):
                ambiguous = True
        if vehicles:
            assignments = [[_vehicle_match(spec,obs.get(field)) for spec,field in zip(vehicles,fields)]
                           for fields in permutations(("nearer_vehicle","farther_vehicle"),len(vehicles))
                           if directions is None or _signed_directions(option,fields,directions,frame_size)]
            if not any(all(state is True for state in row) for row in assignments):
                if not any(False not in row and (True in row or places) for row in assignments):
                    continue
                ambiguous = True
        if ambiguous and not (vehicles or places):
            continue
        key = (tuple(point),tuple(str(obs.get(field) or "").casefold().strip()
                                 for field in ("nearer_vehicle","farther_vehicle","frontage","roadside")))
        if key not in seen:
            (possible if ambiguous else supported).append(option)
            seen.add(key)
    return supported,possible
