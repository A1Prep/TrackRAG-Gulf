"""Resolve short passenger references between clauses, without supplying visual attributes."""
import re


_LOCATION_VERB = r"(?:get|hop|step) (?:out|off)|alight"
_THERE = r"(?:there|at that spot|in that (?:gap|space|spot|opening)|at that (?:place|location))"
_PERSON = r"(?:i|we)(?: (?:would like|want|need|would prefer|can|could|will))?(?: to)?"
_THERE_REQUEST = re.compile(
    rf"(?:(?:that|this|there) (?:is|would be) (?:exactly )?where {_PERSON} (?:{_LOCATION_VERB})|"
    rf"(?:please )?(?:(?:can|could|would) you )?(?:let (?:me|us) (?:out|off)|drop (?:me|us)(?: off)?) {_THERE}|"
    rf"(?:please )?(?:can|could|may) (?:i|we) (?:{_LOCATION_VERB}) {_THERE}|"
    rf"{_PERSON} (?:{_LOCATION_VERB}) {_THERE}|(?:please )?(?:use|take|choose) that (?:gap|space|spot|opening))"
    r"(?: please)?"
)
_PAIR = re.compile(r"(.+?)\s+(?:have|leave|have got)\s+(?:(?:an?|some|a little|a bit of|an open|a clear)\s+)?"
                   r"(?:space|gap|opening|room)\s+(?:in[- ]?)?between them")
_REPLACE_IT = re.compile(r"\b(next to|beside|near|by|alongside|outside|in front of) (?:it|that|there)\b")
_SEE = re.compile(r"(?:(?:(?:can|could) you |do you )?(?:see|notice)|you see) (.+)")
_MY_STOP = re.compile(r"(?:(.+?) (?:is|will be) (?:my|our) (?:stop|destination|drop[- ]?off)|"
                      r"(?:my|our) (?:stop|destination|drop[- ]?off) is (.+))")
_WANT_HERE = re.compile(rf"(.+?) (?:is|would be) (?:exactly )?where {_PERSON} (?:{_LOCATION_VERB})")
_NAME = re.compile(r"[\w' &/-]{1,100}")
_OTHER_CLAIM = re.compile(r"\b(?:but|except|unless|without|instead|not|never|if|only|before|after|behind|"
                          r"opposite|across|turn|continue|drive|then)\b")


def _surface_nouns(text):
    text = re.sub(r"\b(?:patch|area|part|section) of (gravel|sand|dirt|asphalt|concrete|pavers)\b",
                  r"\1 area",text)
    text = re.sub(r"\b(?:flanked|framed) by\b","between",text)
    text = re.sub(r"(\bbetween [^,.!?]{1,90}?) on (?:the )?one side and ([^,.!?]{1,90}?) on the other(?: side)?"
                  r"(?=\s*[,.!?]|\s*$)",r"\1 and \2",text)
    text = re.sub(r"\b(we|i) (?:need|would prefer) to get out\b",r"\1 want to get out",text)
    # A closing courtesy action has no bearing on an already stated, specific location.
    return re.sub(r"\s+and (?:please )?(?:let (?:me|us) (?:out|off)|drop (?:me|us)(?: off)?) there[.!?]*$",
                  "",text)


def _short_form(text):
    text = text.casefold().replace("’","'").replace("–","-").replace("—",", ")
    for short,long in (("that's","that is"),("there's","there is"),("i'll","i will"),("we'll","we will"),
                       ("i'd","i would"),("we'd","we would"),("i'm","i am"),("we're","we are")):
        text = re.sub(r"\b"+re.escape(short)+r"\b",long,text)
    return re.sub(r"\s+"," ",text.strip())


def destination_clause(text, *, expanded=False):
    """Return the complete location request, or the original words when a reference is unresolved."""
    text = _short_form(text)
    if _OTHER_CLAIM.search(text):
        return text
    if expanded:
        text = re.sub(r"^there is (?:some )?room (?=(?:outside|between|beside|near|by|next to)\b)",
                      "there is space ",text)
    clauses = [c.strip(" ,") for c in re.split(r"[.!?;]+",text) if c.strip(" ,")]
    if len(clauses)==1:
        action = _WANT_HERE.fullmatch(clauses[0])
        words = "drop me in "+action[1] if action else text
        return _surface_nouns(re.sub(r"\b(?:hop|step) (?:off|out)|\bget off|\balight\b","get out",words))
    if len(clauses)!=2:
        return text
    first,second = clauses
    if _THERE_REQUEST.fullmatch(second):
        pair = _PAIR.fullmatch(first)
        if pair:
            return "drop me in the gap between "+pair[1]
        place = re.sub(r"^(?:there is|is there|i see|i can see|can you see)\s+","",first)
        if re.search(r"\b(?:space|spot|gap|opening|pull[- ]?in|lay[- ]?by|bay|patch|area)\b",place):
            return "drop me in "+_surface_nouns(place)
    destination = _MY_STOP.fullmatch(first)
    seen = _SEE.fullmatch(first)
    if not (destination or seen):
        return text
    anchor = next(value for value in (destination or seen).groups() if value)
    if not _NAME.fullmatch(anchor):
        return text
    where = _WANT_HERE.fullmatch(second)
    action = "drop me in "+where[1] if where else second
    action = re.sub(r"\b(?:hop|step) (?:off|out)|\bget off|\balight\b","get out",action)
    if not re.search(r"\b(?:drop|stop|park|pull|let (?:me|us) (?:off|out)|get out)\b",action):
        return text
    if _REPLACE_IT.search(action):
        return _surface_nouns(_REPLACE_IT.sub(lambda m:m[1]+" "+anchor,action))
    if re.search(r"\bbetween (?:the two of )?them\b",action):
        return _surface_nouns(re.sub(r"\bbetween (?:the two of )?them\b","between "+anchor,action))
    if not destination and not re.search(r"\bthere\b",action):
        return text
    action = re.sub(r"\b(?:right )?there\b","",action).strip(" ,")
    return _surface_nouns(action)+" near "+anchor


def opposing_image_sides(text):
    """Represent opposite, explicit left/right-of constraints. The caller must check both original boxes."""
    pattern = (r"(?P<lead>.+?)\b(?:to |on )?(?:the )?(?P<s1>left|right) of (?P<a>[^;.!?]+?) and "
               r"(?:to |on )?(?:the )?(?P<s2>left|right) of (?P<b>[^;.!?]+)[.!?]*")
    item = re.fullmatch(pattern,text)
    if not item or item["s1"]==item["s2"]:
        return text,None
    lead = item["lead"].strip()
    preface = re.fullmatch(r"((?:at|outside|near|by|beside|next to|in front of) [^,]+),\s*(.+)",lead)
    place = ""
    if preface:
        place,lead = " "+preface[1],preface[2]
    return f"{lead} between {item['a']} and {item['b']}{place}",(item["s1"],item["s2"])
