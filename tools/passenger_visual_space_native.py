"""Privately review an allowlisted original twice, once for discovery and once for inspection."""
import base64
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

from tools.passenger_roadside35_native import generate, load_runtime
from trackrag.dropoff import visual_space_board as board


def digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()).hexdigest()


def parse_reply(reply):
    try:
        value=json.loads(reply[reply.index("{"):reply.rindex("}")+1])
        return value if isinstance(value,dict) else {}
    except (ValueError,TypeError):
        return {}


def run(folder,weights,revision):
    root=Path(folder)
    rows=json.loads((root/"views.json").read_bytes())
    processor,model=load_runtime(weights)
    first=[]; order=[]; output={}
    for row in rows:
        key=row["source"]
        paved=all(s.get("vehicles") for samples in row["sides"].values() for s in samples)
        extended=all(s.get("native_option") for samples in row["sides"].values() for s in samples)
        packet={"recipe":board.PAVED_RECIPE if paved else board.RECIPE,"revision":revision,"source":key,"context":row["context"],
                "guides":digest([board.REVIEW,board.PAVED_GUIDE] if paved else [board.SELECT,board.REVIEW]),"sides":{}}
        if extended:
            packet.update(recipe=board.EXTENSION_RECIPE,guides=digest([board.REVIEW,board.EXTENSION_GUIDE]))
        for side,samples in row["sides"].items():
            view=(root/f"{key}.{side}.jpg").read_bytes()
            packet["sides"][side]={"view":sha256(view).hexdigest(),"spec":digest(samples),"reviewed":[]}
            if not paved:
                first.append(base64.b64encode(view).decode("ascii"))
            order.append((key,side,samples))
        output[key]=packet
    replies=generate(processor,model,first,board.SELECT,64)
    del first
    second=[]; inspections=[]
    choices=iter(replies)
    for key,side,samples in order:
        paved=output[key]["recipe"] in (board.PAVED_RECIPE,board.EXTENSION_RECIPE)
        chosen=list(range(1,len(samples)+1)) if paved else parse_reply(next(choices)).get("selected")
        if not isinstance(chosen,list):
            continue
        ids=list(dict.fromkeys(i for i in chosen if type(i) is int and 1<=i<=len(samples)))
        if not paved:
            ids=ids[:2]
        for i in ids:
            view=(root/f"{key}.{side}.{i}.jpg").read_bytes()
            entry={"id":i,"view":sha256(view).hexdigest()}
            output[key]["sides"][side]["reviewed"].append(entry)
            second.append(base64.b64encode(view).decode("ascii"))
            inspections.append((entry,output[key]["recipe"]))
    print(f"Selected {len(inspections)} ground locations for a separate original-photo review.",flush=True)
    fields=("left_vehicle","right_vehicle","rear_vehicle","frontage","roadside")
    contexts={}
    for recipe,guide in ((board.PAVED_RECIPE,board.PAVED_GUIDE),(board.EXTENSION_RECIPE,board.EXTENSION_GUIDE)):
        context_inputs=[image for image,(_,kind) in zip(second,inspections,strict=True) if kind==recipe]
        contexts[recipe]=iter(generate(processor,model,context_inputs,guide,290))
    for (entry,recipe),raw in zip(inspections,generate(processor,model,second,board.REVIEW,290),strict=True):
        reply=parse_reply(raw)
        status=str(reply.get("status","")).strip().upper()
        edge=str(reply.get("edge","")).strip().casefold()
        labels=parse_reply(next(contexts[recipe])) if recipe in contexts else reply
        safe={k:re.sub(r"[\w.+-]+@[\w.-]+|(?:\+?\d[\s()+-]?){5,}","",str(labels.get(k) or "NONE"))[:180] for k in fields}
        entry["description"]={**safe,"edge":edge if edge in {"painted bay","recessed curb","different surface","parked row"} else "none",
                              "status":status if status in {"OPEN","BLOCKED","LANE","SEPARATED","UNCLEAR"} else "UNCLEAR"}
    (root/"packets.json").write_text(json.dumps(output),encoding="utf8")


if __name__=="__main__":
    run(*sys.argv[1:])
