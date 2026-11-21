"""Shared, cutoff-bounded candidates for explicitly selected model comparisons.

No reference key, expected answer, or pretrained model is read here. This extends
the Task 4 short-window comparator without silently changing its published pool.
"""
import math
import re
from collections import Counter
from hashlib import sha256

from trackrag.retrieval import build_index as task4_index,candidate_windows
from trackrag.evidence_time import requested_frame_ids


def build_index(recording,tracks):
    return task4_index(recording,tracks)


def candidate_catalog(question,index,*,expanded=True):
    windows=candidate_windows(question,index)
    entries={}
    def add(track,rows,start,end,kind):
        key=(track,start,end)
        if key not in entries:
            entries[key]={'recording_id':question['recording_id'],'source':index.source,
                'track_id':track,'start_frame':start,'end_frame':end,'kind':kind,
                'rows':list(rows)}
    for window in windows:
        add(window.track_id,window.observations,window.start_frame,window.end_frame,'short observed window')
    if expanded:
        eligible=[]
        requested=requested_frame_ids(question['question'])
        for history in index.histories.values():
            rows=[row for row in history if row['frame_id']<=question['query_frame']]
            if not rows:
                continue
            eligible.extend(rows)
            add(rows[0]['track_id'],rows,rows[0]['frame_id'],rows[-1]['frame_id'],'full observed history')
            by_frame={row['frame_id']:row for row in rows}
            present=[frame for frame in requested if frame in by_frame]
            for frame in present:
                add(rows[0]['track_id'],[by_frame[frame]],frame,frame,'requested observed snapshot')
            if len(present)>1:
                start,end=min(present),max(present)
                add(rows[0]['track_id'],[row for row in rows if start<=row['frame_id']<=end],start,end,'requested observed comparison')
        # A future-only identity cannot enable a source-dependent scene candidate.
        if eligible:
            first=next(index.recording.iter_frames())['frame_id']
            for frame in sorted({first,question['query_frame'],*requested}):
                if frame>question['query_frame']:
                    continue
                try:
                    index.recording.get_frame(frame)
                except KeyError:
                    continue
                add(None,[row for row in eligible if row['frame_id']==frame],frame,frame,'full scene')
    result=list(entries.values())
    result.sort(key=lambda row:(row['track_id'] is None,-1 if row['track_id'] is None else row['track_id'],row['start_frame'],row['end_frame']))
    for row in result:
        token=f"{row['recording_id']}:{row['source']}:{row['track_id']}:{row['start_frame']}:{row['end_frame']}"
        row['candidate_id']=sha256(token.encode()).hexdigest()[:24]
    return result


def describe(candidate,recording,*,temporal=True):
    rows=candidate['rows']
    start,end=candidate['start_frame'],candidate['end_frame']
    if candidate['track_id'] is None:
        counts=Counter(row['class_name'].replace('_',' ') for row in rows)
        observed=', '.join(f'{name}: {count}' for name,count in sorted(counts.items())) or 'none'
        return f'Full road scene snapshot at frame {start}. Source observations: {observed}. Observation counts do not prove complete visibility. Image appearance and sign text are not described by these labels.'
    first,last=rows[0],rows[-1]
    frame=recording.get_frame(last['frame_id'])
    width,height=frame['width'],frame['height']
    label=Counter(row['class_name'] for row in rows).most_common(1)[0][0].replace('_',' ')
    left,top,right,bottom=last['bbox_xyxy']
    cx,cy=(left+right)/2/width,(top+bottom)/2/height
    horizontal='left' if cx<1/3 else 'right' if cx>2/3 else 'center'
    vertical='upper' if cy<1/3 else 'lower' if cy>2/3 else 'middle'
    text=f'Observed {label} road user. Evidence frames {start} to {end}. Last observed image position: {horizontal}, {vertical}. '
    if temporal:
        a,b,c,d=first['bbox_xyxy']
        dx=((left+right)-(a+c))/2/width
        dy=((top+bottom)-(b+d))/2/height
        motion_x='leftward' if dx<-.01 else 'rightward' if dx>.01 else 'horizontally stable'
        motion_y='upward' if dy<-.01 else 'downward' if dy>.01 else 'vertically stable'
        ratio=(right-left)*(bottom-top)/((c-a)*(d-b))
        growth='larger growing' if ratio>1.2 else 'smaller shrinking' if ratio<.8 else 'stable image size'
        gaps=sum(y['frame_id']-x['frame_id']-1 for x,y in zip(rows,rows[1:]))
        text+=f'Observed image motion: {motion_x}, {motion_y}; {growth}. Observations: {len(rows)}, missing frames inside history: {gaps}. First observation is not proof of entry or prior absence. '
    return text+'Colour, body subtype and hidden states are not supplied by the observation labels.'


def tokens(text):
    return Counter(re.findall(r'[a-z0-9]+',text.lower()))


def tfidf_scores(query,descriptions):
    documents=[tokens(text) for text in descriptions]
    frequency=Counter(word for doc in documents for word in doc)
    idf={word:math.log((1+len(documents))/(1+count))+1 for word,count in frequency.items()}
    question=tokens(query)
    qnorm=math.sqrt(sum((count*idf.get(word,0))**2 for word,count in question.items()))
    values=[]
    for doc in documents:
        norm=math.sqrt(sum((count*idf[word])**2 for word,count in doc.items()))
        dot=sum(question[word]*count*idf[word]**2 for word,count in doc.items())
        values.append(dot/(norm*qnorm) if norm and qnorm else 0.0)
    return values


def ranked_intervals(candidates,scores,k=3):
    if type(k) is not int or not 0<=k<=3 or len(candidates)!=len(scores):
        raise ValueError('Expected matching candidate/score arrays and k from0 to3')
    if any(not math.isfinite(float(score)) for score in scores):
        raise ValueError('Ranking scores must be finite')
    ordered=sorted(zip(candidates,scores),key=lambda item:(-float(item[1]),item[0]['candidate_id']))[:k]
    return [{**{key:row[key] for key in ('recording_id','source','track_id','start_frame','end_frame')},
             'rank':rank,'score':float(score)} for rank,(row,score) in enumerate(ordered,1)]


def reserve_current_scene(ranked,candidates,k=3):
    scenes=[row for row in candidates if row['track_id'] is None]
    if not scenes or not k:
        return ranked[:k]
    current=max(scenes,key=lambda row:row['start_frame'])
    chosen=[row for row in ranked if not (row['track_id'] is None and row['start_frame']==current['start_frame'])][:k-1]
    chosen.append({**{key:current[key] for key in ('recording_id','source','track_id','start_frame','end_frame')},'rank':1,'score':0.0})
    return [dict(row,rank=i) for i,row in enumerate(chosen,1)]
