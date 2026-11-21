"""Actual local embedding HTTP calls with a content/model-bound vector cache."""
import array
import json
import math
import sqlite3
import time
import urllib.parse
import urllib.request
from hashlib import sha256
from pathlib import Path


class LocalEmbeddings:
    def __init__(self,url,model_sha256,cache_path,*,batch_size=8,runtime_id='llama-b12359-vulkan-last-v1'):
        parsed=urllib.parse.urlparse(url)
        if parsed.scheme!='http' or parsed.hostname not in {'127.0.0.1','localhost','::1'}:
            raise ValueError('The benchmark client accepts a local model server only')
        if len(model_sha256)!=64 or batch_size<1:
            raise ValueError('A checkpoint SHA256 and positive batch size are required')
        self.url=url.rstrip('/')
        self.model_sha256=model_sha256
        self.batch_size=batch_size
        self.runtime_id=runtime_id
        Path(cache_path).parent.mkdir(parents=True,exist_ok=True)
        self.cache=sqlite3.connect(cache_path)
        self.cache.execute('CREATE TABLE IF NOT EXISTS vectors(key TEXT PRIMARY KEY,text TEXT,vector BLOB,dimensions INTEGER,seconds REAL)')
        self.http_calls=0
        self.new_vectors=0
        self.events=[]

    def embed(self,texts):
        result={}
        missing={}
        for text in texts:
            key=sha256((self.model_sha256+'|'+self.runtime_id+'|last-pooling|'+text).encode()).hexdigest()
            stored=self.cache.execute('SELECT vector,dimensions FROM vectors WHERE key=?',(key,)).fetchone()
            if stored:
                values=array.array('f')
                values.frombytes(stored[0])
                if len(values)!=stored[1]:
                    raise ValueError('Corrupt embedding cache dimensions')
                result[text]=list(values)
            else:
                missing.setdefault(text,key)
        items=list(missing)
        for offset in range(0,len(items),self.batch_size):
            batch=items[offset:offset+self.batch_size]
            payload=json.dumps({'input':batch,'encoding_format':'float'}).encode()
            started=time.perf_counter()
            request=urllib.request.Request(self.url+'/v1/embeddings',data=payload,headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=600) as response:
                raw=response.read()
            elapsed=time.perf_counter()-started
            response=json.loads(raw)
            data=sorted(response['data'],key=lambda row:row['index'])
            if len(data)!=len(batch) or [row['index'] for row in data]!=list(range(len(batch))):
                raise ValueError('Embedding server omitted or duplicated an input')
            for text,row in zip(batch,data):
                vector=row['embedding']
                if not isinstance(vector,list) or not vector or any(type(value) not in (int,float) or not math.isfinite(value) for value in vector):
                    raise ValueError('Embedding server returned an invalid vector')
                norm=math.sqrt(sum(value*value for value in vector))
                if not norm:
                    raise ValueError('Embedding server returned a zero vector')
                vector=[value/norm for value in vector]
                blob=array.array('f',vector).tobytes()
                self.cache.execute('INSERT INTO vectors VALUES(?,?,?,?,?)',(missing[text],text,blob,len(vector),elapsed/len(batch)))
                # Round newly computed values identically to later cache reads.
                result[text]=list(array.array('f',vector))
            self.cache.commit()
            self.http_calls+=1
            self.new_vectors+=len(batch)
            self.events.append({'request_sha256':sha256(payload).hexdigest(),'response_sha256':sha256(raw).hexdigest(),
                'input_count':len(batch),'dimensions':len(data[0]['embedding']),'seconds':elapsed,'reported_model':response.get('model')})
            print(f'LOCAL QWEN: {self.new_vectors} new vectors, {self.http_calls} real requests; batch {elapsed:.2f}s',flush=True)
        return [result[text] for text in texts]

    def rank(self,question,documents):
        query='Instruct: Given a driving-footage review question, retrieve observed road-user and frame evidence needed to answer it.\nQuery: '+question
        vectors=self.embed([query,*documents])
        q=vectors[0]
        if any(len(doc)!=len(q) for doc in vectors[1:]):
            raise ValueError('Embedding dimensions disagree')
        return [sum(a*b for a,b in zip(q,doc)) for doc in vectors[1:]]

    def close(self):
        self.cache.close()
