import os
import sys
from pathlib import Path
from groq import Groq
from sentence_transformers import SentenceTransformer
import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance,VectorParams,PointStruct,Filter,FieldCondition,MatchValue,PayloadSchemaType,FilterSelector
from dotenv import load_dotenv
import uuid


load_dotenv()

llm='openai/gpt-oss-120b'
embeddings_model='all-MiniLM-L6-v2'
whisper='large-v3-turbo'  # faster-whisper resolves this to mobiuslabsgmbh/faster-whisper-large-v3-turbo

groq_api_key=os.getenv("GROQ_API_KEY")
qdrant_api_key=os.getenv("QDRANT_API_KEY")
qdrant_url=os.getenv("QDRANT_URL")

groq_client=Groq(api_key=groq_api_key)
qdrant_client=QdrantClient(api_key=qdrant_api_key,url=qdrant_url)
embeddings_client=SentenceTransformer(embeddings_model)

COLLECTION_NAME="video transcript"
EMBEDDING_SIZE=384
CHUNK_CHARS=1200
CHUNK_OVERLAP=2
CACHE_DIR=Path("cache")
DATA_DIR=Path("data")
VIDEO_SUFFIXES={".mp4",".mkv",".webm",".m4a",".mp3",".wav",".flac"}

# whisper runs locally, so it is loaded once on first use instead of at import
_whisper_model=None

# CTranslate2 needs the CUDA dlls that ship in the nvidia-* wheels to be on the dll path
def enable_cuda_libs():
    if not hasattr(os,"add_dll_directory"):
        return
    try:
        import nvidia
    except ImportError:
        return
    # nvidia is a namespace package, so it has __path__ but no __file__
    lib_dirs=[]
    for root in nvidia.__path__:
        root=Path(root)
        lib_dirs+=[str(d) for d in sorted(root.glob("*/bin"))+sorted(root.glob("*/lib"))]

    # CTranslate2 loads cublas lazily by name, which only searches PATH,
    # so PATH has to be set before ctranslate2 is imported
    os.environ["PATH"]=os.pathsep.join(lib_dirs)+os.pathsep+os.environ["PATH"]
    for lib_dir in lib_dirs:
        try:
            os.add_dll_directory(lib_dir)
        except OSError:
            pass

# loads whisper on the GPU, falling back to the CPU when CUDA is unavailable
def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        enable_cuda_libs()
        # imported here so the dll path above is in place first
        from faster_whisper import WhisperModel
        import ctranslate2

        device,compute_type=("cuda","float16") if ctranslate2.get_cuda_device_count()>0 else ("cpu","int8")
        try:
            _whisper_model=WhisperModel(whisper,device=device,compute_type=compute_type)
        except (RuntimeError,ValueError) as e:
            # only a cuda or compute-type problem is worth retrying on the cpu; a bad
            # model id or a failed download fails identically on either device
            if device=="cpu":
                raise
            print(f"CUDA unusable ({type(e).__name__}: {e}); falling back to CPU int8.")
            device,compute_type="cpu","int8"
            _whisper_model=WhisperModel(whisper,device=device,compute_type=compute_type)

        print(f"Whisper loaded on {device} ({compute_type}).")
    return _whisper_model

# transcribes a media file locally with whisper, cached as json per file
def transcript(path):
    path=Path(path)
    CACHE_DIR.mkdir(exist_ok=True)
    transcript_path=CACHE_DIR/f"{path.stem}.json"

    if transcript_path.exists():
        with open(transcript_path,"r",encoding="utf-8") as f:
            return json.load(f)

    # whisper decodes the video directly and windows long audio itself,
    # so there is no upload limit to split around
    segments,_info=get_whisper().transcribe(
        str(path),
        language="en",
        beam_size=5,
        condition_on_previous_text=False
    )
    all_segments=[{"start":s.start,"end":s.end,"text":s.text} for s in segments]

    with open(transcript_path,"w",encoding="utf-8") as f:
        json.dump(all_segments,f,ensure_ascii=False,indent=2)

    return all_segments

# split the source text into smaller parts
def chunking(segments,source,course=""):
    chunks=[]
    current=[]
    current_len=0
    new_since_flush=0

    for segment in segments:
        current.append(segment)
        current_len+=len(segment["text"])
        new_since_flush+=1

        if current_len>=CHUNK_CHARS:
            chunks.append({
                "text":" ".join(s["text"] for s in current),
                "start":current[0]["start"],
                "end":current[-1]["end"],
                "source":source,
                "course":course,
                "chunk_index":len(chunks)
            })
            current=current[-CHUNK_OVERLAP:]
            current_len=sum(len(s["text"]) for s in current)
            new_since_flush=0

    # skip the tail when it is only the overlap already inside the last chunk
    if new_since_flush:
        chunks.append({
            "text":" ".join(s["text"] for s in current),
            "start":current[0]["start"],
            "end":current[-1]["end"],
            "source":source,
            "course":course,
            "chunk_index":len(chunks)
        })

    return chunks

# converts chunks of text into vector embeddings
def embedd(chunks):
    texts=[chunk["text"] for chunk in chunks]
    embeddings=embeddings_client.encode(texts)
    return embeddings.tolist()

# upsert the points to vectorDB
def qdrant_upload(embeddings,chunks):
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_SIZE,distance=Distance.COSINE)
        )

    # idempotent, so it also backfills collections created before the index existed
    for field in ("source","course"):
        qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD
        )

    # drop this video's old points first, otherwise re-ingesting a file that now
    # produces fewer chunks leaves the extra ones behind as stale hits
    if chunks:
        qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=FilterSelector(filter=Filter(must=[
                FieldCondition(key="source",match=MatchValue(value=chunks[0]["source"]))
            ]))
        )

    points=[]
    for i,chunk in enumerate(chunks):
        point_id=str(uuid.uuid5(uuid.NAMESPACE_URL,f"{chunk['source']}:{chunk['chunk_index']}"))
        points.append(PointStruct(
            id=point_id,
            vector=embeddings[i],
            payload=chunk
        ))

    qdrant_client.upsert(collection_name=COLLECTION_NAME,points=points)

# searches the vectorDB for valid context related to query
def search(query,source=None,course=None,top_k=5):
    conditions=[]
    if source is not None:
        conditions.append(FieldCondition(key="source",match=MatchValue(value=source)))
    if course is not None:
        conditions.append(FieldCondition(key="course",match=MatchValue(value=course)))
    query_filter=Filter(must=conditions) if conditions else None

    query_vector=embeddings_client.encode(query).tolist()
    results=qdrant_client.query_points(collection_name=COLLECTION_NAME,
                                    query=query_vector,
                                    limit=top_k,
                                    with_payload=True,
                                    query_filter=query_filter).points

    # the video name travels with the timestamp, otherwise a timestamp
    # means nothing across a library this size
    lines=[]
    for hit in results:
        minutes,seconds=divmod(int(hit.payload["start"]),60)
        lines.append(f"[{hit.payload['source']} @ {minutes:02d}:{seconds:02d}] {hit.payload['text']}")
    return "\n".join(lines)

# the exact messages the deployed path sends, so the evaluator can replay them
# through its own rate limiting without the two prompts drifting apart
def answer_messages(query,context):
    sys_prompt=f"""
        You are a transcription reader assistant.

        Your JOB is to answer the questions related to the concepts discussed in 
        video transcripts using only the information passed as context.
        CONTEXT:{context}

        Each context line is formatted as [video name @ MM:SS] followed by the transcript text.

        The OUTPUT should include:
        a short version of what was discussed
        the name of the video and the timestamp where the concept was discussed

        Incase the user asks query regarding information outside the context 
        do not provide an answer. Respond by saying : I dont have information about that.
    """
    return [{"role":"system","content":sys_prompt},{"role":"user","content":query}]

# passes context to LLM and asks the query
def ask_llm(query,context):
    response=groq_client.chat.completions.create(
        model=llm,messages=answer_messages(query,context),temperature=0.0
    )
    return response.choices[0].message.content

# runs the full pipeline for one media file: transcript -> chunks -> vectors
def ingest(path):
    path=Path(path)
    source=path.name
    # the folder a video sits in names its course
    course=path.parent.name if path.parent!=DATA_DIR else ""

    segments=transcript(path)
    chunks=chunking(segments,source,course)
    embeddings=embedd(chunks)
    qdrant_upload(embeddings,chunks)

    print(f"Ingested {source}: {len(segments)} segments, {len(chunks)} chunks")

# ingests every media file under data/, skipping nothing but re-using cached transcripts
def ingest_all(root=DATA_DIR):
    root=Path(root)
    paths=sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES)

    if not paths:
        print(f"No media files found under {root}")
        return

    print(f"Found {len(paths)} files under {root}")
    failed=[]
    for i,path in enumerate(paths,1):
        print(f"[{i}/{len(paths)}] {path}")
        try:
            ingest(path)
        except Exception as e:
            print(f"  FAILED {type(e).__name__}: {e}")
            failed.append(path)

    print(f"Done: {len(paths)-len(failed)}/{len(paths)} ingested")
    for path in failed:
        print(f"  failed: {path}")

def chat():
    print("Ask about your lectures.")
    print("Commands: 'ingest all' | 'ingest <path>' | 'source <file>' | 'course <name>'")
    print("('source' or 'course' alone clears that filter, 'exit' quits)")

    source=None
    course=None

    while True:
        user_input=input("You: ").strip()

        if not user_input:
            continue
        if user_input.lower()=="exit":
            break
        if user_input.lower()=="ingest all":
            ingest_all()
            continue
        if user_input.lower().startswith("ingest "):
            path=user_input[len("ingest "):].strip()
            try:
                ingest(path)
            except Exception as e:
                print(f"Could not ingest {path}: {e}")
            continue
        if user_input.lower()=="source":
            source=None
            print("Searching all videos.")
            continue
        if user_input.lower().startswith("source "):
            source=user_input[len("source "):].strip()
            print(f"Searching {source} only.")
            continue
        if user_input.lower()=="course":
            course=None
            print("Searching all courses.")
            continue
        if user_input.lower().startswith("course "):
            course=user_input[len("course "):].strip()
            print(f"Searching the {course} course only.")
            continue

        context=search(user_input,source=source,course=course)
        print("Bot:",ask_llm(user_input,context))

def main():
    # the llm emits characters cp1252 cannot encode, which crashes print() on a windows console
    sys.stdout.reconfigure(encoding="utf-8",errors="replace")
    # `python video_transcript_rag.py ingest` runs the batch and exits, for long unattended runs
    if len(sys.argv)>1 and sys.argv[1]=="ingest":
        ingest_all(sys.argv[2] if len(sys.argv)>2 else DATA_DIR)
        return
    chat()

if __name__=="__main__":
    main()
