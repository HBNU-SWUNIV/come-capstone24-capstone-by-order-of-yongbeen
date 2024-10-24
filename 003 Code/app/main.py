# app/main.py

import random
from fastapi import FastAPI, HTTPException, Request, Form, Depends, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from .database import SessionLocal, Base, engine
from .models import User, Document
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import jwt
from jwt import PyJWTError
from datetime import datetime, timedelta
from passlib.context import CryptContext
from fastapi.middleware.cors import CORSMiddleware
import openai
from dotenv import load_dotenv
import os

load_dotenv()

RUNPOD_api_key = os.getenv("RUNPOD_api_key")
MODEL_SERVER = os.getenv(
    "RUNPOD_URL",
)

SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI()

# CORS 설정
origins = [
    "http://localhost",
    "http://localhost:8000",
    # 필요에 따라 추가
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# 정적 파일 및 템플릿 설정

templates = Jinja2Templates(directory="app/templates")

# 데이터베이스 초기화
Base.metadata.create_all(bind=engine)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def hash_password(password: str):
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)


@app.get("/register", response_class=HTMLResponse)
async def get_register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


# 회원가입 처리
@app.post("/register")
def register(
    username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)
):
    # 사용자명 중복 확인
    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")

    # 비밀번호 해싱 및 사용자 생성
    hashed_pw = hash_password(password)
    new_user = User(username=username, hashed_password=hashed_pw)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # 회원가입 후 토큰 발급 (자동 로그인)
    access_token = create_access_token(data={"sub": new_user.username})
    return {"access_token": access_token, "token_type": "bearer"}


# 토큰 생성
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# 토큰 검증 및 디코딩
def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except PyJWTError:
        return None


@app.get("/login", response_class=HTMLResponse)
async def get_login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/token")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


async def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception
    username: str = payload.get("sub")
    if username is None:
        raise credentials_exception
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user


@app.get("/users/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return {"username": current_user.username}


# 메인 페이지
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# 문서 생성
@app.post("/documents/")
def create_document(
    title: str = Form(...),
    content: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    new_doc = Document(title=title, content=content, owner=current_user)
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)
    return {"id": new_doc.id, "title": new_doc.title, "content": new_doc.content}


# 문서 목록 조회
@app.get("/documents/")
def read_documents(
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    documents = db.query(Document).filter(Document.owner == current_user).all()
    return [
        {"id": doc.id, "title": doc.title, "content": doc.content} for doc in documents
    ]


# 특정 문서 조회
@app.get("/documents/{document_id}")
def read_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    doc = (
        db.query(Document)
        .filter(Document.id == document_id, Document.owner == current_user)
        .first()
    )
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"id": doc.id, "title": doc.title, "content": doc.content}


# 문서 수정
@app.put("/documents/{document_id}")
def update_document(
    document_id: int,
    title: str = Form(...),
    content: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    doc = (
        db.query(Document)
        .filter(Document.id == document_id, Document.owner == current_user)
        .first()
    )
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    doc.title = title
    doc.content = content
    db.commit()
    db.refresh(doc)
    return {"id": doc.id, "title": doc.title, "content": doc.content}


# 문서 삭제
@app.delete("/documents/{document_id}")
def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    doc = (
        db.query(Document)
        .filter(Document.id == document_id, Document.owner == current_user)
        .first()
    )
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    db.delete(doc)
    db.commit()
    return {"message": "Document deleted"}


client = openai.OpenAI(
    api_key=RUNPOD_api_key,
    base_url=MODEL_SERVER,
)


# LLM 추론 엔드포인트
@app.post("/inference")
async def run_inference(
    content: str = Form(...),
):

    inst_list = [
        "아래의 글의 문법을 교정하고 종합적인 수준과 점수를 매겨줘.",
        "아래 글의 문법을 교정하고 전체적인 수준과 점수를 평가해줘.",
        "아래의 글의 맞춤법을 교정하고, 전반적인 내용의 질에 따른 수준과 점수를 평가해줘.",
        "아래 글을 보고, 글의 문법을 교정하고 글의 전체에 대한 수준과 점수를 평가해줘.",
        "이 글을 검토하고, 맞춤법을 교정하고 종합적인 수준과 점수를 통해 평가해줘.",
        "다음에 제시된 글을 읽고, 글의 문법을 교정하고 종합적인 품질에 대한 수준과 점수를 매겨줘.",
    ]
    instruction = random.choice(inst_list)

    messages = [{"role": "user", "content": f"{instruction}\n{content}"}]

    try:
        response = client.chat.completions.create(
            model="capstonedesignwithlyb/bllossom3.1_ga",
            messages=messages,
            temperature=0.5,
        )

        result = response.choices[0].message.content

        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


app.mount("/static", StaticFiles(directory="app/static"), name="static")
