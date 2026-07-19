from datetime import datetime, timedelta
from typing import Optional

import httpx
from fastapi import Depends, HTTPException, Cookie
from jose import jwt, JWTError
from sqlalchemy.orm import Session

from config import Config
from database import get_db
from models import User

config = Config()


def create_access_token(data: dict) -> str:
    payload = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=config.jwt.expire_minutes)
    payload["exp"] = expire
    return jwt.encode(payload, config.jwt.secret_key, algorithm=config.jwt.algorithm)


def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(token, config.jwt.secret_key, algorithms=[config.jwt.algorithm])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired session")


def get_current_user(
    access_token: Optional[str] = Cookie(None),
    db: Session = Depends(get_db),
) -> User:
    if not access_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    claims = decode_access_token(access_token)
    user_id = claims.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token claims")
    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


async def exchange_google_code(code: str) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": config.google_oauth.client_id,
                "client_secret": config.google_oauth.client_secret,
                "redirect_uri": config.google_oauth.redirect_uri,
                "grant_type": "authorization_code",
            },
        )
        resp.raise_for_status()
        return resp.json()


async def get_google_userinfo(access_token: str) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            config.google_oauth.userinfo_url,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        resp.raise_for_status()
        return resp.json()
