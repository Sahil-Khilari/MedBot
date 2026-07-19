from typing import Optional
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy.orm import Session

from auth import create_access_token, exchange_google_code, get_current_user, get_google_userinfo
from config import Config
from database import get_db
from models import User

router = APIRouter(prefix="/auth", tags=["auth"])
config = Config()

FRONTEND_URL = "http://localhost:5173"


@router.get("/google")
def google_login():
    params = {
        "client_id": config.google_oauth.client_id,
        "redirect_uri": config.google_oauth.redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "select_account",
    }
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)
    return RedirectResponse(url=url)


@router.get("/google/callback")
async def google_callback(
    code: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    if error or not code:
        return RedirectResponse(url=f"{FRONTEND_URL}?error=auth_failed")

    try:
        token_data = await exchange_google_code(code)
        access_token = token_data.get("access_token")
        if not access_token:
            return RedirectResponse(url=f"{FRONTEND_URL}?error=no_token")

        userinfo = await get_google_userinfo(access_token)
    except Exception:
        return RedirectResponse(url=f"{FRONTEND_URL}?error=google_failed")

    google_id = userinfo.get("sub")
    email = userinfo.get("email", "")
    name = userinfo.get("name", email)
    picture = userinfo.get("picture")

    user = db.query(User).filter(User.google_id == google_id).first()
    if user:
        user.name = name
        user.picture = picture
    else:
        user = User(google_id=google_id, email=email, name=name, picture=picture)
        db.add(user)
    db.commit()
    db.refresh(user)

    jwt_token = create_access_token({"sub": str(user.id)})
    response = RedirectResponse(url=FRONTEND_URL)
    response.set_cookie(
        key="access_token",
        value=jwt_token,
        httponly=True,
        samesite="lax",
        secure=False,  # set True in production with HTTPS
        max_age=config.jwt.expire_minutes * 60,
    )
    return response


@router.post("/logout")
def logout(current_user: User = Depends(get_current_user)):
    response = JSONResponse({"message": "Logged out"})
    response.delete_cookie("access_token")
    return response


@router.get("/me")
def me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "email": current_user.email,
        "name": current_user.name,
        "picture": current_user.picture,
    }
