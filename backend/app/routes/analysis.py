from fastapi import APIRouter

router = APIRouter(prefix="/analysis")

@router.get("/compare/{state}")
async def compare_elections(state: str):
    pass

@router.get("/ood/{state}")
async def get_ood_seats(state: str):
    pass