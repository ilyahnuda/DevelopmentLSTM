from ..Db import Database
from sqlalchemy.orm import Session

from ..models.Company import Company


class CompanyRep:
    __db = None

    def __init__(self, db: Database):
        self.__db = db

    def get_all(self):
        session = Session(bind=self.__db.get_engine())
        companies = session.query(Company).all()
        session.close()
        return companies

    def get_all_company_symbol(self):
        session = Session(bind=self.__db.get_engine())
        companies = session.query(Company.id, Company.symbol,Company.name).all()
        session.close()
        return companies
