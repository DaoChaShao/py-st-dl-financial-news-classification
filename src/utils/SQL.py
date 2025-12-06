#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/3 19:27
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   SQL.py
# @Desc     :
import sqlite3
from pathlib import Path
from sqlite3 import connect

from src.configs.cfg_base import CONFIG

WIDTH: int = 64


class SQLiteIII:
    """ SQLiteIII Class for Database """

    def __init__(self, table: str, col: str, db_path: Path | str | None = None):
        self._connection = None
        self._cursor = None
        self._db = str(db_path) if db_path else CONFIG.FILEPATHS.SQLITE
        self._table = table
        self._col = col

    def __enter__(self):
        if self._connection is None:
            self._connection = connect(self._db)
            self._cursor = self._connection.cursor()

            # Create a table in database
            self._cursor.execute(f"""
                create table if not exists {self._table} (
                    id integer primary key autoincrement,
                    {self._col} text not null,
                    created_at timestamp default current_timestamp
                )
            """)
            self._connection.commit()

            print("*" * WIDTH)
            print("SQLite III")
            print("-" * WIDTH)
            print(f"Congratulations! {self._table} table connected and initialised")
            print("*" * WIDTH)
            print()

        return self

    def connect(self):
        """ Connect to the database without using 'with' """
        return self.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

        return False

    def close(self):
        """ Close database connection manually """
        if self._connection:
            self._connection.close()
            self._connection = None
            self._cursor = None

    def insert(self, data: list):
        params = [(value,) for value in data]
        self._connection.executemany(f"insert into {self._table} ({self._col}) values (?)", params)
        self._connection.commit()

        print(f"{len(data)} rows inserted")

    def count(self) -> int:
        self._cursor.execute(f"select count(*) from {self._table}")

        return self._cursor.fetchone()[0]

    def get_all_data(self) -> list[str]:
        self._cursor.execute(f"select {self._col} from {self._table} order by id")

        return [row[0] for row in self._cursor.fetchall()]

    def remove_by_id(self, id: int):
        self._cursor.execute(f"delete from {self._table} where id = ?", (id,))
        self._connection.commit()

        deleted = self._cursor.rowcount
        if deleted > 0:
            print(f"Deleted {deleted} row(s)")
        else:
            print(f"No row found with id {id}")

    def search(self, keyword: str) -> list[str]:
        self._cursor.execute(
            f"select {self._col} from {self._table} where {self._col} like ?", (f"%{keyword}%" if keyword else "",)
        )

        return [row[0] for row in self._cursor.fetchall()]

    def clear(self):
        self._cursor.execute(f"delete from {self._table}")
        self._connection.commit()

    @property
    def table_name(self) -> str:
        """ Get the table name """
        return self._table

    @property
    def column_name(self) -> str:
        """ Get the column name """
        return self._col

    @property
    def database_path(self) -> str:
        """ Get the database file path """
        return self._db


if __name__ == "__main__":
    pass
