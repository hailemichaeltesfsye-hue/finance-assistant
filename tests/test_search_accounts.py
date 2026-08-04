from db import search_accounts_db


def test_search_accounts_finds_existing_account():
    matches = search_accounts_db("Alice")
    assert len(matches) >= 1
    assert any(account.owner_name == "Alice" for account in matches)
