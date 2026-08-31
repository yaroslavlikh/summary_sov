from database.db import get_conn


def _normalize_name(name):
    return name.strip().lower()


def _normalize_username(username):
    username = username.strip()
    return username if username.startswith('@') else f'@{username}'


def get_group(chat_id, name):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT username FROM mention_groups WHERE chat_id = %s AND group_name = %s ORDER BY username",
            (chat_id, _normalize_name(name)),
        )
        rows = cursor.fetchall()
    return [row[0] for row in rows] or None


def list_groups(chat_id):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT group_name, COUNT(*) FROM mention_groups WHERE chat_id = %s "
            "GROUP BY group_name ORDER BY group_name",
            (chat_id,),
        )
        return cursor.fetchall()


def add_to_group(chat_id, name, usernames):
    name = _normalize_name(name)
    with get_conn() as conn:
        cursor = conn.cursor()
        for username in usernames:
            cursor.execute(
                "INSERT INTO mention_groups (chat_id, group_name, username) VALUES (%s, %s, %s) "
                "ON CONFLICT (chat_id, group_name, username) DO NOTHING",
                (chat_id, name, _normalize_username(username)),
            )
        conn.commit()


def remove_from_group(chat_id, name, usernames):
    name = _normalize_name(name)
    normalized = [_normalize_username(u) for u in usernames]
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM mention_groups WHERE chat_id = %s AND group_name = %s AND username = ANY(%s)",
            (chat_id, name, normalized),
        )
        conn.commit()


def delete_group(chat_id, name):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM mention_groups WHERE chat_id = %s AND group_name = %s",
            (chat_id, _normalize_name(name)),
        )
        conn.commit()
