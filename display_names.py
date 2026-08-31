DISPLAY_NAMES = {
    "Beeeeeeeee1E": "Ярик Комаров",
    "geniymover": "Тимофей",
    "Goshanka200": "Игорь",
    "Ivanushka_Internetional": "Ваня",
    "p34108": "Влад",
    "presccode80": "Ivjenin",
    "Sahaaai": "Сахаи / Саша Резаков",
    "Super_rumit": "Гордей",
    "tigmen": "tigmen / Саша Тигмен",
    "yaroslavlikh": "Ярик Лихачев",
}

# Fallback for senders with no @username set on their Telegram account, keyed
# by the raw first_name Telegram gives us instead.
DISPLAY_NAMES_BY_FIRST_NAME = {
    "mosha": "Тимофей",
}


def resolve_display_name(username, fallback):
    if username and username.lstrip('@') in DISPLAY_NAMES:
        return DISPLAY_NAMES[username.lstrip('@')]
    if fallback and fallback.lower() in DISPLAY_NAMES_BY_FIRST_NAME:
        return DISPLAY_NAMES_BY_FIRST_NAME[fallback.lower()]
    return fallback
